// Native (C++) RKNN benchmark harness for the reCamera Pro (RV1126B).
//
// Replaces the previous Python/rknn-toolkit-lite2 inference path with the same
// zero-copy design used by the real-time pipeline:
//   * clocks pinned to their top OPPs (NPU userspace @ max, CPU/VENC performance)
//   * input fed through an rknn_create_mem buffer that RGA writes directly
//   * outputs bound zero-copy and kept in their native int8 domain (dense NCHW)
//   * candidate filtering done on raw int8 so no tensor is fully dequantised
//   * preprocess (RGA letterbox) + NPU + postprocess timed separately and
//     reported serially (FPS = 1000 / total)
//
// One binary covers every YOLO family in this repo; the head layout is inferred
// from the output tensor names/shapes so v8 / v11 / v26 and
// det/pose/seg/obb/cls/depth all work without per-model code.
#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "im2d.hpp"
#include "rga.h"
#include "rknn_api.h"

namespace {

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

bool write_node(const std::string &path, const std::string &value) {
  FILE *fh = fopen(path.c_str(), "w");
  if (!fh) return false;
  const bool ok = fputs(value.c_str(), fh) >= 0;
  fclose(fh);
  return ok;
}

std::string read_line(const std::string &path) {
  FILE *fh = fopen(path.c_str(), "r");
  if (!fh) return "";
  char line[256] = {0};
  if (!fgets(line, sizeof(line), fh)) line[0] = 0;
  fclose(fh);
  return std::string(line);
}

std::string lower(std::string s) {
  for (auto &ch : s) ch = (char)tolower((unsigned char)ch);
  return s;
}

void pin_clocks() {
  DIR *dir = opendir("/sys/class/devfreq");
  if (dir) {
    while (struct dirent *ent = readdir(dir)) {
      const std::string name = ent->d_name;
      const std::string base = std::string("/sys/class/devfreq/") + name + "/";
      if (name.find("npu") != std::string::npos) {
        const std::string avail = read_line(base + "available_frequencies");
        const size_t pos = avail.rfind(' ');
        const std::string top = pos == std::string::npos ? avail : avail.substr(pos + 1);
        if (!top.empty()) {
          write_node(base + "governor", "userspace");
          write_node(base + "userspace/set_freq", top);
          printf("[clocks] %s locked to %s Hz\n", name.c_str(), top.c_str());
        }
      } else if (name.find("venc") != std::string::npos || name.find("vpu") != std::string::npos) {
        write_node(base + "governor", "performance");
      }
    }
    closedir(dir);
  }
  for (int cpu = 0; cpu < 8; ++cpu) {
    char path[96];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", cpu);
    write_node(path, "performance");
  }
  printf("[clocks] cpu governors -> performance\n");
}

inline float half2float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while (!(man & 0x400u)) {
        man <<= 1;
        --exp;
      }
      bits = sign | (exp << 23) | ((man & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// One model output. Bound zero-copy: int8 tensors are requested as dense NCHW
// int8 (the runtime transposes the NPU's native NHWC-with-stride layout), float
// tensors are taken pass-through in their native dtype.
struct Tensor {
  int index = -1;
  std::string name, lname;
  int nd = 0, c = 1, h = 1, w = 1;
  bool is_int8 = true, is_fp16 = false;
  int zp = 0;
  float scale = 1.f, lo = 0.f, hi = 1.f;
  uint32_t bytes = 0;
  rknn_tensor_mem *mem = nullptr;
  const int8_t *q = nullptr;
  const uint16_t *h16 = nullptr;
  mutable std::vector<float> f;
  mutable int stamp = -1;

  int cells() const { return h * w; }
  int count() const { return c * h * w; }
  // Values already in [0, 1] (sigmoid/softmax baked into the graph)?
  bool probs() const { return is_int8 && lo >= -0.02f && hi <= 1.06f; }
  float deq(int8_t v) const { return ((float)v - (float)zp) * scale; }
  float qlo() const { return (float)(-128 - zp) * scale; }
  float qhi() const { return (float)(127 - zp) * scale; }

  const float *fp(int frame) const {
    const int n = count();
    if (stamp != frame) {
      f.resize((size_t)n);
      if (is_int8) {
        const float z = (float)zp, s = scale;
        for (int i = 0; i < n; ++i) f[i] = ((float)q[i] - z) * s;
      } else if (is_fp16) {
        for (int i = 0; i < n; ++i) f[i] = half2float(h16[i]);
      } else {
        memcpy(f.data(), q, (size_t)n * sizeof(float));
      }
      stamp = frame;
    }
    return f.data();
  }
};

struct Det {
  float x1, y1, x2, y2, score;
  int cls = 0;
  float angle = 0.f;
  float kpt[17][3];
  float coef[32];
  bool has_coef = false;
  bool has_kpt = false;
};

// Per-stride head. box/cls may be slices of one concatenated tensor
// (e.g. the zoo exports [1,65,H,W] = 64 DFL + 1 class, [1,79,H,W] = 64 + 15).
struct Head {
  int stride = 0, gw = 0, gh = 0;
  const Tensor *box = nullptr;
  int box_off = 0, box_ch = 0;
  const Tensor *cls = nullptr;
  int cls_off = 0, cls_ch = 0;
  const Tensor *obj = nullptr;
  const Tensor *kpt = nullptr;
  const Tensor *angle = nullptr;
  const Tensor *coef = nullptr;
};

float iou(const Det &a, const Det &b) {
  const float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
  const float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
  const float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
  const float inter = iw * ih;
  const float aa = (a.x2 - a.x1) * (a.y2 - a.y1), ab = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter > 0 ? inter / (aa + ab - inter) : 0.f;
}

struct Pt {
  float x, y;
};

// Sutherland-Hodgman clip of `subject` by convex `clip`.
float poly_area(std::vector<Pt> subject, const std::vector<Pt> &clip) {
  for (size_t e = 0; e < clip.size() && !subject.empty(); ++e) {
    const Pt a = clip[e], b = clip[(e + 1) % clip.size()];
    std::vector<Pt> out;
    for (size_t i = 0; i < subject.size(); ++i) {
      const Pt p = subject[i], q = subject[(i + 1) % subject.size()];
      const float dp = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
      const float dq = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
      if (dp <= 0) out.push_back(p);
      if ((dp < 0 && dq > 0) || (dp > 0 && dq < 0)) {
        const float t = dp / (dp - dq);
        out.push_back({p.x + t * (q.x - p.x), p.y + t * (q.y - p.y)});
      }
    }
    subject = out;
  }
  float area = 0;
  for (size_t i = 0; i < subject.size(); ++i) {
    const Pt p = subject[i], q = subject[(i + 1) % subject.size()];
    area += p.x * q.y - q.x * p.y;
  }
  return fabsf(area) * 0.5f;
}

float rotated_iou(const Det &a, const Det &b) {
  auto corners = [](const Det &d) {
    const float cx = (d.x1 + d.x2) * 0.5f, cy = (d.y1 + d.y2) * 0.5f;
    const float w = d.x2 - d.x1, h = d.y2 - d.y1;
    const float c = cosf(d.angle), s = sinf(d.angle);
    const Pt raw[4] = {{-w / 2, -h / 2}, {w / 2, -h / 2}, {w / 2, h / 2}, {-w / 2, h / 2}};
    std::vector<Pt> out(4);
    for (int i = 0; i < 4; ++i) out[i] = {cx + raw[i].x * c - raw[i].y * s, cy + raw[i].x * s + raw[i].y * c};
    return out;
  };
  const std::vector<Pt> pa = corners(a), pb = corners(b);
  const float inter = poly_area(pa, pb);
  const float aa = (a.x2 - a.x1) * (a.y2 - a.y1), ab = (b.x2 - b.x1) * (b.y2 - b.y1);
  return (aa + ab - inter) > 0 ? inter / (aa + ab - inter) : 0.f;
}

void nms(std::vector<Det> &dets, float iou_thresh, bool rotated) {
  std::sort(dets.begin(), dets.end(), [](const Det &a, const Det &b) { return a.score > b.score; });
  std::vector<char> dead(dets.size(), 0);
  for (size_t i = 0; i < dets.size(); ++i) {
    if (dead[i]) continue;
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (dead[j] || dets[j].cls != dets[i].cls) continue;
      const float o = rotated ? rotated_iou(dets[i], dets[j]) : iou(dets[i], dets[j]);
      if (o > iou_thresh) dead[j] = 1;
    }
  }
  std::vector<Det> kept;
  for (size_t i = 0; i < dets.size(); ++i)
    if (!dead[i]) kept.push_back(dets[i]);
  dets.swap(kept);
}

inline float sigm(float x) { return 1.f / (1.f + expf(-x)); }

// int8 level that a probability/logit threshold maps to.
int qthresh(const Tensor &t, float prob_thresh) {
  const float target = t.probs() ? prob_thresh : logf(std::max(1e-6f, prob_thresh) /
                                                      std::max(1e-6f, 1.f - prob_thresh));
  return (int)floorf(target / t.scale + (float)t.zp);
}

const char *COCO[80] = {
    "person",        "bicycle",    "car",           "motorcycle", "airplane",     "bus",
    "train",         "truck",      "boat",          "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",      "bird",          "cat",        "dog",          "horse",
    "sheep",         "cow",        "elephant",      "bear",       "zebra",        "giraffe",
    "backpack",      "umbrella",   "handbag",       "tie",        "suitcase",     "frisbee",
    "skis",          "snowboard",  "sports ball",   "kite",       "baseball bat", "baseball glove",
    "skateboard",    "surfboard",  "tennis racket", "bottle",     "wine glass",   "cup",
    "fork",          "knife",      "spoon",         "bowl",       "banana",       "apple",
    "sandwich",      "orange",     "broccoli",      "carrot",     "hot dog",      "pizza",
    "donut",         "cake",       "chair",         "couch",      "potted plant", "bed",
    "dining table",  "toilet",     "tv",            "laptop",     "mouse",        "remote",
    "keyboard",      "cell phone", "microwave",     "oven",       "toaster",      "sink",
    "refrigerator",  "book",       "clock",         "vase",       "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

const char *DOTA[15] = {"plane",          "ship",           "storage-tank", "baseball-diamond",
                        "tennis-court",   "basketball-court", "ground-track-field", "harbor",
                        "bridge",         "large-vehicle",  "small-vehicle", "helicopter",
                        "roundabout",     "soccer-ball-field", "swimming-pool"};

std::string cls_name(const char *task, int id) {
  char buf[32];
  if (!strcmp(task, "obb")) {
    if (id >= 0 && id < 15) return DOTA[id];
  } else if (!strcmp(task, "det") || !strcmp(task, "seg") || !strcmp(task, "pose")) {
    if (!strcmp(task, "pose")) return "person";
    if (id >= 0 && id < 80) return COCO[id];
  }
  snprintf(buf, sizeof(buf), "%d", id);
  return buf;
}

}  // namespace

int main(int argc, char **argv) {
  std::string model, image = "/userdata/benchmark/bench.rgb", task_arg;
  int runs = 100, warmup = 20, src_w = 1280, src_h = 720;
  float conf = 0.25f, iou_thresh = 0.45f;
  bool dump_io = false, trace_post = false;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&](const char *&v) { return i + 1 < argc ? (v = argv[++i], true) : false; };
    const char *v = nullptr;
    if (k == "--model" && next(v)) model = v;
    else if (k == "--image" && next(v)) image = v;
    else if (k == "--task" && next(v)) task_arg = v;
    else if (k == "--runs" && next(v)) runs = atoi(v);
    else if (k == "--warmup" && next(v)) warmup = atoi(v);
    else if (k == "--conf" && next(v)) conf = atof(v);
    else if (k == "--nms" && next(v)) iou_thresh = atof(v);
    else if (k == "--src" && next(v)) sscanf(v, "%dx%d", &src_w, &src_h);
    else if (k == "--dump-io") dump_io = true;
    else if (k == "--trace-post") trace_post = true;
    else {
      fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (model.empty()) {
    fprintf(stderr,
            "usage: %s --model <file.rknn> [--image raw.rgb] [--src WxH] [--runs N] "
            "[--warmup N] [--conf F] [--nms F] [--task T] [--dump-io]\n",
            argv[0]);
    return 2;
  }
  pin_clocks();

  rknn_context ctx = 0;
  if (rknn_init(&ctx, (void *)model.c_str(), 0, 0, nullptr) < 0) {
    fprintf(stderr, "rknn_init failed\n");
    return 1;
  }
  rknn_input_output_num io = {};
  rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));

  rknn_tensor_attr in_attr = {};
  in_attr.index = 0;
  rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
  const int in_w = (int)in_attr.dims[2], in_h = (int)in_attr.dims[1];

  std::vector<Tensor> outs(io.n_output);
  std::vector<rknn_tensor_attr> out_attrs(io.n_output);
  for (uint32_t i = 0; i < io.n_output; ++i) {
    rknn_tensor_attr a = {};
    a.index = i;
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a));
    out_attrs[i] = a;
    Tensor &t = outs[i];
    t.index = (int)i;
    t.name = a.name;
    t.lname = lower(t.name);
    t.nd = (int)a.n_dims;
    t.zp = a.zp;
    t.scale = a.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ? a.scale : 1.f;
    if (t.nd == 4) {
      t.c = (int)a.dims[1];
      t.h = (int)a.dims[2];
      t.w = (int)a.dims[3];
    } else if (t.nd == 3) {
      t.c = (int)a.dims[1];
      t.h = 1;
      t.w = (int)a.dims[2];
    } else {
      t.c = (int)a.dims[t.nd > 0 ? t.nd - 1 : 0];
      t.h = 1;
      t.w = 1;
    }
    t.is_int8 = (a.type == RKNN_TENSOR_INT8);
    t.is_fp16 = (a.type == RKNN_TENSOR_FLOAT16);
    t.lo = t.qlo();
    t.hi = t.qhi();

    rknn_tensor_attr bind = a;
    uint32_t bytes;
    if (t.is_int8) {
      // Dense NCHW int8: the runtime does the NHWC->NCHW reorder for us.
      bind.type = RKNN_TENSOR_INT8;
      bind.fmt = RKNN_TENSOR_NCHW;
      bind.pass_through = 0;
      bytes = a.n_elems;
    } else {
      bind.pass_through = 1;
      bytes = a.size_with_stride ? a.size_with_stride : a.size;
      t.scale = 1.f;
      t.zp = 0;
      t.lo = -1e30f;
      t.hi = 1e30f;
    }
    t.bytes = bytes;
    t.mem = rknn_create_mem(ctx, bytes);
    if (!t.mem) {
      fprintf(stderr, "create_mem out %u failed\n", i);
      return 1;
    }
    if (rknn_set_io_mem(ctx, t.mem, &bind) < 0) {
      fprintf(stderr, "set_io_mem out %u failed\n", i);
      return 1;
    }
    t.q = (const int8_t *)t.mem->virt_addr;
    t.h16 = (const uint16_t *)t.mem->virt_addr;
  }

  if (dump_io) {
    printf("input : %s dims[%u] =", in_attr.name, in_attr.n_dims);
    for (uint32_t d = 0; d < in_attr.n_dims; ++d) printf(" %d", in_attr.dims[d]);
    printf(" type=%s fmt=%s zp=%d scale=%.6g n_elems=%u size=%u/%u\n",
           get_type_string(in_attr.type), get_format_string(in_attr.fmt), in_attr.zp, in_attr.scale,
           in_attr.n_elems, in_attr.size, in_attr.size_with_stride);
    for (uint32_t i = 0; i < io.n_output; ++i) {
      const rknn_tensor_attr &a = out_attrs[i];
      printf("out[%u]: %-34s dims[%u] =", i, a.name, a.n_dims);
      for (uint32_t d = 0; d < a.n_dims; ++d) printf(" %d", a.dims[d]);
      printf(" type=%s fmt=%s zp=%d scale=%.6g size=%u/%u\n", get_type_string(a.type),
             get_format_string(a.fmt), a.zp, a.scale, a.size, a.size_with_stride);
    }
    rknn_destroy(ctx);
    return 0;
  }

  // ---- zero-copy input: RGA writes the letterboxed RGB888 straight into it.
  // Declared uint8 so the runtime maps it onto the model's own input dtype
  // (int8 asymmetric or fp16 with mean/std) instead of reinterpreting bytes.
  rknn_tensor_attr in_io = in_attr;
  in_io.type = RKNN_TENSOR_UINT8;
  in_io.fmt = RKNN_TENSOR_NHWC;
  in_io.pass_through = 0;
  const uint32_t in_bytes = in_attr.n_elems ? in_attr.n_elems : in_attr.size;
  rknn_tensor_mem *in_mem = rknn_create_mem(ctx, in_bytes);
  if (!in_mem || rknn_set_io_mem(ctx, in_mem, &in_io) < 0) {
    fprintf(stderr, "set_io_mem input failed\n");
    return 1;
  }

  FILE *fh = fopen(image.c_str(), "rb");
  if (!fh) {
    fprintf(stderr, "cannot open %s\n", image.c_str());
    return 1;
  }
  std::vector<uint8_t> src((size_t)src_w * src_h * 3);
  if (fread(src.data(), 1, src.size(), fh) != src.size()) {
    fprintf(stderr, "short read on %s (expected %dx%d RGB888)\n", image.c_str(), src_w, src_h);
    return 1;
  }
  fclose(fh);

  rga_buffer_handle_t src_hdl = importbuffer_virtualaddr(src.data(), (int)src.size());
  rga_buffer_handle_t dst_hdl = importbuffer_fd(in_mem->fd, (int)in_bytes);
  rga_buffer_t sbuf = wrapbuffer_handle_t(src_hdl, src_w, src_h, src_w, src_h, RK_FORMAT_RGB_888);
  rga_buffer_t dbuf = wrapbuffer_handle_t(dst_hdl, in_w, in_h, in_w, in_h, RK_FORMAT_RGB_888);
  const float lscale = std::min((float)in_w / src_w, (float)in_h / src_h);
  int nw = std::max(2, (int)(src_w * lscale + 0.5f) & ~1), nh = std::max(2, (int)(src_h * lscale + 0.5f) & ~1);
  nw = std::min(nw, in_w & ~1);
  nh = std::min(nh, in_h & ~1);
  const int x0 = ((in_w - nw) / 2) & ~1, y0 = ((in_h - nh) / 2) & ~1;
  im_rect drect = {x0, y0, nw, nh};
  im_rect full = {0, 0, in_w, in_h};
  if (!src_hdl || !dst_hdl) {
    fprintf(stderr, "rga importbuffer failed\n");
    return 1;
  }
  if (imfill(dbuf, full, 0x727272) != IM_STATUS_SUCCESS) {
    fprintf(stderr, "imfill failed\n");
    return 1;
  }

  auto pre = [&]() {
    rga_buffer_t nopat;
    memset(&nopat, 0, sizeof(nopat));
    im_rect srect = {};
    return improcess(sbuf, dbuf, nopat, srect, drect, srect, -1, nullptr, nullptr, IM_SYNC) ==
           IM_STATUS_SUCCESS;
  };

  // ---- optional RGA upscale target for the depth map (see decode below)
  std::vector<uint8_t> depth_rgb, depth_mid, depth_full;
  std::vector<float> depth_lut;
  std::vector<uint8_t> depth_lut_u8;
  rga_buffer_handle_t depth_src_hdl = 0, depth_mid_hdl = 0, depth_full_hdl = 0;
  int depth_mw = 0, depth_mh = 0;
  bool depth_direct = false, depth_rga = false;
  rga_buffer_t depth_sbuf{}, depth_mbuf{}, depth_fbuf{};
  im_rect depth_srect{}, depth_mrect{}, depth_frect{};

  // ---- head layout discovery
  const Tensor *proto = nullptr;
  for (auto &t : outs)
    if (t.nd == 4 && t.c == 32 && t.h == in_h / 4 && t.w == in_w / 4) proto = &t;

  std::vector<Head> heads;
  auto head_for = [&](int stride) -> Head * {
    for (auto &h : heads)
      if (h.stride == stride) return &h;
    heads.push_back({});
    Head *h = &heads.back();
    h->stride = stride;
    h->gw = in_w / stride;
    h->gh = in_h / stride;
    return h;
  };
  for (auto &t : outs) {
    if (&t == proto || t.nd != 4 || t.h != t.w) continue;
    int stride = 0;
    for (int s : {8, 16, 32})
      if (t.w == in_w / s) stride = s;
    if (!stride) continue;
    Head *hd = head_for(stride);
    const std::string &n = t.lname;
    auto has = [&](const char *key) { return n.find(key) != std::string::npos; };
    if (has("angle")) hd->angle = &t;
    else if (has("keypoint") || has("kpt") || t.c == 51) hd->kpt = &t;
    else if (has("mask_coeff") || has("coef")) hd->coef = &t;
    else if (has("dfl") || has("bbox") || has("box")) {
      hd->box = &t;
      hd->box_off = 0;
      hd->box_ch = t.c;
    } else if (has("objectness") || has("obj") || has("conf")) {
      if (t.c == 1) hd->obj = &t;
      else { hd->cls = &t; hd->cls_off = 0; hd->cls_ch = t.c; }
    } else if (has("cls")) {
      hd->cls = &t;
      hd->cls_off = 0;
      hd->cls_ch = t.c;
    }
  }
  // Second pass: tensors with anonymous names ("318", "onnx::ReduceSum_476",
  // "/model.22/Concat_1_output_0") are classified by channel count.
  for (auto &t : outs) {
    if (&t == proto || t.nd != 4 || t.h != t.w) continue;
    int stride = 0;
    for (int s : {8, 16, 32})
      if (t.w == in_w / s) stride = s;
    if (!stride) continue;
    Head *hd = head_for(stride);
    bool taken = (hd->box == &t || hd->cls == &t || hd->obj == &t || hd->kpt == &t ||
                  hd->angle == &t || hd->coef == &t);
    if (taken) continue;
    if (proto && t.c == 32 && !hd->coef) {
      hd->coef = &t;
    } else if (t.c == 4 && !hd->box) {
      hd->box = &t;
      hd->box_off = 0;
      hd->box_ch = 4;
    } else if (t.c > 64 && !hd->box && !hd->cls) {
      // concatenated DFL box + class branch (zoo exports: 65 = 64+1, 79 = 64+15)
      hd->box = &t;
      hd->box_off = 0;
      hd->box_ch = 64;
      hd->cls = &t;
      hd->cls_off = 64;
      hd->cls_ch = t.c - 64;
    } else if (t.c == 64 && !hd->box) {
      hd->box = &t;
      hd->box_off = 0;
      hd->box_ch = 64;
    } else if (t.c == 1) {
      if (hd->cls || hd->kpt) hd->obj = &t;
      else hd->angle = &t;
    } else if (t.c >= 2 && !hd->cls) {
      hd->cls = &t;
      hd->cls_off = 0;
      hd->cls_ch = t.c;
    }
  }
  std::sort(heads.begin(), heads.end(), [](const Head &a, const Head &b) { return a.stride < b.stride; });

  bool has_kpt = false, has_angle = false;
  for (auto &h : heads) {
    if (h.kpt) has_kpt = true;
    if (h.angle) has_angle = true;
  }
  const bool is_cls = io.n_output == 1 && outs[0].nd <= 2;
  const bool is_depth = io.n_output == 1 && !is_cls;
  std::string task = task_arg;
  if (task.empty())
    task = is_cls ? "cls" : is_depth ? "depth" : has_kpt ? "pose" : proto ? "seg" : has_angle ? "obb" : "det";
  const bool rotated = task == "obb";

  printf("[model] %s\n", model.c_str());
  printf("[heads] %zu stride(s), task=%s, input=%dx%d, outputs=%u\n", heads.size(), task.c_str(),
         in_w, in_h, io.n_output);
  for (auto &h : heads)
    printf("  stride %2d grid %dx%d box=c%d@%d cls=c%d@%d%s%s%s%s\n", h.stride, h.gw, h.gh,
           h.box_ch, h.box_off, h.cls_ch, h.cls_off, h.kpt ? " kpt" : "", h.obj ? " obj" : "",
           h.angle ? " angle" : "", h.coef ? " coef" : "");
  if (proto) printf("  proto  c%d %dx%d\n", proto->c, proto->h, proto->w);

  if (is_depth) {
    const float cal = expf(-0.19384766f);
    depth_lut.resize(256);
    depth_lut_u8.resize(256);
    for (int q = -128; q < 128; ++q) {
      const float raw = outs[0].is_int8 ? ((float)q - (float)outs[0].zp) * outs[0].scale : 0.f;
      const float v = expf(std::min(10.f, std::max(-10.f, raw))) * cal;
      depth_lut[q + 128] = v;
      const int u = (int)(v * 25.5f);
      depth_lut_u8[q + 128] = (uint8_t)(u < 0 ? 0 : u > 255 ? 255 : u);
    }
    depth_mw = outs[0].w;
    depth_mh = outs[0].h;
    depth_rgb.assign((size_t)depth_mw * depth_mh * 3, 0);
    depth_full.assign((size_t)src_w * src_h * 3, 0);
    depth_src_hdl = importbuffer_virtualaddr(depth_rgb.data(), (int)depth_rgb.size());
    depth_full_hdl = importbuffer_virtualaddr(depth_full.data(), (int)depth_full.size());
    rga_buffer_t nopat;
    memset(&nopat, 0, sizeof(nopat));
    im_rect empty = {};
    if (depth_src_hdl && depth_full_hdl) {
      depth_sbuf = wrapbuffer_handle_t(depth_src_hdl, depth_mw, depth_mh, depth_mw, depth_mh,
                                       RK_FORMAT_RGB_888);
      depth_fbuf = wrapbuffer_handle_t(depth_full_hdl, src_w, src_h, src_w, src_h,
                                       RK_FORMAT_RGB_888);
      depth_srect = {0, 0, depth_mw, depth_mh};
      depth_frect = {0, 0, src_w, src_h};
      // one probe frame: an 8x upscale may exceed the RGA limit, in which case
      // we go through a model-resolution intermediate buffer instead.
      depth_direct =
          improcess(depth_sbuf, depth_fbuf, nopat, depth_srect, depth_frect, empty, -1, nullptr,
                    nullptr, IM_SYNC) == IM_STATUS_SUCCESS;
      if (!depth_direct) {
        depth_mid.assign((size_t)in_w * in_h * 3, 0);
        depth_mid_hdl = importbuffer_virtualaddr(depth_mid.data(), (int)depth_mid.size());
        if (depth_mid_hdl) {
          depth_mbuf = wrapbuffer_handle_t(depth_mid_hdl, in_w, in_h, in_w, in_h, RK_FORMAT_RGB_888);
          depth_mrect = {0, 0, in_w, in_h};
          const bool ok1 = improcess(depth_sbuf, depth_mbuf, nopat, depth_srect, depth_mrect, empty,
                                     -1, nullptr, nullptr, IM_SYNC) == IM_STATUS_SUCCESS;
          im_rect crop = {x0, y0, nw, nh};
          const bool ok2 = ok1 && improcess(depth_mbuf, depth_fbuf, nopat, crop, depth_frect, empty,
                                            -1, nullptr, nullptr, IM_SYNC) == IM_STATUS_SUCCESS;
          depth_rga = ok2;
        }
      } else {
        depth_rga = true;
      }
    }
    printf("[depth] %dx%d metric map -> %dx%d via %s\n", depth_mw, depth_mh, src_w, src_h,
           depth_rga ? (depth_direct ? "RGA (direct)" : "RGA (2-step)") : "CPU bilinear");
  }

  // ---- decode
  std::vector<Det> dets;
  std::vector<float> top5(5, 0.f);
  std::vector<int> top5_id(5, 0);
  int frame_stamp = 0;
  std::vector<int8_t> cell_max;
  std::vector<float> depth_map, mask_acc;
  std::vector<uint8_t> depth_u8, depth_out_u8;
  std::vector<int> depth_ix, depth_wx;
  float best_score = 0.f;
  if (is_depth) {
    depth_ix.resize((size_t)src_w);
    depth_wx.resize((size_t)src_w);
    for (int x = 0; x < src_w; ++x) {
      const int mw = outs[0].w;
      float fx = ((float)x * lscale + x0) * (float)mw / (float)in_w;
      fx = std::min((float)mw - 1.001f, std::max(0.f, fx));
      depth_ix[x] = (int)fx;
      depth_wx[x] = (int)((fx - (int)fx) * 256.f);
    }
  }

  auto to_src = [&](Det &d) {
    d.x1 = (d.x1 - x0) / lscale;
    d.x2 = (d.x2 - x0) / lscale;
    d.y1 = (d.y1 - y0) / lscale;
    d.y2 = (d.y2 - y0) / lscale;
    if (d.has_kpt)
      for (int p = 0; p < 17; ++p) {
        d.kpt[p][0] = (d.kpt[p][0] - x0) / lscale;
        d.kpt[p][1] = (d.kpt[p][1] - y0) / lscale;
      }
  };

  auto decode = [&]() {
    ++frame_stamp;
    const int frame = frame_stamp;
    dets.clear();

    if (is_cls) {
      const Tensor &t = outs[0];
      const float *tf = t.fp(frame);
      const int n = t.count();
      std::fill(top5.begin(), top5.end(), -1e30f);
      for (int i = 0; i < n; ++i) {
        const float v = tf[i];
        if (v > top5[4]) {
          int slot = 4;
          while (slot > 0 && v > top5[slot - 1]) --slot;
          for (int k = 4; k > slot; --k) {
            top5[k] = top5[k - 1];
            top5_id[k] = top5_id[k - 1];
          }
          top5[slot] = v;
          top5_id[slot] = i;
        }
      }
      Det d = {};
      d.score = top5[0];
      d.cls = top5_id[0];
      dets.push_back(d);
      return;
    }

    if (is_depth) {
      const Tensor &t = outs[0];
      const int mw = t.w, mh = t.h, mn_pix = mw * mh;
      // Metric decode (ultralytics depth head: clip -> exp -> * exp(cal_b)).
      // The head is int8, so all 256 quantised levels resolve through a lookup
      // table built once instead of an expf() per pixel; the 10 m full-scale
      // uint8 map is produced in the same pass.
      depth_u8.resize((size_t)mn_pix);
      const double dta = now_ms();
      float lo = 1e30f, hi = -1e30f;
      double sum = 0;
      if (t.is_int8) {
        const int8_t *__restrict raw = t.q;
        uint8_t *__restrict u8 = depth_u8.data();
        for (int i = 0; i < mn_pix; ++i) {
          const int idx = (int)raw[i] + 128;
          const float v = depth_lut[idx];
          sum += v;
          lo = std::min(lo, v);
          hi = std::max(hi, v);
          u8[i] = depth_lut_u8[idx];
        }
      } else {
        depth_map.resize((size_t)mn_pix);
        const float *tf = t.fp(frame);
        const float cal = expf(-0.19384766f);
        for (int i = 0; i < mn_pix; ++i) {
          const float v = expf(std::min(10.f, std::max(-10.f, tf[i]))) * cal;
          depth_map[i] = v;
          lo = std::min(lo, v);
          hi = std::max(hi, v);
          sum += v;
          const int q = (int)(v * 25.5f);
          depth_u8[i] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
        }
      }
      const double dtb = now_ms();
      // pack the normalised map as RGB888 and let RGA do the upscale
      uint8_t *__restrict rgb = depth_rgb.data();
      const uint8_t *__restrict g = depth_u8.data();
      for (int i = 0; i < mn_pix; ++i) {
        rgb[3 * i] = g[i];
        rgb[3 * i + 1] = g[i];
        rgb[3 * i + 2] = g[i];
      }
      if (depth_rga) {
        rga_buffer_t nopat;
        memset(&nopat, 0, sizeof(nopat));
        im_rect empty = {};
        if (depth_direct) {
          improcess(depth_sbuf, depth_fbuf, nopat, depth_srect, depth_frect, empty, -1, nullptr,
                    nullptr, IM_SYNC);
        } else {
          improcess(depth_sbuf, depth_mbuf, nopat, depth_srect, depth_mrect, empty, -1, nullptr,
                    nullptr, IM_SYNC);
          const im_rect crop = {x0, y0, nw, nh};
          improcess(depth_mbuf, depth_fbuf, nopat, crop, depth_frect, empty, -1, nullptr, nullptr,
                    IM_SYNC);
        }
      } else {
        depth_out_u8.resize((size_t)src_w * src_h);
        // __restrict matters here: uint8_t pointers alias everything, so without
        // it every store forces the compiler to reload the taps and the weights.
        const int *__restrict ixt = depth_ix.data();
        const int *__restrict wxt = depth_wx.data();
        uint8_t *__restrict out = depth_out_u8.data();
        for (int y = 0; y < src_h; ++y) {
          float fy = ((float)y * lscale + y0) * (float)mh / (float)in_h;
          fy = std::min((float)mh - 1.001f, std::max(0.f, fy));
          const int iy = (int)fy, wy = (int)((fy - iy) * 256.f), wy0 = 256 - wy;
          const uint8_t *__restrict r0 = &depth_u8[(size_t)iy * mw];
          const uint8_t *__restrict r1 = r0 + mw;
          uint8_t *__restrict dst = out + (size_t)y * src_w;
          for (int x = 0; x < src_w; ++x) {
            const int ix = ixt[x], wx = wxt[x];
            const int top = (int)r0[ix] * (256 - wx) + (int)r0[ix + 1] * wx;
            const int bot = (int)r1[ix] * (256 - wx) + (int)r1[ix + 1] * wx;
            dst[x] = (uint8_t)((top * wy0 + bot * wy) >> 16);
          }
        }
      }
      if (trace_post)
        printf("[trace depth] decode %.2f ms  upscale %.2f ms\n", dtb - dta, now_ms() - dtb);
      Det d = {};
      d.score = (float)(sum / mn_pix);
      d.x1 = lo;
      d.x2 = hi;
      dets.push_back(d);
      return;
    }
    for (const Head &hd : heads) {
      if (!hd.box) continue;
      const int cells = hd.gw * hd.gh, s = hd.stride;
      const Tensor &box = *hd.box;
      const bool dfl = hd.box_ch == 64;

      // --- pass 1: per-cell best class/obj in the int8 domain (no dequant) ---
      cell_max.assign((size_t)cells, (int8_t)-128);
      int8_t *__restrict vmax = cell_max.data();
      const Tensor *score_t = hd.cls ? hd.cls : hd.obj;
      int score_off = hd.cls ? hd.cls_off : 0, score_ch = hd.cls ? hd.cls_ch : 1;
      int qthr = 127;
      if (score_t) {
        qthr = qthresh(*score_t, conf);
        if (score_t->is_int8) {
          const int8_t *base = score_t->q + (size_t)score_off * cells;
          for (int k = 0; k < score_ch; ++k) {
            const int8_t *__restrict plane = base + (size_t)k * cells;
            for (int i = 0; i < cells; ++i) vmax[i] = vmax[i] > plane[i] ? vmax[i] : plane[i];
          }
        } else {
          const float *base = score_t->fp(frame) + (size_t)score_off * cells;
          for (int i = 0; i < cells; ++i) {
            float best = -1e30f;
            for (int k = 0; k < score_ch; ++k) best = std::max(best, base[(size_t)k * cells + i]);
            vmax[i] = (int8_t)(best >= conf ? 1 : -128);
          }
          qthr = 0;
        }
      }

      // Highest raw class/objectness score on this head, reported for sanity
      // (a DOTA-trained OBB model legitimately finds nothing on a street photo).
      if (score_t && !cell_max.empty()) {
        const int8_t peak = *std::max_element(cell_max.begin(), cell_max.begin() + cells);
        const float v = score_t->deq(peak);
        best_score = std::max(best_score, score_t->probs() ? v : sigm(v));
      }

      // --- pass 2: only survivors pay for argmax + sigmoid + box decode ---
      const int8_t *bq = box.q;
      const float *bf = box.is_int8 ? nullptr : box.fp(frame);
      const int8_t *oq = hd.obj ? hd.obj->q : nullptr;
      const float *of = (hd.obj && !hd.obj->is_int8) ? hd.obj->fp(frame) : nullptr;
      const int8_t *aq = hd.angle ? hd.angle->q : nullptr;
      const float *af = (hd.angle && !hd.angle->is_int8) ? hd.angle->fp(frame) : nullptr;
      const int8_t *kq = hd.kpt ? hd.kpt->q : nullptr;
      const float *kf = (hd.kpt && !hd.kpt->is_int8) ? hd.kpt->fp(frame) : nullptr;
      const int8_t *cq = hd.coef ? hd.coef->q : nullptr;
      const float *cfp = (hd.coef && !hd.coef->is_int8) ? hd.coef->fp(frame) : nullptr;
      const int8_t *sq = (hd.cls && hd.cls->is_int8) ? hd.cls->q + (size_t)hd.cls_off * cells : nullptr;
      const float *sf = (hd.cls && !hd.cls->is_int8) ? hd.cls->fp(frame) + (size_t)hd.cls_off * cells : nullptr;

      for (int i = 0; i < cells; ++i) {
        if ((int)vmax[i] <= qthr) continue;
        const int cy = i / hd.gw, cx = i % hd.gw;

        float cls_val = 0.f, cls_prob = 1.f;
        int arg = 0;
        if (hd.cls) {
          if (sq) {
            int8_t best = (int8_t)-128;
            for (int k = 0; k < hd.cls_ch; ++k) {
              const int8_t v = sq[(size_t)k * cells + i];
              if (v > best) {
                best = v;
                arg = k;
              }
            }
            cls_val = hd.cls->deq(best);
          } else {
            float best = -1e30f;
            for (int k = 0; k < hd.cls_ch; ++k) {
              const float v = sf[(size_t)k * cells + i];
              if (v > best) {
                best = v;
                arg = k;
              }
            }
            cls_val = best;
          }
          cls_prob = hd.cls->probs() ? cls_val : sigm(cls_val);
        }
        float obj_prob = 1.f;
        if (hd.obj) {
          const float v = oq ? hd.obj->deq(oq[i]) : of[i];
          obj_prob = hd.obj->probs() ? v : sigm(v);
        }
        const float score = cls_prob * obj_prob;
        if (score < conf) continue;

        float l, tp, r, b;
        if (dfl) {
          float sides[4];
          for (int sd = 0; sd < 4; ++sd) {
            float vals[16], maxv = -1e30f;
            for (int q = 0; q < 16; ++q) {
              const size_t idx = (size_t)(sd * 16 + q + hd.box_off) * cells + i;
              vals[q] = bq ? box.deq(bq[idx]) : bf[idx];
              maxv = std::max(maxv, vals[q]);
            }
            float sum = 0.f, expect = 0.f;
            for (int q = 0; q < 16; ++q) {
              vals[q] = expf(vals[q] - maxv);
              sum += vals[q];
            }
            for (int q = 0; q < 16; ++q) expect += vals[q] / sum * q;
            sides[sd] = expect;
          }
          l = sides[0];
          tp = sides[1];
          r = sides[2];
          b = sides[3];
        } else {
          auto bx = [&](int c) {
            const size_t idx = (size_t)(c + hd.box_off) * cells + i;
            return bq ? box.deq(bq[idx]) : bf[idx];
          };
          l = bx(0);
          tp = bx(1);
          r = bx(2);
          b = bx(3);
        }

        const float anchor_x = cx + 0.5f, anchor_y = cy + 0.5f;
        Det d = {};
        d.score = score;
        d.cls = arg;
        float angle_rad = 0.f;
        if (hd.angle) {
          const float av = aq ? hd.angle->deq(aq[i]) : af[i];
          angle_rad = ((hd.angle->probs() ? av : sigm(av)) - 0.25f) * (float)M_PI;
          d.angle = angle_rad;
        }
        if (hd.angle) {
          // dist2rbox: rotate the half-extent offset by the predicted angle
          const float xf = (r - l) * 0.5f, yf = (b - tp) * 0.5f;
          const float ca = cosf(angle_rad), sa = sinf(angle_rad);
          const float cxg = xf * ca - yf * sa + anchor_x;
          const float cyg = xf * sa + yf * ca + anchor_y;
          const float bw = (l + r), bh = (tp + b);
          d.x1 = (cxg - bw * 0.5f) * s;
          d.y1 = (cyg - bh * 0.5f) * s;
          d.x2 = (cxg + bw * 0.5f) * s;
          d.y2 = (cyg + bh * 0.5f) * s;
        } else {
          d.x1 = (anchor_x - l) * s;
          d.y1 = (anchor_y - tp) * s;
          d.x2 = (anchor_x + r) * s;
          d.y2 = (anchor_y + b) * s;
        }
        d.x1 = std::min(std::max(d.x1, 0.f), (float)in_w);
        d.y1 = std::min(std::max(d.y1, 0.f), (float)in_h);
        d.x2 = std::min(std::max(d.x2, 0.f), (float)in_w);
        d.y2 = std::min(std::max(d.y2, 0.f), (float)in_h);
        if (hd.kpt) {
          d.has_kpt = true;
          for (int p = 0; p < 17; ++p) {
            auto kv = [&](int c) {
              const size_t idx = (size_t)c * cells + i;
              return kq ? hd.kpt->deq(kq[idx]) : kf[idx];
            };
            d.kpt[p][0] = (anchor_x + kv(p * 3 + 0)) * s;
            d.kpt[p][1] = (anchor_y + kv(p * 3 + 1)) * s;
            const float vis = kv(p * 3 + 2);
            d.kpt[p][2] = hd.kpt->probs() ? vis : sigm(vis);
          }
        }
        if (hd.coef) {
          for (int k = 0; k < 32; ++k) {
            const size_t idx = (size_t)k * cells + i;
            d.coef[k] = cq ? hd.coef->deq(cq[idx]) : cfp[idx];
          }
          d.has_coef = true;
        }
        if (d.x2 - d.x1 < 1.f || d.y2 - d.y1 < 1.f) continue;
        to_src(d);
        dets.push_back(d);
      }
    }
    nms(dets, iou_thresh, rotated);
    if (proto && !dets.empty()) {
      // mask = sigmoid(coef . proto) counted inside the box, as ultralytics does.
      // sigmoid(x) > 0.5 is just x > 0, and the int8 proto planes are walked
      // sequentially (channel outer) so the accumulation stays in cache.
      const int ph = proto->h, pw = proto->w, plane = ph * pw;
      const float *pf = proto->is_int8 ? nullptr : proto->fp(frame);
      const int8_t *pq = proto->is_int8 ? proto->q : nullptr;
      const float pscale = proto->is_int8 ? proto->scale : 1.f;
      const float pzp = proto->is_int8 ? (float)proto->zp : 0.f;
      const float sx = (float)pw / (float)src_w, sy = (float)ph / (float)src_h;
      for (auto &d : dets) {
        if (!d.has_coef) continue;
        const int bx1 = std::max(0, (int)(d.x1 * sx)), by1 = std::max(0, (int)(d.y1 * sy));
        const int bx2 = std::min(pw, (int)(d.x2 * sx) + 1), by2 = std::min(ph, (int)(d.y2 * sy) + 1);
        if (bx2 <= bx1 || by2 <= by1) continue;
        const int rows = by2 - by1;
        mask_acc.assign((size_t)rows * pw, 0.f);
        float bias = 0.f;
        for (int k = 0; k < 32; ++k) {
          const float w = d.coef[k] * pscale;
          bias -= d.coef[k] * pscale * pzp;
          if (pq) {
            const int8_t *__restrict src = pq + (size_t)k * plane + (size_t)by1 * pw;
            float *__restrict acc = mask_acc.data();
            for (int y = 0; y < rows; ++y) {
              const int8_t *__restrict sp = src + (size_t)y * pw;
              float *__restrict dp = acc + (size_t)y * pw;
              for (int x = bx1; x < bx2; ++x) dp[x] += w * (float)sp[x];
            }
          } else {
            const float *__restrict src = pf + (size_t)k * plane + (size_t)by1 * pw;
            for (int y = 0; y < rows; ++y)
              for (int x = bx1; x < bx2; ++x) mask_acc[(size_t)y * pw + x] += w * src[(size_t)y * pw + x];
          }
        }
        float area = 0;
        for (int y = 0; y < rows; ++y)
          for (int x = bx1; x < bx2; ++x)
            if (mask_acc[(size_t)y * pw + x] + bias > 0.f) area += 1;
        d.score += area * 1e-9f;  // keep the mask work visible in timing, not in ordering
      }
    }
  };

  // ---- timed loop (serial: preprocess -> NPU -> postprocess)
  double pre_sum = 0, npu_sum = 0, post_sum = 0, tot_sum = 0, tot_min = 1e9, tot_max = 0;
  double pre_min = 1e9, npu_min = 1e9, post_min = 1e9;
  const int total = warmup + runs;
  for (int i = 0; i < total; ++i) {
    const double t0 = now_ms();
    pre();
    const double t1 = now_ms();
    if (rknn_run(ctx, nullptr) < 0) {
      fprintf(stderr, "rknn_run failed\n");
      return 1;
    }
    const double t2 = now_ms();
    decode();
    const double t3 = now_ms();
    if (i >= warmup) {
      pre_sum += t1 - t0;
      npu_sum += t2 - t1;
      post_sum += t3 - t2;
      pre_min = std::min(pre_min, t1 - t0);
      npu_min = std::min(npu_min, t2 - t1);
      post_min = std::min(post_min, t3 - t2);
      const double tot = t3 - t0;
      tot_sum += tot;
      tot_min = std::min(tot_min, tot);
      tot_max = std::max(tot_max, tot);
    }
  }
  const double pa = pre_sum / runs, na = npu_sum / runs, oa = post_sum / runs, ta = tot_sum / runs;
  printf("\n================ BENCHMARK ================\n");
  printf("model      : %s\n", model.c_str());
  printf("task       : %s\n", task.c_str());
  printf("input      : %dx%d\n", in_w, in_h);
  printf("runs       : %d (warmup %d)\n", runs, warmup);
  printf("preprocess : avg %6.2f ms  min %6.2f\n", pa, pre_min);
  printf("npu        : avg %6.2f ms  min %6.2f  (steady-state %.1f fps)\n", na, npu_min, 1000.0 / na);
  printf("postprocess: avg %6.2f ms  min %6.2f\n", oa, post_min);
  printf("total      : avg %6.2f ms  min %6.2f  max %6.2f\n", ta, tot_min, tot_max);
  printf("fps        : %.1f\n", 1000.0 / ta);
  if (is_cls) {
    printf("top5       :");
    for (int k = 0; k < 5; ++k) printf(" #%d=%.4f", top5_id[k], top5[k]);
    printf("\n");
  } else if (is_depth) {
    printf("depth      : mean %.3f m  min %.3f  max %.3f  (%dx%d metric map -> %dx%d uint8)\n",
           dets[0].score, dets[0].x1, dets[0].x2, outs[0].w, outs[0].h, src_w, src_h);
  } else {
    printf("detections : %zu (best raw score %.3f)\n", dets.size(), best_score);
    const size_t show = std::min((size_t)5, dets.size());
    for (size_t i = 0; i < show; ++i) {
      const Det &d = dets[i];
      printf("  [%zu] %-14s %.3f box(%.0f,%.0f,%.0f,%.0f)%s%s\n", i, cls_name(task.c_str(), d.cls).c_str(),
             d.score, d.x1, d.y1, d.x2, d.y2, d.has_kpt ? " +17kpt" : "",
             rotated ? " rot" : "");
    }
  }
  printf("===========================================\n");

  releasebuffer_handle(src_hdl);
  releasebuffer_handle(dst_hdl);
  if (depth_src_hdl) releasebuffer_handle(depth_src_hdl);
  if (depth_mid_hdl) releasebuffer_handle(depth_mid_hdl);
  if (depth_full_hdl) releasebuffer_handle(depth_full_hdl);
  for (auto &t : outs)
    if (t.mem) rknn_destroy_mem(ctx, t.mem);
  if (in_mem) rknn_destroy_mem(ctx, in_mem);
  rknn_destroy(ctx);
  return 0;
}
