"""Model contracts. Size is width × height. Audio size is the number of PCM samples."""

from typing import NamedTuple


class Model(NamedTuple):
    file: str
    name: str
    task: str
    corpus: str
    size: tuple
    pixels: str
    decoder: str
    layout: str = "nchw"


MODELS = [
    Model(
        "object_detection_nanodet_2022nov",
        "NanoDet-Plus-m",
        "目标检测",
        "coco",
        (416, 416),
        "nanodet",
        "nanodet",
    ),
    Model(
        "object_detection_yolox_2022nov",
        "YOLOX-s",
        "目标检测",
        "coco",
        (640, 640),
        "rgb",
        "yolox",
    ),
    Model("gghl-dota", "GGHL", "旋转目标检测", "ships", (512, 512), "rgb01", "gghl"),
    Model(
        "yolov5s-seg",
        "YOLOv5s-seg",
        "实例分割",
        "voc",
        (640, 640),
        "rgb01",
        "instances",
    ),
    Model(
        "human_segmentation_pphumanseg_2023mar",
        "PP-HumanSeg",
        "人像分割",
        "voc",
        (192, 192),
        "rgb11",
        "semantic",
    ),
    Model(
        "pp_liteseg_cityscapes",
        "PP-LiteSeg",
        "场景语义分割",
        "street",
        (512, 512),
        "imagenet",
        "semantic",
    ),
    Model(
        "depth_midas_v21_small_256_fp16",
        "MiDaS v2.1 Small",
        "单目相对深度",
        "depth",
        (256, 256),
        "rgb8",
        "depth",
        "nhwc",
    ),
    Model(
        "person_detection_mediapipe_2023mar",
        "BlazePose Detector",
        "人体检测",
        "pose",
        (224, 224),
        "rgb11",
        "person",
    ),
    Model(
        "pose_estimation_mediapipe_2023mar",
        "BlazePose Landmarks",
        "人体姿态",
        "pose",
        (256, 256),
        "rgb01",
        "pose",
        "nhwc",
    ),
    Model(
        "palm_detection_mediapipe_2023feb",
        "BlazePalm",
        "手掌检测",
        "hands",
        (192, 192),
        "rgb01",
        "palm",
        "nhwc",
    ),
    Model(
        "handpose_estimation_mediapipe_2023feb",
        "MediaPipe Hand Landmarks",
        "手部关键点",
        "hands",
        (224, 224),
        "rgb01",
        "hand",
        "nhwc",
    ),
    Model(
        "object_tracking_vittrack_2023sep",
        "VitTrack",
        "单目标跟踪",
        "track",
        (256, 256),
        "track",
        "track",
    ),
    Model(
        "person_reid_youtu_2021nov",
        "Youtu ReID",
        "行人重识别",
        "reid",
        (128, 256),
        "imagenet",
        "embedding",
    ),
    Model(
        "face_detection_yunet_2023mar",
        "YuNet",
        "人脸检测",
        "faces",
        (640, 640),
        "bgr",
        "yunet",
    ),
    Model(
        "RetinaFace_mobile320",
        "RetinaFace MobileNet",
        "人脸检测",
        "faces",
        (320, 320),
        "retina",
        "retina",
    ),
    Model(
        "scrfd500m_640_fp16",
        "SCRFD-500M",
        "人脸检测",
        "faces",
        (640, 640),
        "rgb8",
        "scrfd",
        "nhwc",
    ),
    Model(
        "face_recognition_sface_2021dec",
        "SFace",
        "人脸特征",
        "portrait",
        (112, 112),
        "rgb",
        "embedding",
    ),
    Model(
        "arcface_mbf_fp16",
        "ArcFace MobileFaceNet",
        "人脸特征",
        "portrait",
        (112, 112),
        "rgb8",
        "embedding",
        "nhwc",
    ),
    Model(
        "facial_expression_recognition_mobilefacenet_2022july",
        "MobileFaceNet FER",
        "表情识别",
        "portrait",
        (112, 112),
        "rgb11",
        "class",
    ),
    Model(
        "ediffiqa_tiny_jun2024",
        "EDiffIQA Tiny",
        "人脸质量",
        "quality",
        (112, 112),
        "rgb11",
        "scalar",
    ),
    Model(
        "face_landmark_fp16",
        "MediaPipe FaceMesh",
        "人脸关键点",
        "mesh",
        (192, 192),
        "rgb8",
        "mesh",
        "nhwc",
    ),
    Model(
        "liveness_minifasnet_fp16",
        "MiniFASNet V2",
        "人脸防伪",
        "spoof27",
        (80, 80),
        "bgr8",
        "class",
        "nhwc",
    ),
    Model(
        "liveness_minifasnet_v1se_fp16",
        "MiniFASNet V1SE",
        "人脸防伪",
        "spoof40",
        (80, 80),
        "bgr8",
        "class",
        "nhwc",
    ),
    Model(
        "license_plate_detection_lpd_yunet_2023mar",
        "LPD-YuNet",
        "车牌四角检测",
        "vehicle",
        (320, 240),
        "bgr",
        "platebox",
    ),
    Model("lprnet", "LPRNet", "车牌字符识别", "plate", (94, 24), "bgr11", "plate"),
    Model(
        "text_detection_en_ppocrv3_2023may",
        "PP-OCRv3 EN Det",
        "文字定位",
        "textline",
        (320, 320),
        "db",
        "textboxes",
    ),
    Model(
        "text_detection_cn_ppocrv3_2023may",
        "PP-OCRv3 CN Det",
        "文字定位",
        "textline",
        (320, 320),
        "db",
        "textboxes",
    ),
    Model(
        "text_recognition_CRNN_EN_2021sep",
        "CRNN EN",
        "文字识别",
        "words",
        (100, 32),
        "gray11",
        "word",
    ),
    Model("espcn-x3", "ESPCN ×3", "图像超分辨率", "sr", (224, 224), "y01", "sr"),
    Model("yamnet_3s", "YAMNet", "声音事件分类", "sound", (48000,), "pcm", "sound"),
    Model(
        "edge_detection_dexined_2024sep",
        "DexiNed",
        "边缘检测",
        "edges",
        (640, 480),
        "edge",
        "edge",
    ),
    Model(
        "inpainting_lama_2025jan",
        "LaMa",
        "图像修复",
        "inpaint",
        (512, 512),
        "inpaint",
        "inpaint",
    ),
    Model(
        "deblurring_nafnet_2025may",
        "NAFNet",
        "运动去模糊",
        "deblur",
        (384, 384),
        "rgb01",
        "restore",
    ),
    Model(
        "nemo-ctc-6s",
        "NeMo Conformer Small CTC",
        "语音转文字",
        "speech",
        (96000,),
        "fbank",
        "subword",
    ),
    Model(
        "citrinet-6s",
        "NeMo Citrinet-512",
        "语音转文字",
        "speech",
        (96000,),
        "fbank",
        "subword",
    ),
    Model(
        "wav2vec2-6s",
        "Wav2Vec2 Base 960h · 6s",
        "语音转文字",
        "speech",
        (96000,),
        "wave_norm",
        "speech",
    ),
]

from extra_tasks import contracts

MODELS.extend(contracts(Model))

from extended_models import contracts as extended_contracts

MODELS.extend(extended_contracts(Model))

from field_models import contracts as field_contracts

MODELS.extend(field_contracts(Model))

from sequence_models import contracts as sequence_contracts

MODELS.extend(sequence_contracts(Model))

from architecture_models import contracts as architecture_contracts

MODELS.extend(architecture_contracts(Model))
