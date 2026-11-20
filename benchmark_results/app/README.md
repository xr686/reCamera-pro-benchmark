# RV1126B 真机模型与算法基准

在 reCamera Pro 固件 V1.3.2 上，应用以 batch 1 完成 **270 个模型配置、100 个任务标签、21,171 次计时调用**。ImageNet 分类共 59 个配置，按主干合并为 **50 个代表模型**。同一骨干在不同数据集上训练或换用任务头时，配置数和代表模型数分别统计。

| 要查的内容 | 入口 |
|---|---|
| 按架构或任务找算法 | [50 个代表分类模型与其他 99 个任务的算法清单](RV1126B-tested-algorithms.md) |
| 查看模型变体和训练权重来源 | [代表模型及实测变体](RV1126B-ImageNet-model-index.csv) · [59 个分类配置的来源](classification-sources.csv) |
| 查一个模型的输入、FP16 体积、前后处理、推理均值与 P95、FPS、任务精度 | [270 项实测 CSV](measurements.csv) |
| 在同一批 100 张图上比较最近 16 个分类架构和原始 ONNX | [专项数据](RV1126B-architectures.csv) · [转换记录与精度差异](../../apps/npu-benchmark/ARCHITECTURES-1.3.2.md) |
| 为单个模型构建应用 | [270 个逐模型 Demo](../../examples/MODELS.md) · [安装和读取预测结果](../../examples/README.md) |
| 下载和复测 | [24 个实测 RKNN 权重](../../models/app-architectures/README.md) · [算法实现、模型转换和应用打包](../../apps/npu-benchmark/README.md) |
| 从汇总追到真机测量 | [70 份分组记录](runs) · [模型 SHA-256](model-hashes.csv) · [100 图标签、哈希和原始 ONNX 输出指纹](imagenette-100) |

## 读数口径

- 每行对应一个实际运行的模型 ID。`size` 为输入尺寸或序列长度，`precision` 为模型数值格式。`quality` 为本次样本上的任务指标，例如 Top-1、IoU 或字错率。`cpu_metrics` 是有记录时相同样本的主机对照。`unique` 和 `calls` 分别表示不同输入数和计时调用数。不同数据集和任务的分数应结合样本量、标签和指标定义解读。[质量指标与子集说明](../../apps/npu-benchmark/tools/evaluation-notes.md)
- `pre_ms`、`infer_ms`、`post_ms` 分别为前处理、提交输入至同步取回输出、后处理的均值。`p95_ms` 为推理时延的第 95 百分位。串行 `fps` 等于 1000 除以三段均值之和，`with_gc` 另计每次释放输出及 Python 垃圾回收。三个早期基础分类配置没有单独记录回收时间。此处的 FPS 属于应用内处理流水线，摄像头采集与显示在计时范围之外。
- 测量前通过 AppMgr 获取 NPU 独占租约，等连续三次负载为零再预热。70 份记录保留初始负载采样及模型内持有者检查。2,637 次检查仅观察到基准 Python 进程和固件 broker `rkipc`。按 CSV 的 `series/run` 可在 `runs/` 定位记录，`times_ms` 每行为前处理、推理和后处理三段毫秒数。设备频率与温度随每次检查记录。
- 最近 16 个架构共用同一批 Imagenette 验证图，每个模型预热 5 次，分类候选为 1000 类。VOLO 有四个转置回退到 CPU，板端 Top-1 为 82%，原始 ONNX 为 87%。历史分类配置的样本量保留在模型索引中。[架构专项报告](../../apps/npu-benchmark/ARCHITECTURES-1.3.2.md)

从仓库根目录运行以下命令，可用 Python 标准库从逐次时延复算 CSV 并核对任务评分、模型哈希和架构归属。

```sh
python3 benchmark_results/app/verify.py
```

SmolVLM-256M-Instruct 和 InternVL3-1B 的场景描述延迟另行测量，独立于上述 270 项。[VLM 应用测量](vlm.md)
