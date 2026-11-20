"""Fixed-length supervised signal classifiers and their public archive datasets."""

# Dataset, algorithm, task, channels, time steps.
SIGNALS = [
    ("ECG5000", "residual", "心搏波形分类", 1, 140),
    ("FordA", "fcn", "发动机噪声故障分类", 1, 500),
    ("Wafer", "dilated", "晶圆工艺异常分类", 1, 152),
    ("ElectricDevices", "inception", "用电设备识别", 1, 96),
    ("BasicMotions", "separable", "腕部活动识别", 6, 100),
    ("Epilepsy", "residual", "模拟动作识别", 3, 206),
    ("Handwriting", "inception", "手写轨迹识别", 3, 152),
    ("Coffee", "lda", "咖啡品种光谱分类", 1, 286),
    ("Strawberry", "ridge", "果泥光谱鉴别", 1, 235),
    ("Meat", "rbf", "肉类光谱分类", 1, 448),
    ("ItalyPowerDemand", "mlp", "用电曲线季节分类", 1, 24),
    ("Earthquakes", "separable", "地震信号分类", 1, 512),
    ("InsectWingbeatSound", "fcn", "昆虫翅振声分类", 1, 256),
    ("FaceAll", "mlp", "人脸轮廓序列分类", 1, 131),
]
ARCH_NAMES = {
    "residual": "Residual temporal CNN",
    "fcn": "Temporal FCN",
    "dilated": "Dilated temporal CNN",
    "inception": "Multiscale Inception CNN",
    "separable": "Depthwise temporal CNN",
    "lda": "Shrinkage LDA",
    "ridge": "Ridge classifier",
    "rbf": "RBF prototype network",
    "mlp": "Two-layer MLP",
}
