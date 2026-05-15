import sys

import torch

# ========== 配置路径 ==========
PT_PATH = r"E:\Desktop\DL\yolov5-master\yolov5s1.pt"
ONNX_PATH = r"C:\Users\陈卓钺\Desktop\yolov5s1.onnx"  # 输出到桌面
YOLOV5_DIR = r"E:\Desktop\DL\yolov5-master"

# 将 yolov5 目录加入 Python 路径
sys.path.insert(0, YOLOV5_DIR)

# ========== 加载模型 ==========
device = torch.device("cpu")  # CPU 更稳定
model = torch.hub.load(YOLOV5_DIR, "custom", path=PT_PATH, source="local", force_reload=True).to(device).eval()

# ========== 导出 ONNX ==========
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"转换完成！ONNX 已保存到: {ONNX_PATH}")
