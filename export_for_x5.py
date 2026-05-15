"""
加载 yolov5s1.pt（Ultralytics 解耦头格式），替换 Detect 头后导出 ONNX
适配 RDK X5 (bayes-e) BPU 量化
"""
import sys
import torch
import torch.nn as nn

# ========== 路径配置 ==========
PT_PATH = r"E:\Desktop\DL\yolov5-master\yolov5s1.pt"
ONNX_OUT = r"E:\Desktop\DL\yolov5-master\yolov5s_x5_deploy.onnx"
IMG_SIZE = 640
# ==============================


def load_model_weights(weights_path, device='cpu'):
    ckpt = torch.load(weights_path, map_location=device)
    if 'model' in ckpt:
        model = ckpt['model']
    elif 'ema' in ckpt:
        model = ckpt['ema']
    else:
        model = ckpt
    model = model.float().eval().to(device)
    return model


def patch_detect_head(model):
    """
    Ultralytics 解耦头: cv2[i]=bbox分支, cv3[i]=cls分支
    修改为只输出原始卷积结果，去掉 DFL/Decode/Concat 等后处理
    """
    for m in model.modules():
        if type(m).__name__ == 'Detect':
            nc = getattr(m, 'nc', 80)
            nl = getattr(m, 'nl', 3)
            reg_max = getattr(m, 'reg_max', 16)
            print(f"[INFO] 找到 Detect 头: nl={nl}, nc={nc}, reg_max={reg_max}")

            if not hasattr(m, 'cv2') or not hasattr(m, 'cv3'):
                raise RuntimeError("该模型不是解耦头格式，没有 cv2/cv3 属性")

            def new_forward(self, x):
                outs = []
                for i in range(self.nl):
                    bbox = self.cv2[i](x[i])  # [B, 64, H, W]  (reg_max*4)
                    cls = self.cv3[i](x[i])   # [B, nc, H, W]
                    out = torch.cat([bbox, cls], dim=1)  # [B, 64+nc, H, W]
                    out = out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]  BPU友好
                    outs.append(out)
                return outs

            import types
            m.forward = types.MethodType(new_forward, m)
            return True

    raise RuntimeError("找不到 Detect 头")


def export_onnx(model, output_path, img_size=640):
    dummy = torch.zeros(1, 3, img_size, img_size).to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(dummy)
    print(f"[INFO] 模型输出数量: {len(out)}")
    for i, o in enumerate(out):
        print(f"  out{i}: {o.shape}")

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=11,
        input_names=['images'],
        output_names=['out0', 'out1', 'out2'],
        dynamic_axes=None,
    )
    print(f"\n[SUCCESS] ONNX 已保存: {output_path}")

    import onnx
    m = onnx.load(output_path)
    onnx.checker.check_model(m)
    print("ONNX 验证通过!")
    print("输出节点:")
    for o in m.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  {o.name}: {shape}")


def main():
    print(f"[INFO] 加载模型: {PT_PATH}")
    model = load_model_weights(PT_PATH)

    print("[INFO] 修改 Detect 头...")
    patch_detect_head(model)

    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = False

    print(f"[INFO] 导出 ONNX...")
    export_onnx(model, ONNX_OUT, IMG_SIZE)


if __name__ == '__main__':
    main()
