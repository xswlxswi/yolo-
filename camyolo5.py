import cv2
from ultralytics import YOLO

# ========== 配置 ==========
# 直接加载你用 ultralytics 训练好的模型
MODEL_PATH = r"E:\Desktop\DL\yolov5-master\yolov5s1.pt"
model = YOLO(MODEL_PATH)

# ========== 打开摄像头 ==========
# 加入 cv2.CAP_DSHOW 标志位
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("无法打开摄像头！")
    exit()

# 调高分辨率，有助于看清远处的网球
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

print("按 'Q' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ========== 推理 ==========
    # conf 设置置信度，device=0 确保调用你的 RTX 3060
    results = model.predict(source=frame, conf=0.25, device=0, verbose=False)

    # ========== 获取结果并绘制 ==========
    # ultralytics 自带了画图封装，一行代码自动画框和写标签，不需要手动遍历坐标
    annotated_frame = results[0].plot()

    # 显示自定义的帧率和检测信息
    detections_count = len(results[0].boxes)
    fps_info = f"Ultralytics | CUDA:0 | Detected: {detections_count} tennis balls"
    cv2.putText(annotated_frame, fps_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ========== 显示 ==========
    cv2.imshow("Tennis Ball Tracker", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
