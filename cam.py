import cv2
import torch

# 加载你自己的模型（替换成你的文件路径）
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s1.pt", force_reload=True)

# 或者使用本地路径直接加载
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/path/to/your/model.pt')

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测
    results = model(frame)

    # 显示结果
    cv2.imshow("Detection", results.render()[0])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
