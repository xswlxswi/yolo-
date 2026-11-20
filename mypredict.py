from ultralytics import YOLO

model = YOLO(r"E:\Desktop\DL\yolov5-master\yolov5s1.pt")  # r后边为模型
model.predict(
    source=r"E:\Desktop\assets",  # r后边为分析文件位置
    save=True,  # 保存预测结果
    show=False,  # 不用立刻显示结果这俩至少有一个ture
    line_width=8,  # 画的线变粗
)
