from ultralytics import YOLO

if __name__ == "__main__":
    medol = YOLO(r"yolov5s.pt")
    medol.train(
        data=r"E:\Desktop\DL\yolov5-master\data\tennis.yaml",
        epochs=30,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=1,)