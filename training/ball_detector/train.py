from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"runs\detect\train\weights\last.pt")
    model.train(
        data=r"C:\Users\user\OneDrive\Documents\padelsense\Padel Ball.v2-initial-data-set-raw.yolov11\data.yaml",
        epochs=150,
        imgsz=640,
        device="cuda",
        hsv_h=0.015,
        hsv_s=0.7,
        degrees=10,
        scale=0.5,
        mosaic=1.0,
        workers=4
    )