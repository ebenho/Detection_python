import cv2
from ultralytics import YOLO
import os

# Hàm load YOLOv8n model (dùng chung)
def load_model(model_name="yolov8n.pt"):
    return YOLO(model_name)

# Hàm detect và vẽ bounding box trên 1 ảnh
def detect_image(model, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: could not read {img_path}")
        return None
    results = model.predict(img)
    annotated_img = results[0].plot()
    return annotated_img

# Hàm lưu ảnh kết quả
def save_image(img, output_path):
    cv2.imwrite(output_path, img)
    print(f"Saved result to {output_path}")

