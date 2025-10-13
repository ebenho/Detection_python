import os
import cv2
from ultralytics import YOLO
from utils.image_utils import detect_image, save_image


# --- Đường dẫn chuẩn ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # detectors/
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # project_root/
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
MODEL_PATH = os.path.join(PROJECT_ROOT, "assets", "models", "yolov8n.pt")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load YOLO model ---
model = YOLO(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")


def detect_from_folder():
    """Nhận diện tất cả ảnh trong thư mục inputs và lưu vào outputs."""
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)

            annotated = detect_image(model, input_path)
            if annotated is not None:
                save_image(annotated, output_path)
                print(f"{filename} -> Done")
            else:
                print(f"Bỏ qua: {filename}")
    print("Hoàn tất nhận diện thư mục inputs!")
    
def detect_single_image(image_path):
    """Nhận diện một ảnh và lưu vào thư mục outputs."""
    if not os.path.isfile(image_path):
        print(f"File không tồn tại: {image_path}")
        return None

    filename = os.path.basename(image_path)
    output_path = os.path.join(OUTPUT_DIR, filename)

    annotated = detect_image(model, image_path)
    if annotated is not None:
        save_image(annotated, output_path)
        print(f"{filename} -> Done")
        return output_path
    else:
        print(f"Không thể đọc ảnh: {filename}")
        return None


# Chạy trực tiếp nếu gọi file này
if __name__ == "__main__":
    detect_from_folder()
