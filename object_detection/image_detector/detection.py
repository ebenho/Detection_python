import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # image_detector
PARENT_DIR = os.path.dirname(BASE_DIR)                 # object_detection
sys.path.append(PARENT_DIR)

from utils.image_utils import load_model, detect_image, save_image

input_folder = os.path.join(BASE_DIR, "inputs")
output_folder = os.path.join(BASE_DIR, "outputs")
os.makedirs(output_folder, exist_ok=True)

model = load_model()

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_folder, filename)

        # Tạo tên output: input1.jpg -> output1.jpg
        name, ext = os.path.splitext(filename)
        if name.startswith("input") and name[5:].isdigit():
            output_name = f"output{name[5:]}{ext}"
        else:
            output_name = f"output_{filename}"  # fallback
        output_path = os.path.join(output_folder, output_name)

        annotated_img = detect_image(model, input_path)
        if annotated_img is not None:
            save_image(annotated_img, output_path)
