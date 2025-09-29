import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

# Load YOLO model (COCO pre-trained)
model = YOLO("yolov8n.pt")  # file model bạn đã có ở thư mục gốc

# Tạo thư mục output nếu chưa có
os.makedirs("outputs/results", exist_ok=True)

def detect_image():
    # Chọn file ảnh
    file_path = filedialog.askopenfilename(
        initialdir="data/images",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return
    
    # Chạy YOLO detect
    results = model(file_path, save=True, project="outputs", name="results", exist_ok=True)
    
    # Lấy file kết quả YOLO vẽ bounding box
    output_file = os.path.join("outputs/results", os.path.basename(file_path))
    
    # Hiển thị ảnh kết quả
    img = Image.open(output_file)
    img = img.resize((500, 400))  # resize cho vừa cửa sổ
    tk_img = ImageTk.PhotoImage(img)
    panel.config(image=tk_img)
    panel.image = tk_img

    # Hiển thị số đối tượng
    num_objects = len(results[0].boxes)
    info_label.config(text=f"✅ Phát hiện {num_objects} đối tượng.")

# =================== Giao diện Tkinter ===================
root = tk.Tk()
root.title("🚀 Object Detection App (Tkinter)")
root.geometry("650x550")
root.configure(bg="#f5f5f5")

title = Label(root, text="Ứng dụng Nhận dạng Đối tượng", font=("Arial", 18, "bold"), bg="#f5f5f5", fg="#333")
title.pack(pady=10)

btn = Button(root, text="📂 Chọn ảnh để nhận dạng", command=detect_image, font=("Arial", 12), bg="#007BFF", fg="white")
btn.pack(pady=10)

info_label = Label(root, text="", font=("Arial", 12), bg="#f5f5f5", fg="green")
info_label.pack(pady=5)

panel = Label(root, bg="#f5f5f5")
panel.pack(pady=10)

root.mainloop()
