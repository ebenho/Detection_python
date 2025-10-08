import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2, os, time

from utils.image_utils import load_model, detect_image, save_image

# BASE_DIR = image_detector
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_detector")
BASE_DIR = os.path.abspath(BASE_DIR)  # chuẩn hóa

# Load YOLO model
model = YOLO("yolov8n.pt")

# Khởi tạo window
root = tb.Window(themename="cosmo")
root.title("🚀 Object Detection App")
root.geometry("1100x700")

# ========== ẢNH NỀN ==========
bg_image = Image.open("object_detection\Background.png").resize((1100, 700))
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tb.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # full background

# ========== MENU DỌC ==========
menu_frame = tb.Frame(root, bootstyle="secondary", width=200)
menu_frame.pack(side="left", fill="y")

# ========== KHU VỰC HIỂN THỊ KẾT QUẢ ==========
lbl = tb.Label(root, background="", borderwidth=0)
lbl.place(x=200, y=0, relwidth=0.8, relheight=0.95)

# Set ảnh mặc định để tránh nền trắng
default_img = Image.open("object_detection\Background.png").resize((850, 600))
default_photo = ImageTk.PhotoImage(default_img)
lbl.config(image=default_photo)
lbl.image = default_photo

# Status bar
status = tb.Label(root, text="Sẵn sàng", anchor="w", bootstyle="dark")
status.pack(side="bottom", fill="x")

cap = None

def update_status(msg):
    status.config(text=msg)

# ========== XỬ LÝ ẢNH ==========
def detect_image_gui():
    global cap
    if cap: cap.release()
    file_path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    # Gọi lại hàm detect_image trong utils
    annotated = detect_image(model, file_path)

    if annotated is None:
        update_status("❌ Không đọc được ảnh")
        return

    # Lưu bằng save_image trong utils
    output_folder = os.path.join(BASE_DIR, "outputs")  # đúng folder image_detector/outputs
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    save_image(annotated, output_path)
 

    # Hiển thị kết quả lên GUI
    img = Image.open(output_path).resize((850, 600))
    imgtk = ImageTk.PhotoImage(img)
    lbl.config(image=imgtk)
    lbl.image = imgtk
    update_status(f"Ảnh: {os.path.basename(file_path)} | Đã lưu: {output_path}")

# ========== XỬ LÝ VIDEO ==========
def detect_video():
    global cap
    if cap: cap.release()
    file_path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov")])
    if not file_path: return
    cap = cv2.VideoCapture(file_path)
    update_status(f"Video: {os.path.basename(file_path)}")
    show_frame()

# ========== XỬ LÝ CAMERA ==========
def detect_camera():
    global cap
    if cap: cap.release()
    cap = cv2.VideoCapture(0)
    update_status("Camera bật")
    show_frame()

def show_frame():
    global cap
    if cap is None: return
    ret, frame = cap.read()
    if ret:
        start = time.time()
        results = model(frame)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + 1e-6)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((850, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)
        update_status(f"Camera/Video | FPS: {fps:.2f} | Đối tượng: {len(results[0].boxes)}")
    lbl.after(20, show_frame)

# ========== NÚT TRONG MENU ==========
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image_gui, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)

# ========== CHẠY APP ==========
root.mainloop()
