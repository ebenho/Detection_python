import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2, os, time

# Load YOLO
model = YOLO("yolov8n.pt")

# ===== GIAO DIỆN CHÍNH =====
root = tb.Window(themename="cosmo")
root.title("🚀 Object Detection App")
root.geometry("1100x700")

# ===== ẢNH NỀN =====
try:
    bg_path = os.path.join(os.path.dirname(__file__), "assets", "Background.png")
    bg_image = Image.open(bg_path).resize((1100, 700))
    bg_photo = ImageTk.PhotoImage(bg_image)
    tb.Label(root, image=bg_photo).place(x=0, y=0, relwidth=1, relheight=1)
except FileNotFoundError:
    messagebox.showwarning("Cảnh báo", "Không tìm thấy ảnh nền Background.png trong thư mục assets!")
    root.configure(bg="#1e1e1e")  # fallback màu xám

# ===== MENU DỌC =====
menu_frame = tb.Frame(root, bootstyle="secondary", width=200)
menu_frame.pack(side="left", fill="y")

# ===== KHU HIỂN THỊ KẾT QUẢ =====
lbl = tb.Label(root, background="", borderwidth=0)
lbl.place(x=200, y=0, relwidth=0.8, relheight=0.95)

# ===== ẢNH MẶC ĐỊNH =====
try:
    default_img = Image.open(bg_path).resize((1100, 700))
    default_photo = ImageTk.PhotoImage(default_img)
    lbl.config(image=default_photo)
    lbl.image = default_photo
except:
    lbl.config(text="Không tìm thấy ảnh nền", font=("Arial", 18), anchor="center")

status = tb.Label(root, text="Sẵn sàng", anchor="w", bootstyle="dark")
status.pack(side="bottom", fill="x")

cap = None
frame_count = 0
running_mode = None
after_id = None

def update_status(msg):
    status.config(text=msg)

def stop_current():
    """Dừng mọi video/camera đang chạy"""
    global cap, running_mode, after_id
    running_mode = None
    if after_id:
        root.after_cancel(after_id)
    if cap and cap.isOpened():
        cap.release()
    cap = None
    update_status("⏹ Đã dừng video/camera.")

# ===== XỬ LÝ ẢNH =====
def detect_image():
    stop_current()
    file_path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg;*.jpeg;*.png")])
    if not file_path: return
    results = model(file_path, save=True, project="outputs", name="results", exist_ok=True)
    output_file = os.path.join("outputs/results", os.path.basename(file_path))
    img = Image.open(output_file).resize((850, 600))
    imgtk = ImageTk.PhotoImage(img)
    lbl.config(image=imgtk)
    lbl.image = imgtk
    update_status(f"📸 Ảnh: {os.path.basename(file_path)} | Đối tượng: {len(results[0].boxes)}")

# ===== XỬ LÝ VIDEO =====
def detect_video():
    stop_current()
    global cap, frame_count, running_mode
    running_mode = "video"
    file_path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov")])
    if not file_path: return
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        update_status("❌ Không thể mở video.")
        return
    frame_count = 0
    update_status(f"🎥 Video: {os.path.basename(file_path)}")
    process_stream()

# ===== XỬ LÝ CAMERA =====
def detect_camera():
    stop_current()
    global cap, running_mode
    running_mode = "camera"
    cap = cv2.VideoCapture(0)
    if not cap or not cap.isOpened():
        update_status("❌ Không thể mở camera.")
        return
    update_status("📡 Camera bật.")
    process_stream()

# ===== XỬ LÝ CHUNG =====
def process_stream():
    global cap, frame_count, running_mode, after_id
    if cap is None or running_mode not in ("video", "camera"):
        return

    ret, frame = cap.read()
    if not ret:
        update_status("📁 Video/camera kết thúc hoặc mất tín hiệu.")
        stop_current()
        return

    frame_count += 1
    if frame_count % 5 == 0:
        frame_small = cv2.resize(frame, (320, 320))
        start = time.time()
        results = model(frame_small, verbose=False)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + 1e-6)
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((850, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)
        update_status(f"{running_mode.upper()} | FPS: {fps:.2f} | Đối tượng: {len(results[0].boxes)}")

    after_id = root.after(20, process_stream)

# ===== MENU =====
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)

root.mainloop()
