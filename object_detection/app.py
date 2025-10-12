import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
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
bg_image = Image.open("Background.png").resize((1100, 700))
bg_photo = ImageTk.PhotoImage(bg_image)
tb.Label(root, image=bg_photo).place(x=0, y=0, relwidth=1, relheight=1)

# ===== MENU DỌC =====
menu_frame = tb.Frame(root, bootstyle="secondary", width=200)
menu_frame.pack(side="left", fill="y")

# ===== KHU HIỂN THỊ KẾT QUẢ =====
lbl = tb.Label(root, background="", borderwidth=0)
lbl.place(x=200, y=0, relwidth=0.8, relheight=0.95)

# Set ảnh mặc định để tránh nền trắng
default_img = Image.open("object_detection/assets/Background.png").resize((1100, 700))

default_photo = ImageTk.PhotoImage(default_img)
lbl.config(image=default_photo)
lbl.image = default_photo

status = tb.Label(root, text="Sẵn sàng", anchor="w", bootstyle="dark")
status.pack(side="bottom", fill="x")

cap = None
frame_count = 0
running_mode = None
after_id = None  # để hủy vòng lặp Tkinter

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
   

# ===== XỬ LÝ VIDEO =====
def detect_video():
    stop_current()
    global cap, frame_count, running_mode
    running_mode = "video"
    file_path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov")])
    if not file_path:
        return
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

# ===== XỬ LÝ CHUNG (VIDEO/CAMERA) =====
def process_stream():
    global cap, frame_count, running_mode, after_id, paused
    if cap is None or running_mode not in ("video", "camera"):
        return

    ret, frame = cap.read()
    if not ret:
        update_status("📁 Video/camera kết thúc hoặc mất tín hiệu.")
        stop_current()
        return

    frame_count += 1
    if frame_count % 5 == 0:  # giảm tải CPU
        frame_small = cv2.resize(frame, (320, 320))
        start = time.time()
        results = model(frame_small, verbose=False)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + 1e-6)
        info = count_objects(results)

        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((850, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)

        update_status(f"{running_mode.upper()} | FPS: {fps:.2f} | {info}") #FPS là tốc độ xử lý video (frame/giây).

    after_id = root.after(20, process_stream)

# ===== MENU =====
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)

# ===== NÚT ĐIỀU KHIỂN VIDEO =====
btn_pause = tb.Button(video_control_frame, text="⏸ Tạm dừng", bootstyle="warning-outline", command=toggle_pause)
btn_pause.pack(side="left", padx=10)

btn_replay = tb.Button(video_control_frame, text="🔁 Phát lại", bootstyle="info-outline", command=replay_video)
btn_replay.pack(side="left", padx=10)

root.mainloop()
