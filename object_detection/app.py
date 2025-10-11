import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2, os, time
import shutil  # thêm ở đầu file

from utils.image_utils import  detect_image, save_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "inputs")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
MODEL_PATH = os.path.join(BASE_DIR, "assets", "models", "yolov8n.pt")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")
root = tb.Window(themename="cosmo")
root.title("🚀 Object Detection App")
root.geometry("1100x700")

# ===== ẢNH NỀN =====
bg_image = Image.open("object_detection/assets/Background.png").resize((1100, 700))
bg_photo = ImageTk.PhotoImage(bg_image)
tb.Label(root, image=bg_photo).place(x=0, y=0, relwidth=1, relheight=1)

# ===== MENU DỌC =====
menu_frame = tb.Frame(root, bootstyle="secondary", width=200)
menu_frame.pack(side="left", fill="y")

# ===== KHU HIỂN THỊ KẾT QUẢ =====
lbl = tb.Label(root, background="", borderwidth=0)
lbl.place(x=200, y=0, relwidth=0.8, relheight=0.95)
# ===== KHUNG NÚT VIDEO (Pause / Replay) =====
video_control_frame = tb.Frame(root, bootstyle="light")
video_control_frame.place(x=450, y=630)  # vị trí dưới khung video, có thể chỉnh

btn_pause = None
btn_replay = None

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
paused = False

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
    output_folder = OUTPUT_DIR
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    save_image(annotated, output_path)

    # Sao chép ảnh gốc vào thư mục inputs (nếu chưa có)
    dest_input_path = os.path.join(INPUT_DIR, os.path.basename(file_path))
    if not os.path.exists(dest_input_path):
        try:
            shutil.copy(file_path, dest_input_path)
            update_status(f"Ảnh gốc đã được sao chép vào: {dest_input_path}")
        except Exception as e:
            update_status(f"Lỗi khi copy ảnh: {e}")

    # Hiển thị kết quả lên GUI
    img = Image.open(output_path).resize((850, 600))
    imgtk = ImageTk.PhotoImage(img)
    lbl.config(image=imgtk)
    lbl.image = imgtk
    update_status(f"Ảnh: {os.path.basename(file_path)} | Đã lưu: {output_path}")

def stop_current():
    """Dừng tất cả video/camera đang chạy."""
    global cap, running_mode, after_id
    running_mode = None
    if after_id:
        root.after_cancel(after_id)
    if cap and cap.isOpened():
        cap.release()
    cap = None
    update_status("⏹ Đã dừng video/camera.")
   
# ====== HÀM PAUSE / REPLAY ======
def toggle_pause():
    global paused
    paused = not paused
    if paused:
        update_status("⏸ Video tạm dừng.")
        btn_pause.config(text="▶ Tiếp tục", bootstyle="success-outline")
    else:
        update_status("🎬 Tiếp tục phát video...")
        btn_pause.config(text="⏸ Tạm dừng", bootstyle="warning-outline")

def replay_video():
    global cap, running_mode, frame_count, paused
    if running_mode != "video" or cap is None:
        update_status("⚠️ Không có video nào để phát lại.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    paused = False
    btn_pause.config(text="⏸ Tạm dừng", bootstyle="warning-outline")
    update_status("🔁 Phát lại video từ đầu.")
    
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
    update_status("🎥 Đang phát video...")
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

# ===== VÒNG LẶP XỬ LÝ CHUNG =====
def process_stream():
    global cap, frame_count, running_mode, after_id, paused
    if cap is None or running_mode not in ("video", "camera"):
        return
    
    if paused:  # 🟡 Khi pause thì không đọc frame mới
        after_id = root.after(100, process_stream)
        return
    
    ret, frame = cap.read()
    if not ret:
        update_status("📁 Kết thúc hoặc mất tín hiệu.")
        stop_current()
        return

    frame_count += 1
    if frame_count % 5 == 0:  # skip frame để nhẹ CPU
        frame_small = cv2.resize(frame, (320, 320))
        start = time.time()
        results = model(frame_small, verbose=False)
        annotated = results[0].plot()
        fps = 1 / (time.time() - start + 1e-6)
        img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((850, 600))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk = imgtk
        lbl.config(image=imgtk)
        update_status(f"{running_mode.upper()} | FPS: {fps:.2f} | Đối tượng: {len(results[0].boxes)}")

    after_id = root.after(10, process_stream)


# ========== NÚT TRONG MENU ==========
tb.Button(menu_frame, text="📷 Ảnh", bootstyle=SUCCESS, command=detect_image_gui, width=15).pack(pady=15)
tb.Button(menu_frame, text="🎥 Video", bootstyle=INFO, command=detect_video, width=15).pack(pady=15)
tb.Button(menu_frame, text="📡 Camera", bootstyle=PRIMARY, command=detect_camera, width=15).pack(pady=15)
tb.Button(menu_frame, text="❌ Thoát", bootstyle=DANGER, command=root.destroy, width=15).pack(pady=15)

# ===== NÚT ĐIỀU KHIỂN VIDEO =====
btn_pause = tb.Button(video_control_frame, text="⏸ Tạm dừng", bootstyle="warning-outline", command=toggle_pause)
btn_pause.pack(side="left", padx=10)

btn_replay = tb.Button(video_control_frame, text="🔁 Phát lại", bootstyle="info-outline", command=replay_video)
btn_replay.pack(side="left", padx=10)

root.mainloop()
