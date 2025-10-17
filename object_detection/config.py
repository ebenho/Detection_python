import os

# Configuration settings for the object detection app
class Config:
    # Window settings
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 700
    MENU_WIDTH = 300

    # YOLO model settings
    MODELS_DIR = os.path.dirname(__file__)
    MODEL_PATH = "yolov8n.pt"  # Sử dụng tên mô hình thống nhất
    OUTPUT_DIR = "outputs"
    RESULTS_SUBDIR = "results"

    # Image processing settings
    IMAGE_W = 1000  # Thêm để khớp với app.py
    IMAGE_H = 700  # Thêm để khớp với app.py
    IMAGE_RESIZE_WIDTH = 1000
    IMAGE_RESIZE_HEIGHT = 700

    # Frame rate and timing
    FRAME_DELAY = 20  # milliseconds
    MIN_FPS = 1e-6

    # File types
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
    
    # Đường dẫn cơ bản
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUTS = os.path.join(BASE_DIR, "input")
    OUTPUTS = os.path.join(BASE_DIR, "output")
    MODELS_DIR = os.path.join(BASE_DIR, "assets", "models")
    
    # Ngưỡng độ tin cậy
    CONFIDENCE_THRESHOLD = 0.5

# Kiểm tra và tạo thư mục nếu chưa tồn tại
os.makedirs(Config.INPUTS, exist_ok=True)
os.makedirs(os.path.join(Config.OUTPUTS, Config.RESULTS_SUBDIR), exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)