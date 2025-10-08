# Configuration settings for the object detection app
class Config:
    # Window settings
    WINDOW_WIDTH = 1100
    WINDOW_HEIGHT = 700
    MENU_WIDTH = 200

    # YOLO model settings
    MODEL_PATH = "yolov8n.pt"
    OUTPUT_DIR = "outputs"
    RESULTS_SUBDIR = "results"

    # Image processing settings
    IMAGE_RESIZE_WIDTH = 850
    IMAGE_RESIZE_HEIGHT = 600

    # Frame rate and timing
    FRAME_DELAY = 20  # milliseconds
    MIN_FPS = 1e-6

    # File types
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")
