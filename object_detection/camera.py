import cv2
import time
from PIL import Image, ImageTk
from config import Config

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.is_running = False

    def start_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            return True
        return False

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.is_running = False
            self.cap = None

    def get_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        self.stop_camera()

    def get_processed_frame(self, model):
        frame = self.get_frame()
        if frame is not None:
            start = time.time()
            results = model(frame)
            annotated = results[0].plot()
            fps = 1 / (time.time() - start + Config.MIN_FPS)
            img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((Config.IMAGE_RESIZE_WIDTH, Config.IMAGE_RESIZE_HEIGHT))
            return ImageTk.PhotoImage(image=img), fps, len(results[0].boxes)
        return None, 0, 0
