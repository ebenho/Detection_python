# ===============================
# tests/test_video.py
# Kiểm thử chức năng nhận dạng video bằng YOLOv8
# ===============================
import sys
import os

# Thêm thư mục gốc (DETECTION_PYTHON) vào sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_detection.detectors.video_detector import detect_video
import os

if __name__ == "__main__":
    test_video_path = "test_video.mp4"  # đặt video này cùng cấp thư mục gốc (Detection_python/)

    if not os.path.exists(test_video_path):
        print(f"⚠️ Không tìm thấy file '{test_video_path}'.")
        print("👉 Hãy đặt 1 file video vào thư mục gốc project và đổi tên nó thành 'test_video.mp4'.")
    else:
        print("🚀 Bắt đầu nhận dạng video bằng YOLOv8...\n")
        detect_video(
            video_path=test_video_path,
            output_path="outputs/results/result_video.mp4",
            model_path="yolov8n.pt"
        )
        print("\n✅ Hoàn tất kiểm thử video!")
