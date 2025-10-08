# ===============================
# video_detector.py
# Nhận dạng đối tượng trong video bằng YOLOv8
# ===============================

from ultralytics import YOLO
import cv2
import os

def detect_video(video_path, output_path="outputs/result_video.mp4", model_path="yolov8n.pt"):
    """
    Hàm nhận dạng đối tượng trong video.
    
    Args:
        video_path (str): Đường dẫn đến video cần nhận dạng.
        output_path (str): Đường dẫn lưu video kết quả (mp4).
        model_path (str): File model YOLO (vd: yolov8n.pt, yolov8s.pt,...)
    """

    # Kiểm tra video tồn tại
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    # Load model YOLO
    print(f"🚀 Đang load model YOLO từ {model_path} ...")
    model = YOLO(model_path)

    # Mở video
    cap = cv2.VideoCapture(video_path)

    # Lấy thông tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Tạo writer để lưu video kết quả
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("🎥 Bắt đầu xử lý video... (nhấn Q để thoát sớm)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dự đoán bằng YOLO
        results = model(frame, verbose=False)

        # Vẽ khung nhận dạng
        annotated_frame = results[0].plot()

        # Ghi vào video output
        out.write(annotated_frame)

        # Hiển thị trực tiếp
        cv2.imshow("YOLO Video Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng bộ nhớ
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Hoàn thành! Video kết quả lưu tại: {output_path}")
    

if __name__ == "__main__":
    # Ví dụ chạy trực tiếp file này
    test_video = "test_video.mp4"  # Đặt file video test của bạn ở thư mục gốc
    detect_video(test_video)
