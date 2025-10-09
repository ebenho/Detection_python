from ultralytics import YOLO
import cv2, os, time

def detect_video(video_path, model_path="yolov8n.pt"):
    """
    Nhận dạng đối tượng trong video bằng YOLOv8 (tối ưu cho CPU)
    """
    if not os.path.exists(video_path):
        print(f"❌ Không tìm thấy video: {video_path}")
        return

    print("🚀 Đang load model YOLO (CPU)...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không thể mở video.")
        return

    frame_count = 0
    print("🎥 Bắt đầu nhận dạng realtime (nhấn Q để thoát)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📁 Video đã phát hết.")
            break

        frame_count += 1
        # Chỉ xử lý 1/5 frame để giảm tải CPU
        if frame_count % 5 != 0:
            continue

        # Giảm kích thước input (tăng tốc)
        frame_resized = cv2.resize(frame, (320, 320))

        start_time = time.time()
        results = model(frame_resized, verbose=False)
        annotated_frame = results[0].plot()

        fps = 1 / (time.time() - start_time + 1e-6)

        # Hiển thị FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 CPU Video Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Dừng video theo yêu cầu người dùng.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Kết thúc nhận dạng video.")

if __name__ == "__main__":
    test_video = "test_video.mp4"  # đổi tên video của bạn
    detect_video(test_video)
