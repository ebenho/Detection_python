from ultralytics import YOLO
import cv2, os, time
from object_detection.utils.save_log import save_detection_log

def detect_video(video_path, model_path="yolov8n.pt"):
    """
    Nhận dạng đối tượng trong video.
    Chỉ lưu log CSV, không sinh video/ảnh.
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
    results_summary = []  # lưu nhãn & độ chính xác

    print("🎥 Bắt đầu nhận dạng realtime (nhấn Q để thoát)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("📁 Video đã phát hết.")
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        start_time = time.time()
        results = model(frame, verbose=False)

        # ✅ Thu thập kết quả
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                results_summary.append((label, conf))

        fps = 1 / (time.time() - start_time + 1e-6)
        cv2.imshow("YOLO Video Detection", results[0].plot())

        

    # ✅ Ghi log
    save_detection_log("Video", video_path, results_summary)
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Kết thúc nhận dạng video và đã lưu log.")
