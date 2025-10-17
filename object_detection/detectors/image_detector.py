import os
import cv2
from PIL import Image
from object_detection.utils.save_log import save_detection_log


def detect_single_image(image_path, model):
    """
    Nhận diện 1 ảnh bằng YOLO, hiển thị bounding box và lưu log.
    """
    if not os.path.isfile(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return None

    try:
        # ✅ Nhận diện
        results = model(image_path, verbose=False)
    except Exception as e:
        print(f"⚠️ Lỗi khi chạy YOLO: {e}")
        return None

    # ✅ Vẽ bounding box (annotated image)
    annotated_img = results[0].plot()

    # ✅ Lưu ảnh tạm (không cần output folder)
    output_path = os.path.splitext(image_path)[0] + "_detected.jpg"
    cv2.imwrite(output_path, annotated_img)

    # ✅ Ghi log
    detected_objects = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            detected_objects.append((label, conf))

    save_detection_log("Ảnh", image_path, detected_objects)

    print(f"✅ Đã nhận diện xong {os.path.basename(image_path)}:")
    for label, conf in detected_objects:
        print(f"   - {label}: {conf:.2f}")

    # ✅ Trả về đường dẫn ảnh có khung
    return output_path
