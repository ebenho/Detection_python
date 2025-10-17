import csv
from datetime import datetime
import os

# Lưu file CSV ở thư mục gốc
LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "detection_log.csv"
)

def save_detection_log(mode, file_name, detected_objects):
    """
    Ghi log dạng bảng (CSV)
    mode: Ảnh / Video / Camera
    file_name: tên tệp hoặc "Live Camera"
    detected_objects: [(label, confidence), ...]
    """
    # Nếu file chưa tồn tại → tạo header
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Thời gian", "Chức năng", "Tên tệp", "Đối tượng", "Độ chính xác"])

        # Ghi từng dòng đối tượng
        for label, conf in detected_objects:
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                mode,
                os.path.basename(file_name),
                label,
                round(float(conf), 2)
            ])

    print(f"[LOG] Đã lưu dữ liệu vào {LOG_FILE}")
