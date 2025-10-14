import os

def detect_single_image(image_path, model):
    """
    Nhận diện 1 ảnh bằng model YOLO đã load sẵn trong app.py
    Trả về đường dẫn ảnh kết quả (outputs/results/xxx.jpg)
    """
    if not os.path.isfile(image_path):
        print(f"❌ File không tồn tại: {image_path}")
        return None

    # Xác định đường dẫn tuyệt đối đến thư mục outputs/results
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
    output_dir = os.path.join(base_dir, "outputs", "results")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # YOLO tự động lưu kết quả vào outputs/results
        results = model.predict(
            source=image_path,
            project=os.path.join(base_dir, "outputs"),
            name="results",
            exist_ok=True,
            save=True
        )
    except Exception as e:
        print(f"⚠️ Lỗi khi chạy YOLO: {e}")
        return None

    # Đường dẫn ảnh kết quả
    output_file = os.path.join(output_dir, os.path.basename(image_path))

    # Một số bản YOLO không tự ghi đè ảnh, cần kiểm tra lại
    if not os.path.exists(output_file):
        # lấy ảnh chú thích từ YOLO rồi lưu thủ công
        annotated = results[0].plot()
        from PIL import Image
        import cv2
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        Image.fromarray(annotated).save(output_file)

    if os.path.exists(output_file):
        print(f"✅ Đã xử lý: {output_file}")
        return output_file
    else:
        print("⚠️ Không tìm thấy ảnh kết quả.")
        return None
