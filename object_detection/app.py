import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os

# Load YOLO model (COCO pre-trained)
model = YOLO("yolov8n.pt")  

st.set_page_config(page_title="Object Detection App", page_icon="🚀", layout="wide")
st.title("🚀 Object Detection App")
st.write("Nhận dạng **người, đồ vật, loài vật** trong ảnh, video và webcam.")

# Sidebar chọn chế độ
mode = st.sidebar.radio("Chọn chế độ:", ["Ảnh", "Video", "Webcam"])

# ========== ẢNH ==========
if mode == "Ảnh":
    uploaded_file = st.file_uploader("📷 Chọn ảnh...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Lưu file tạm
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # Chạy detection
        results = model(temp_file.name)

        # Hiển thị kết quả
        for r in results:
            st.image(r.plot(), caption="Kết quả nhận dạng", use_container_width=True)

        os.remove(temp_file.name)

# ========== VIDEO ==========
elif mode == "Video":
    uploaded_video = st.file_uploader("🎥 Chọn video...", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Lưu file video tạm
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        temp_video.close()

        # Xuất video kết quả
        output_path = temp_video.name + "_out.mp4"
        results = model.predict(source=temp_video.name, save=True, project="outputs", name="video_results")

        # Hiển thị video
        st.video(f"outputs/video_results/{os.path.basename(temp_video.name)}")

# ========== WEBCAM ==========
elif mode == "Webcam":
    st.write("⚡ Webcam realtime detection (chạy bằng OpenCV)")
    run = st.checkbox("Bật Webcam")
    if run:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        cap.release()
