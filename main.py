


import io
import time
import threading

import av
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode


st.set_page_config(
    page_title="AI Vision Lab",
    page_icon="👁️",
    layout="wide"
)

MODEL_LOCK = threading.Lock()


@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")




class YOLOVideoProcessor:
    def __init__(self, confidence=0.45):
        self.confidence = confidence
        self.last_inference = 0.0
        self.last_output = None
        self.interval = 0.4
        self.model = load_model()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(image, format="bgr24")
        original_height, original_width = image.shape[:2]

        # Keep a copy of the original camera frame
        original = image.copy()

        now = time.monotonic()

        if now - self.last_inference >= self.interval:
            self.last_inference = now

            try:
                # Resize for faster inference while preserving aspect ratio
                max_dimension = 640
                scale = min(
                    max_dimension / original_width,
                    max_dimension / original_height,
                    1.0
                )

                infer_width = max(1, int(original_width * scale))
                infer_height = max(1, int(original_height * scale))

                inference_image = cv2.resize(
                    original,
                    (infer_width, infer_height),
                    interpolation=cv2.INTER_AREA
                )

                with MODEL_LOCK:
                    result = self.model.predict(
                        inference_image,
                        imgsz=416,
                        conf=self.confidence,
                        device="cpu",
                        verbose=False
                    )[0]

                annotated = result.plot()

                # Return the annotated frame at the camera's exact dimensions
                self.last_output = cv2.resize(
                    annotated,
                    (original_width, original_height),
                    interpolation=cv2.INTER_LINEAR
                )

            except Exception as error:
                # Show inference errors in the video instead of hiding them
                self.last_output = original.copy()
                cv2.putText(
                    self.last_output,
                    "Detection error",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

        if self.last_output is None:
            self.last_output = original

        return av.VideoFrame.from_ndarray(
            self.last_output,
            format="bgr24"
        )


if "history" not in st.session_state:
    st.session_state.history = []


st.title("👁️ AI Vision Lab")
st.caption("AI-powered Object Detection & Visual Analytics")

with st.sidebar:
    st.header("Vision Controls")
    page = st.radio(
        "Workspace",
        [
            "Dashboard",
            "Image Detection",
            "Live Camera",
            "Analytics",
            "Settings"
        ]
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05
    )


if page == "Dashboard":
    st.header("Welcome to Vision Lab")

    st.write(
        "Detect and analyze objects in images and live camera feeds."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Images processed", len(st.session_state.history)
    )
    col2.metric(
        "Objects detected",
        sum(x["objects"] for x in st.session_state.history)
    )
    col3.metric("Detection model", "YOLO11n")

    st.info("Choose Image Detection or Live Camera from the sidebar.")


elif page == "Image Detection":
    st.header("Image Object Detection")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        st.image(
            image, caption="Original image", width="stretch"
        )

        if st.button("Run Detection", type="primary"):
            try:
                model = load_model()

                start = time.perf_counter()

                with MODEL_LOCK:
                    result = model.predict(
                        image,
                        imgsz=640,
                        conf=confidence,
                        device="cpu",
                        verbose=False
                    )[0]

                elapsed = time.perf_counter() - start
                annotated = result.plot()

                st.session_state.history.append({
                    "objects": len(result.boxes),
                    "seconds": round(elapsed, 3)
                })

                left, right = st.columns(2)

                with left:
                    st.subheader("Detection Result")
                    st.image(
                        annotated[:, :, ::-1],
                        width="stretch"
                    )

                with right:
                    st.subheader("Detection Summary")
                    st.metric("Objects found", len(result.boxes))
                    st.metric("Inference time", f"{elapsed:.2f} s")

                    rows = []

                    for box in result.boxes:
                        class_id = int(box.cls[0].item())

                        rows.append({
                            "Object": model.names[class_id],
                            "Confidence": round(
                                float(box.conf[0].item()), 3
                            )
                        })

                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows),
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.warning("No objects detected.")

                    buffer = io.BytesIO()

                    Image.fromarray(
                        annotated[:, :, ::-1]
                    ).save(buffer, format="JPEG")

                    st.download_button(
                        "Download annotated image",
                        data=buffer.getvalue(),
                        file_name="vision_result.jpg",
                        mime="image/jpeg"
                    )

            except Exception as error:
                st.error(f"Detection failed: {error}")


elif page == "Live Camera":
    st.header("Real-Time Object Detection")

    st.write(
        "Start the camera to detect everyday objects with YOLO11n."
    )

    st.caption(
        "For performance, detection is limited to approximately "
        "5 updates per second. Actual speed depends on server load "
        "and network conditions."
    )

    st.info(
        "Allow camera access when Safari asks. Use a secure HTTPS "
        "connection and a stable internet connection."
    )

    webrtc_streamer(
        key="vision-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {
                    "urls": ["stun:stun.l.google.com:19302"]
                }
            ]
        },
        media_stream_constraints={
            "video": {
                "facingMode": {"ideal": "environment"},
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },
        video_processor_factory=lambda: YOLOVideoProcessor(
            confidence=confidence
        ),
        async_processing=True
    )

    st.caption(
        "If the camera does not connect, check browser permissions "
        "and try again. Some networks restrict WebRTC connections."
    )


elif page == "Analytics":
    st.header("Detection Analytics")

    history = st.session_state.history

    if history:
        df = pd.DataFrame(history)

        col1, col2 = st.columns(2)

        col1.metric("Images processed", len(df))
        col2.metric(
            "Objects detected", int(df["objects"].sum())
        )

        st.line_chart(df["objects"])
        st.dataframe(df, use_container_width=True)

    else:
        st.info("Run image detection to populate analytics.")


elif page == "Settings":
    st.header("Detection Settings")

    st.write(f"Confidence threshold: {confidence:.0%}")
    st.write("Model: YOLO11n")
    st.write("Live camera inference size: 416 pixels")
    st.write("Live camera target: approximately 5 updates/sec")

    st.caption(
        "The first model load may take longer. Detection history "
        "currently lasts only for the active Streamlit session."
    )

