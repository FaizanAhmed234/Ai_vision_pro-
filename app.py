import av
import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Live Detection", page_icon="🎯", layout="centered")

RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

MODELS = {
    "YOLOv8n (fastest)": "yolov8n.pt",
    "YOLOv8s (more accurate)": "yolov8s.pt",
}


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


class YoloProcessor(VideoProcessorBase):
    def __init__(self, model: YOLO):
        self.model = model
        self.conf = 0.4
        self.imgsz = 320

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(
            img, conf=self.conf, imgsz=self.imgsz, verbose=False, device="cpu"
        )
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


st.title("🎯 YOLO Live Detection")
st.caption("Real-time object detection from your phone camera.")

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(MODELS.keys()))
    conf = st.slider("Confidence", 0.1, 0.9, 0.4, 0.05)
    imgsz = st.select_slider("Inference size (smaller = faster)", [224, 320, 416, 640], value=320)
    facing = st.radio("Camera", ["Back", "Front"], horizontal=True)

model = load_model(MODELS[model_label])

ctx = webrtc_streamer(
    key=f"yolo-{model_label}-{facing}",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=lambda: YoloProcessor(model),
    media_stream_constraints={
        "video": {
            "facingMode": "environment" if facing == "Back" else "user",
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15},
        },
        "audio": False,
    },
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.conf = conf
    ctx.video_processor.imgsz = imgsz

st.info("Tap START, then allow camera access when Safari asks.")
