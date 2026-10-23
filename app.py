import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av
import cv2 

st.set_page_config(page_title="Vision", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; font-weight: 300;'>Live Object Detection</h3>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

class VideoProcessor:
    def __init__(self):
        self.conf_threshold = 0.15 

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        results = model.predict(img, conf=self.conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        
        # Diagnostic HUD
        num_objects = len(results[0].boxes)
        cv2.putText(
            annotated_frame, 
            f"Objects Found: {num_objects}", 
            (20, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            (0, 255, 0), 
            3
        )
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Upgraded from just STUN to include a free public TURN relay for 5G mobile networks
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
})

ctx = webrtc_streamer(
    key="vision-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": {"facingMode": "environment"}, "audio": False},
    video_html_attrs={
        "style": {"width": "100%", "border-radius": "10px"},
        "playsinline": True,
        "autoplay": True,
        "muted": True
    },
    async_processing=True,
)

if ctx.video_processor:
    st.markdown("<br>", unsafe_allow_html=True)
    conf_slider = st.slider("Confidence Threshold", 0.05, 1.0, 0.15, 0.05)
    ctx.video_processor.conf_threshold = conf_slider
