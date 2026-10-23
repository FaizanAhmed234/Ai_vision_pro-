import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av

# --- 1. MINIMAL UI CONFIGURATION ---
st.set_page_config(page_title="Vision", layout="centered")

# Inject custom CSS to hide the default Streamlit menus and footers for a sleek look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; font-weight: 300; margin-bottom: 0;'>Live Object Detection</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 0.9rem;'>Point your camera at objects to detect them in real-time.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- 2. MODEL CACHING ---
@st.cache_resource(show_spinner=False)
def load_model():
    # The nano model (yolo11n) is the most efficient for live cloud processing
    return YOLO("yolo11n.pt")

model = load_model()

# --- 3. VIDEO PROCESSING LOGIC ---
class VideoProcessor:
    def __init__(self):
        self.conf_threshold = 0.5

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Run inference (verbose=False keeps server logs clean)
        results = model.predict(img, conf=self.conf_threshold, verbose=False)
        
        # Draw the sleek bounding boxes provided by ultralytics
        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# --- 4. WEBRTC STREAM ---
# Google STUN server ensures the video connects cleanly over cellular data
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Render the camera widget
ctx = webrtc_streamer(
    key="vision-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "facingMode": "environment" # Defaults to the rear camera on mobile
        },
        "audio": False
    },
    async_processing=True,
)

# --- 5. MINIMAL CONTROLS ---
# Only show the slider when the camera stream is actively running
if ctx.video_processor:
    st.markdown("<br>", unsafe_allow_html=True)
    conf_slider = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
    )
    ctx.video_processor.conf_threshold = conf_slider
