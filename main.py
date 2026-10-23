
import streamlit as st

st.set_page_config(
    page_title="AI Vision Lab",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ AI Vision Lab")
st.caption("Intelligent Object Detection & Visual Analytics")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a workspace",
    [
        "Dashboard",
        "Image Detection",
        "Video Detection",
        "Analytics",
        "Settings"
    ]
)

if page == "Dashboard":
    st.header("Welcome to AI Vision Lab")

    st.write(
        "An AI-powered computer vision platform "
        "for detecting and analyzing objects in images and videos."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Objects Detected", "—")
    col2.metric("Images Processed", "0")
    col3.metric("Detection Status", "Not Started")

    st.divider()

    st.subheader("Get Started")
    st.write("Select Image Detection from the sidebar to begin.")

elif page == "Image Detection":
    st.header("Image Detection")
    st.info("The YOLO detection engine will be added next.")

elif page == "Video Detection":
    st.header("Video Detection")
    st.info("Video processing will be added in a later stage.")

elif page == "Analytics":
    st.header("Detection Analytics")
    st.info("Your detection statistics will appear here.")

elif page == "Settings":
    st.header("Settings")
    st.write("Configure your detection preferences here.")
