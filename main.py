

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time

st.set_page_config(
    page_title="AI Vision Lab",
    page_icon="👁️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

st.title("👁️ AI Vision Lab")
st.caption("AI-powered Object Detection & Visual Analytics")

with st.sidebar:
    st.header("Vision Controls")
    page = st.radio(
        "Workspace",
        ["Dashboard", "Image Detection", "Analytics", "Settings"]
    )
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05
    )

if "history" not in st.session_state:
    st.session_state.history = []

if page == "Dashboard":
    st.header("Welcome to Vision Lab")
    st.write(
        "Detect and analyze objects in images using a "
        "pretrained YOLO computer vision model."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Images processed", len(st.session_state.history))
    col2.metric(
        "Objects detected",
        sum(item["objects"] for item in st.session_state.history)
    )
    col3.metric("Model", "YOLO11n")

    st.info("Open Image Detection in the sidebar to get started.")

elif page == "Image Detection":
    st.header("Image Object Detection")
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Original image", width="stretch")

        if st.button("Run Detection", type="primary"):
            try:
                with st.spinner("Loading model and detecting objects..."):
                    model = load_model()
                    start = time.perf_counter()

                    results = model.predict(
                        image,
                        conf=confidence,
                        verbose=False
                    )

                    elapsed = time.perf_counter() - start
                    result = results[0]
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
                        caption="Detected objects",
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
                        st.warning("No objects detected at this threshold.")

                    
                    import io

                    buffer = io.BytesIO()
                    Image.fromarray(
                        annotated[:, :, ::-1]
                    ).convert("RGB").save(buffer, format="JPEG")

                    st.download_button(
                        "Download annotated image",
                        data=buffer.getvalue(),
                        file_name="vision_result.jpg",
                        mime="image/jpeg"
                    )

            except Exception as error:
                st.error(f"Detection failed: {error}")

elif page == "Analytics":
    st.header("Detection Analytics")

    history = st.session_state.history
    if history:
        df = pd.DataFrame(history)
        col1, col2 = st.columns(2)
        col1.metric("Images processed", len(df))
        col2.metric("Total objects detected", int(df["objects"].sum()))
        st.line_chart(df["objects"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run image detection to populate analytics.")

elif page == "Settings":
    st.header("Detection Settings")
    st.write(f"Confidence threshold: {confidence:.0%}")
    st.write("Model: YOLO11n pretrained object detector")
    st.caption(
        "The first inference may download the model weights. "
        "Performance depends on available compute resources."
    )

