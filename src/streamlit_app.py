import os
import io
import hashlib
import json
from datetime import datetime
import gc

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ExifTags
import pandas as pd

# ===========================
# CONFIG
# ===========================
MODEL_PATH = "src/oak_wilt_3.h5"

CLASSIFICATION_CATEGORIES = {
    "THIS PICTURE HAS OAK WILT": {"min_conf": 99.5, "color": "#FF0000"},
    "HIGH CHANCE OF OAK WILT": {"min_conf": 90, "max_conf": 99.5, "color": "#FF6600"},
    "CHANGES OF COLORS ON TREE LEAVES": {"min_conf": 70, "max_conf": 90, "color": "#FFAA00"},
    "Not an Oak Wilt": {"max_conf": 70, "color": "#00AA00"}
}

IMG_SIZE = 256
MAX_UPLOAD = 75
THUMB_WIDTH = 600


# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(layout="wide", page_title="Oak-Wilt Detector")
st.title("Grand Haven Parks Oak-Wilt Detector")
st.markdown("Advanced 4-category Oak Wilt classification system | MAX UPLOADS: 75 Images")

st.markdown("""
<style>
/* "Loading images..." spinner under the uploader while files upload */
[data-testid="stFileUploader"]:has([role="progressbar"]) {
    position: relative;
    padding-bottom: 2.25rem;
}
[data-testid="stFileUploader"]:has([role="progressbar"])::before {
    content: "";
    position: absolute;
    left: 0;
    bottom: 0.4rem;
    width: 1.1rem;
    height: 1.1rem;
    border: 3px solid rgba(128, 128, 128, 0.3);
    border-top-color: #1E8E3E;
    border-radius: 50%;
    animation: upload-spin 0.8s linear infinite;
}
[data-testid="stFileUploader"]:has([role="progressbar"])::after {
    content: "Loading images...";
    position: absolute;
    left: 1.8rem;
    bottom: 0.35rem;
    font-weight: 600;
}
@keyframes upload-spin {
    to { transform: rotate(360deg); }
}

/* Green feedback message above the Good/Bad buttons */
.fb-close {
    display: none;
}
.fb-banner {
    position: relative;
    background: #1E8E3E;
    color: #FFFFFF;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 0.6rem 2rem 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
    animation: fb-hide 0s 3s forwards;
}
.fb-x {
    position: absolute;
    top: 0.15rem;
    right: 0.5rem;
    cursor: pointer;
    font-size: 1.1rem;
    line-height: 1.2;
}
.fb-close:checked + .fb-banner {
    display: none;
}
@keyframes fb-hide {
    to { visibility: hidden; height: 0; padding: 0; margin: 0; overflow: hidden; }
}
</style>
""", unsafe_allow_html=True)


# ===========================
# MODEL LOADING
# ===========================
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.isfile(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


model = load_model()


# ===========================
# SESSION STATE
# ===========================
if "results" not in st.session_state:
    st.session_state.results = []
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()
if "failed_files" not in st.session_state:
    st.session_state.failed_files = []
if "feedback_count" not in st.session_state:
    st.session_state.feedback_count = 0


# ===========================
# HELPER FUNCTIONS
# ===========================
def classify_prediction(confidence):
    confidence = confidence * 100
    if confidence > 99.5:
        return "THIS PICTURE HAS OAK WILT"
    elif 90 < confidence <= 99.5:
        return "HIGH CHANCE OF OAK WILT"
    elif 70 < confidence <= 90:
        return "CHANGES OF COLORS ON TREE LEAVES"
    else:
        return "Not an Oak Wilt"


def convert_to_degrees(value):
    d, m, s = value
    return d + (m / 60.0) + (s / 3600.0)


def get_gps_data(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes))
        exif_data = img._getexif()
        if not exif_data:
            return None
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = ExifTags.GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]
                if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                    lat = convert_to_degrees(gps_data["GPSLatitude"])
                    lon = convert_to_degrees(gps_data["GPSLongitude"])
                    if gps_data.get("GPSLatitudeRef") != "N":
                        lat = -lat
                    if gps_data.get("GPSLongitudeRef") != "E":
                        lon = -lon
                    return (lat, lon)
    except Exception:
        pass
    return None


def process_image(img_bytes):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    # Small JPEG thumbnail for display; the full-size upload is never kept
    h, w = img.shape[:2]
    if w > THUMB_WIDTH:
        thumb = cv2.resize(img, (THUMB_WIDTH, int(h * THUMB_WIDTH / w)), interpolation=cv2.INTER_AREA)
    else:
        thumb = img
    _, thumb_buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
    thumbnail_bytes = thumb_buf.tobytes()

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img, axis=0)
    prediction = float(model(img_input, training=False)[0][0])
    classification = classify_prediction(prediction)
    gps = get_gps_data(img_bytes)
    del img_array, img, img_input, thumb, thumb_buf
    return classification, prediction * 100, gps, thumbnail_bytes


def generate_csv(results):
    positive = [r for r in results if r["classification"] != "Not an Oak Wilt"]
    if not positive:
        return None
    rows = []
    for r in positive:
        rows.append({
            "filename": r["filename"],
            "classification": r["classification"],
            "confidence": r["confidence"],
            "latitude": r["gps"][0] if r["gps"] else "",
            "longitude": r["gps"][1] if r["gps"] else "",
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def generate_geojson(results):
    positive = [r for r in results if r["classification"] != "Not an Oak Wilt" and r["gps"]]
    if not positive:
        return None
    geojson = {"type": "FeatureCollection", "features": []}
    for r in positive:
        lat, lon = r["gps"]
        geojson["features"].append({
            "type": "Feature",
            "properties": {
                "filename": r["filename"],
                "confidence": f"{r['confidence']:.2f}%",
                "classification": r["classification"]
            },
            "geometry": {"type": "Point", "coordinates": [lon, lat]}
        })
    return json.dumps(geojson, indent=2).encode("utf-8")

def feedback_banner(message, row):
    # Alternate the outer tag on each click so the browser builds a fresh box,
    # restarting the 3-second timer and clearing an earlier close click
    tag = "section" if st.session_state.feedback_count % 2 else "div"
    box_id = f"fb-close-{row}"
    return (
        f'<{tag}>'
        f'<input type="checkbox" id="{box_id}" class="fb-close">'
        f'<div class="fb-banner">{message}'
        f'<label for="{box_id}" class="fb-x">&times;</label>'
        f'</div>'
        f'</{tag}>'
    )


def render_results(results):
    for i, result in enumerate(results):
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

        if result["classification"] not in CLASSIFICATION_CATEGORIES:
            result["classification"] = "Not an Oak Wilt"

        with col1:
            st.image(result["thumbnail"], caption=result["filename"], width="stretch")

        with col2:
            st.write("**Classification**")
            color = CLASSIFICATION_CATEGORIES[result["classification"]]["color"]
            st.markdown(
                f'<span style="color:{color};font-weight:bold">{result["classification"]}</span>',
                unsafe_allow_html=True
            )

        with col3:
            st.write("**Confidence of OW**")
            st.write(f"{result['confidence']:.2f}%")

        with col4:
            st.write("**Coordinates**")
            if result["gps"]:
                st.write(f"{result['gps'][0]:.5f}, {result['gps'][1]:.5f}")
            else:
                st.write("No GPS")

        with col5:
            st.write("**Feedback**")
            feedback_msg = st.empty()
            col_good, col_bad = st.columns(2)
            with col_good:
                if st.button("Good", key=f"good_{result['id']}", help="Correct prediction"):
                    st.session_state.feedback_count += 1
                    feedback_msg.markdown(
                        feedback_banner("Thanks! Prediction was correct.", i),
                        unsafe_allow_html=True
                    )
            with col_bad:
                if st.button("Bad", key=f"bad_{result['id']}", help="Incorrect prediction"):
                    st.session_state.feedback_count += 1
                    feedback_msg.markdown(
                        feedback_banner("Thanks! Prediction was incorrect.", i),
                        unsafe_allow_html=True
                    )

        st.markdown("---")


# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.header("Classification Info")
    for category, config in CLASSIFICATION_CATEGORIES.items():
        if "min_conf" in config and "max_conf" in config:
            conf_range = f"{config['min_conf']}-{config['max_conf']}%"
        elif "min_conf" in config:
            conf_range = f">{config['min_conf']}%"
        else:
            conf_range = f"≤{config['max_conf']}%"
        st.markdown(category)
        st.caption(f"Confidence: {conf_range}")
        st.markdown("---")


# ===========================
# MAIN UI
# ===========================
files = st.file_uploader(
    "Upload JPG/PNG images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if files:
    if len(files) > MAX_UPLOAD:
        st.error(f"Cannot upload more than {MAX_UPLOAD} images. Please select fewer images.")
        st.stop()

    # Deduplicate by content, so different photos that share a filename are all kept
    unique_files = {}
    for f in files:
        unique_files.setdefault(hashlib.md5(f.getvalue()).hexdigest(), f)

    # Only re-process if the uploaded file set has changed
    uploaded_ids = set(unique_files)
    if uploaded_ids != st.session_state.processed_ids:
        st.session_state.results = []
        st.session_state.failed_files = []
        st.session_state.processed_ids = uploaded_ids

        progress = st.progress(0)

        with st.spinner("Analyzing images..."):
            for i, (file_id, file) in enumerate(unique_files.items()):
                img_bytes = file.getvalue()
                try:
                    classification, confidence, gps, thumbnail = process_image(img_bytes)
                except Exception:
                    st.session_state.failed_files.append(file.name)
                    progress.progress((i + 1) / len(unique_files))
                    continue
                st.session_state.results.append({
                    "id": file_id,
                    "filename": file.name,
                    "thumbnail": thumbnail,
                    "classification": classification,
                    "confidence": confidence,
                    "gps": gps
                })
                progress.progress((i + 1) / len(unique_files))
                del img_bytes

        gc.collect()
        progress.empty()
else:
    # Uploads cleared: drop the previous batch
    st.session_state.results = []
    st.session_state.failed_files = []
    st.session_state.processed_ids = set()

if st.session_state.failed_files:
    st.warning(
        "Could not read these images, so they were skipped: "
        + ", ".join(st.session_state.failed_files)
    )

if st.session_state.results:
    results = st.session_state.results

    # Filter dropdown
    filter_options = ["All"] + list(CLASSIFICATION_CATEGORIES.keys())
    selected_filter = st.selectbox("Filter by classification", filter_options)

    filtered = results if selected_filter == "All" else [
        r for r in results if r["classification"] == selected_filter
    ]

    st.write(f"Showing {len(filtered)} of {len(results)} images")
    st.markdown("---")

    render_results(filtered)

    # Export
    st.subheader("Export Results")
    csv_data = generate_csv(results)
    geojson_data = generate_geojson(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Stacked download buttons
    if csv_data:
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"oak_wilt_results_{timestamp}.csv",
            mime="text/csv",
            key="dl_csv"
        )
    else:
        st.button("Download CSV", disabled=True, key="dl_csv_disabled")

    if geojson_data:
        st.download_button(
            "Download GeoJSON",
            data=geojson_data,
            file_name=f"oak_wilt_map_{timestamp}.geojson",
            mime="application/geo+json",
            key="dl_geo"
        )
    else:
        st.button("Download GeoJSON", disabled=True, key="dl_geo_disabled")