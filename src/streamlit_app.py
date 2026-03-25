import os
import io
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
RESULTS_DIR = "results"
MAX_UPLOAD = 75


# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(layout="wide", page_title="Oak-Wilt Detector")
st.title("Grand Haven Parks Oak-Wilt Detector")
st.markdown("Advanced 4-category Oak Wilt classification system | MAX UPLOADS: 75 Images")


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
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = set()


# ===========================
# HELPER FUNCTIONS
# ===========================
def classify_prediction(confidence):
    confidence = confidence * 100
    if confidence > 99.5:
        return "THIS PICTURE HAS OAK WILT"
    elif 90 < confidence <= 99.5:
        return "HIGH CHANCE OF OAK WILTS"
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
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img_input = np.expand_dims(img, axis=0)
    prediction = model.predict(img_input, verbose=0)[0][0]
    classification = classify_prediction(prediction)
    gps = get_gps_data(img_bytes)
    del img_array, img, img_input
    gc.collect()
    return classification, prediction * 100, gps


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
            "lat": r["gps"][0] if r["gps"] else "",
            "lon": r["gps"][1] if r["gps"] else "",
        })
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"oak_wilt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(path, index=False)
    return path


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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"oak_wilt_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson")
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    return path

def render_results(results):
    for i, result in enumerate(results):
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

        if result["classification"] not in CLASSIFICATION_CATEGORIES:
            result["classification"] = "Not an Oak Wilt"

        with col1:
            st.image(result["img_bytes"], caption=result["filename"], use_container_width=True)

        with col2:
            st.write("**Classification**")
            color = CLASSIFICATION_CATEGORIES[result["classification"]]["color"]
            st.markdown(
                f'<span style="color:{color};font-weight:bold">{result["classification"]}</span>',
                unsafe_allow_html=True
            )

        with col3:
            st.write("**Probability of OW**")
            st.write(f"{result['confidence']:.2f}%")

        with col4:
            st.write("**Coordinates**")
            if result["gps"]:
                st.write(f"{result['gps'][0]:.5f}, {result['gps'][1]:.5f}")
            else:
                st.write("No GPS")

        with col5:
            st.write("**Feedback**")
            col_good, col_bad = st.columns(2)
            with col_good:
                if st.button("Good", key=f"good_{i}_{result['filename']}", help="Correct prediction"):
                    st.toast("Thanks! Prediction was correct.")
            with col_bad:
                if st.button("Bad", key=f"bad_{i}_{result['filename']}", help="Incorrect prediction"):
                    st.toast("Thanks! Prediction was incorrect.")

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

    # Deduplicate
    seen = set()
    unique_files = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)

    # Only re-process if the uploaded file set has changed
    uploaded_names = {f.name for f in unique_files}
    if uploaded_names != st.session_state.processed_filenames:
        st.session_state.results = []
        st.session_state.processed_filenames = uploaded_names

        progress = st.progress(0)

        with st.spinner("Analyzing images..."):
            for i, file in enumerate(unique_files):
                img_bytes = file.read()
                classification, confidence, gps = process_image(img_bytes)
                st.session_state.results.append({
                    "filename": file.name,
                    "img_bytes": img_bytes,
                    "classification": classification,
                    "confidence": confidence,
                    "gps": gps
                })
                progress.progress((i + 1) / len(unique_files))
                del img_bytes
                gc.collect()

        progress.empty()

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
    csv_path = generate_csv(results)
    geojson_path = generate_geojson(results)

    # Stacked download buttons
    if csv_path:
        st.download_button(
            "Download CSV",
            data=open(csv_path, "rb").read(),
            file_name=os.path.basename(csv_path),
            mime="text/csv",
            key="dl_csv"
        )
    else:
        st.button("Download CSV", disabled=True, key="dl_csv_disabled")

    if geojson_path:
        st.download_button(
            "Download GeoJSON",
            data=open(geojson_path, "rb").read(),
            file_name=os.path.basename(geojson_path),
            mime="application/geo+json",
            key="dl_geo"
        )
    else:
        st.button("Download GeoJSON", disabled=True, key="dl_geo_disabled")