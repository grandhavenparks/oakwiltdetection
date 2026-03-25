---
title: Oakwiltdetection
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Advanced 4-category Oak Wilt classification system
---

# Grand Haven Parks Oak-Wilt Detector

A Streamlit-based application that classifies oak trees into four categories related to oak wilt, using a TensorFlow model. It also extracts GPS data from images and provides CSV/GeoJSON export functionality for detected cases.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Model Placement](#model-placement)
6. [Running Locally](#running-locally)
7. [Pushing to Hugging Face Spaces](#pushing-to-hugging-face-spaces)
8. [Deployment Considerations](#deployment-considerations)
9. [App Usage](#app-usage)
10. [Known Limitations](#known-limitations)
11. [File Structure](#file-structure)

---

## Project Overview

This app allows users to upload images of oak trees to detect oak wilt disease. It:

* Predicts oak wilt probability using a pre-trained TensorFlow model.
* Categorizes images into four confidence-based classes.
* Extracts GPS coordinates if available in EXIF metadata.
* Generates CSV and GeoJSON files for mapping or record-keeping.

---

## Features

* Four-category classification:

  * **THIS PICTURE HAS OAK WILT** (>99.5%)
  * **HIGH CHANCE OF OAK WILT** (90–99.5%)
  * **CHANGES OF COLORS ON TREE LEAVES** (70–90%)
  * **Not an Oak Wilt** (<70%)
* Handles up to 75 images per upload.
* Deduplicates images by filename.
* Allows users to give feedback on predictions.
* Provides sidebar with confidence info for categories.
* CSV and GeoJSON export for further analysis.

---

## Requirements

* Python ≥ 3.10
* Streamlit
* TensorFlow
* OpenCV (`cv2`)
* NumPy
* Pillow
* pandas

Install via:

```bash
pip install streamlit tensorflow opencv-python-headless numpy pillow pandas
```

> **Note:** Use `opencv-python-headless` for environments without GUI support (like Hugging Face Spaces).

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. Ensure the model is in the correct path (see [Model Placement](#model-placement)).

3. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Placement

* Place your TensorFlow model in `src/oak_wilt_3.h5`.
* Ensure the filename matches `MODEL_PATH` in the code.

> Hugging Face Spaces requires **all models under your repository**, or you can load large models from the Hub using `from_pretrained()`.

---

## Running Locally

```bash
streamlit run app.py
```

* Visit `http://localhost:8501` to use the app.
* Upload images (JPG/PNG, max 75).
* Review classification, probability, GPS coordinates, and export results.

---

## Pushing to Hugging Face Spaces

1. **Create a Space** on Hugging Face:

   * Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   * Create a **Streamlit** space.

2. **Add your Hugging Face remote**:

```bash
git remote add hf https://huggingface.co/spaces/<username>/<space-name>
```

3. **Push your code**:

```bash
git add .
git commit -m "Initial push"
git push hf main
```

> Hugging Face Spaces automatically deploys after a successful push to `main`.
> Any subsequent pushes to `main` will trigger redeployment.

4. **Tips for deployment**:

   * Include `requirements.txt` with all Python dependencies.
   * Keep your model under `src/` or use a link to HF Hub for large models.
   * Avoid heavy files >500 MB if possible—consider model compression.

---

## Deployment Considerations

* **Resource limits**: Hugging Face Spaces free tier limits RAM (~4GB). Ensure model + uploads fit within this.
* **GPU acceleration**: Use `tensorflow` with CPU if GPU is unavailable. Large image batches may crash CPU-only environments.
* **Session state & caching**: Streamlit `st.cache_resource` is used for model loading to prevent repeated expensive loading.
* **Garbage collection**: Images are explicitly deleted after processing (`gc.collect()`) to save memory.

---

## App Usage

1. Upload images (JPG/PNG).
2. Monitor upload progress.
3. Filter results by classification.
4. Click **Good/Bad** feedback buttons for predictions.
5. Export results via CSV or GeoJSON.

---

## Known Limitations

* Only images with GPS EXIF data will populate GeoJSON coordinates.
* Maximum upload of 75 images to prevent memory overload.
* Very large images may slow processing—consider resizing before upload.
* Model predictions are probabilistic and may require human verification.

---

## File Structure

```
├── app.py                 # Main Streamlit app
├── src/
│   └── oak_wilt_3.h5      # TensorFlow model
├── results/               # Exported CSV and GeoJSON
├── requirements.txt       # Dependencies
└── README.md              # This file
```

