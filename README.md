# 🛡️ Real-Time Spatiotemporal Deepfake Detector

An enterprise-grade, real-time computer vision system designed to detect AI-generated deepfakes. This system utilizes a **Spatiotemporal Vision Transformer (ViT)** architecture combined with live biometric analysis to identify synthetic manipulation in video streams.

## 🚀 Key Features

* **Spatiotemporal Analysis:** Detects inconsistencies across both space (pixel manipulation) and time (frame-to-frame coherence).
* **Live Biometric Diagnostics:** Tracks **Eye Aspect Ratio (EAR)** for blink analysis and **Photoplethysmography (rPPG)** for physiological pulse signal variance.
* **Explainable AI (XAI):** Real-time heatmap overlays generated via attention maps, visualizing exactly where the model detects "fake" structural artifacts.
* **Adaptive Denoising:** Employs **Bilateral Filtering** to neutralize webcam ISO grain, lighting glares, and background artifacts, ensuring the model sees clear, structural data.
* **Production-Ready Pipeline:** Streamlined for low-latency inference using batch processing and frame-skipping techniques.

## 🛠️ Tech Stack

* **Framework:** Streamlit (Frontend/Orchestration)
* **AI Engine:** PyTorch (Spatiotemporal ViT)
* **Computer Vision:** OpenCV & MediaPipe (Face Landmarks/Biometrics)
* **Streaming:** WebRTC (for cross-platform browser support)
* **Deployment:** GitHub Actions & Hugging Face Spaces

## 🏗️ System Pipeline

1.  **Ingestion:** Browser-based video capture via WebRTC.
2.  **Preprocessing:** Bilateral filter application to minimize "Webcam Domain Gap."
3.  **Biometric Extraction:** Real-time facial landmark mesh tracking.
4.  **Inference:** Neural network evaluation via Spatiotemporal Transformer.
5.  **Visualization:** Real-time Red/Green bounding box updates with XAI heatmaps.

## 📦 Requirements

Your deployment environment must include both Python libraries and Linux system dependencies.

### `requirements.txt`
```text
streamlit
streamlit-webrtc
opencv-python-headless
torch
torchvision
transformers
mediapipe
pandas
av