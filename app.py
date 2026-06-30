import streamlit as st
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import deque

from config import Config
from model import SpatiotemporalViT
from biometrics import BiometricTracker

st.set_page_config(page_title="Spatiotemporal Deepfake Detector", layout="wide")
st.title("🛡️ Real-Time Spatiotemporal Deepfake Detector")
st.markdown("This analyzer captures spatiotemporal inconsistencies, neural network structural artifacts, and biometrics.")

@st.cache_resource
def load_deepfake_model():
    model = SpatiotemporalViT().to(Config.DEVICE)
    model.eval()
    return model

model = load_deepfake_model()
tracker = BiometricTracker()

FAKE_CLASS_INDEX = 0

# We only need the buffers for the biometrics charts now
ear_history = deque(maxlen=100)
rppg_history = deque(maxlen=100)

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Live Biometric Diagnostics")
    score_metric = st.metric(label="Deepfake Confidence Score", value="0.0%")
    blink_counter_metric = st.metric(label="Calculated Blink Rate State", value="Normal")
    
    st.markdown("**Real-Time Photoplethysmography (rPPG) Pulse Signal**")
    pulse_chart = st.empty()
    
    st.markdown("**Eye Aspect Ratio (EAR) Trend Line**")
    ear_chart = st.empty()

with col1:
    st.subheader("📹 Monitored Stream Processing")
    run_detection = st.checkbox("Initialize Detection Pipeline", value=False)
    video_placeholder = st.empty()

def generate_xai_overlay(attention_map, face_crop):
    fh, fw, _ = face_crop.shape
    side = int(np.sqrt(attention_map.shape[-1]))
    
    attn_matrix = attention_map.reshape(side, side).cpu().numpy()
    attn_matrix = cv2.resize(attn_matrix, (fw, fh))
    attn_matrix = (attn_matrix - attn_matrix.min()) / (attn_matrix.max() - attn_matrix.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_matrix), cv2.COLORMAP_JET)
    overlayed_face = cv2.addWeighted(face_crop, 0.6, heatmap, 0.4, 0)
    return overlayed_face

if run_detection:
    cap = cv2.VideoCapture(0)
    
    # === HIGH-SPEED STATE VARIABLES ===
    last_fake_prob = 0.0
    last_attn_map = None
    frame_counter = 0
    # ==================================
    
    while cap.isOpened() and run_detection:
        ret, frame = cap.read()
        if not ret:
            st.error("Hardware Video Capture Pipeline Inaccessible.")
            break
            
        frame_counter += 1
        h, w, c = frame.shape
        metrics = tracker.process_frame(frame)
        
        if metrics["bbox"] is not None:
            x1, y1, x2, y2 = metrics["bbox"]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size != 0:
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                clean_face = cv2.bilateralFilter(face_crop_rgb, d=5, sigmaColor=50, sigmaSpace=50)
                
                resized_face = cv2.resize(clean_face, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
                normalized_face = resized_face.astype(np.float32) / 255.0
                
                ear_history.append(metrics["ear"])
                rppg_history.append(metrics["rppg_val"])
                
                # === THE INSTANT FIX: Evaluate 1 frame at a time, every 2nd frame ===
                if frame_counter % 2 == 0:
                    # Instantly convert the current frame to the 5D tensor required by the model
                    single_frame_tensor = torch.tensor(normalized_face).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
                    
                    with torch.inference_mode():
                        predictions = model(single_frame_tensor)
                        probabilities = F.softmax(predictions, dim=1)
                        current_prob = probabilities[0][FAKE_CLASS_INDEX].item()
                        
                        # Smooth the numbers slightly so they don't visually flicker wildly
                        if last_fake_prob == 0.0:
                            last_fake_prob = current_prob
                        else:
                            last_fake_prob = (0.6 * last_fake_prob) + (0.4 * current_prob)
                            
                        last_attn_map = model.get_last_attention_map()[0]

                # Instantly Update the Dashboard UI Score
                score_metric.metric(label="Deepfake Confidence Score", value=f"{last_fake_prob * 100:.2f}%")
                
                # Instantly Draw the Red/Green Boxes
                if last_attn_map is not None:
                    if last_fake_prob >= Config.CONFIDENCE_THRESHOLD:
                        label_color = (0, 0, 255)
                        frame[y1:y2, x1:x2] = generate_xai_overlay(last_attn_map, face_crop)
                        cv2.putText(frame, f"FAKE CONFIDENCE: {last_fake_prob*100:.1f}%", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
                    else:
                        label_color = (0, 255, 0)
                        cv2.putText(frame, "VERIFIED REAL", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)

                # Process Biometric UI
                if metrics["ear"] < Config.EAR_THRESHOLD:
                    blink_counter_metric.metric(label="Calculated Blink Rate State", value="Blink Tracked", delta="Closed")
                else:
                    blink_counter_metric.metric(label="Calculated Blink Rate State", value="Tracking Open")

                ear_chart.line_chart(pd.DataFrame(list(ear_history), columns=["Eye Aspect Ratio"]))
                rppg_signals = list(rppg_history)
                standardized_rppg = (np.array(rppg_signals) - np.mean(rppg_signals)).tolist() if len(rppg_signals) > 1 else rppg_signals
                pulse_chart.line_chart(pd.DataFrame(standardized_rppg, columns=["Pulse Variance Profile"]))

        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
    cap.release()
else:
    st.info("Toggle 'Initialize Detection Pipeline' to connect local live hardware resources.")