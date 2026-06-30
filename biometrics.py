import cv2
import numpy as np
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import Config

class BiometricTracker:
    def __init__(self):
        # 1. Automatically download the required Face Landmarker model 
        self.model_path = "face_landmarker.task"
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe Face model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 
                self.model_path
            )
            
        # 2. Initialize the modern MediaPipe Tasks API (Python 3.12 Compatible)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # Eye landmarks indices for EAR calculations
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        # Forehead landmark index for rPPG tracking
        self.FOREHEAD = 10 

    def calculate_ear(self, landmarks, eye_indices, img_w, img_h):
        """Calculates Eye Aspect Ratio (EAR) to measure eye opening state."""
        pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h]) for i in eye_indices]
        d_v1 = np.linalg.norm(pts[1] - pts[5])
        d_v2 = np.linalg.norm(pts[2] - pts[4])
        d_h = np.linalg.norm(pts[0] - pts[3])
        return (d_v1 + d_v2) / (2.0 * d_h + 1e-6)

    def extract_rppg_signal(self, frame, landmarks, img_w, img_h):
        """Extracts the average normalized green-channel intensity from the forehead region."""
        fh_x = int(landmarks[self.FOREHEAD].x * img_w)
        fh_y = int(landmarks[self.FOREHEAD].y * img_h)
        
        roi_radius = 10
        roi = frame[max(0, fh_y-roi_radius):min(img_h, fh_y+roi_radius), 
                    max(0, fh_x-roi_radius):min(img_w, fh_x+roi_radius)]
        if roi.size == 0:
            return 0.0
        return np.mean(roi[:, :, 1])

    def process_frame(self, frame):
        """Processes a single frame and returns tracking metadata alongside bounding boxes."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert standard frame to MediaPipe Image format required by the new API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        metrics = {"ear": 0.3, "rppg_val": 0.0, "bbox": None, "landmarks": None}
        
        # Check if any faces were detected
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            metrics["landmarks"] = landmarks
            
            # Compute EAR
            left_ear = self.calculate_ear(landmarks, self.LEFT_EYE, w, h)
            right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, w, h)
            metrics["ear"] = (left_ear + right_ear) / 2.0
            
            # Compute rPPG signal point
            metrics["rppg_val"] = self.extract_rppg_signal(frame, landmarks, w, h)
            
            # Form standard face Bounding Box
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]
            metrics["bbox"] = (int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords)))
            
        return metrics