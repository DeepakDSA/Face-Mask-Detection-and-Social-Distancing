# app.py
import base64
import time
import cv2
import numpy as np
import os
from collections import deque
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Import the updated utility functions
from utils import bbox_overlap, calculate_distance

# ---------------- CONFIG ----------------
MODELS_DIR = "models"
MASK_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
PERSON_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "handdsa.pt")

# --- Processing & Accuracy Tuning ---
RUN_SIZE_LIVE = 640
RUN_SIZE_UPLOAD = 1024
CONF_PERSON = 0.50
CONF_HAND = 0.50
# This is the overlap percentage from your script
HAND_ON_MOUTH_THRESHOLD = 0.40 

# --- Tracking & Smoothing ---
SMOOTH_WINDOW = 5
TRACK_MATCH_DIST = 80
TRACK_MAX_AGE = 2.0

app = Flask(__name__)

# --- LOAD MODELS ---
print("Loading YOLO models...")
mask_model = YOLO(MASK_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)
hand_model = YOLO(HAND_MODEL_PATH)
print("Models ready.")

# --- TRACKING STATE (in-memory) ---
tracks = {}
_next_track_id = 0
def get_next_id():
    global _next_track_id
    _next_track_id += 1
    return _next_track_id

def extract_boxes(res):
    """Helper to get box data from YOLO results."""
    items = []
    try:
        for r in res:
            for box in r.boxes:
                items.append({
                    'box': tuple(map(int, box.xyxy[0].tolist())),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0])
                })
    except Exception:
        return []
    return items

def prune_tracks():
    now = time.time()
    to_remove = [tid for tid, tr in tracks.items() if now - tr['last_seen'] > TRACK_MAX_AGE]
    for tid in to_remove:
        del tracks[tid]

def match_track(centroid):
    best_id, best_dist = None, float('inf')
    cx, cy = centroid
    for tid, tr in tracks.items():
        tcx, tcy = tr['centroid']
        dist_sq = (tcx - cx)**2 + (tcy - cy)**2
        if dist_sq < best_dist:
            best_dist = dist_sq
            best_id = tid
    if best_id is not None and best_dist <= TRACK_MATCH_DIST**2:
        return best_id
    return None

def get_smoothed_label(history):
    if not history: return "without_mask", 0.0
    labels = [label for label, conf in history]
    best_label = max(set(labels), key=labels.count)
    avg_conf = np.mean([conf for label, conf in history if label == best_label])
    return best_label, float(avg_conf)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        payload = request.get_json()
        img_b64 = payload["image"].split(",", 1)[1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        
        # --- Run all detections ---
        # Your script runs detections on the full frame, so we will do the same.
        person_results = person_model(frame, classes=[0], verbose=False)
        mask_results = mask_model(frame, verbose=False)
        hand_results = hand_model(frame, verbose=False)

        person_detections = extract_boxes(person_results)
        mask_detections = extract_boxes(mask_results)
        hand_detections = extract_boxes(hand_results)

        # --- LOGIC TO MERGE MASK AND HAND DETECTIONS ---
        # This new list will hold the final, combined detections
        final_face_detections = []
        
        # Create a list of hand boxes for easy iteration
        hand_boxes = [h['box'] for h in hand_detections]

        for mask_det in mask_detections:
            face_box = mask_det['box']
            label = mask_model.names.get(mask_det['class'], 'unknown')
            confidence = mask_det['confidence']

            # Check for hand overlap, just like in your script
            is_hand_on_mouth = False
            for hand_box in hand_boxes:
                if bbox_overlap(face_box, hand_box) > HAND_ON_MOUTH_THRESHOLD:
                    is_hand_on_mouth = True
                    break
            
            # If overlap, override the label
            if is_hand_on_mouth:
                label = "Hand on Mouth"

            final_face_detections.append({
                'box': face_box,
                'label': label,
                'confidence': confidence
            })

        # --- Person tracking for social distancing ---
        # We use the separate person detections for stable social distance lines
        now = time.time()
        seen_track_ids = set()
        active_people = []

        for p_det in person_detections:
            box = p_det['box']
            x1, y1, x2, y2 = box
            centroid = (int((x1 + x2) / 2), y2) # Bottom-center for better distance
            height = y2 - y1

            # We don't need complex tracking here, just a list of current people
            active_people.append({'centroid': centroid, 'height_px': height})

        # --- Calculate Social Distancing ---
        social_distancing = []
        for i in range(len(active_people)):
            for j in range(i + 1, len(active_people)):
                p1 = active_people[i]
                p2 = active_people[j]
                dist = calculate_distance(p1['centroid'], p1['height_px'], p2['centroid'], p2['height_px'])
                social_distancing.append({
                    'from': p1['centroid'],
                    'to': p2['centroid'],
                    'distance': f"{dist:.1f} cm",
                    'safe': dist >= 150.0 # Standard 1.5 meter distance
                })

        # The final JSON payload for the frontend
        return jsonify({
            'face_detections': final_face_detections,
            'person_boxes': [p['box'] for p in person_detections], # Send person boxes for drawing
            'social_distancing': social_distancing
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
