# app.py
import base64
import cv2
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Import the utility functions
from utils import bbox_overlap, calculate_distance

# ---------------- CONFIG ----------------
# This line tells the app to look for models in the 'models' folder
MODELS_DIR = "models"
MASK_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")
PERSON_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "handdsa.pt")

# This is the overlap percentage from your script
HAND_ON_MOUTH_THRESHOLD = 0.40 

app = Flask(__name__)

# --- LOAD MODELS ---
print("Loading YOLO models...")
mask_model = YOLO(MASK_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)
hand_model = YOLO(HAND_MODEL_PATH)
print("Models ready.")

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        payload = request.get_json()
        img_b64 = payload["image"].split(",", 1)[1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)
        
        # --- Run all detections on the full frame ---
        person_results = person_model(frame, classes=[0], verbose=False)
        mask_results = mask_model(frame, verbose=False)
        hand_results = hand_model(frame, verbose=False)

        person_detections = extract_boxes(person_results)
        mask_detections = extract_boxes(mask_results)
        hand_detections = extract_boxes(hand_results)

        # --- LOGIC TO MERGE MASK AND HAND DETECTIONS ---
        final_face_detections = []
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
            
            if is_hand_on_mouth:
                label = "Hand on Mouth"

            final_face_detections.append({
                'box': face_box,
                'label': label,
                'confidence': confidence
            })

        # --- Person tracking for social distancing ---
        active_people = []
        for p_det in person_detections:
            box = p_det['box']
            x1, y1, x2, y2 = box
            centroid = (int((x1 + x2) / 2), y2)
            height = y2 - y1
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
                    'safe': dist >= 150.0
                })

        return jsonify({
            'face_detections': final_face_detections,
            'person_boxes': [p['box'] for p in person_detections],
            'social_distancing': social_distancing
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
