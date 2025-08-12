import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from utils import bbox_overlap, calculate_distance
import time

# Load models once
MASK_MODEL_PATH = "models/best.pt"
PERSON_MODEL_PATH = "models/yolov8n.pt"
HAND_MODEL_PATH = "models/handdsa.pt"

mask_model = YOLO(MASK_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)
hand_model = YOLO(HAND_MODEL_PATH)

HAND_ON_MOUTH_THRESHOLD = 0.40

def process_frame(frame):
    # Run detections
    person_results = person_model(frame, classes=[0], verbose=False)
    mask_results = mask_model(frame, verbose=False)
    hand_results = hand_model(frame, verbose=False)

    def extract_boxes(res):
        items = []
        for r in res:
            for box in r.boxes:
                items.append({
                    'box': tuple(map(int, box.xyxy[0].tolist())),
                    'confidence': float(box.conf[0]),
                    'class': int(box.cls[0])
                })
        return items

    person_detections = extract_boxes(person_results)
    mask_detections = extract_boxes(mask_results)
    hand_detections = extract_boxes(hand_results)
    hand_boxes = [h['box'] for h in hand_detections]

    final_face_detections = []
    for mask_det in mask_detections:
        face_box = mask_det['box']
        label = mask_model.names.get(mask_det['class'], 'unknown')
        confidence = mask_det['confidence']

        # Check hand overlap
        is_hand_on_mouth = any(bbox_overlap(face_box, hb) > HAND_ON_MOUTH_THRESHOLD for hb in hand_boxes)
        if is_hand_on_mouth:
            label = "Hand on Mouth"

        final_face_detections.append({'box': face_box, 'label': label, 'confidence': confidence})

    # Social distancing calculations
    active_people = []
    for p_det in person_detections:
        box = p_det['box']
        x1, y1, x2, y2 = box
        centroid = (int((x1 + x2) / 2), y2)
        height = y2 - y1
        active_people.append({'centroid': centroid, 'height_px': height})

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

    # Draw on frame
    # Draw social distancing lines
    for line in social_distancing:
        color = (0,255,0) if line['safe'] else (0,0,255)
        cv2.line(frame, line['from'], line['to'], color, 2)
        mid = ((line['from'][0] + line['to'][0])//2, (line['from'][1] + line['to'][1])//2)
        cv2.putText(frame, line['distance'], mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw person boxes
    for p in person_detections:
        box = p['box']
        cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), (255,191,0), 2)  # deep sky blue

    # Draw face detections
    color_map = {
        "without_mask": (0,0,255),
        "mask_weared_incorrect": (0,255,255),
        "with_mask": (0,255,0),
        "Hand on Mouth": (255,0,255)
    }
    for det in final_face_detections:
        box = det['box']
        label = det['label']
        conf = det['confidence']
        c = color_map.get(label, (200,200,200))
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), c, 3)
        text = f"{label} {conf*100:.0f}%"
        cv2.putText(frame, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    return frame

iface = gr.Interface(fn=process_frame, inputs=gr.Image(source="webcam", streaming=True), outputs="image",
                     title="Face Mask and Social Distancing Detector",
                     description="Uses YOLOv8 models to detect masks and calculate social distancing.")

if __name__ == "__main__":
    iface.launch()
