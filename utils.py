# utils.py
import math

def bbox_overlap(face_box, hand_box):
    """
    Calculates the percentage of the face_box area that is covered by the hand_box.
    This matches the logic from your local script.
    """
    fx1, fy1, fx2, fy2 = face_box
    hx1, hy1, hx2, hy2 = hand_box

    # Calculate the intersection area
    inter_x1 = max(fx1, hx1)
    inter_y1 = max(fy1, hy1)
    inter_x2 = min(fx2, hx2)
    inter_y2 = min(fy2, hy2)

    interArea = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate the area of the face box
    faceBoxArea = (fx2 - fx1) * (fy2 - fy1)

    # If the face area is zero, avoid division by zero
    if faceBoxArea == 0:
        return 0.0

    # Return the percentage of the face covered by the hand
    return interArea / float(faceBoxArea)

# --- YOUR DISTANCE CALIBRATION LOGIC ---
# Using the exact functions and constants from your script
KNOWN_DISTANCE_CM = 100
KNOWN_HEIGHT_PX = 390

def estimate_distance_cm(height_px):
    """Estimates depth (distance from camera) based on a person's pixel height."""
    if height_px == 0:
        return float('inf')
    return (KNOWN_DISTANCE_CM * KNOWN_HEIGHT_PX) / height_px

def calculate_distance(p1, h1, p2, h2):
    """
    Estimates the 3D distance between two people using your proven logic.
    """
    d1 = estimate_distance_cm(h1)
    d2 = estimate_distance_cm(h2)
    dx_px = abs(p1[0] - p2[0])
    
    # Use an average pixel-to-cm ratio for horizontal distance
    avg_height_px = (h1 + h2) / 2
    if avg_height_px == 0: return float('inf')
    px_per_cm_at_known_dist = KNOWN_HEIGHT_PX / KNOWN_DISTANCE_CM
    # Scale this ratio based on the average estimated distance
    # This is a simplification; a full perspective transform would be more accurate
    # but this is a good approximation.
    avg_dist = (d1 + d2) / 2
    px_per_cm = px_per_cm_at_known_dist * (KNOWN_DISTANCE_CM / avg_dist) if avg_dist > 0 else px_per_cm_at_known_dist

    dx_cm = dx_px / px_per_cm if px_per_cm > 0 else 0
    
    dz = abs(d1 - d2)
    
    return round(math.hypot(dx_cm, dz), 1)
