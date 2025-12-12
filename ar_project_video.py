import cv2
import cvzone
import math
import numpy as np
from collections import deque
from ultralytics import YOLO
from cvzone.FaceMeshModule import FaceMeshDetector

# --- SETUP ---
VIDEO_SOURCE = "test_video.mp4" 

model = YOLO('yolov8s.pt') 
face_detector = FaceMeshDetector(maxFaces=5, minDetectionCon=0.5, minTrackCon=0.5)

cap = cv2.VideoCaptu2re(VIDEO_SOURCE)
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

# --- STATE ---
selected_obj_id = None
paused = False
img_raw = None       
box_history = {}     
current_objects = [] 

cv2.namedWindow("AR Assessment Project")

def mouse_callback(event, x, y, flags, param):
    global selected_obj_id
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = False
        for obj in current_objects:
            coords = obj["coords"]
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            if x1 < x < x2 and y1 < y < y2:
                selected_obj_id = obj["id"] 
                clicked = True
                break
        if not clicked: selected_obj_id = None

cv2.setMouseCallback("AR Assessment Project", mouse_callback)

# --- HELPER FUNCTIONS ---

def get_face_center(landmarks):
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    cx = (left_eye[0] + right_eye[0]) // 2
    cy = (left_eye[1] + right_eye[1]) // 2
    return (cx, cy)

def analyze_logic(face_landmarks):
    up = face_landmarks[13]
    low = face_landmarks[14]
    left = face_landmarks[61]
    right = face_landmarks[291]
    
    width = math.dist(left, right)
    if width == 0: width = 1
    ratio = math.dist(up, low) / width
    
    activity = "Neutral"
    if ratio > 0.6: activity = "Surprised"
    elif ratio > 0.35: activity = "Talking"
    elif ratio > 0.15: activity = "Laughing"
    
    return activity

def draw_face_skeleton(img, landmarks):
    """Draws the main facial features connecting lines."""
    
    # Indices for facial features (Standard MediaPipe Topology)
    contours = [
        # Face Oval
        [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
        # Lips
        [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
        [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191],
        # Left Eye
        [33, 160, 158, 133, 153, 144],
        # Right Eye
        [362, 385, 387, 263, 373, 380],
        # Left Eyebrow
        [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        # Right Eyebrow
        [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
        # Nose Bridge
        [168, 6, 197, 195, 5, 4]
    ]

    for contour in contours:
        # Get coordinates for this contour
        pts = np.array([landmarks[i] for i in contour], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        
    # Draw simple dots for eyes to make it look "alive"
    # Left Iris approx (468) and Right Iris approx (473) - but using center pupil logic
    le = landmarks[468] if len(landmarks) > 468 else landmarks[159]
    re = landmarks[473] if len(landmarks) > 473 else landmarks[386]
    cv2.circle(img, le, 3, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, re, 3, (0, 0, 255), cv2.FILLED)

# --- MAIN LOOP ---
success, img_raw = cap.read()
if not success:
    print("Error: Could not read video file.")
    exit()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('p') or key == 32: paused = not paused

    if not paused:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        img_raw = frame.copy() 
        current_objects = []   

        # A. Detect
        results = model.track(img_raw, persist=True, tracker="bytetrack.yaml", verbose=False)
        _, faces = face_detector.findFaceMesh(img_raw, draw=False)
        
        # B. Prepare Faces
        available_faces = []
        if faces:
            for face in faces:
                center = get_face_center(face)
                act = analyze_logic(face)
                available_faces.append({
                    "center": center, "data": face, "activity": act, "matched": False
                })

        # C. Process Objects
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.id is None: continue
                
                obj_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # Smoothing
                if obj_id not in box_history: box_history[obj_id] = deque(maxlen=5)
                box_history[obj_id].append((x1,y1,x2,y2))
                x1, y1, x2, y2 = np.mean(box_history[obj_id], axis=0).astype(int)
                w, h = x2-x1, y2-y1

                info_text = "Static Object"
                face_to_draw = None 

                if label == 'person':
                    head_area_x = x1 + w // 2
                    head_area_y = y1 + h // 4
                    
                    best_face = None
                    min_dist = 99999
                    
                    for face_obj in available_faces:
                        if face_obj["matched"]: continue
                        fx, fy = face_obj["center"]
                        
                        if (x1 - 50 < fx < x2 + 50) and (y1 - 50 < fy < y2 + 50):
                            dist = math.hypot(head_area_x - fx, head_area_y - fy)
                            if dist < min_dist:
                                min_dist = dist
                                best_face = face_obj
                    
                    if best_face:
                        best_face["matched"] = True
                        info_text = f"Activity: {best_face['activity']}"
                        face_to_draw = best_face 
                    else:
                        info_text = "Activity: Unknown"
                
                current_objects.append({
                    "coords": (x1, y1, x2, y2, w, h),
                    "id": obj_id,
                    "label": label,
                    "info": info_text,
                    "face_data": face_to_draw
                })

    # --- DRAWING PHASE ---
    if img_raw is not None:
        img_final = img_raw.copy()
        
        if paused:
             cv2.putText(img_final, "- PAUSED -", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)

        for obj in current_objects:
            x1, y1, x2, y2, w, h = obj["coords"]
            obj_id = obj["id"]
            label = obj["label"]
            info = obj["info"]
            face_data = obj["face_data"]
            
            color = (255, 100, 0)
            if label == 'person': color = (255, 0, 255)

            # 1. Draw Skeleton (if matched)
            if face_data:
                draw_face_skeleton(img_final, face_data["data"])

            # 2. Draw Bounding Box
            is_selected = (obj_id == selected_obj_id)
            if is_selected:
                color = (0, 255, 0)
                cv2.rectangle(img_final, (x1, y1), (x2, y2), color, 3)
            else:
                cvzone.cornerRect(img_final, (x1, y1, w, h), l=15, rt=2, colorR=color)

            cvzone.putTextRect(img_final, f"{obj_id}: {label}", (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=color)

        # 3. UI Panel
        if selected_obj_id is not None:
            target = next((o for o in current_objects if o["id"] == selected_obj_id), None)
            if target:
                cv2.rectangle(img_final, (20, 20), (300, 150), (0,0,0), cv2.FILLED)
                cv2.rectangle(img_final, (20, 20), (300, 150), (0,255,0), 2)
                
                cv2.putText(img_final, "INSPECTION MODULE", (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0,255,0), 1)
                cv2.putText(img_final, f"ID: {target['id']}", (30, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
                cv2.putText(img_final, f"Type: {target['label'].upper()}", (30, 105), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
                cv2.putText(img_final, target['info'], (30, 130), cv2.FONT_HERSHEY_PLAIN, 1.2, (100,255,255), 1)

        cv2.imshow("AR Assessment Project", img_final)

cap.release()
cv2.destroyAllWindows()