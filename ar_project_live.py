import cv2
import cvzone
import math
import numpy as np
from collections import deque
from ultralytics import YOLO
from cvzone.FaceMeshModule import FaceMeshDetector

# --- SETUP ---
print("Initializing Model...")
model = YOLO('yolov8s.pt') 
face_detector = FaceMeshDetector(maxFaces=5, minDetectionCon=0.5, minTrackCon=0.5)

print("Starting Camera...")
# Try index 0 first, if black screen change to 1
cap = cv2.VideoCapture(0) 
cap.set(3, 1280)
cap.set(4, 720)

# --- STATE ---
selected_obj_id = None
box_history = {}    
current_objects = [] 

cv2.namedWindow("AR Assessment Project")

def mouse_callback(event, x, y, flags, param):
    global selected_obj_id
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = False
        for obj in current_objects:
            x1, y1, x2, y2, obj_id, _, _ = obj
            if x1 < x < x2 and y1 < y < y2:
                selected_obj_id = obj_id 
                clicked = True
                break
        if not clicked: selected_obj_id = None

cv2.setMouseCallback("AR Assessment Project", mouse_callback)

# --- HELPER FUNCTIONS ---

def get_face_center(landmarks):
    # Ensure we only use the first 2 coordinates (x, y)
    left_eye = landmarks[33][:2]
    right_eye = landmarks[263][:2]
    cx = (left_eye[0] + right_eye[0]) // 2
    cy = (left_eye[1] + right_eye[1]) // 2
    return (cx, cy)

def analyze_logic(face_landmarks):
    # Ensure we use [x,y] slicing to avoid 3D crashes
    up = face_landmarks[13][:2]
    low = face_landmarks[14][:2]
    left = face_landmarks[61][:2]
    right = face_landmarks[291][:2]
    
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
    try:
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
            # CRITICAL FIX: explicit slicing [:2] to ensure we only get x,y (not z)
            # This prevents the memory crash on Mac M1/M2
            pts_list = [landmarks[i][:2] for i in contour]
            pts = np.array(pts_list, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
            
        # Draw pupils
        le = landmarks[468][:2] if len(landmarks) > 468 else landmarks[159][:2]
        re = landmarks[473][:2] if len(landmarks) > 473 else landmarks[386][:2]
        
        cv2.circle(img, (le[0], le[1]), 3, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (re[0], re[1]), 3, (0, 0, 255), cv2.FILLED)
    except Exception as e:
        print(f"Drawing Error (Skipped): {e}")

# --- MAIN LOOP ---
while True:
    success, img = cap.read()
    if not success: 
        print("Failed to read from camera. Check permissions or index.")
        break
    
    current_objects = []

    # A. Detect
    results = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
    img, faces = face_detector.findFaceMesh(img, draw=False)
    
    # B. Prepare Faces
    available_faces = []
    if faces:
        for face in faces:
            # Face might be list of [x,y,z], we handle that in helper functions
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
            
            if obj_id not in box_history: box_history[obj_id] = deque(maxlen=5)
            box_history[obj_id].append((x1,y1,x2,y2))
            x1, y1, x2, y2 = np.mean(box_history[obj_id], axis=0).astype(int)
            w, h = x2-x1, y2-y1

            info_text = "Static Object"
            color = (255, 100, 0) 

            if label == 'person':
                color = (255, 0, 255) 
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
                    draw_face_skeleton(img, best_face["data"])
                else:
                    info_text = "Activity: Unknown"

            is_selected = (obj_id == selected_obj_id)
            if is_selected:
                color = (0, 255, 0) 
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            else:
                cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=2, colorR=color)

            cvzone.putTextRect(img, f"{obj_id}: {label}", (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=color)
            current_objects.append([x1, y1, x2, y2, obj_id, label, info_text])

    # D. Inspection Panel
    if selected_obj_id is not None:
        target = next((o for o in current_objects if o[4] == selected_obj_id), None)
        if target:
            _, _, _, _, oid, lbl, nfo = target
            cv2.rectangle(img, (20, 20), (300, 150), (0,0,0), cv2.FILLED) 
            cv2.rectangle(img, (20, 20), (300, 150), (0,255,0), 2)       
            
            cv2.putText(img, "INSPECTION MODULE", (30, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0,255,0), 1)
            cv2.putText(img, f"ID: {oid}", (30, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
            cv2.putText(img, f"Type: {lbl.upper()}", (30, 105), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 1)
            cv2.putText(img, nfo, (30, 130), cv2.FONT_HERSHEY_PLAIN, 1.2, (100,255,255), 1)

    cv2.imshow("AR Assessment Project", img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()