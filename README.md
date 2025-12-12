# AR Assessment Project: Interactive Object & Face Analysis

This project is a Computer Vision application that combines **YOLOv8 Object Detection** with **MediaPipe Face Mesh** (via `cvzone`) to create an interactive Augmented Reality (AR) experience.

The system detects people and objects, analyzes facial expressions in real-time to determine activity (e.g., Talking, Laughing), and renders context-aware AR hats that rotate and scale with the user's head movement. It includes an interactive "Inspection Module" allowing users to click on objects to view detailed metadata.

## 📂 Project Structure

* **`ar_project_live.py`**: Runs the application using the **Webcam** (Live Feed). Ideal for real-time demonstrations.
* **`ar_project_video.py`**: Runs the application on a **Video File**. Includes advanced playback controls (Pause/Play, Auto-Rewind) to allow easier inspection of specific frames.

## 🚀 Features

### 1. Hybrid Detection Pipeline
* **YOLOv8s:** Used for robust bounding box detection of objects (specifically optimized for 'person').
* **Face Mesh:** High-fidelity facial landmark detection to identify eyes, mouth, and head tilt.
* **Smart Matching:** A custom logic layer that associates a generic YOLO "Person" box with specific Face Mesh data by calculating Euclidean distance between the box's "head area" and the face center.

### 2. Logic-Driven AR Overlay
The application analyzes facial landmarks to determine the state of the subject and applies a specific AR Hat:
* **Neutral:** 🟡 **Gold Crown** (Standard state).
* **Talking:** ⚪ **Viking Helmet** (Triggered when mouth aspect ratio > 0.35).
* **Laughing/Surprised:** 🔵 **Jester Hat** (Triggered when mouth aspect ratio > 0.6).
* *Note: All AR assets are drawn procedurally using OpenCV polygons and rotate dynamically with head tilt.*

### 3. Interactive UI
* **Mouse Interaction:** Click on any detected object to select it.
* **Inspection Module:** A HUD (Heads-Up Display) appears for the selected object, showing its unique ID, Classification, and current Activity State (e.g., "Activity: Talking").

## 🛠️ Installation & Requirements

Ensure you have Python installed (3.8+ recommended).

1.  **Install Dependencies:**
    You will need `opencv-python`, `cvzone`, `ultralytics`, and `numpy`.
    ```bash
    pip install opencv-python cvzone ultralytics numpy
    ```

2.  **YOLO Model:**
    The scripts use `yolov8s.pt`. On the first run, the `ultralytics` library will automatically download this model to your project folder.

## 💻 Usage

### Running the Live Version (Webcam)
```bash
python ar_project_live.py