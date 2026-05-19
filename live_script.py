import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from retinaface import RetinaFace
from collections import deque, Counter
import time

from model import FRITNet

# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "best_frit_weights.pth"

DETECTION_CONFIDENCE_THRESHOLD = 0.20
EMOTION_CONFIDENCE_THRESHOLD = 0.45

FRAME_HISTORY = 5

# =========================================================
# EMOTION LABELS
# =========================================================

EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on: {device}")

# =========================================================
# MODEL
# =========================================================

model = FRITNet(num_classes=7).to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

print("Model loaded successfully")

# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# TEMPORAL SMOOTHING STORAGE
# =========================================================

emotion_history = {}

# =========================================================
# WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press Q to quit")

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    start_time = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------------------------------
    # RetinaFace Detection
    # -----------------------------------------------------

    try:
        detections = RetinaFace.detect_faces(frame)

    except:
        detections = {}

    # -----------------------------------------------------
    # Process Faces
    # -----------------------------------------------------

    if isinstance(detections, dict):

        for face_id, face_data in detections.items():

            confidence = face_data["score"]

            # ---------------------------------------------
            # Skip weak detections
            # ---------------------------------------------

            if confidence < DETECTION_CONFIDENCE_THRESHOLD:
                continue

            # ---------------------------------------------
            # Bounding Box
            # ---------------------------------------------

            x1, y1, x2, y2 = face_data["facial_area"]

            x1 = max(0, x1)
            y1 = max(0, y1)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # ---------------------------------------------
            # Convert to RGB
            # ---------------------------------------------

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(face_rgb)

            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # ---------------------------------------------
            # Inference
            # ---------------------------------------------

            with torch.no_grad():

                logits, _ = model(input_tensor)

                probabilities = torch.softmax(logits, dim=1)

                confidence_score, prediction = torch.max(probabilities, dim=1)

            emotion_confidence = confidence_score.item()

            predicted_class = prediction.item()

            predicted_emotion = EMOTIONS[predicted_class]

            # ---------------------------------------------
            # Confidence Threshold
            # ---------------------------------------------

            if emotion_confidence < EMOTION_CONFIDENCE_THRESHOLD:
                predicted_emotion = "Uncertain"

            # ---------------------------------------------
            # Temporal Smoothing
            # ---------------------------------------------

            unique_face_key = f"{x1}_{y1}_{x2}_{y2}"

            if unique_face_key not in emotion_history:

                emotion_history[unique_face_key] = deque(maxlen=FRAME_HISTORY)

            emotion_history[unique_face_key].append(predicted_emotion)

            smoothed_emotion = Counter(
                emotion_history[unique_face_key]
            ).most_common(1)[0][0]

            # ---------------------------------------------
            # Draw Bounding Box
            # ---------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ---------------------------------------------
            # Draw Label
            # ---------------------------------------------

            label = f"{smoothed_emotion} ({emotion_confidence:.2f})"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # -----------------------------------------------------
    # FPS Calculation
    # -----------------------------------------------------

    fps = 1 / (time.time() - start_time)

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # -----------------------------------------------------
    # Show Window
    # -----------------------------------------------------

    cv2.imshow("Live FER - FRITNet", frame)

    # -----------------------------------------------------
    # Exit
    # -----------------------------------------------------

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()