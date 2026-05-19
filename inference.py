import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from retinaface import RetinaFace

from model import FRITNet

# RAF-DB Labels
EMOTIONS = [
    "Surprise", "Fear", "Disgust",
    "Happiness", "Sadness", "Anger", "Neutral"
]

class FERInference:
    def __init__(self, model_path, device=None, conf_thresh=0.2):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh

        # Load model
        self.model = FRITNet(num_classes=7).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # ✅ Correct transform (same as validation)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect_faces(self, image_path):
        """
        Detect faces using RetinaFace and filter by confidence
        """
        img = cv2.imread(image_path)

        if img is None:
            print("Error: Image not found or cannot be read.")
            return []

        detections = RetinaFace.detect_faces(img_path=image_path)

        faces = []

        if isinstance(detections, dict):
            for key in detections:
                conf = detections[key]["score"]

                # ❌ Reject low-confidence faces
                if conf < self.conf_thresh:
                    continue

                x1, y1, x2, y2 = detections[key]["facial_area"]

                face = img[y1:y2, x1:x2]

                if face is None or face.size == 0:
                    continue

                faces.append((face, conf))

        return faces

    def preprocess_face(self, face):
        """
        Convert face to model input tensor
        """
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face = self.transform(face).unsqueeze(0)
        return face.to(self.device)

    def predict(self, image_path):
        faces = self.detect_faces(image_path)

        if len(faces) == 0:
            print("No valid face detected (confidence < threshold or none found).")
            return None

        print(f"Detected {len(faces)} valid face(s)\n")

        results = []

        for i, (face, det_conf) in enumerate(faces):
            tensor = self.preprocess_face(face)

            with torch.no_grad():
                logits, _ = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            emotion = EMOTIONS[pred]
            emotion_conf = probs[0][pred].item()

            results.append({
                "face_id": i,
                "det_conf": det_conf,
                "emotion": emotion,
                "emotion_conf": emotion_conf
            })

        return results


# =========================
# 🚀 RUN EXAMPLE
# =========================
if __name__ == "__main__":
    model_path = "best_frit_weights.pth"   # put your weights here
    image_path = "test9.jpg"                # test image

    infer = FERInference(model_path)

    results = infer.predict(image_path)

    if results:
        for r in results:
            print(f"Face {r['face_id']}:")
            print(f"  Detection Confidence: {r['det_conf']:.2f}")
            print(f"  Emotion: {r['emotion']}")
            print(f"  Emotion Confidence: {r['emotion_conf']:.2f}")
            print("-" * 30)