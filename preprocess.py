import cv2
import torch
from torchvision import transforms
from retinaface import RetinaFace

class FacePreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
        # We normalize using standard ImageNet mean and std because the 
        # FaceNet backbone was pre-trained on VGGFace2 using these values.
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image_path):
        """
        Input: Raw Image path
        Output: Aligned PyTorch Tensor of shape (1, 3, 224, 224)
        """
        # 1. RetinaFace Detection & 5-Point Affine Alignment
        try:
            faces = RetinaFace.extract_faces(img_path=image_path, align=True)
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

        if len(faces) == 0:
            print("RetinaFace could not detect a face in this image.")
            return None

        # 2. Extract the most prominent face
        face_img = faces[0] 
        
        # 3. Resize to the target dimension for FaceNet (224x224)
        face_resized = cv2.resize(face_img, self.target_size)
        
        # 4. Convert to PyTorch Tensor, Normalize, and add Batch Dimension
        tensor_face = self.transform(face_resized).unsqueeze(0)
        
        return tensor_face