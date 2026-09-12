from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from .config import FACE_MARGIN, TARGET_FACE_SIZE


class FaceProcessor:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # MTCNN expects PIL images
        self.detector = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, bgr_frame: 'np.ndarray') -> Tuple[List[Tuple[int, int, int, int]], Optional[List[float]]]:
        """Detect faces in a BGR OpenCV frame.

        Returns (boxes, probs) where each box is (x1, y1, x2, y2) in pixels
        (face region without margin) and probs are detection confidences.
        Returns ([], None) when no face is found.
        """
        rgb = bgr_frame[:, :, ::-1]
        pil = Image.fromarray(rgb)
        boxes, probs = self.detector.detect(pil)
        if boxes is None or len(boxes) == 0:
            return [], None
        int_boxes = [tuple(int(x) for x in box) for box in boxes]
        prob_list = [float(p) for p in probs] if probs is not None else None
        return int_boxes, prob_list

    def detect_and_crop(self, bgr_frame: 'np.ndarray') -> Optional[List[Image.Image]]:
        """Detect faces in a BGR OpenCV frame and return list of cropped PIL images.

        Returns empty list if no face is detected.
        """
        boxes, _ = self.detect_faces(bgr_frame)
        if not boxes:
            return []

        crops = []
        w, h = bgr_frame.shape[1], bgr_frame.shape[0]
        rgb = bgr_frame[:, :, ::-1]
        pil = Image.fromarray(rgb)
        for box in boxes:
            x1, y1, x2, y2 = [int(x) for x in box]
            # add margin
            x1 = max(0, x1 - FACE_MARGIN)
            y1 = max(0, y1 - FACE_MARGIN)
            x2 = min(w, x2 + FACE_MARGIN)
            y2 = min(h, y2 + FACE_MARGIN)
            crop = pil.crop((x1, y1, x2, y2)).resize(TARGET_FACE_SIZE)
            crops.append(crop)

        return crops

    def preprocess_pil(self, pil_img: Image.Image):
        """Return torch tensor normalized with ImageNet mean/std shape [3,224,224]"""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(TARGET_FACE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(pil_img)
