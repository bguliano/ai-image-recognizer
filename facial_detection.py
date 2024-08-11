from dataclasses import dataclass
from typing import Literal, Generator

import cv2
import numpy as np
from ultralytics import YOLO

from facial_recognition import FaceRecognizer


@dataclass(frozen=True)
class BBox:
    xyxy: list[float]

    @property
    def xs(self) -> tuple[float, float]:
        return self.xyxy[0], self.xyxy[2]

    @property
    def ys(self) -> tuple[float, float]:
        return self.xyxy[1], self.xyxy[3]

    def min_x(self) -> float:
        return min(self.xs)

    def min_y(self) -> float:
        return min(self.ys)

    def max_x(self) -> float:
        return max(self.xs)

    def max_y(self) -> float:
        return max(self.ys)

    def to_vertices(self) -> list[tuple[float, float]]:
        return [
            (self.xyxy[0], self.xyxy[1]),
            (self.xyxy[2], self.xyxy[1]),
            (self.xyxy[2], self.xyxy[3]),
            (self.xyxy[0], self.xyxy[3])
        ]


@dataclass(frozen=True)
class FaceDetection:
    image: np.ndarray
    face_image: np.ndarray
    bbox: BBox
    confidence: float


def get_face_image(original_image: np.ndarray, face_bbox: BBox) -> np.ndarray:
    y = int(face_bbox.min_y())
    x = int(face_bbox.min_x())
    h = int(face_bbox.max_y() - y)
    w = int(face_bbox.max_x() - x)
    return original_image[y:y + h, x:x + w]


class FaceDetector:
    def __init__(self, min_confidence: float, *, device: Literal['cpu', 'mps']):
        self.model = YOLO('yolov8m-face.pt').to(device)
        self.min_confidence = min_confidence

    def generate_detections(self, from_frame: np.ndarray) -> Generator[FaceDetection, None, None]:
        # read frame and make predictions
        face_result = self.model.predict(from_frame, conf=self.min_confidence, verbose=False)[0]

        # create detections
        for i in range(len(face_result)):
            # get vars from result
            original_image = face_result.orig_img
            bbox = BBox(face_result.boxes.xyxy.cpu().numpy()[i].tolist())
            confidence = float(face_result.boxes.conf[i])

            # create face image using bounding box from result
            face_image = get_face_image(original_image, bbox)

            # finally, create FaceDetection object
            yield FaceDetection(original_image, face_image, bbox, confidence)
