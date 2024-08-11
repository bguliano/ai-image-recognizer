import cv2
import numpy as np

from facial_detection import FaceDetection, FaceDetector


def draw_bounding_boxes(base_image: np.ndarray, face_detections: list[FaceDetection]) -> np.ndarray:
    try:
        img = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    except cv2.error:
        img = base_image.copy()

    for detection in face_detections:
        polygon = np.array(detection.bbox.to_vertices(), np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

        caption = f'face {detection.confidence:.3f}'
        min_x, min_y = detection.bbox.min_x(), detection.bbox.min_y()
        cv2.putText(img, caption, (int(min_x), int(min_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img


def test_live_inspection() -> None:
    camera = cv2.VideoCapture(0)
    face_detector = FaceDetector(0.8, device='mps')
    while True:
        _, frame = camera.read()
        frame = cv2.resize(frame, (960, 540))
        detections = list(face_detector.generate_detections(frame))
        image = draw_bounding_boxes(frame, detections)
        cv2.imshow('Test for face recognition', image)
        cv2.waitKey(1)


def main():
    pass


if __name__ == '__main__':
    test_live_inspection()
