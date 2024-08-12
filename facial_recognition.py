from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2
import face_recognition
import numpy as np
from PIL import Image


class FaceRecognizer:
    def __init__(self):
        self.faces: dict[str, Deque[np.ndarray]] = {}

    def label_face(self, face_image: np.ndarray) -> Optional[str]:
        # get latest encoding for each face
        known_encodings = [x[-1] for x in self.faces.values()]

        # perform face recognition
        rgb_face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        try:
            unknown_encoding = face_recognition.face_encodings(rgb_face_image)[0]
        except IndexError:  # no faces found
            return
        comparison = face_recognition.compare_faces(known_encodings, unknown_encoding)

        # face has been detected before, add the new encoding to its entry
        if any(comparison):
            name = list(self.faces)[comparison.index(True)]
            self.faces[name].append(unknown_encoding)
            return name

        # face has never been detected before, so create a new entry in faces
        new_name = f'person{len(self.faces)}'
        print(f'New face added: {new_name}')
        self.faces[new_name] = deque([unknown_encoding], maxlen=5)
        return new_name

    def load_face(self, face_name: str, face_image: np.ndarray):
        encoding = face_recognition.face_encodings(face_image)[0]
        if face_name in self.faces:
            self.faces[face_name].append(encoding)
        else:
            print(f'New face added: {face_name}')
            self.faces[face_name] = deque([encoding])

    def load_face_dir(self, face_dir: str) -> list[str]:
        face_dir = Path(face_dir)
        all_face_names = set()
        for image in face_dir.rglob('[!.]*.jpeg'):
            np_image = np.array(Image.open(image))
            self.load_face(image.parent.name, np_image)
            all_face_names.add(image.parent.name)
        return list(all_face_names)
