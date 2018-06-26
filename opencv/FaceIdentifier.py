from typing import Dict, Tuple, List
import numpy
import cv2
import os, sys, math, datetime, json, logging
from FaceDetector import FaceObject


"""
Based on code from https://github.com/informramiz/opencv-face-recognition-python.
"""
class FaceIdentifierCV2:


    def __init__(self):
        self.dataset_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_file = os.path.join(self.dataset_folder, "face_recognizer.yml")
        if os.path.isfile(self.model_file):
            self.face_recognizer.read(self.model_file)
        else:
            logging.warn(f"{self.model_file} does not exist. Please train first.")


    def train(self, faces, labels):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius = 1,
            neighbors = 8,
            grid_x = 8,
            grid_y = 8,
            threshold = 100
        )

        face_images = [cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY) for f in faces]

        # Labels must be integers for OpenCV.
        label_to_int_map = {}
        labels_as_ints = []
        for label in labels:
            if label not in label_to_int_map:
                label_to_int_map[label] = len(label_to_int_map)
            labels_as_ints.append(label_to_int_map[label])

        self.face_recognizer.train(face_images, numpy.array(labels_as_ints))
        logging.info(f"Saving trained model to {self.model_file}.")
        self.face_recognizer.save(self.model_file)


    def identify_from_image(self, image):
        label, confidence = self.face_recognizer.predict(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))        
        return label, confidence

