import os, logging, math
import dlib
import pickle
from FaceDetector import FaceObject
import numpy as np
import cv2


"""
Base class for any identity trainers.
"""
class FaceIdentityTrainer:
    def __init__(self):
        self.dataset_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        self.image_size = 160


    def train(self, face_objects, labels):
        pass


"""
Train face identifier for dlib vectors.
"""
class DLibFaceIdentityTrainer(FaceIdentityTrainer):
    def __init__(self):
        super().__init__()
        
        self.model_name = 'dlib_face_recognition_resnet_model_v1.dat'
        logging.debug(f"Using dlib model {self.model_name} for face identification.")
        self.face_recognizer = dlib.face_recognition_model_v1(self.model_name)

        self.face_vector_labels = []
        self.face_vector_labels_file = os.path.join(self.dataset_folder, 'dlib_face_recognizer.pkl')
        if os.path.exists(self.face_vector_labels_file):
            self.face_vector_labels = pickle.load(open(self.face_vector_labels_file, 'rb'))
        else:
            logging.warn(f"{self.face_vector_labels_file} does not exist. Please train FaceIdentifierDLib to create it.")


    def train(self, faces, labels):
        self.face_vector_labels = []

        for i in range(0, len(faces)):
            face_vector = self.find_face_vector_from_face(faces[i])
            self.face_vector_labels.append([labels[i], face_vector])

        pickle.dump(self.face_vector_labels, open(self.face_vector_labels_file, 'wb'))


    """
    This method assumes that the entire image is the face.
    """
    def find_face_vector_from_face(self, face: FaceObject):
        return self.face_recognizer.compute_face_descriptor(face.orig_image, face.dlib_shape)