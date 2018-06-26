from typing import Dict, Tuple, List
import numpy
import cv2
import os, sys, math, datetime, json, logging
from FaceDetector import FaceObject
from FaceIdentifyObject import FaceIdentifyObject

from FaceTrainer import DLibFaceIdentityTrainer


"""
Based on code from https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py.
"""
class DLibFaceIdentifier(DLibFaceIdentityTrainer):
    def __init__(self):
        super().__init__()


    def classify(self, face: FaceObject):
        all_classifications = []
        face_vector = self.find_face_vector_from_face(face)
        for identity in self.face_vector_labels:
            dist = self.euclidean_dist(face_vector, identity[1])
            all_classifications.append([identity[0], 1-dist])

        ret = sorted(all_classifications, key=lambda x: x[1], reverse=True)
        return ret


    def euclidean_dist(self, vector_x, vector_y):
        if len(vector_x) != len(vector_y):
            raise Exception('Vectors must be same dimensions')
        return sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x)))


"""
Class for tracking an identity across mutliple face detections.
"""
class FaceIdentifier:
    def __init__(self, identifier=None, threshold=.4):
        self.identifier = identifier
        if self.identifier is None:
            self.identifier = DLibFaceIdentifier()

        self.threshold = threshold
        self.face_identities = []

    
    """
    Given a FaceObject determine who's face it is and if we have seen it before. Will
    return the FaceIdentifyObject that was found.
    """
    def identify_from_face(self, face: FaceObject) -> FaceIdentifyObject:
        face_identity = None

        # See if we already know the face by the position.
        for identity in self.face_identities:
            if identity.should_include(face):
                if identity.is_confirmed():
                    #logging.debug(f"Face was determined to be included in the {identity.get_face_id()} face identity.")
                    face.face_id = identity.get_face_id()
                    face.confidence = 100
                else:
                    face.face_id, face.confidence, face.all_classifications = self.classify_from_face(face)
                    #logging.debug(f"Face identified as {face.face_id} with confidence {face.confidence}, all: {str(face.all_classifications)}.")
                identity.add_face(face)
                face_identity = identity

        # No face found so add a new tracking.
        if face_identity is None:
            face.face_id, face.confidence, face.all_classifications = self.classify_from_face(face)
            logging.debug(f"New identity object for {face.face_id} with confidence {face.confidence}, all: {str(face.all_classifications)}.")

            face_identity = FaceIdentifyObject()
            face_identity.add_face(face)
            self.face_identities.append(face_identity)
            
        return face_identity


    """ 
    Remove any old faces that have not been seen lately.
    """
    def check_for_remove_and_return_expired(self):
        removing = []
        i = 0
        while i < len(self.face_identities):
            if self.face_identities[i].is_expired():
                removed = self.face_identities.pop(i)
                if removed.is_confirmed():
                    removing.append(removed)
                i = i - 1
            i = i + 1
                
        return removing


    """
    Will return a Triple of label, confidence in that label, and all label confidences.
    """
    def classify_from_face(self, face: FaceObject):
        try:
            all_classifications = self.identifier.classify(face)
            if len(all_classifications) == 0:
                return -1, 0, []

            label, confidence = all_classifications[0]
            if confidence < self.threshold:
                return -1, confidence, all_classifications
            else:
                return label, confidence, all_classifications
        except cv2.error as e:
            logging.error(f"Failed to classify image. See failed_image.jpg for the image it failed on.")
            cv2.imwrite('failed_image.png', face.image)
            raise e
