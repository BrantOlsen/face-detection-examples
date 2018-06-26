from typing import Dict, Tuple, List
import numpy
from PIL import Image
import cv2
import os
import math, datetime, logging
import FaceAlign
import dlib


"""
This object describes the face that was detected in the image.
"""
class FaceObject:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.time = datetime.datetime.now()
        self.face_id = None
        self.confidence = None
        self.all_classifications = None
        self.image = None
        self.orig_image = None
        self.eyes = []
        self.dlib_shape = None


"""
This object describes the eye that was detected in the image.
"""
class EyeObject:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.x2 = 0
        self.y2 = 0
        self.width = 0
        self.height = 0

    
    """
    Set the attributes of the eye from the two given points. left_point x values 
    should be less than the right_point x values.
    """
    def set_from_points(self, left_point, right_point):
        self.x = left_point.x
        self.y = left_point.y
        self.x2 = right_point.x
        self.y2 = right_point.y
        self.width = self.x - self.x2
        self.height = self.y - self.y2


    def get_area(self):
        return self.width * self.height


    def get_center_point(self):
        if self.x2 != 0 and self.y2 != 0:
            return numpy.mean([[self.x, self.y], [self.x2, self.y2]], axis=0).astype("int")
        else:
            return (self.x + int(self.width / 2), self.y + int(self.height/ 2))


    def get_points_relative_to_face(self, face: FaceObject) -> Tuple:
        if self.x + face.x > face.width:
            return (self.x, self.y), (self.x2, self.y2)
        else:
            return (self.x + face.x, self.y + face.y), (self.x + self.width + face.x, self.y + self.height + face.y)


"""
CV2 face decetor. Much faster than the Tensorflow one, but not as accurate for training.
"""
class FaceDetectorCV2:
    def __init__(self, align_offset_pct=(.28, .28)):
        self.align_offset_pct = align_offset_pct

        cascade_file = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_alt2.xml')
        if not os.path.exists(cascade_file):
            raise AttributeError(f"Cascade file '{cascade_file}' does not exist.")
        logging.debug(f"Using {cascade_file} for CascadeClassifier.")
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

        eye_cascade_file = os.path.join(os.path.dirname(__file__), 'haarcascade_eye.xml')
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_file)
        

    """
    Detect all faces in the image and return a FaceObject that contains the 
    bouding box around them.
    """
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3, 
            minNeighbors=4, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces = []
        for (x, y, w, h) in detected_faces:
            face = FaceObject()
            face.x = x
            face.y = y
            face.width = w
            face.height = h
            face.eyes = self.detect_eyes(gray)
            self.set_image_and_adjust(image, face)
            faces.append(face)
        return faces


    """
    Find the eyes in the face image and return them as a list.
    """
    def detect_eyes(self, gray_image):
        detected_eyes = self.eye_cascade.detectMultiScale(gray_image)
        eyes = []
        for (x, y, w, h) in detected_eyes:
            eye = EyeObject()
            eye.set_from_points(dlib.point(x,y), dlib.point(x+w,y+h))
            eyes.append(eye)
        return eyes


    """
    Set the image and align it by the eyes.
    """
    def set_image_and_adjust(self, image, face: FaceObject):
        face.orig_image = image
        face.image = face.orig_image[face.y:face.y+face.height, face.x:face.x+face.width]
        self.align_image_by_eyes(face)
    

    """
    Align the image using the eyes for better identification.
    https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#aligning-face-images
    """
    def align_image_by_eyes(self, face: FaceObject):
        eyes_to_use = face.eyes
        if len(eyes_to_use) < 2:
            logging.debug(f"{len(face.eyes)} eyes were found for this face; not 2.")
            return
        
        if len(eyes_to_use) > 2:
            eyes_to_use = face.eyes[:2]
            for eye in face.eyes[2:]:
                if eyes_to_use[0].get_area() > eyes_to_use[1].get_area() and eyes_to_use[1].get_area() < eye.get_area():
                    eyes_to_use[1] = eye
                elif eyes_to_use[0].get_area() < eye.get_area():
                    eyes_to_use[0] = eye

        eye_left = eyes_to_use[0].get_center_point() if eyes_to_use[1].x > eyes_to_use[0].x else eyes_to_use[1].get_center_point()
        eye_right = eyes_to_use[1].get_center_point() if eyes_to_use[1].x > eyes_to_use[0].x else eyes_to_use[0].get_center_point()

        dest_size = 160, 160
        face.image = numpy.array(FaceAlign.CropFace(Image.fromarray(face.orig_image), eye_left=eye_left, eye_right=eye_right, offset_pct=self.align_offset_pct, dest_sz=dest_size))
