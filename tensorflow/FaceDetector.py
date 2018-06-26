from typing import Dict, Tuple, List
import numpy
from PIL import Image
import cv2
import os
import math, datetime, logging
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
Detect faces using
"""
class FaceDetectorDLib:
    def __init__(self, align_offset_pct = (0.28, 0.28)):
        self.align_offset_pct = align_offset_pct
        self.detector = dlib.get_frontal_face_detector()
        predictor_model = os.path.join(os.path.dirname(__file__), 'shape_predictor_5_face_landmarks.dat')
        self.face_feature_predictor = dlib.shape_predictor(predictor_model)

    
    """
    Detect all faces in the image and return a FaceObject that contains the 
    bouding box around them.
    """
    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 0)

        faces = []
        for i, rect in enumerate(dets):
            face = FaceObject()
            face.x = rect.left()
            face.y = rect.top()
            face.width = rect.right() - rect.left()
            face.height = rect.bottom() - rect.top()
            face.eyes, face.dlib_shape = self.detect_eyes(gray, rect, face)
            self.align_face(image, face)
            faces.append(face)
        return faces


    """
    Find the eyes in the face image and return them as a list.
    """
    def detect_eyes(self, gray_image, rect, face: FaceObject):
        shape = self.face_feature_predictor(gray_image, rect)
        eyes = []

        if shape.num_parts == 5:
            left_eye = EyeObject()
            left_eye.set_from_points(shape.part(0), shape.part(1))
            eyes.append(left_eye)

            right_eye = EyeObject()
            right_eye.set_from_points(shape.part(3), shape.part(2))
            eyes.append(right_eye)

        return eyes, shape


    def align_face(self, image, face: FaceObject):
        if len(face.eyes) != 2:
            return

        desiredFaceWidth = 160
        desiredFaceHeight = desiredFaceWidth

        leftEyeCenter = face.eyes[0].get_center_point() 
        rightEyeCenter = face.eyes[1].get_center_point()

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = numpy.degrees(numpy.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.align_offset_pct[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = numpy.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.align_offset_pct[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * self.align_offset_pct[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        face.image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        face.orig_image = image
