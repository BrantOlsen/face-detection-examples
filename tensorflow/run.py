import cv2    
import numpy as np
import time, datetime, os, sys, logging, json
from PIL import Image

from FaceIdentifier import FaceNetFaceIdentifier
from FaceDetector import FaceDetectorDLib


def run():
    capture_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', '.captured'))
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)
    logging.info("Captured Dir: " + capture_dir)

    logging.info("Loading Face Identifier...")
    identifer = FaceNetFaceIdentifier()
    logging.info("Loading Face Detector...")
    detector = FaceDetectorDLib()

    cam = cv2.VideoCapture(0)

    # Start looking for faces.
    while True:
        ret_val, cam_img = cam.read()
        if cam_img.any():
            image_with_rect = cam_img.copy()
            faces = detector.detect_faces(cam_img.copy())
            for face in faces:
                #logging.debug(f"Found face at x={face.x}, y={face.y} with width={face.width}, height={face.height}.")
                cv2.imshow('detected_face', face.image)

                # Draw the face with eyes on the image and display to the operator.
                cv2.rectangle(image_with_rect, (face.x, face.y), (face.x+face.width, face.y+face.height), (0, 255, 0), 2)
                for eye in face.eyes:
                    eye_points = eye.get_points_relative_to_face(face)
                    cv2.rectangle(image_with_rect, eye_points[0], eye_points[1], (255, 0, 0), 2)

                # Find the identity of the face.
                label, conf, the_rest = identifer.classify(face)
                cv2.putText(image_with_rect, f"{label} ({conf})", (face.x, face.y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
              
            cv2.imshow('eyes', image_with_rect)
        else:
            img = np.zeros((512,512,3), np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'No Camera Connected.',(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(img,'Press ESC to Exit',(10,200), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('eyes', img)

        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s][%(levelname)s] %(message)s")
run()