"""
Retrain the FaceIdentifier on the faces found in face.data folder.

Usage: python run.py
"""


import os, sys, logging
import cv2
import numpy as np
from FaceIdentifier import DLibFaceIdentifier, FaceIdentifier
from FaceDetector import FaceDetectorDLib
from FaceTrainer import DLibFaceIdentityTrainer


def run():
    logging.info("Starting...")
    dataset_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    user_folders = os.listdir(dataset_folder)
    faces = []
    labels = []
    detector = FaceDetectorDLib()
                 
    for user in user_folders:
        user_path = os.path.join(dataset_folder, user)
        if os.path.isdir(user_path) and not user.startswith('.') and not 'base_model' in user:
            label = user
            logging.info("Found folder for user id " + str(label))
            for training_file in os.listdir(user_path):
                if '.debug.' not in training_file:
                    image_path = os.path.join(dataset_folder, user, training_file)
                    logging.info("Detecting faces in " + image_path)
                    image = cv2.imread(image_path)
                    face_objects = detector.detect_faces(image)
                    if len(face_objects) > 1:
                        logging.error("Found " + str(len(face_objects)) + " in training image " + image_path + ". Should have only found 1.")
                    elif len(face_objects) == 0:
                        logging.error("Did not find a face in training image "  + image_path + ".")
                    else:
                        image_debug_path = os.path.join(dataset_folder, user, os.path.splitext(training_file)[0] + '.debug.png')
                        cv2.imwrite(image_debug_path, face_objects[0].image)
                        faces.append(face_objects[0])
                        labels.append(label)
                    
    logging.info("Labels: " + str(len(labels)))
    logging.info("Faces: " + str(len(faces)))

    logging.info("Training DLib...")
    trainer1 = DLibFaceIdentityTrainer()
    trainer1.train(faces, labels)

    test_identifier2 = FaceIdentifier(identifier=DLibFaceIdentifier())
    logging.info("Checking the DLib identifier...")
    for i in range(len(faces)):
        label, conf, all_class = test_identifier2.classify_from_face(faces[i])
        if label != labels[i]:
            logging.error(f"Failed to properly classify face {i} for label {labels[i]}. Found label {label} with conf {conf} from {str(all_class)}.")

    logging.info("Training Complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s][%(levelname)s] %(message)s") 
    run()