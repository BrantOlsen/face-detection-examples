from typing import Dict, Tuple, List
import numpy
import cv2
import os, sys, math, datetime, json, logging
from FaceDetector import FaceObject
from FaceTrainer import FaceNetFaceIdentityTrainer


"""
Based on code from https://github.com/davidsandberg/facenet.
"""
class FaceNetFaceIdentifier(FaceNetFaceIdentityTrainer):
    def __int__(self):
        super().__init__()


    def classify(self, face: FaceObject):
        if self.classifier is None:
            raise AttributeError('No classifier has been trained. Please call method train first with all labels and faces.')

        with self.graph.as_default():
            # Run forward pass to calculate embeddings
            images = self.load_data([self.convert_cv2_image_to_what_we_expect(face.image)], False, False, self.image_size)
            feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
            emb_array = numpy.zeros((1, self.embedding_size))
            emb_array[0:1,:] = self.session.run(self.embeddings, feed_dict=feed_dict)
            
            # Classify image
            (model, class_names) = self.classifier
            predictions = model.predict_proba(emb_array)
            best_class_indices = numpy.argmax(predictions, axis=1)
            best_class_probabilities = predictions[numpy.arange(len(best_class_indices)), best_class_indices]
            
            classify_data = []
            for i in range(len(best_class_indices)):
                classify_data.append((class_names[best_class_indices[i]], best_class_probabilities[i]))

            #prediction_data = []
            #for i in range(len(predictions)):
            #    print(class_names[i])
            #    print(predictions[i])
            #     prediction_data.append((class_names[i], predictions[i]))
            #print('PDATA')
            #print(prediction_data)
            return classify_data[0][0], classify_data[0][1], classify_data