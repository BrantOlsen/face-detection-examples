import os, logging, math
import tensorflow as tf
from tensorflow.python.platform import gfile
import dlib
import pickle
from FaceDetector import FaceObject
import numpy as np
import cv2
from sklearn.svm import SVC


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
Training class for using FaceNet as the identifier.
"""
class FaceNetFaceIdentityTrainer(FaceIdentityTrainer):
    def __init__(self):
        super().__init__()
        self.batch_size = 1000

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            self.load_model('20180402-114759.pb')
             # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]
        
        self.classifier_file_name = os.path.join(self.dataset_folder, 'classifier.pkl')
        logging.debug(f"Using {self.classifier_file_name} to load classifier.")
        if os.path.exists(self.classifier_file_name):
            self.classifier = (model, class_names) = pickle.load(open(self.classifier_file_name, 'rb'))
        else:
            self.classifier = None
    

    def load_model(self, model_file):
        if not os.path.isfile(model_file):
            logging.error(f"{model_file} is not a file.")
            raise ValueError(f"model_file must be a file.")

        logging.debug(f"FaceIdentifier Model: {model_file}.")
        with gfile.FastGFile(model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')


    def load_data(self, raw_images, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
        nrof_samples = len(raw_images)
        images = np.zeros((nrof_samples, image_size, image_size, 3))
        for i in range(nrof_samples):
            img = raw_images[i]
            if img.ndim == 2:
                img = self.to_rgb(img)
            if do_prewhiten:
                img = self.prewhiten(img)
            img = self.crop(img, do_random_crop, image_size)
            img = self.flip(img, do_random_flip)
            images[i,:,:,:] = img
        return images

    
    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  


    def crop(self, image, random_crop, image_size):
        if image.shape[1]>image_size:
            sz1 = int(image.shape[1]//2)
            sz2 = int(image_size//2)
            if random_crop:
                diff = sz1-sz2
                (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
            else:
                (h, v) = (0,0)
            image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
        return image
    

    def flip(self, image, random_flip):
        if random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image


    def to_rgb(self, img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret


    """
    OpenCV2 reads image files as BGR color whereas most things expect RGB.
    """
    def convert_cv2_image_to_what_we_expect(self, img):
        cvt_image = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(cvt_image, cv2.COLOR_BGR2RGB)


    def train(self, face_objects, labels):
        logging.debug(f"Labels: {len(labels)}")
        logging.debug(f"Input Faces: {len(face_objects)}")

        with self.graph.as_default():
            # Run forward pass to calculate embeddings
            nrof_images = len(face_objects)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.batch_size))
            emb_array = np.zeros((nrof_images, self.embedding_size))

            faces = []
            for i in range(len(face_objects)):
                faces.append(self.convert_cv2_image_to_what_we_expect(face_objects[i].image))

            for i in range(nrof_batches_per_epoch):
                start_index = i*self.batch_size
                end_index = min((i+1)*self.batch_size, nrof_images)
                logging.debug(f"Loading data for images {start_index} to {end_index}.")
                images = self.load_data(faces[start_index:end_index], False, False, self.image_size)
                feed_dict = { self.images_placeholder:images, self.phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = self.session.run(self.embeddings, feed_dict=feed_dict)
            
            # Train classifier
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Saving classifier model
            class_names = np.unique(labels).tolist()
            logging.debug(f"Class Names = {class_names}")
            with open(self.classifier_file_name, 'wb') as outfile:   
                pickle.dump((model, class_names), outfile)