**Tensorflow (FaceNet) Identification Example**

These files encapsulate logic from https://github.com/davidsandberg/facenet for using the FaceNet algorithm to classify faces in real-time.


**How to Run**
1. Download the following from https://github.com/davisking/dlib-models and extract here.
    - shape_predictor_5_face_landmarks.dat.bz2
2. Download the model from https://github.com/davidsandberg/facenet#pre-trained-models and extract here.
3. Copy folders with image data and named by their with label into the data folder one up from this folder.
4. Run `train.py`.
5. Run `run.py`.


**Notes**
- Must install tensorflow-gpu to get good performance which means NVIDIA GPUs only.
- Pre-trained model allowed for accurate results with only a few images of each label.
- Code is low enough level that it allows you to see the nuts and bolts of the learning algorithm.