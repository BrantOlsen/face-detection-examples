**DLib Face Detection and Identification Example**

These files encapsulte logic found in http://dlib.net/face_detector.py.html for detecting faces in real-time.


**How to Run**
1. Download the following from https://github.com/davisking/dlib-models and extract here.
    - shape_predictor_5_face_landmarks.dat.bz2
    - dlib_face_recognition_resnet_model_v1.dat.bz2
2. Copy folders with image data and named by their with label into the data folder one up from this folder.
3. Run `train.py`.
4. Run `run.py`.


**Notes**
- Fast with an i7 CPU and easy to use. 
- Pre-trained model allowed for accurate results with only a few images of each label.
- Has addtional features like earlob detection and nose detection for future use.