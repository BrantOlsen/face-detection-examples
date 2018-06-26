**OpenCV Face Detection and Identification Example**

These files encapsulte logic found in https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html for detecting faces in real-time.


**How to Run**
1. Go to https://github.com/opencv/opencv/tree/master/data/haarcascades and download the following files to this folder.
    - haarcascade_eye.xml
    - haarcascade_frontalface_alt2.xml
2. Copy folders with image data and named by their with label into the data folder one up from this folder.
3. Run `train.py`
4. Run `run.py` 


**Notes**
- Face detection was easy to use and implement.
- Face identification had no pre-trained model, so large amounts of training data will be needed.
- Face detection was not as accurate as DLib's.