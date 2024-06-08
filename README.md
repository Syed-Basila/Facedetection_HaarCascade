# Facedetection_HaarCascade
# Face Detection using openCV with HaarCascade 
# Description
This project demonstrates a simple face detection application using OpenCV and Python. It captures video from the webcam, detects faces in real-time, and draws rectangles around detected faces.

# Features
- Real-time face detection
- Drawing rectangles around detected faces
# Installation
# Prerequisites
- Python 3.x
- OpenCV
  
Install required packages:
-- sh
pip install opencv-python
# Download Haar Cascade file:
Download the haarcascade_frontalface_default.xml file from OpenCV GitHub repository and place it in the project directory.

# Usage
Run the following command to start the face detection application:

--sh
Copy code
python face_detection.py
Face Detection Script
python
Copy code
import cv2 # openCV

alg= "haarcascade_frontalface_default.xml" #accessed the model file
haar_cascade=cv2.CascadeClassifier(alg) #loading the model with cv2

cam = cv2.VideoCapture(0) #intializing camera

while True:
    _,img = cam.read() #read the frame from the camera
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting color into gray scale
    face = haar_cascade.detectMultiScale(grayImg,1.3,4) #get coordinates of face
    for (x,y,w,h) in face: #segregating x,y,w,h.
        cv2.rectangle(img,(x,y),(x+w, y+h), (0,255,0),2)
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(10)
    if key ==27:#esc button to return
        break

cam.release()
cv2.destroyAllWindows()
Haar Cascades
Haar Cascades are a popular object detection method used in computer vision, developed by Paul Viola and Michael Jones. They are especially known for their use in real-time face detection. The technique involves training a cascade function with a large number of positive and negative images. The trained model can then detect objects in new images.

In this project, we use the haarcascade_frontalface_default.xml file, which is pre-trained to detect faces.

Screenshots

Output Video
You can view a demo of the face detection application here.

Contributing
Contributions are welcome! Please create an issue to discuss any changes or improvements.
