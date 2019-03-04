# import the necessary packages
from keras.preprocessing.image import img_to_array
#used to convert each individual frame from video stream to properly channel ordered array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

cascade = "haarcascade_frontalface_default.xml"
#path to where the Haar face cascade resides
model_cnn = "model/smiles_lenet.hdf5"
#path to wear the model is saved
video = ""#"data/test1.mp4"
#path to wear the video is

#load the face detector cascade and smile detetctor CNN
detector = cv2.CascadeClassifier(cascade)
model = load_model(model_cnn)

#take web cam output else take video from path
if(video):
    camera = cv2.VideoCapture(video) #pre saved video file
else:
    camera = cv2.VideoCapture(0) #webcam

#main processing pipeline
while True:
    #grab the current frame
    (grabbed, frame) = camera.read()
    
    #if we are viewing a video and didn't grab a frame, 
    #then we have reached the end of thr video
    if(video) and not grabbed:
        break 
        
    #resize the frame, convert to grayscale then clone the OG frame
    frame = imutils.resize(frame, width = 300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    
    #detect faces in the input frame
    rects = detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)
    #the face must have a min width of 30x30
    #minneighbors helps prune false positives
    #scalefactor controls the number of image pyramids generated
    #returns a list of 4-tuples that form a rectangle 
    #that bounds the face in the frame (x,y,h,w)
    for (fX, fY, fW, fH) in rects:
        #extract the ROI of face
        #resize to 28x28 and send to CNN
        roi = gray[fY:fY + fH, fX: fX + fW]
        roi = cv2.resize(roi, (28,28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis = 0) #padding the image with an extra dimension
        
        #determine prob of smiling and not
        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"
        
        #display the label and bounding box rectangle on subject
        if label == "Not Smiling":
        	colorOverlay = (0,0,255)
        else:
        	colorOverlay = (0,255, 0)

        cv2.putText(frameClone, label, (fX, fY - 10), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.45, colorOverlay, 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), colorOverlay,2)
    
    cv2.imshow("Face", frameClone)
        
    #if 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

