#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import argparse
import numpy as np
import pyrealsense2 as rs
from time import sleep

def init_realsense():
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    rs_pipeline.start(rs_config)
    return rs_pipeline

def get_image(color):
    img = np.asanyarray(color.get_data())
    # img = cv2.resize(img, (240, 240))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.rotate(img, cv2.ROTATE_180)
    return img

def get_frame(rs_pipeline):
    # Get frame from pipeline
    frames = rs_pipeline.wait_for_frames()
    
    # Get depth image
    color = frames.get_color_frame()
    if not color: 
        return
    
    img = get_image(color)
    return img

def highlightPerson(hog, frame):
    boxes, _ = hog.detectMultiScale(frame, winStride=(8,8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    return frame, boxes

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

# faceProto="opencv_face_detector.pbtxt"
# faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

# faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# out = cv2.VideoWriter('result/output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

padding=20

rs_pipeline = init_realsense()

try:
    while cv2.waitKey(1) < 0 :

        frame = get_frame(rs_pipeline)

        if frame is None:
            print("No frame received...")
            continue
        
        resultImg, personBoxes = highlightPerson(hog, frame)
        
        if not len(personBoxes):
            cv2.imshow("Detecting age and gender", resultImg)

        for faceBox in personBoxes:
            face=frame[max(0,faceBox[1]-padding):
                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                    :min(faceBox[2]+padding, frame.shape[1]-1)]

            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            # out.write(resultImg)
            cv2.imshow("Detecting age and gender", resultImg)
except KeyboardInterrupt:
    print("User interrupted. Closing stream...")
    # out.release()
