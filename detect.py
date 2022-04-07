#A Gender and Age Detection program by Mahesh Sawant

import cv2
import argparse
import numpy as np
import time
import pyrealsense2 as rs
from mtcnn_cv2 import MTCNN

def init_realsense():
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    rs_pipeline.start(rs_config)
    return rs_pipeline

def get_image(color):
    img = np.asanyarray(color.get_data())
    img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.rotate(img, cv2.ROTATE_180)
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

def highlightFaceNew(net, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    result = net.detect_faces(frame)
    faceBoxes=[]
    for res in result:
        if res["confidence"] >= conf_threshold:
            # keypoints = res["keypoints"]
            bbox = res["box"]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), int(round(frameHeight/150)), 8)
            faceBoxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    
    return frame, faceBoxes

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # blob=cv2.dnn.blobFromImage(frameOpencvDnn)
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (224, 224), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceModel="mobilenetssd_facedetector/res10_300x300_ssd_iter_140000.caffemodel"
faceProto="mobilenetssd_facedetector/deploy.prototxt"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

# faceNet=cv2.dnn.readNet(faceModel, faceProto)
faceNet = MTCNN()
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20

rs_pipeline = init_realsense()

while cv2.waitKey(1)<0 :
    start_time = time.time()

    # hasFrame, frame = video.read()
    frame = get_frame(rs_pipeline)
            
    if frame is None:
        continue
    
    resultImg,faceBoxes=highlightFaceNew(faceNet,frame)

    if not len(faceBoxes):
        cv2.imshow("Detecting age and gender", resultImg)
        print("No face detected")
        continue

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}, Conf: {genderPreds[0][genderPreds[0].argmax()]}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years, Conf: {agePreds[0][agePreds[0].argmax()]}')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
