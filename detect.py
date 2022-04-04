#A Gender and Age Detection program by Mahesh Sawant

import cv2
import math
import argparse
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from time import sleep
import time


def init_realsense():
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    rs_pipeline.start(rs_config)
    return rs_pipeline

def get_image(color):
    img = np.asanyarray(color.get_data())
    # img = cv2.resize(img, (256, 256))
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


def highlightPerson(frame, personNet, min_area_k = 0.001, thr=0.3):
    blob = cv2.dnn.blobFromImage(
        frame, 1.0/127.5, 
        (300, 300), 
        (127.5, 127.5, 127.5), 
        swapRB=True, 
        crop=False
    )
    personNet.setInput(blob)
    out = personNet.forward()
    rows = frame.shape[0]
    cols = frame.shape[1]
    r = np.array([cols, rows, cols, rows])

    boxes = []
    scores = []
    for d in out[0, 0, :, :]:
        score = float(d[2])
        cls = int(d[1])
        area_k = (d[5] - d[3]) * (d[6] - d[4])
        
        if cls != 1 or score < thr or area_k < min_area_k:
            continue

        box = d[3:7] * r
        box[2] -= box[0]
        box[3] -= box[1]
        boxes.append(box.astype("int"))
        scores.append(score)

    dxs = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.1)
    
    if not len(dxs):
        return frame, []
    
    persons = [boxes[i] for i in dxs.flatten()]
    for p in persons:
        clr = (255, 0, 0)
        cv2.rectangle(frame, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), clr, thickness=2)

    return frame, [(p[0], p[1], p[0] + p[2], p[1] + p[3]) for p in persons]


def highlightFace(frame, net, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

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
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight /150)), 8)
    return frameOpencvDnn, faceBoxes

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

# faceProto="opencv_face_detector.pbtxt"
# faceModel="opencv_face_detector_uint8.pb"
faceProto="face_detector/deploy.prototxt"
faceModel="face_detector/res10_300x300_ssd_iter_140000.caffemodel"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
personModel = "ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb"
personProto = "ssd_mobilenet_v1_coco_11_06_2017/ssd_mobilenet_v1_coco.pbtxt"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
personNet = cv2.dnn.readNetFromTensorflow(personModel, personProto)

padding=20
conf_threshold = 0.7

rs_pipeline = init_realsense()

try:
    while cv2.waitKey(1) < 0 :
        start_time = time.time()
        frame = get_frame(rs_pipeline)

        if frame is None:
            continue
        
        resultImg, personBoxes = highlightPerson(frame, personNet)
        
        if not len(personBoxes):
            cv2.putText(resultImg, f'No person found', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)
            continue

        for personBox in personBoxes:
            person=frame[max(0, personBox[1]-padding):
                    min(personBox[3]+padding,frame.shape[0]-1),max(0,personBox[0]-padding)
                    :min(personBox[2]+padding, frame.shape[1]-1)]
            
            person, faceBoxes = highlightFace(person, faceNet)
            
            if not len(faceBoxes):
                cv2.putText(resultImg, f'No face found', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Detecting age and gender", resultImg)
                continue

            for faceBox in faceBoxes:

                face=person[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding // 2)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')

                age = None
                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                if agePreds[0].max() > conf_threshold:
                    age=ageList[agePreds[0].argmax()]
                    print(f'Age: {age[1:-1]} years')

                cv2.putText(resultImg, f'{gender}, {age if age is not None else "N/A"}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Detecting age and gender", resultImg)

        end_time = time.time()
        print(f"Time taken to process a single frame: {end_time - start_time}")

except KeyboardInterrupt:
    print("User interrupted. Closing stream...")
