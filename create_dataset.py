# Creating dataset from video feed by recognising people
import cv2
import time
import shutil
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
from mtcnn_cv2 import MTCNN
from skimage.metrics import structural_similarity as ssim

personModel = "ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb"
personProto = "ssd_mobilenet_v1_coco_11_06_2017/ssd_mobilenet_v1_coco.pbtxt"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)
SAVE_FLAG = True
WINNOW_THRESHOLD = None

def init_realsense():
    rs_pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
    rs_pipeline.start(rs_config)
    return rs_pipeline

def get_image(color):
    img = np.asanyarray(color.get_data())
    # img = cv2.resize(img, (256, 256))
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
    frameCopy = frame.copy()
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
        cv2.rectangle(frameCopy, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), clr, thickness=2)

    return frameCopy, [(p[0], p[1], p[0] + p[2], p[1] + p[3]) for p in persons]

def highlightFace(frame, net, conf_threshold=0.7):
    frameCopy = frame.copy()
    frameHeight = frame.shape[0]
    result = net.detect_faces(frame)
    faceBoxes=[]
    for res in result:
        if res["confidence"] >= conf_threshold:
            bbox = res["box"]
            cv2.rectangle(frameCopy, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), int(round(frameHeight/150)), 8)
            faceBoxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    
    return frameCopy, faceBoxes

def winnow_images():
    images = list(DATA_DIR.glob("*.png"))
    to_delete = set()

    for i in range(len(images) - 1):
        img1 = cv2.imread(images[i].as_posix())
        for j in range(i, len(images)):
            if images[j] not in to_delete:
                img2 = cv2.imread(images[j].as_posix())

                score = ssim(img1, img2)
                if score > WINNOW_THRESHOLD:
                    to_delete.add(images[j])
    
    MOVE_DIR = DATA_DIR.parent / "copies"
    for fname in to_delete:
        shutil.move(fname.as_posix(), MOVE_DIR / fname.name)

    return
    
def main():
    personNet = cv2.dnn.readNetFromTensorflow(personModel, personProto)
    faceNet = MTCNN()
    rs_pipeline = init_realsense()
    last_save = time.time()
    frame_count = 0

    try:
        while cv2.waitKey(1) < 0 :
            start_time = time.time()
            frame = get_frame(rs_pipeline)

            if frame is None:
                continue
            
            resultImg, personBoxes = highlightPerson(frame, personNet)
            # resultImg, personBoxes   = highlightFace(frame, faceNet)
            
            cv2.putText(resultImg, f'{len(personBoxes)} people found', \
                (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Video Feed x01", resultImg)
            
            # save a frame if person is found and time has exceeded by 2 seconds
            if SAVE_FLAG and len(personBoxes):
                cv2.imwrite(f"{DATA_DIR.as_posix()}/frame-{frame_count}.png", frame)
                last_save = time.time()

            end_time = time.time()
            print(f"Time taken to process a single frame: {end_time - start_time}")
            frame_count += 1

    except KeyboardInterrupt:
        print("User interrupted. Closing stream...")
    
if __name__ == "__main__":
    main()