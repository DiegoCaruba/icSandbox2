import cv2
import numpy as np

# img = cv2.imread('pothole.png')
threshold = 0.5  # Threshold to detect object
nms_threshold = 0.1
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().splitlines()
print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
wightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(wightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=threshold)
    # print(classIds, bbox)

    """bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)
    print("Indices: ", indices)"""

    """try:
        for i in indices:
            x = i[0]
            box = bbox[x]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w), (y + h), color=(0, 0, 255), thickness=1)
            cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    except:
        print("...")"""

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 0, 255), thickness=1)
            print(classId)
            cv2.putText(img, classNames[classId-1],
                        (box[0], box[1]-5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 255), 1)
            cv2.putText(img, str(round(confidence * 100, 2)),
                        (box[0] + 200, box[1] - 5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 255), 1)

    cv2.imshow("Output", img)
    cv2.waitKey(1)


