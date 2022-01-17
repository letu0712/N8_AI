import cv2
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift


def convert(list_hist):
    result = []
    for hist in list_hist:
        result.append(hist[0])
    return np.array(result)

def cal_euclid(x, y):
    sum_square = 0
    for i in range(len(x)):
        sum_square += (x[i] - y[i])**2
    return np.sqrt(sum_square)

#PATH_CHECKPOINT = "shufflenetv2k16-211120-101235-wholebody.pkl.epoch045"
PATH_CHECKPOINT = "shufflenetv2k16-wholebody"
PATH_VIDEO = "UploadServer/TestTracking2.mp4"
PATH_JSON_OUTPUT = "JsonOutput/jsonvideo.json"
PATH_VIDEO_OUTPUT = "VideoOutput/videooutput.mp4"

os.system("python3 -m openpifpaf.video --source="+PATH_VIDEO+
         " --checkpoint="+PATH_CHECKPOINT+
         " --line-width 1 --json-output "+PATH_JSON_OUTPUT+
         " --video-output "+PATH_VIDEO_OUTPUT+
         " --show-box")
print("python3 -m openpifpaf.video --source="+PATH_VIDEO+
         " --checkpoint="+PATH_CHECKPOINT+
         " --line-width 1 --json-output "+PATH_JSON_OUTPUT+
         " --video-output "+PATH_VIDEO_OUTPUT+
         " --show-box")

tweets = []
for line in open(PATH_JSON_OUTPUT, "r"):
    tweets.append(json.loads(line))

print("Num Frame: ", len(tweets))

video = cv2.VideoCapture(PATH_VIDEO)
result = cv2.VideoWriter('VideoResult/VideoResult.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (int(video.get(3)), int(video.get(4))))

width = int(video.get(3))
height = int(video.get(4))
print("width: ", width)
print("height: ", height)

result_notracking = cv2.VideoWriter("VideoResult/VideoResult_NoTracking.avi",
                                    cv2.VideoWriter_fourcc(*"MJPG"),
                                    30, (int(video.get(3)), int(video.get(4))))

result_histogram_per1 = cv2.VideoWriter('VideoResult/Histogram_per1_NoTracking.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640, 480))

result_histogram_per2 = cv2.VideoWriter('VideoResult/Histogram_per2_NoTracking.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640, 480))
k = 1   #Index Frame

colors = [(0,255,0), (0,0,255),(255,0,0),(255,255,0),(0,255,255)]

list_histogram = []


while True:
    ret, image = video.read()

    if ret == False:
        break

    if(k == len(tweets)):
        break
    print("Frame: ",k)
    for i in range(len(tweets[k]['predictions'])):

        person = tweets[k]['predictions'][i]
        
        list_keypoint = person['keypoints']
        x_coord = [int(list_keypoint[x]) for x in range(0, len(list_keypoint), 3)]
        y_coord = [int(list_keypoint[y]) for y in range(1, len(list_keypoint), 3)]

        bbox = person["bbox"]

        bbox = [int(bbox[b]) for b in range(0, len(bbox))]
        # for j in range(0, len(x_coord)):
        #     cv2.circle(image, (x_coord[j], y_coord[j]), 1, (255,0,0), -1)
        cv2.rectangle(image, (bbox[0]-40, bbox[1]-40), (bbox[0] + bbox[2]+40, bbox[1] + bbox[3]+40), (0,255,0), 1)
        cv2.putText(image, "ID: "+ str(i), (bbox[0]-40, bbox[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

        ROI = image[bbox[1]-40:bbox[1]+bbox[3]+40, bbox[0]-40:bbox[0]+bbox[2]+40]
        
        ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([ROI],[0], None, [256], [0,256])
        hist = convert(hist)

        list_histogram.insert(0, hist.tolist())

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(hist)
        fig.savefig("histogram.png")
        plt.close(fig)

        histogram = cv2.imread("histogram.png")
                

        if(i == 0):
            result_histogram_per1.write(histogram)
        if(i == 1):
            result_histogram_per2.write(histogram)
        
        result_notracking.write(image)

    cv2.waitKey(1)
    k += 1

ms = MeanShift()
ms.fit(list_histogram)
cluster_centers = ms.cluster_centers_
print("cluster.shape, ",cluster_centers.shape)

print("ListHistogram.shape: ",np.array(list_histogram).shape)


result_histogram_per1_tracking = cv2.VideoWriter('VideoResult/Histogram_per1_Tracking.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640, 480))

result_histogram_per2_tracking = cv2.VideoWriter('VideoResult/Histogram_per2_Tracking.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640, 480))

t = 1

video = cv2.VideoCapture(PATH_VIDEO_OUTPUT)

while True:
    ret, image = video.read()
    
    image = cv2.resize(image,(width, height))
    if ret == False:
        break

    distance = []
    for i in range(len(tweets[t]['predictions'])):
        distance.append(float("inf"))   
    
    if(t == len(tweets)):
        break
    print("Frame: ",t)

    for i in range(len(tweets[t]['predictions'])):

        person = tweets[t]['predictions'][i]

        bbox = person["bbox"]

        bbox = [int(bbox[b]) for b in range(0, len(bbox))]
        cv2.imwrite("image.jpg", image)
        ROI = image[bbox[1]-40:bbox[1]+bbox[3]+40, bbox[0]-40:bbox[0]+bbox[2]+40]

        ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([ROI],[0], None, [256], [0,256])
        hist = convert(hist)
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(hist)
        fig.savefig("histogram.png")
        plt.close(fig)

        histogram = cv2.imread("histogram.png")

        for j in range(len(tweets[t]["predictions"])):    #Calculate distance euclid with each cluster_center
            distance[j] = cal_euclid(cluster_centers[j], hist)

        ID = np.argmin(distance)
        if(ID == 0):
            result_histogram_per1_tracking.write(histogram)
        if(ID == 1):
            result_histogram_per2_tracking.write(histogram)
            

        #cv2.rectangle(image, (bbox[0]-40, bbox[1]-40), (bbox[0] + bbox[2]+40, bbox[1] + bbox[3]+40), (0,255,0), 1)        
        cv2.putText(image, "ID: "+ str(ID), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        result.write(image)

    cv2.waitKey(1)
    t += 1


video.release()
result.release()












