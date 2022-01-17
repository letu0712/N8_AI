import cv2
import os
import json
import math
import numpy as np

x_coord = 0
y_coord = 0

def check_handup():
    if((y_coord[9] < y_coord[5] and y_coord[9]*y_coord[5]>0) and (y_coord[10] < y_coord[6] and y_coord[10]*y_coord[6] >0)):
        return "Handup two hands"
    elif(y_coord[9] < y_coord[5] and y_coord[9]*y_coord[5]>0):
        return "Handup Left"
    elif(y_coord[10] < y_coord[6] and y_coord[10]*y_coord[6]>0):
        return "Handup Right"
    elif((y_coord[9] > y_coord[5] and y_coord[9]*y_coord[5]>0) and (y_coord[10] > y_coord[6] and y_coord[10]*y_coord[6]>0)):
        return "No Handup"
    else:
        return "Unknown"


def cal_distance(i, j):
    return math.sqrt((x_coord[i] - x_coord[j])**2 + (y_coord[i] - y_coord[j])**2)


def cal_angle(i, j , k):
    try:
        angle_radian = math.acos((cal_distance(i,j)**2 + cal_distance(i,k)**2
        - cal_distance(j,k)**2)/(2*cal_distance(i,j)*cal_distance(i,k)))       #Radian

        angle = angle_radian * 180 / math.pi
    except:
        angle = 0
    return angle


def check_standing():
    angle_hip_right = cal_angle(12, 6, 14)
    angle_hip_left = cal_angle(11, 5, 13)
    # if((angle_hip_right > 150 and y_coord[6] < y_coord[12] and y_coord[6]*y_coord[12]>0)
    #       or (angle_hip_left > 150 and y_coord[5] < y_coord[11] and y_coord[5]*y_coord[11]>0)):
    #     result = "Standing"

    if((abs(y_coord[14]-y_coord[12]) >= (cal_distance(6, 12) / 3) and y_coord[14]*y_coord[12]>0)
       or (abs(y_coord[13]-y_coord[11]) >= (cal_distance(5, 11)/3) and y_coord[13]*y_coord[11]>0)):
       result = "Standing"
    elif((abs(y_coord[14]-y_coord[12]) < (cal_distance(6, 12) / 3) and y_coord[14]*y_coord[12]>0)
       or (abs(y_coord[13]-y_coord[11]) < (cal_distance(5, 11)/3) and y_coord[13]*y_coord[11]>0)):
        result = "Sitting"
    else:
        result = "Unknown"
    return result


def relu(x):
    if(x >= 0):
        return x
    else:
        return 0

PATH_CHECKPOINT = "shufflenetv2k16-211120-101235-wholebody.pkl.epoch045"
#PATH_VIDEO = "VideoTest/UploadServer/tracking1.mp4"
PATH_VIDEO = "VideoTest/Video/test1.mp4"
PATH_JSON_OUTPUT = "VideoTest/JsonVideo/jsonvideo1.json"
PATH_VIDEO_OUTPUT = "VideoTest/VideoOutput/output1.mp4"


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

video = cv2.VideoCapture(PATH_VIDEO_OUTPUT)
result = cv2.VideoWriter('VideoTest/Video/VideoResult1.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (int(video.get(3)), int(video.get(4))))
width = int(video.get(3))
height = int(video.get(4))

k = 1   #Index Frame

colors = [(0,255,0), (0,0,255),(255,0,0),(255,255,0),(0,255,255)]

def cal_distance_euclid(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

list_keypoint_shoulder = []
distance = []
for i in range(len(tweets[k]['predictions'])):
    list_keypoint_shoulder.append([])
    distance.append(float("inf"))
    
while video.isOpened():
    _, image = video.read()
    
    print("Frame: ",k)
    if(k == 50):
        break

    if(k == 1):
        for i in range(len(tweets[k]['predictions'])):
            person = tweets[k]['predictions'][i]   
            list_keypoint = person['keypoints']
            x_coord = [relu(int(list_keypoint[x])) for x in range(0, len(list_keypoint), 3)]
            y_coord = [relu(int(list_keypoint[y])) for y in range(1, len(list_keypoint), 3)]

            list_keypoint_shoulder[i].append([x_coord[5], y_coord[5]])
    
    for i in range(len(tweets[k]['predictions'])):
        person = tweets[k]['predictions'][i]   
        list_keypoint = person['keypoints']
        x_coord = [relu(int(list_keypoint[x])) for x in range(0, len(list_keypoint), 3)]
        y_coord = [relu(int(list_keypoint[y])) for y in range(1, len(list_keypoint), 3)]
        print(x_coord[5], y_coord[5])

        for j in range(len(list_keypoint_shoulder)):
            print(list_keypoint_shoulder[j][-1][0], list_keypoint_shoulder[j][-1][1])
            distance[j] = cal_distance_euclid(x_coord[5], y_coord[5], list_keypoint_shoulder[j][-1][0], list_keypoint_shoulder[j][-1][1])
            
        
        print(distance)
        index_min = np.argmin(distance, 0)
        print(index_min)
        list_keypoint_shoulder[index_min].append([x_coord[5], y_coord[5]])
        print(list_keypoint_shoulder[index_min])    

        bbox = person["bbox"]
        bbox = [int(bbox[b]) for b in range(0, len(bbox))]
        # for j in range(0, len(x_coord)):
        #     cv2.circle(image, (x_coord[j], y_coord[j]), 1, (255,0,0), -1)
        # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0,255,0), 1)
        

        cv2.putText(image, check_standing(), (bbox[0]+20, bbox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)          #colors[i]
        cv2.putText(image, check_handup(), (bbox[0]+20, bbox[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)                     #colors[i]

    for x in range(len(list_keypoint_shoulder)):
    
        cv2.putText(image, "ID: " + str(x), (list_keypoint_shoulder[x][k][0] - 20, list_keypoint_shoulder[x][k][1] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
        (255,0,0), 2)   

    result.write(image)
    cv2.waitKey(1)
    k += 1


video.release()
result.release()














