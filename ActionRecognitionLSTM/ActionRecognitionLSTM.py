import cv2
import os
import json 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


#Corredsponding result.left_hand_landmarks.landmark
def exact_keypoints(x_coord, y_coord):
    left_hand = np.array([[x_coord[i], y_coord[i]] for i in range(112,133)]).flatten()

    #Corredsponding result.right_hand_landmarks.landmark
    right_hand = np.array([[x_coord[i], y_coord[i]] for i in range(91,112)]).flatten()

    shoulder_arm = np.array([[x_coord[i], y_coord[i]] for i in range(5,11)]).flatten()

    #Face
    face = np.array([[x_coord[i], y_coord[i]] for i in range(23,91)]).flatten()
    
    return np.concatenate([face, shoulder_arm, right_hand, left_hand])


actions = np.array(["Like", "Hello"])
no_sequences = 30
sequence_length = 30

#Build Model
def build_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30,232)))
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(actions.shape[0], activation="softmax"))
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model

model = build_model()
model.load_weights("action3.h5")

PATH_CHECKPOINT = "shufflenetv2k16-wholebody"
PATH_VIDEO_TEST = "UploadServer/VideoTest.avi"
PATH_JSON_OUTPUT = "VideoJson/jsonvideo.json"
PATH_VIDEO_KEYPOINT = "VideoOutput/outputkeypoint.mp4"

print("time CUDA_VISIBLE_DEVICES=1 python3 -m openpifpaf.video --source="+PATH_VIDEO_TEST+
         " --checkpoint="+PATH_CHECKPOINT+
         " --line-width 1 --json-output "+PATH_JSON_OUTPUT+
         " --video-output "+PATH_VIDEO_KEYPOINT+
         " --show-box")

os.system("time CUDA_VISIBLE_DEVICES=1 python3 -m openpifpaf.video --source="+PATH_VIDEO_TEST+
         " --checkpoint="+PATH_CHECKPOINT+
         " --line-width 1 --json-output "+PATH_JSON_OUTPUT+
         " --video-output "+PATH_VIDEO_KEYPOINT+
         " --show-box")



#List contain keypoints of each frame
tweets = []
for line in open(PATH_JSON_OUTPUT, "r"):
    tweets.append(json.loads(line))

video = cv2.VideoCapture(PATH_VIDEO_KEYPOINT)
result = cv2.VideoWriter('VideoResult.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (int(video.get(3)), int(video.get(4))))

k = 1   #Index Frame

sequence = []
sentence = []

while True:
    if (k == len(tweets)):
        break
    _, image = video.read()
    height, width, _ = image.shape
    print("Frame: ",k)
    for i in range(len(tweets[k]['predictions'])):
        person = tweets[k]['predictions'][i]
        
        list_keypoint = person['keypoints']
        x_coord = [int(list_keypoint[i])/width for i in range(0, len(list_keypoint), 3)]
        y_coord = [int(list_keypoint[i])/height for i in range(1, len(list_keypoint), 3)]

        keypoints = exact_keypoints(x_coord, y_coord)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]
        
        cv2.rectangle(image, (0,0), (width, 60), (255, 0, 0), -1)
        if len(sequence) == 30:   
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(res)
            print(actions[np.argmax(res)])
            cv2.putText(image,"Predict Action: "+ actions[np.argmax(res)], (int(width/2) - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
          
    result.write(image)
    cv2.waitKey(1)
    k += 1
video.release()
result.release()
cv2.destroyAllWindows()

print("End===================================")
input_path = "VideoResult.avi"
output_path = "VideoSegmentation/OutputSegmentation.mp4"
PATH_MODEL_YOLACT = "/workspace/Tu2/yolact/weights/yolact_resnet50_54_800000.pth"



