# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import pygame
import pprint

from mlx90614 import mlx90614_reader
from tflite_speech_recognition import speech_command_recognizer
from app.model import controllers

COVID_QUERY="/home/pi/Downloads/Please-answer-following-questi1592997390.mp3"
WELCOME="/home/pi/Downloads/Hello-Welcome-to-target--Tha1592969510.mp3"
TEMP_QUERY="/home/pi/Downloads/Please-bring-your-forehead-in1593020270.mp3"
TEMP_MSG="/home/pi/Downloads/Temperature-measured-successfu1593020307.mp3"
THANKS_MSG="/home/pi/Downloads/Thanks-you-for-the-screening-1592997554.mp3"
FAILED_MSG="/home/pi/Downloads/You-have-failed--the-screening1592997835.mp3"
PASS_MSG="/home/pi/Downloads/You-have-pass--the-screening-p1592997882.mp3"
MASK_MSG="/home/pi/Downloads/You-have-not-wearing-the-mask1592996442.mp3"




def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = np.array([])

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        preds = maskNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)




# # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--face", type=str,
# 	default="face_detector",
# 	help="path to face detector model directory")
# ap.add_argument("-m", "--model", type=str,
# 	default="mask_detector.model",
# 	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# # load our serialized face detector model from disk
# print("[INFO] loading face detector model...")
# prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
# weightsPath = os.path.sep.join([args["face"],
# 	"res10_300x300_ssd_iter_140000.caffemodel"])
# faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# # load the face mask detector model from disk
# print("[INFO] loading face mask detector model...")
# maskNet = load_model(args["model"])





# loop over the frames from the video stream
def mask_detection(res):
    print("Result of mask:   ",res)
    return res

def speech_recognizer(interpreter,input_details,output_details):
    print("[INFO]:! Start Voice Processing")
    speech_commands=speech_command_recognizer.voice_inference(interpreter,input_details,output_details)
#     print("[PERSON SPEECH_COMMAND] ",speech_commands)
    return speech_commands
    
    
    
def temperature():
    print("[INFO]: Start Temperature Processing")
    temp=mlx90614_reader.temperature()
    print("[PERSON TEMP] ",temp)
    return temp

def play_audio(audio_path):
    
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    


def screening(mask_res,temperature,speech_command):
    print("[INFO]:! Start Screening Processing")
    # DO some processing 



def robotic_states_machine():
    robotic_state={
        "state_1":{"status":False,"result":'',"name":"mask_detection"},
        "state_2":{"status":False,"result":0.0,'name':"persion_temperature"},
        "state_3":{"status":False,"result":'','name':"speech_command"},
        }
    return robotic_state

# robotic_states={
#     "state_1":{"status":False,"result":''},
#     "state_2":{"status":False,"result":''},
#     "state_3":{"status":False,"result":''},
#     "state_4":{"status":False,"result":''}
#     }


def init_video_streaming(faceNet, maskNet,interpreter,input_details,output_details):

    mask_count=0
    state_count=0
    screening_status=False
    wait=False
    robotic_states=robotic_states_machine()

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=1000)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
#         print("---",locs, type(preds.shape),preds.shape)
        # loop over the detected face locations and their corresponding
        # locations

        if preds.shape[0]:
            robotic_states["state_1"]["status"]=True
            print("[STATE-1----STATE-1]")
            state_count+=1
        else:
            mask_count=0
            state_count=0
            robotic_states["state_1"]["status"]=False


        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            if robotic_states["state_1"]["status"]:
                if label == "Mask":
                    mask_count+=1
            if state_count>=6:
                if mask_count>3:
                    print("--------Mask---",pred)
                    play_audio(WELCOME)
                    time.sleep(5)
                    mask_res=mask_detection(True)
                    robotic_states["state_1"]['result']=mask_res
                    play_audio(TEMP_QUERY)
                    time.sleep(10)
                    temp_detected=temperature()
                    play_audio(TEMP_MSG)
                    time.sleep(5)
                    robotic_states["state_2"]['result']=float(temp_detected['PersionTemp'])
                    play_audio(COVID_QUERY)
                    time.sleep(15)
                    speech_command=speech_recognizer(interpreter,input_details,output_details)
                    screening(mask_res,temp_detected,speech_command)
                    robotic_states["state_3"]['result']=speech_command
                    print("[INFO ]: robotic_states -----",)
                    pprint.pprint(robotic_states)
                    play_audio(THANKS_MSG)
                    time.sleep(5)
                    
                    if speech_command == 'no':
                        play_audio(PASS_MSG)
                        time.sleep(5)
                    else:
                        play_audio(FAILED_MSG)
                        time.sleep(5)
                    controllers.db_insert(robotic_states)
                    
                    robotic_states=robotic_states_machine()
#                     data_insert(robotic_states)
                    mask_count=0
                    wait=True
                else:
                    play_audio(MASK_MSG)
                    time.sleep(5)
                    controllers.db_insert(robotic_states)
                state_count=0
            



            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            if screening_status:
                label = "{}: {:.2f}% TEMP: {:.2f} COVID: {}".format(label, max(mask, withoutMask) * 100,temp_detected['PersionTemp'],speech_command)
            else:
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            break


        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()