# -*- coding: utf-8 -*-
# @Author  : Suresh Saini
# @Email   : suresh.saini@target.com
# @Site    : https://git.target.com/z003ljx/Review_comments_classification
# @File    : app.py
# @IDE: PyCharm Community Edition
from flask import Flask, request, jsonify, Response
import flask
import io
import json
import requests
import time,sys
import argparse
import logging


# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
import argparse
import imutils
import time
import cv2
import os
from flask_cors import CORS


from face_mask_detector import detect_mask_video

application = Flask(__name__)
CORS(application)
logger = logging.getLogger(__name__)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_mask_detector/face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="face_mask_detector/mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])

print("[INFO] loading ")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])


detect_mask_video.init_video_streaming(faceNet, maskNet)



# HTTP Errors handlers
@application.errorhandler(404)
def url_error(e):
  return """
  Wrong URL!
  <pre>{}</pre>""".format(e), 404

@application.errorhandler(500)
def server_error(e):
  return """
  An internal error occurred: <pre>{}</pre>
  See logs for full stacktrace.
  """.format(e), 500


@application.route('/upload', methods = ['POST'])  
def api(vpc=False):  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        image_path=str(f.filename)
        print('image_path',image_path)
        res=inference.run_inference(ARGS,image_path,LOADED_MODEL)
        
        print("res",res)
        response=jsonify({"result":res})   
        response.status_code=200
        return response
 

if __name__ == '__main__':
#     LOAD_NET=inference.load_net(config["cfg_path"],config["weight_path"],config["data_path"])
    

    application.run(host='0.0.0.0', port = 8088, threaded=True, debug=True, use_reloader=False)
    # setting debug = True is good for debugging, but it results in the model 
    # being loaded twice. There is no easy way to get rid of this without 
    # sacrificing auto reload which is useful in developement. 
    # Remove debug=True in  production.