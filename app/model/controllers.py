from flask import request, render_template, flash, g, session, redirect, url_for, jsonify
import sys
# Import the database object from the main app module
from app import app, db
# Import module models (i.e. Covidinfo), if need be this object can be used to validate or modify info
from app.model.models import Covidinfo

human_info = {'is_human': True}
mask_info = {'mask_status': True}
temp_info = {'value': 100}
voice_info = {'value': 'ans'}  #test this with False as well and it can have False value as well
audio2_q1= {"value":1}
audio2_q2= {"value":1}
audio2_q3= {"value": 1}


#just for playing around and testing
@app.route('/')
def hello():
	list = [
            {'a': 1, 'b': 2},
            {'a': 5, 'b': 10}
           ]
	return jsonify({'name': list})


#health check in case this needs to be on TAP
@app.route('/health')
def health():
	return jsonify({'Health':"Health is all GOOD"})


def is_human():
	if human_info == 'True':
		print('You are good to go')
	else:
		sys.exit("Error message, non human responce.")

def mask_check():
	if mask_info['mask_status'] == 'True':
		print ('You are good to go')
	else:
		sys.exit("Error message: No mask found, please make sure you are wearing a mask")

def temp_check():
    threshold = 100
    if temp_info['value'] >= threshold:
        sys.exit("Error message, seems like you have fever")
    else:
        print ('you can move ahead')


def trigger_voice(audio_input):
    #play command 1
    play_audio_1()

def voice_get_info(audio_file):
    attempt = 0
    while voice_info['value'] == False:
        attempt += 1
        trigger_voice(audio_file)   #call raspi funcition for playing the audio
        if attempt == 3:
            sys.exit("Error message, you have reached max attempts for screening")
    if voice_info['value'] == 'yes':
        return voice_info['value']
    else:
        return voice_info['value']

def screening():
	answer1 = voice_get_info(audio2_q1)
	answer2 = voice_get_info(audio2_q2)
	answer3 = voice_get_info(audio2_q3)
	list = [ answer1, answer2, answer3 ]
	return list


#fix this, error with boolean and string

def screening_response():
    answers = screening()
    if any(False in i for i in answers):
        print ('You can enter')
    else:
        print ('sorry you cannot enter')

def data_insert(data_dict):
    """
    @params:
        data_dict={
        "state_1":{"status":False,"result":'',"name":"mask_detection"},
        "state_2":{"status":False,"result":'','name':"persion_temperature"},
        "state_3":{"status":False,"result":'','name':"speech_command"},
        }
    """
    pass
        
def main_loop():
	count = 0
	while True:
		count+=1
		try:
			print('stage1')
			is_human()
			print('Stage2')
			mask_check()
			print('Stage3')
			temp_check()
			print('Stage4')
			screening()
			print('Stage5')
			screening_response()
		except Exception as e:
			count=0

@app.route('/test')
def test():
	is_human()
	return response

#markup stage and execute only when called.
#jsonify all the responces
#CHANGE USING LIST comprehension. line 77 to 82