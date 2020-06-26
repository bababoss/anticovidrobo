from flask import request, render_template, flash, g, session, redirect, url_for, jsonify
import sqlite3
# Import the database object from the main app module
from app import app, db
# Import module models (i.e. Covidinfo), if need be this object can be used to validate or modify info
from app.model.models import Covidinfo
#from face_mask_detector import detect_mask_video

temp_threshold = 37.6 # Temperature threshold in Celsius to decide if person is sick
#data_dict = detect_mask_video.robotic_states_machine()

# data_dict={
# 	     "state_1":{"status":False,"result":True,"name":"mask_detection"},
# 	     "state_2":{"status":False,"result":36.6,'name':"persion_temperature"},
# 	     "state_3":{"status":False,"result":'yes','name':"speech_command"},
# 	     }

# result_dict = {key: value['result'] for key, value in data_dict.items()}

# health check in case this needs to be on TAP
@app.route('/health')
def health():
    return jsonify({'Health': "Health is all GOOD"})


# /selectdata endpoint can be used to present data in UI
@app.route('/selectdata')
def selectdata():
	conn = sqlite3.connect('covidinfo.db')
	cur = conn.cursor()
	cur.execute('select * from covidinfo_table;')
	colnames = [desc[0] for desc in cur.description]
	collength = len(colnames)
	result_list=cur.fetchall()
	cur.execute('''select count(*) as Total, (select count(*) from covidinfo_table where screening_result='Pass') as Safe,
				(select count(*) from covidinfo_table where screening_result='Failed') as notSafe from covidinfo_table;''')
	colnames1 = [desc[0] for desc in cur.description]
	result_list1=cur.fetchall()

	cur.execute('Select  date_created, count(*), screening_result  from covidinfo_table GROUP BY screening_result, date_created ORDER BY date_created ;')
	colnames2 = [desc[0] for desc in cur.description]
	result_list2=cur.fetchall()
	return render_template('index.html', colnames=colnames, tableout=result_list, collength=collength,
		colnames1=colnames1, tableout1=result_list1, colnames2=colnames2, tableout2=result_list2, topic= 'Covid Screening info')



def temp_decision(data_dict):
    result_dict = {key: value['result'] for key, value in data_dict.items()}
    temp_low = True
    if result_dict['state_2'] > temp_threshold:
        temp_low = False
    return temp_low

# endpoint to check

def screening_decision(data_dict):
    result_dict = {key: value['result'] for key, value in data_dict.items()}
    is_temp_low=temp_decision(data_dict)
    is_mask_on = result_dict['state_1']
    if result_dict['state_3'] == 'no':
        no_symptoms = True
    else:
        no_symptoms = False
    decision_list = [is_mask_on, is_temp_low, no_symptoms]
    if all(decision_list):
        return ("Pass")
    else:
        return ("Failed")

# For inserting inputs from robotic_state_machine() into the database



def db_insert(data_dict):
    result_dict = {key: value['result'] for key, value in data_dict.items()}
    screening_result=screening_decision(data_dict)
    try:
        covidinfo_now = Covidinfo(result_dict['state_1'], result_dict['state_2'], {"Voice Input" :result_dict['state_3']}, screening_result)
        db.session.add(covidinfo_now)
        db.session.commit()
        db.session.close()
        return jsonify({'message': 'One record has been inserted into the database'})
    except Exception as e:
        return e


# db_insert(data_dict)
	
# Voice team to call db_insert() function at the end of while loop