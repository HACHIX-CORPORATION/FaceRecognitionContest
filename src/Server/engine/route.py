import flask
from flask import Blueprint
from flask_uploads import UploadSet, DATA, DOCUMENTS
from flask import jsonify, render_template, request

import numpy as np
from PIL import Image
import io
from pdb import set_trace

import os
import base64
import glob
import datetime
import sys
import time
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.facenet.facenet_engine_v2 import FacenetEngine
# from engine.facenet.facenet_engine import FacenetEngine
from database.db_utilities import DBUtility

facenet = FacenetEngine()
db_util = DBUtility()

engine = Blueprint('engine', __name__)
files = UploadSet('files', DATA + DOCUMENTS)


@engine.route("/api/engine/predict", methods=['GET', 'POST'])
def recognize():
    """
    Upload binary image API with format: <image_source>
    And process recognition

    """
    # Set directory for recognition image
    cur_path = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(cur_path, "recognition_images")
    if os.path.exists(img_dir) is False:
        os.mkdir(img_dir)

    if request.method == 'POST':
        start = time.time()
        try:
            # save recognition data
            if request.stream:
                data = request.stream.read()
            else:
                data = request.data
            data = data[data.find(b'/9'):]
            img_data = base64.b64decode(data)
            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)

            img_data = Image.open(io.BytesIO(img_data))

            # save image
            now = datetime.datetime.now()
            img_file_name = "{}.jpeg".format(now.timestamp())
            img_file_path = os.path.join(img_dir, img_file_name)
            img_data.save(img_file_path)

            # with open(img_file_path, "wb+") as g:
            #     g.write(img_data)
            #     g.close()

            # # load image from file
            # image = Image.open(img_file_path)
            # print(image.size)

            # Recognition
            errcode, name, user_id, department = facenet.recognize(os.path.abspath(img_file_path), image_data=img_data)
            # errcode, name, user_id, department = facenet.recognize(os.path.abspath(img_file_path))

            if errcode is 0:
                label = {
                    "name": name,
                    "id": user_id,
                    "department": department,
                    "img_file_name": img_file_name
                }
                print(label)

                response = jsonify({
                    "errcode": 0,
                    "msg": label
                })
            else:
                response = jsonify({
                    "errcode": errcode,
                    "msg": "写真には1つの顔のみを許可します。再度ご確認ください。"
                })
        except Exception as e:
            print(e)
            response = jsonify({
                "errcode": -1,
                "msg": "Error: {}".format(e)
            })
        end = time.time()
        print("execute time: {}".format(end-start))
    else:
        response = jsonify({
            "errcode": -1,
            "msg": "get method is not supported"
        })

    return response


@engine.route("/api/engine/feedback", methods=['GET', 'POST'])
def feedback():
    """
    Catch feedback from user and post to database

    """
    # Set directory for recognition image
    cur_path = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(cur_path, "recognition_images")
    if os.path.exists(img_dir) is False:
        os.mkdir(img_dir)

    if request.method == 'POST':
        start = time.time()
        try:
            data = request.json
            keys = ['id', 'name', 'department', 'img_file_name', 'wrong_id', 'wrong_name']
            for key in keys:
                if key not in data:
                    raise KeyError("body not has key {}".format(key))

            img_file_name = data['img_file_name']
            img_file_path = os.path.join(img_dir, img_file_name)
            # Make encode
            errcode, img_encode = facenet.make_encode(img_file_path)
            if errcode is 0:
                # Post encode and true label to db
                encode = img_encode.flatten()
                encode = encode.tolist()
                # Insert encode to make new label data for train
                errcode, msg = db_util.insert_encode(name=data['name'], id=data['id'], department=data['department'], encode=encode)

                # now ts
                now_ts = datetime.datetime.now().timestamp()

                # Insert feedback
                errcode, msg = db_util.insert_feedback(file_name=data['img_file_name'],
                                                    correct_id=data['id'],
                                                    correct_name=data['name'],
                                                    wrong_id=data['wrong_id'],
                                                    wrong_name=data['wrong_name'],
                                                    ts=now_ts)

                # Copy a false recognition image to Client/static/assets
                if data['id'] != data['wrong_id']:
                    des_img_dir = "../Client/static/assets/FalseRecognition"
                    if os.path.exists(des_img_dir) is False:
                        os.mkdir(des_img_dir)
                    des_img_path = os.path.join(des_img_dir, img_file_name)
                    shutil.copy(img_file_path, des_img_path)

                response = jsonify({
                    "errcode": errcode,
                    "msg": msg
                })
            else:
                response = jsonify({
                    "errcode": errcode,
                    "msg": "写真には1つの顔のみを許可します。再度ご確認ください。"
                })
        except Exception as e:
            print(e)
            response = jsonify({
                "errcode": -1,
                "msg": "Error: {}".format(e)
            })
        end = time.time()
        print("execute time: {}".format(end-start))
    else:
        # GET METHOD
        # Get statistic of feedback from db
        try:
            errcode , msg = db_util.get_statistic_of_feedback()
            response = jsonify({
                "errcode": errcode,
                "msg": msg
            })
        except Exception as e:
            print(e)
            response = jsonify({
                "errcode": -1,
                "msg": "Error: {}".format(e)
            })
        
    return response


@engine.route("/api/engine/train", methods=['GET', 'POST'])
def train():
    """
    Upload binary image API with format: <image_source>
    And process train
    Dummy method
    """
    # Set directory for recognition image
    cur_path = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(cur_path, "dataset")
    if os.path.exists(img_dir) is False:
        os.mkdir(img_dir)

    if request.method == 'POST':
        try:
            # save training data
            data = request.stream.read()
            data = data[data.find(b'/9'):]
            img_data = base64.b64decode(data)
            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)

            # save image
            now = datetime.datetime.now()
            img_file_name = "{}.jpeg".format(now.timestamp())
            img_file_path = os.path.join(img_dir, img_file_name)

            with open(img_file_path, "wb+") as g:
                g.write(img_data)
                g.close()

            response = jsonify({
                "errcode": 0,
                "msg": "dummy traning method"
            })
        except Exception as e:
            print(e)
            response = jsonify({
                "errcode": -1,
                "msg": "Error: {}".format(e)
            })
    else:
        response = jsonify({
            "errcode": -1,
            "msg": "get method is not supported"
        })

    return response
