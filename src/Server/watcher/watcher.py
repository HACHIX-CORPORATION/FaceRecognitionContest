# coding: utf-8
"""
Watcher class

Abstract::
    - This class watches changes and take action if neccessary

History::
    - Ver.      Date            Author        History
    - [1.0.0]   2020/02/19      Trung Pham        New

Copyright (C) 2020 HACHIX Corporation. All Rights Reserved.
"""
import pymongo
import json
from collections import OrderedDict
from configparser import ConfigParser
# from pdb import set_trace
import os
# from os import path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.facenet.facenet_engine import FacenetEngine
from database.database import DbController

db = DbController()


def read_ini():
    root = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(root, 'watcher', 'watcher_config.ini')
    config = ConfigParser()
    config.read(file_path)
    config.sections()
    return config


class Watcher(object):
    def __init__(self):
        config_data = read_ini()
        self.threshold = config_data['ENGINE']['threshold']

    def watch_for_retraining_facenet(self):
        """"""
        # Read previous_item_count
        prev_count = db.get_prev_count_encode()

        # Get count item in DB 
        curr_count = db.get_curr_count_encode()

        if curr_count < prev_count:
            print("WARNING ! current count is smaller than previous count ! Please reset previous count")

        # Check if percent of change is larger than threshold
        change_percent = abs((curr_count - prev_count)/prev_count) * 100
        
        if change_percent > int(self.threshold):
            # Call retrain
            facenet = FacenetEngine()
            facenet.train()

            # Update previous item count
            db.update_prev_count_encode(curr_count)
            print("Previous count updated !")


if __name__ == '__main__':
    watcher = Watcher()