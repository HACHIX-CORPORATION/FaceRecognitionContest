# coding: utf-8
"""
DBUtility class

Abstract::
    - for file controlling

History::
    - Ver.      Date            Author        History
    - [1.0.0]   2020/02/05      Pham        New

Copyright (C) 2020 HACHIX Corporation. All Rights Reserved.
"""
import os
import sys
from .database import DbController
import pandas as pd
import datetime
from bson.json_util import dumps
from bson.objectid import ObjectId

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DBUtility(object):
    # クラス変数
    __DEBUG = True

    def __init__(self):
        self._db_ctrl = DbController()

    def get_all_staff(self):
        """
        Get staff information from DB
        Args::
              query (dict)         : query: E.g: {"id":1234}

        Return::
            All document in talent5__staff
        """
        res = self._db_ctrl.find({})
        return list(res)

    def get_encode(self, query):
        """
        Get encode information
        Args::
            query (dict)         : query: E.g: {"id":1234}

        Return::
            res (list)          : list of doc
        """
        res = self._db_ctrl.find(query, collection_name="talent5__encode_collection")
        return list(res)

    def get_all_encode(self):
        """
        Get all encode from database

        """
        res = self._db_ctrl.find({}, collection_name="talent5__encode_collection")
        return list(res)

    def insert_encode(self, name, id, department, encode):
        """
        Insert encode to DB

        Args:
            body (dict)         : body of document

        Return:
            msg (str)           : msg
        """
        body = {
            "_id": str(ObjectId.from_datetime(datetime.datetime.now())),
            "id": id,
            "name": name,
            "department": department,
            "encode": encode
        }
        print(body)
        errcode, msg = self._db_ctrl.insert_encode(body)
        return errcode, msg

    def insert_feedback(self, file_name, wrong_id, wrong_name, correct_id, correct_name):
        """
        Insert to feedback collection
        Args:
            body (dict)         : body of document

        Return:
            msg (str)           : msg
        """
        body = {
            "_id": str(ObjectId.from_datetime(datetime.datetime.now())),
            "file_name": file_name,
            "wrong_id": wrong_id,
            "wrong_name": wrong_name,
            "correct_id": correct_id,
            "correct_name": correct_name,
        }
        errcode, msg = self._db_ctrl.insert_feedback(body)
        return errcode, msg

    def get_statistic_of_feedback(self):
        """
        Get all feedback from feedback coll

        Return:
            errcode (int)           : -1 if get error
            data(dict)              :  - correct_ratio(int): correct recoginiton ratio
                                       - wrong_list(list): list of wrong recognition file name 

        """
        errcode, res = 0, {
            "correct_ratio": 0,
            "wrong_list": []
        }

        try:
            query = {
              "$expr": {"$eq": ["$correct_id", "$wrong_id"] }
            }
            correct_length = self._db_ctrl.find(query, collection_name="talent5__feedback_collection").count()

            query2 = {
            "$expr": {"$ne": ["$correct_id", "$wrong_id"] }
            }
            wrong_list = list(self._db_ctrl.find(query2, collection_name="talent5__feedback_collection"))

            length = correct_length + len(wrong_list)

            if length > 0:
                correct_ratio = correct_length/length
                res = {
                    "correct_ratio": correct_ratio,
                    "wrong_list": wrong_list
                }
        except Exception as e:
            msg = "Error when get feedback: {}".format(e)
            print("Error when get feedback: {}".format(e))
            errcode = -1
            res = msg

        return errcode, res

    def find_one_encode(self, id):
        """
        Find one encode with id

        Args:
            id (int)        : id number

        Return:
            res (dict)          : one record
        """
        res = self._db_ctrl.find_one({'id': id}, collection_name="talent5__encode_collection")
        return res