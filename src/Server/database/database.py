# coding: utf-8
"""
DBUtility class

Abstract::
    - for file controlling

History::
    - Ver.      Date            Author        History
    - [1.0.0]   2020/02/19     Pham        New

Copyright (C) 2020 HACHIX Corporation. All Rights Reserved.
"""
import pymongo
import json
from collections import OrderedDict
from configparser import ConfigParser
import os
import time
from os import path
import sys
from pdb import set_trace
from threading import Lock

# Common error code
ERRMSG = {
    0: None,
    -1: 'Internal Error'
}


class SingletonMetaDB(type):
    """
    DBを制御するインスタンスはSingleton　Thread-safeに適用するクラス
    """
    _instance = None
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class DbController(metaclass=SingletonMetaDB):
    client = None
    retry_times = 5
    retry_interval = 5   # 単位: ms

    def __init__(self):
        """
        コンストラクタ
        """
        # 設定ファイル読み込む
        current_dir = path.dirname(path.abspath(__file__))
        config_file = os.path.join(current_dir, 'db_config.ini')
        abs_path_config = os.path.abspath(config_file)

        config_data = ConfigParser()
        config_data.optionxform = str  # differ with large and small character
        config_data.read(abs_path_config, encoding="utf-8")

        # connect to DB once
        if DbController.client is None:
            count = 0
            while True:
                if count > DbController.retry_times:
                    break

                try:
                    client = pymongo.MongoClient(config_data['DB_INFO']['uri'])
                    print(client)

                    if client is not None:
                        DbController.client = client
                        break
                except Exception as ex:
                    print("exception = {}".format(ex))
                    count += 1
                    time.sleep(DbController.retry_interval)

        print(config_data['DB_INFO']['database'])
        db = DbController.client[config_data['DB_INFO']['database']]

        # 対象collectionを設定する。
        self.talent5__staff_collection = db[config_data['DB_INFO']['talent5__staff_collection']]
        self.talent5__staff_collection.create_index("id", unique=True)

        self.talent5__encode_collection = db[config_data['DB_INFO']['talent5__encode_collection']]
        self.talent5__feedback_collection = db[config_data['DB_INFO']['talent5__feedback_collection']]
        self.talent5__count_collection = db[config_data['DB_INFO']['talent5__count_collection']]

    @staticmethod
    def set_schema_collection(db, schema_file, collection_name):
        """
        Collectionのスキーマを設定する
        """
        current_dir = path.dirname(path.abspath(__file__))
        json_path = os.path.join(current_dir, schema_file)
        abs_json_path = os.path.abspath(json_path)
        with open(abs_json_path, 'r') as j:
            validator = json.loads(j.read())

        query = [('collMod', collection_name),
                 ('validator', validator),
                 ('validationLevel', 'moderate')]

        query = OrderedDict(query)
        db.command(query)

    def get_collection(self, collection_name):
        """
        対象のcollectionを選択する
        """
        collection = None
        if collection_name == 'talent5__staff_collection':
            collection = self.talent5__staff_collection
        elif collection_name == 'talent5__encode_collection':
            collection = self.talent5__encode_collection
        elif collection_name == 'talent5__count_collection':
            collection = self.talent5__count_collection
        elif collection_name == 'talent5__feedback_collection':
            collection = self.talent5__feedback_collection

        return collection

    def insert_staff(self, data):
        """
        - data: insert data
        - collection: (string)
        """
        print("Inserting staff data")
        try:
            collection_name = 'talent5__staff_collection'
            collection = self.get_collection(collection_name)

            if collection is not None:
                collection.insert(data)
                print("The insert staff is done!")
                return 0
            else:
                print('collection is None')
                return -1
        except:
            print("Exception:", sys.exc_info())
            return -1

    def insert_encode(self, data):
        """
        - data: insert data
        - collection: (string)
        """
        print("Inserting encode data")
        try:
            collection_name = 'talent5__encode_collection'
            collection = self.get_collection(collection_name)
            if collection is not None:
                collection.insert(data)
                print("The insert encode is done!")
                return 0, "The insert encode is done!"
            else:
                print('collection is None')
                return -1, 'collection is None'
        except Exception as e:
            msg = "Exception: {}".format(e)
            print("Exception:", sys.exc_info())
            return -1, msg

    def insert_feedback(self, data):
        """
        - data: insert data
        """
        print("insert_feedback started")
        try:
            collection_name = 'talent5__feedback_collection'
            collection = self.get_collection(collection_name)
            if collection is not None:
                collection.insert(data)
                print("The insert feedback is done!")
                return 0, "The insert feedback is done!"
            else:
                print('collection is None')
                return -1, 'collection is None'
        except Exception as e:
            msg = "Exception: {}".format(e)
            print("Exception:", sys.exc_info())
            return -1, msg

    def get_coll_length(self, collection_name):
        try:
            collection = self.get_collection(collection_name)
            if collection is not None:
                return 0, collection.count_documents()
            else:
                return -1, "collection {} is None".format(collection_name)
        except Exception as e:
            msg = "Error when get coll length: {}".format(e)
            return -1, msg

    def find_one(self, query, collection_name='talent5__staff_collection'):
        """
        API for find single object
        Args::
            - query(Obj)    : {"key": "value"}
        """
        collection = self.get_collection(collection_name)

        if collection is None:
            return None
        else:
            return collection.find_one(query)

    def find(self, query, collection_name='talent5__staff_collection'):
        """
        API for find many objects
        Args::
            - query(Obj)    : MongoDB query. Eg:  {"key": "value"}
        """
        collection = self.get_collection(collection_name)

        if collection is None:
            return None
        else:
            return collection.find(query, {'_id': False})

    def get_prev_count_encode(self):
        """"""
        collection_name = 'talent5__count_collection'
        collection = self.get_collection(collection_name)
        item = list(collection.find({}, {'_id': False}).sort([('_id', -1)]).limit(1))
        if len(item) == 0:
            count = 0
        else:
            count = item[0]['count']
        return count

    def get_curr_count_encode(self):
        """"""
        collection_name = 'talent5__encode_collection'
        collection = self.get_collection(collection_name)
        count = collection.count_documents({})
        return count

    def update_prev_count_encode(self, count):
        collection_name = 'talent5__count_collection'
        collection = self.get_collection(collection_name)
        collection.insert({'count': int(count)})
        return None

    def delete_null_id_record(self):
        collection_name = 'talent5__encode_collection'
        collection = self.get_collection(collection_name)
        res = collection.delete_many({"id": ""})
        return None

    def add_ts_to_feedback_coll(self):
        """
        add ts field to feedback coll
        """
        collection_name = 'talent5__feedback_collection'
        collection = self.get_collection(collection_name)
        print("collection {}".format(collection))
        cursor = collection.aggregate([
            {"$addFields":{
                "ts": {"$arrayElemAt":[
                    {"$split": ["$file_name", ".j"]}, 0]}
            }
            }
        ])
        # print(list(cursor))
        for doc in list(cursor):
            print(doc['_id'])
            collection.find_one_and_update({"_id": doc['_id']},
            {"$set": {
                "ts": doc['ts']
            }})
        return list(cursor)
        

if __name__ == '__main__':
    db_ctrl = DbController()
    # db_ctrl.delete_null_id_record()
    db_ctrl.add_ts_to_feedback_coll()
