import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './..')))
from database import DbController
import time
import datetime

import unittest


class TestDataBase(unittest.TestCase):
    """
    Test database.py
    """

    today = datetime.datetime.timestamp(datetime.datetime.now())

    @classmethod
    def setUpClass(cls):
        """
        begin all tests
        :return:
        """
        cls.db_ctrl = DbController()

    def test_001_insert_staff(self):
        """
        test insert function: 正常系
        """
        test_data = {
            "_id": str(self.today),
            "id": int(self.today),
            "name": "hachix",
            "department": "development"
        }
        res = self.db_ctrl.insert_staff(test_data)
        self.assertEqual(res, 0)

    def test_005_insert_staff(self):
        """
        test insert function: dupplicated id
        """
        test_data = {
            "_id": str(self.today),
            "id": int(self.today),
            "name": "hachixxx",
            "department": "development01"
        }
        res = self.db_ctrl.insert_staff(test_data)
        self.assertEqual(res, -1)

    def test_010_find_one(self):
        """
        test find one function: 正常系
        """
        query = {"id": int(self.today)}
        res = self.db_ctrl.find_one(query)
        self.assertEqual(res["name"], "hachix")

    def test_015_find(self):
        """
        test find function: 正常系
        ショップ名で検索する。
        """
        query = {"id": int(self.today)}
        res = self.db_ctrl.find(query)
        for doc in res:
            self.assertEqual(doc["name"], "hachix")

    """
        Test encode database
    """

    def test_101_insert_encode(self):
        """
        test insert function: 正常系
        """
        test_data = {
            "_id": str(self.today),
            "id": str(self.today),
            "name": "hachix",
            "department": "development",
            "encode": [1, 2, 3, 4.5]
        }
        errcode, msg = self.db_ctrl.insert_encode(test_data)
        self.assertEqual(errcode, 0)

    def test_105_insert_encode(self):
        """
        test insert function: dupplicated id
        """
        test_data = {
            "_id": str(self.today),
            "id": str(self.today),
            "name": "hachixxx",
            "department": "development01",
            "encode": [75135.53125, -53125, -5135.5125, 53.5135, -5.52315, -0.5135134]
        }
        errcode, msg = self.db_ctrl.insert_encode(test_data)
        self.assertEqual(errcode, -1)

    def test_110_find_one(self):
        """
        test find one function: 正常系
        """
        query = {"id": int(self.today)}
        res = self.db_ctrl.find_one(query, collection_name='talent5__encode_collection')
        self.assertEqual(res["name"], "hachix")
        self.assertEqual(res["encode"], [1, 2, 3, 4])

    def test_115_find(self):
        """
        test find function: 正常系
        ショップ名で検索する。
        """
        query = {"id": int(self.today)}
        res = self.db_ctrl.find(query, collection_name='talent5__encode_collection')
        for doc in res:
            self.assertEqual(doc["name"], "hachix")
            self.assertEqual(doc["encode"], [1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main(failfast=True)
