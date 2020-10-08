import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './..')))
from db_utilities import DBUtility
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
        cls.db_util = DBUtility()

    # def test_105_insert_encode(self):
    #     """
    #     test insert function: dupplicated id
    #     """

    #     name = "hachixx"
    #     id = str(self.today)
    #     department = "dev"
    #     encode = [75135.53125, -53125, -5135.5125, 53.5135, -5.52315, -0.5135134]
    #     errcode, msg = self.db_util.insert_encode(name, id, department, encode)
    #     self.assertEqual(errcode, -1)

    def test_100_get_feedback(self):
        """
        get feedback informations
        """
        _, data = self.db_util.get_statistic_of_feedback()
        print(data)


if __name__ == '__main__':
    unittest.main(failfast=True)
