from random import choice 
from numpy import load 
from numpy import expand_dims 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer 
from sklearn.svm import SVC 
from matplotlib import pyplot as plt
import os
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))

from facenet_engine import FacenetEngine

import unittest


class TestFacenetEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        begin all tests
        :return:
        """
        cls.engine = FacenetEngine()

    @classmethod
    def tearDownClass(cls):
        """End all tests"""
        pass

    def setUp(self):
        """Begin each test"""
        pass

    def tearDown(self):
        """Start each test"""
        pass

    def test_load_data_set(self):
        """
        test load data set
        :return:
        """
        X, Y = self.engine.load_data_set()
        print(X[0].shape)
        print(Y[0])

    def test_make_faces_encoding_labels(self):
        """
        test make_faces_encoding_labels
        :return:
        """
        # AAA
        # Arrange

        # Action
        faces_encoding, labels = self.engine.make_faces_encoding_labels()

        # Assert
        self.assertEqual(labels, "ABC")

        print(labels)

    def test_train(self):
        """
        test train
        :return:
        """
        self.engine.train()

    def test_predict(self):
        """
        test predict
        :return:
        """
        predictions = self.engine.predict('./test_predict.jpg')
        print("predictions = {}".format(predictions))

        predictions = self.engine.predict('./test_predict2.jpg')
        print("prediction2 = {}".format(predictions))

    def test_recognize(self):
        name, user_id, department = self.engine.recognize('./test_predict.jpg')
        print('user_id = {}'.format(user_id))

        name, user_id, department = self.engine.recognize('./test_predict2.jpg')
        print('user_id = {}'.format(user_id))

        name, user_id, department = self.engine.recognize('./test_predict3.jpg')
        print('user_id = {}'.format(user_id))


if __name__ == '__main__':
    unittest.main(failfast=True)