# coding: utf-8
"""
test for log

Abstract::
    Test for log

History::
    - Ver.      Date           Author     History
    - [1.0.0]   2019/03        Pham      New

Copyright (C) 2019 HACHIX Corporation. All Rights Reserved.
"""
import os
import sys
import unittest
import datetime
import csv
import time
import json

from os import path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './../../')))
from log.logger import SystemLogger


class TestLog(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        begin all tests
        :return:
        """
        cls.obj = SystemLogger()

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

    def execute_test(self, test_func, test_param):
        """
        execute each test

        Calling::
            obj.execute_test(
                test_func,       (i) test function
                test_param,      (i) test param
            )

        Returns::
            - errmsg (unicode)   : error msg

        Details::

        """
        raise_flg = test_param['raise']
        IN = test_param['IN']
        if 'OUT' in test_param:
            OUT = test_param['OUT']
        else:
            OUT = None

        # print_msg = 'func = {}; in = {}; out = {}; raise = {}'
        # print_msg = print_msg.format(test_func.__name__, IN, OUT, raise_flg)
        # print(print_msg)

        if raise_flg is True:
            with self.assertRaises(Exception) as ex:
                test_func(**IN)
            print('test errmsg: {}'.format(ex.exception))
            errmsg = 'exception.message : {}'.format(ex.exception)
            self.assertTrue(errmsg.find(OUT) > -1)

            return None
        else:
            result = test_func(**IN)
            if OUT is not None:
                self.assertEqual(result[:-1], OUT[:-1])

            return result

    def test00_001_zip(self):
        """
        test normal test00_001_zip

        """
        print('******************************')
        print('Start test00_001_zip')

        test_func = self.obj.zip_log_file

        test_params = (
            {'raise': False,
             'IN': {},
             'OUT': None},
        )

        for test_param in test_params:
            result = self.execute_test(test_func, test_param)
            print(result)

        print('End test00_001_zip')

    def test10_001_error(self):
        """
        test normal test10_001_error

        """
        print('******************************')
        print('Start test10_001_error')

        test_func = self.obj.error

        test_params = (
            {'raise': False,
             'IN': {'msg':"test error test10_001_error"},
             'OUT': None},
        )

        for test_param in test_params:
            result = self.execute_test(test_func, test_param)
            print(result)

        print('End test10_001_error')

    def test20_001_info(self):
        """
        test normal test20_001_info

        """
        print('******************************')
        print('Start test20_001_info')

        test_func = self.obj.info

        test_params = (
            {'raise': False,
             'IN': {'msg':"test info: test20_001_info"},
             'OUT': None},
        )

        for test_param in test_params:
            result = self.execute_test(test_func, test_param)
            print(result)

        print('End test20_001_info')


    def test30_001_warning(self):
        """
        test normal test30_001_warning

        """
        print('******************************')
        print('Start test30_001_warning')

        test_func = self.obj.warning

        test_params = (
            {'raise': False,
             'IN': {'msg':"test info: test30_001_warning"},
             'OUT': None},
        )

        for test_param in test_params:
            result = self.execute_test(test_func, test_param)
            print(result)

        print('End test30_001_warning')


    def test40_001_write(self):
        """
        test normal test40_001_write

        """
        print('******************************')
        print('Start test20_001_info')

        test_func = self.obj.write

        test_params = (
            {'raise': False,
             'IN': {'msg':"test info: test40_001_write"},
             'OUT': None},
        )

        for test_param in test_params:
            result = self.execute_test(test_func, test_param)
            print(result)

        print('End test40_001_write')



if __name__ == '__main__':
    unittest.main(failfast=True)
