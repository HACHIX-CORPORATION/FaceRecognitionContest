# coding: utf-8
"""
data transformer

Abstract::
    - Handle logs

History::
    - Ver.      Date            Author          History
    - [1.0.0]   2019/03/24      Pham          New

Copyright (C) 2019 HACHIX Corporation. All Rights Reserved.
"""
import shutil
import datetime
import os
import sys
import configparser
import logging
from logging.handlers import RotatingFileHandler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SystemLogger(object):
    """
    SystemLogger
    """

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
    def __init__(self):
        """
        Constructor

        Calling::
            obj.DataTransformer()

        Args::

        Returns::
            _ None

        Raises::
            - None

        Details::
            -
        """

        path = os.path

        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = self.cur_dir + '/files/'

        # Read config
        current_dir = path.dirname(path.abspath(__file__))
        config_file = current_dir + '/logger_config.ini'
        abs_path_config = os.path.abspath(config_file)

        self._config = configparser.ConfigParser()
        self._config.optionxform = str  # differ with large and small character
        self._config.read(abs_path_config, encoding="utf-8")

        # check config
        chk_sections = ['LOG_FILES']
        for section in chk_sections:
            if section not in self._config.sections():
                raise ValueError('invalid ini file has no section {}'.format(section))

        self._LOG_FILES = dict(self._config.items('LOG_FILES'))

        # check key in section
        chk_key = ['LOG_FILE']
        for key in chk_key:
            if key not in self._LOG_FILES:
                raise ValueError('invalid LOG_FILES section has no key {}'.format(key))

        log_file_name = self._LOG_FILES['LOG_FILE']

        self._logger_file = self.log_dir + log_file_name

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self._logger_file):
            with open(self._logger_file, "a") as f:
                f.write("start log")

        log_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')

        handler = RotatingFileHandler(filename=self._logger_file, mode='a', maxBytes= 500 * 1024,
                                      backupCount=1, encoding=None, delay=0)
        handler.setFormatter(log_formatter)
        handler.setLevel(logging.INFO)

        self._logger = logging.getLogger( __name__ )
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)

    def error(self,msg):
        self._logger.error(msg)

    def info(self,msg):
        self._logger.info(msg)

    def warning(self,msg):
        self._logger.warning(msg)

    def write(self,msg):
        self._logger.log(logging.DEBUG,msg)





    def zip_log_file(self):
        """
        Zip all file in ./files

        Calling::

        Args::

        Returns::
            - path_to_zip(string) :path to zipped file

        Raises:: None

        Details::
            - Create a zip of log with datetime string 
        """
        today = datetime.date.today().strftime("%d-%m-%Y")
        path_to_zip = self.cur_dir + '/zips/log-' + today
        shutil.make_archive(path_to_zip, 'zip', self.log_dir)
        return path_to_zip + '.zip'