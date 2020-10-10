#coding: utf-8
"""
face extraction tool
Abstract::
    - extracting faces from a dataset 
History::
    - Ver.      Date            Author          History
    - [1.0.0]   2020/01/19      Tran          New
Copyright (C) 2018 HACHIX Corporation. All Rights Reserved.
"""
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray 
from os import listdir 
from os.path import isdir
import os
from matplotlib import pyplot as plt
from numpy import savez_compressed 
from numpy import asarray 
# Fixed bug
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ExtractFace: 
    def __init__(self, filename):
        self.filename = filename

    def extract_face(self, filepath, required_size=(160, 160)):
        """
        extract face for further steps 
        Calling::
        faces  = extract_face(filepath)
        Args::
            _ filename: path of images file 
            _ require_size: required size of training model 
            
        Returns::
            _ face_array: Numpy array contains bounding box information  
            -
        Details::
            - get_trained_data
        """
        # load image from file 
        image = Image.open(filepath)
        # convert to RGB, if needed
        img = image.convert('RGB')
        # conver to array 
        pixels = asarray(img)
        # create new detector, using default weights from mtcnn 
                   
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box
        x1, y1, width, height = results[0]['box']
        # resize pixels to the model size
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face 
        face = pixels[y1:y2, x1:x2]
        # resize pixels to required size of further steps 
        img = Image.fromarray(face)
        img = img.resize(required_size)
        face_array = asarray(img)
        return face_array 
    
    def load_dataset(self, require_size=(160, 160)):
        """
            Load face locations from parent dataset 
            Calling::
                faces = load_faces(directory)
            Args::
                _  
                
            Returns::
                _ asarray(X): Numpy array contains bounding box information  
                _ asarray(Y): Numpy array contains labels  
            Raises::
                -
            Details::
                - get_trained_data
        """
        X, Y = list(), list()
        # enumerate folders, on per class 
        for subdir in listdir(self.filename):
            faces = list()
            # path 
            path = self.filename + subdir + '/'
            # skip any files that might be in the dir 
            if not isdir(path):
                continue
            for name in listdir(path):
                filepath = path + name
                print(filepath)
                # extract face 
                face = self.extract_face(filepath)
                faces.append(face)
            # create labels 
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' %(len(faces), subdir))
            # storing faces 
            X.extend(faces)
            Y.extend(labels)
        return asarray(X), asarray(Y)

