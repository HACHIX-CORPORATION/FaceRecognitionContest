# coding: utf-8
import json

from keras import Model, optimizers, regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator
from numpy import asarray
import os
import sys
from os import listdir
from os.path import isdir
import os.path as osp
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import numpy as np
from scipy.spatial import distance
from glob import glob
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from database.db_utilities import DBUtility

db_util = DBUtility()


class FacenetEngine(object):
    """
    facenet engine class
    """
    # クラス変数
    # encodeのベクトルサイズ = 128, それぞれの値は-2~2である。
    # そのため、max_distance = 11.3
    __distance_threshold = 11.0
    __debug_mode = True
    __bound = 18
    # __encode_features_vector_length = 128
    # NOTE:Encodes of new model have 512-dimension
    __encode_features_vector_length = 512

    def __init__(self):
        """
        コンストラクト
        """
        cur_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

        # facenet model　path
        # NOTE: 2015 model
        model_path = "./model/keras-facenet/model/facenet_keras.h5"
        # NOTE: 2018 model
        # reference: https://github.com/davidsandberg/facenet
        # model_path = "./model/20200622-keras-facenet/model/facenet_keras.h5"
        model_path = os.path.join(cur_dir, model_path)

        # モデルパスの存在チェック
        if osp.exists(model_path) is False:
            raise ValueError('{} not Exist'.format(model_path))

        # set for facenet model
        self.__model_path = model_path
        self.model = load_model(self.__model_path, compile=False)

        # transfer learning model
        self.transfer_model = self.make_transfer_learning_model()
        print(self.transfer_model.input)
        print(self.transfer_model.output)

        # create new detector, using default weights from mtcnn
        self.__detector = MTCNN()

        # set classifier model
        classifier_filename = "./model/SVM_classifer.pkl"
        classifier_filename = os.path.join(cur_dir, classifier_filename)
        self.__classifier_filename = classifier_filename

    # --------------------------------------------------------------------------------
    # Public function
    # --------------------------------------------------------------------------------
    def recognize(self, image_path):
        """
        指定するイメージを認証する
        :param image_path:
        :return: name
        """
        errcode, name, user_id, department = 0, 'Unknown', -1, 'Unknown'

        # check arguments
        if osp.exists(image_path) is False:
            raise ValueError('file not found {}'.format(image_path))

        # make encode from image_path
        errcode, img_encode = self.make_encode(image_path)

        if errcode is 0:
            # get all encodes fromdb
            all_anchors = db_util.get_all_encode()

            distances = list()
            for anchor in all_anchors:
                if len(anchor['encode']) == FacenetEngine.__encode_features_vector_length:
                    distances.append(distance.euclidean(img_encode, anchor['encode']))
            print(distances)

            if FacenetEngine.__debug_mode is True:
                print('img_coode = {}'.format(type(img_encode)))
                print("length of all encode in database: {}".format(len(all_anchors)))
                print("min of distances = {}".format(min(distances)))
                print("max of distances = {}".format(max(distances)))

            if np.max(distances) < FacenetEngine.__bound:
                if min(distances) < FacenetEngine.__distance_threshold:
                    anchor_idx = distances.index(min(distances))
                    name = all_anchors[anchor_idx]['name']
                    user_id = all_anchors[anchor_idx]['id']
                    department = all_anchors[anchor_idx]['department']
                else:
                    print(distances)
                    print(FacenetEngine.__distance_threshold)

            if FacenetEngine.__debug_mode is True:
                print('name = {}, id = {}, department = {}'.format(name, user_id, department))

        return errcode, name, user_id, department

    '''
    
    Training SVM 
    
    '''

    def extract_face(self, file_path, required_size=(160, 160)):
        """
        extract face for further steps
        Calling::
        faces  = extract_face(file_path)
        Args::
            _ filename: path of images file
            _ require_size: required size of training model

        Returns::
            _ face_array: Numpy array contains bounding box information
            -
        Details::
            - get_trained_data
        """
        errcode, face_array = 0, np.array([])
        # load image from file
        image = Image.open(file_path)
        # convert to RGB, if needed
        img = image.convert('RGB')
        # conver to array
        pixels = asarray(img)
        # detect faces in the image
        results = self.__detector.detect_faces(pixels)
        if len(results) == 0:
            errcode = -1
        else:
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
        return errcode, face_array

    def extract_face_for_preprocessing(self, file_path, required_size=(160, 160)):
        """
        extract face for further steps
        Calling::
        faces  = extract_face(file_path)
        Args::
            _ filename: path of images file
            _ require_size: required size of training model

        Returns::
            _ face_array: Numpy array contains bounding box information
            -
        Details::
            - get_trained_data
        """
        errcode, face_array = 0, np.array([])
        # load image from file
        image = Image.open(file_path)
        # convert to RGB, if needed
        img = image.convert('RGB')
        # conver to array
        pixels = asarray(img)
        # detect faces in the image
        results = self.__detector.detect_faces(pixels)
        if len(results) < 1:
            errcode = -1
        else:
            # extract the bounding box
            x1, y1, width, height = results[0]['box']
            # resize pixels to the model size
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # TODO:  Add with debug mode
            # cv2.imwrite('check.jpg', face)
            # resize pixels to required size of further steps
            img = Image.fromarray(face)
            img = img.resize(required_size)
            face_array = asarray(img)
        return errcode, face_array

    def load_data_set(self, require_size=(160, 160)):
        """
        Load face locations from data_set
        Calling::
            faces = load_faces(directory)
        Args::
            -

        Returns::
            - asarray (X): Numpy array contains bounding box information for face position
            - asarray(Y):  Numpy array contains labels

        Raises::
            -
        Details::
            - Load face locations from data_set
        """
        X, Y = list(), list()

        # enumerate folders, on per class
        for subdir in listdir(self.__data_set_path):
            faces = list()
            # path
            path = self.__data_set_path + subdir + '/'

            # skip any files that might be in the dir
            if not isdir(path):
                continue
            for name in listdir(path):
                file_path = path + name
                print(file_path)
                # extract face
                face = self.extract_face(file_path)
                faces.append(face)

            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            # storing faces
            X.extend(faces)
            Y.extend(labels)

        return asarray(X), asarray(Y)

    def convert(self, faces):
        """
        Load faces dataset (160, 160, 3) to encode into embedding 128d vector
        """

        new = list()
        # Training dataset
        # Convert each face to an encoding
        for face in faces:
            embed = self.encoding(self.model, face)
            new.append(embed)
        new = np.asarray(new)
        # Checking new dataset dimemsion
        return new

    @staticmethod
    def encoding(model, faces):
        """
        Load facenet pretrained model and encoding using predict function of Keras
        """
        # Scale pixel values
        faces = faces.astype('float32')
        # Standardize pixel value across channels (global)
        mean, std = faces.mean(), faces.std()
        faces = (faces - mean) / std
        # Transform face into one sample
        samples = np.expand_dims(faces, axis=0)
        # Make prediction to get encoding
        Y_hat = model.predict(samples)
        # TODO: Normalizationが必要かどうかを要検討
        # Y_hat_norm = [((i - min(Y_hat[0])) / (max(Y_hat[0]) - min(Y_hat[0]))) for i in Y_hat[0]]

        return Y_hat[0]

    @staticmethod
    def l2_normalizer(x, axis=-1, epsilon=1e-10):
        """
        標準化
        """
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def make_faces_encoding_labels(self):
        """
        画像より128次元の特徴値に変換する
        """
        faces, labels = self.load_data_set()

        print(faces.shape)  # データ数, 160, 160, 3
        print(labels.shape)  # データ数

        # Encoding faces
        faces_encoding = self.convert(faces)
        print(faces_encoding.shape)  # データ数, 128

        # Normalize
        faces_encoding = self.l2_normalizer(faces_encoding)

        return faces_encoding, labels

    def train(self):
        """
        Train SVM model on given dataset
        """
        encodes = db_util.get_all_encode()

        faces_encoding = []
        labels = []
        for encode in encodes:
            if len(encode['encode']) == FacenetEngine.__encode_features_vector_length:
                faces_encoding.append(encode['encode'])
                labels.append(encode['id'])
            else:
                print("length is not {} encode: {}".format(FacenetEngine.__encode_features_vector_length,
                                                           len(encode['encode'])))

        # Label encode targets
        encoder = LabelEncoder()
        encoder.fit(labels)
        normalized_labels = encoder.transform(labels)

        normalized_labels = np.array(normalized_labels)
        faces_encoding = np.array(faces_encoding)

        # Fit into SVM model
        model = SVC(kernel='linear', probability=True)
        model.fit(faces_encoding, normalized_labels)
        joblib.dump(model, self.__classifier_filename)
        print('Save')

    def preprocessing(self, input_folder, output_folder):
        """
        Extract face from input image and save as output image

        Args:
            - input_folder(str)   :     path of input data folder (will process all image in all sub dir of input folder)
            - output_folder(str)  :    path of output data folder (output folder structure is the same as structure
                                            of input data folder)

        Details:
            - input images size   : any
            - output images size  : 160*160*3 (RGB)
        """
        for cur, dirs, _ in os.walk(input_folder):
            for sub_dir in dirs:
                for curDir, subDirs, files in os.walk(os.path.join(input_folder, sub_dir)):
                    for file in files:
                        file_path = os.path.join(curDir, file)
                        filename, file_extension = os.path.splitext(file_path)
                        out_path = os.path.join(output_folder, sub_dir)
                        if not os.path.exists(out_path):
                            os.mkdir(out_path)
                        output_file_path = os.path.join(out_path, file)
                        if 'jpeg' in file_extension:
                            errcode, face = self.extract_face_for_preprocessing(file_path)
                            if errcode is 0:
                                try:
                                    pil_img = Image.fromarray(face)
                                    pil_img.save(output_file_path)
                                except Exception as e:
                                    print("process image {} get error {}".format(file, e))
                            else:
                                print("process image {} get error when extract face".format(file))

    def make_transfer_learning_model(self):
        """
        making transfer learning model from facenet
        input: 160,160,3
        output: 128
        """
        model = self.model
        # Freeze the layers
        for layer in model.layers[:424]:
            layer.trainable = False
        model.layers.pop()
        # Adding custom Layers
        x = model.layers[-1].output
        predictions = Dense(26, activation="softmax", kernel_regularizer=regularizers.l2(0.01))(x)

        # creating the final model
        model_final = Model(input=model.input, output=predictions)

        return model_final

    def transfer_learning(self, train_data_dir, validation_data_dir, epochs):
        # compile the model
        self.transfer_model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                                    metrics=["accuracy"])

        # Initiate the train and test generators with data Augumentation
        # Save the model according to the conditions
        checkpoint = ModelCheckpoint("facenet_transfer_weight.h5", monitor='val_accuracy', verbose=2,
                                     save_best_only=True,
                                     save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=100, verbose=1, mode='auto')

        temp_path = os.path.join(os.getcwd(), "temp")
        train_data_path = os.path.join(temp_path, "train")
        val_data_path = os.path.join(temp_path, "val")

        # doing preprocessing when temp dir not exist
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
            if not os.path.exists(train_data_path):
                os.mkdir(train_data_path)

            if not os.path.exists(val_data_path):
                os.mkdir(val_data_path)

            self.preprocessing(train_data_dir, train_data_path)
            self.preprocessing(validation_data_dir, val_data_path)

        train_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_path,
            target_size=(160, 160),
            batch_size=32,
            class_mode="categorical")

        validation_generator = test_datagen.flow_from_directory(
            val_data_path,
            target_size=(160, 160),
            class_mode="categorical")

        # Train the model
        history = self.transfer_model.fit_generator(
            train_generator,
            steps_per_epoch=2,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=2,
            callbacks=[checkpoint, early])

        return history

    '''
    Predicting  
    '''

    def make_encode(self, input_image):
        """
        Make embedding vector (128-dimensions) from one image
        """
        errcode, embed = 0, np.array([])
        errcode, face = self.extract_face(input_image)

        # TODO: CongThanh test performance
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        if errcode is 0:
            faces = []
            faces.append(face)
            embed = self.convert(faces)
        return errcode, embed

    def predict(self, input_image):
        """
        Predicting the class of input image using pretrained model on Japanese dataset
        """
        errcode, predictions = 0, None
        errcode, embed = self.make_encode(input_image)
        if errcode is 0:
            model = joblib.load(self.__classifier_filename)
            predictions = model.predict_proba(embed)

        return errcode, predictions


if __name__ == "__main__":
    face_net_eval = FacenetEngine()

    history = face_net_eval.transfer_learning(train_data_dir="./dataset", validation_data_dir="./valdata", epochs=500)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    #  "Accuracy"
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # "Loss"
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
