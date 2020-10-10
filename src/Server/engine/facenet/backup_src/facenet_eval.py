# coding: utf-8

import argparse
from numpy import asarray
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


class FaceNetEval(object):
    """
    facenetのモデルを評価する
    """
    # クラス変数
    __images_per_person = 5        # 一人に対しての画像枚数
    __anchor_idx = (0, )
    __evals_idx = (1, 2, 3, 4)
    __distance_threshold = 0.95

    def __init__(self, data_set_path, model_path):
        """
        コンストラクト
        """
        # 引数チェック
        if osp.exists(data_set_path) is False:
            raise ValueError('{} not Exist'.format(data_set_path))

        if osp.exists(model_path) is False:
            raise ValueError('{} not Exist'.format(model_path))

        self.__data_set_path = data_set_path
        self.__model_path = model_path

        # get sub directory
        sub_dirs = glob(self.__data_set_path + '*/')

        # get list of name
        self.__people = [sub_dir.split('/')[-2] for sub_dir in sub_dirs]

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

    @staticmethod
    def extract_face(file_path, required_size=(160, 160)):
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
        # load image from file
        image = Image.open(file_path)
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

    def convert(self, faces):
        # Load the FaceNet model
        model = load_model(self.__model_path, compile=False)

        new = list()
        # Training dataset
        # Convert each face to an encoding
        for face in faces:
            embed = self.encoding(model, face)
            new.append(embed)
        new = np.asarray(new)
        # Checking new dataset dimemsion
        return new

    @staticmethod
    def encoding(model, faces):
        # Scale pixel values
        faces = faces.astype('float32')
        # Standardize pixel value across channels (global)
        mean, std = faces.mean(), faces.std()
        faces = (faces - mean) / std
        # Transfomr face into one sample
        samples = np.expand_dims(faces, axis=0)
        # Make prediction to get encoding
        Y_hat = model.predict(samples)
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


    def create_label2idx(self, labels):
        """
        ラベルとindexの対応付リストを生成する。
        """
        label2idx = []
        n = self.__images_per_person
        for i in range(int(len(labels) / n)):
            label2idx.append([i * n, i * n + 1, i * n + 2, i * n + 3, i * n + 4])

        return label2idx

    @staticmethod
    def split_face_encodings(encodes_labels_list):
        """
        画像特徴リストより、anchorsと評価用に分割するkento-yamazaki
        Args::
            - encodes_labels_list: list of dict

        Returns::
            - anchors: list of object for anchors

            - evals: list of object for evals
        """
        anchors = []
        evals = []
        for encode_label in encodes_labels_list:
            if (encode_label['id'] % FaceNetEval.__images_per_person) in FaceNetEval.__anchor_idx:
                anchors.append(encode_label)

            if (encode_label['id'] % FaceNetEval.__images_per_person) in FaceNetEval.__evals_idx:
                evals.append(encode_label)

        return anchors, evals

    @staticmethod
    def evaluate(anchors, evals):
        """
        anchorsとevalsで評価する。
        """
        bound = 1.5
        unknown = 0
        positive = 0
        negative = 0
        for k in range(len(evals)):
            distances = []
            max_distances = []
            for i in range(len(anchors)):
                distances.append(distance.euclidean(evals[k]['encode'], anchors[i]['encode']))

            if np.max(distances) > bound:
                unknown += 1
            else:
                anchor_idx = distances.index(min(distances))
                anchor_name = anchors[anchor_idx]['label']
                eval_name = evals[k]['label']

                if anchor_name == eval_name:
                    positive += 1
                else:
                    print('anchor_name = {}, eval_name = {}'.format(anchor_name, eval_name))
                    negative += 1

        accuracy = positive / len(evals)

        print('unknown = {}, positive = {}, negative = {}, accuracy = {}'.format(unknown, positive, negative, accuracy))

        return unknown, accuracy

    @staticmethod
    def convert_dict_to_asarray(dict_list, component):
        data_list = []
        for index in range(len(dict_list)):
            # print(index)
            data_list.append(dict_list[index][component])
        data_list = np.asarray(data_list)
        return data_list

    def train_SVM(self, train_dataset, test_dataset):
        labels = []
        for index in range(len(test_dataset)):
            labels.append(test_dataset[index]['label'])
        # Convert dataset from dict to numpy array
        train_encodes = self.convert_dict_to_asarray(train_dataset, 'encode')
        train_labels = self.convert_dict_to_asarray(train_dataset, 'label')
        test_encodes = self.convert_dict_to_asarray(test_dataset, 'encode')
        test_labels = self.convert_dict_to_asarray(test_dataset, 'label')

        # Label encode targets
        encoder = LabelEncoder()
        encoder.fit(test_labels)
        test_labels = encoder.transform(test_labels)
        train_labels = encoder.transform(train_labels)

        # Fit into SVM model
        model = SVC(kernel='linear', probability=True)
        model.fit(train_encodes, train_labels)


        # Prediction
        predictions = model.predict_proba(test_encodes)
        best_class_indices = np.argmax(predictions, axis=1)
        print(best_class_indices)
        predicted_class = []
        wrong = 0
        for index in range(len(best_class_indices)):
            predicted_class.append(labels[best_class_indices[index]])
            if best_class_indices[index] != test_labels[index]:
                print(best_class_indices[index])
                print(test_labels[index])
                wrong = wrong + 1
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        return best_class_probabilities, predicted_class, wrong


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', metavar='PATH', dest='data_set')
    parser.add_argument('--model_path', metavar='PATH', dest='model_path')
    parser.add_argument('--using_SVM', type=int, dest='classifier', default=0)
    args = parser.parse_args()

    print(args.data_set)
    print(args.model_path)

    face_net_eval = FaceNetEval(data_set_path=args.data_set, model_path=args.model_path)

    # make faces_encodings, labels
    faces_encoding, labels = face_net_eval.make_faces_encoding_labels()
    #print(labels)
    print('faces_encoding shape = {}, labels shape = {}'.format(faces_encoding.shape, labels.shape))

    encodes_labels_list = []

    for i in range(len(labels)):
        encode_label_obj = dict()
        encode_label_obj['id'] = i
        encode_label_obj['encode'] = faces_encoding[i]
        encode_label_obj['label'] = labels[i]
        encodes_labels_list.append(encode_label_obj)

    # indices which belong to label
    # label2idx = face_net_eval.create_label2idx(labels)
    # print(label2idx)
    if (args.classifier == 0):
        # 画像特徴リストより、anchorsと評価用に分割する
        anchors, evals = face_net_eval.split_face_encodings(encodes_labels_list)
        print('len of anchors = {}, len of evals = {}'.format(len(anchors), len(evals)))

        # 評価する
        unknown, acc = face_net_eval.evaluate(anchors=anchors, evals=evals)
        print('unknown:', unknown)
        print('acc:', acc)
    else:
        '''
        SVM Classifier 
        '''
        # trainとtestのデータセット分割する
        test, train = face_net_eval.split_face_encodings(encodes_labels_list)
        predictions, predicted_class, wrong = face_net_eval.train_SVM(train, test)
        acc = (10 - wrong)/10
        print(acc)
        for index in range(len(predicted_class)):
            print('Prediction for test image {} is {} with confidence of {}'.format(index, predicted_class[index], predictions[index]))




