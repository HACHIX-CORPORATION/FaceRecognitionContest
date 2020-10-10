# coding: utf-8
import sys
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
import argparse
import time
import os.path as osp
from secret import CONFIG      # azure face apiの情報


class AzureFaceEval(object):
    """
    Azure Face Apiを利用して、学習する
    azure face api url
    https://azure.microsoft.com/ja-jp/services/cognitive-services/face/
    """
    # クラス変数
    __DEBUG_MODE = True

    # Set the FACE_SUBSCRIPTION_KEY environment variable with your key as the value.
    # This key will serve all examples in this document.
    __KEY = CONFIG['FACE_SUBSCRIPTION_KEY']

    # Set the FACE_ENDPOINT environment variable with the endpoint from your Face service in Azure.
    # This endpoint will be used in all examples in this quick start.
    __ENDPOINT = CONFIG['FACE_ENDPOINT']

    print('key = {}, endpoint = {}'.format(__KEY, __ENDPOINT))

    # Used in the Person Group Operations,  Snapshot Operations, and Delete Person Group examples.
    # You can call list_person_groups to print a list of preexisting PersonGroups.
    # SOURCE_PERSON_GROUP_ID should be all lowercase and alphanumeric. For example, 'mygroupname' (dashes are OK).
    __PERSON_GROUP_ID = 'omni_test'

    __TRAIN_IMG_IDX = ('5', )       # 学習で利用する画像のindex
    __IDENTIFY_IMG_IDX = ('1', '2', '3', '4')  # 学習で利用する画像のindex
    __IMG_EXT = '.jpg'              # 画像の拡張子

    def __init__(self, data_set_path):
        """
        コンストラクト
        """
        # 引数チェック
        if osp.exists(data_set_path) is False:
            raise ValueError('{} not Exist'.format(data_set_path))

        self.__data_set_path = data_set_path

        # 学習で利用する人の名前リスト
        self.__person_name_list = ('aoi-yu', 'erika-sawajiri', 'hamabe-minami', 'kentaro-sakaguchi',
                                   'kento-yamazaki', 'kitagawa-keiko', 'kyoko-fukada', 'nagasawa-masami',
                                   'oguri-shun', 'tadanobu-asano')

        # face clientを生成する。
        self.__face_client = FaceClient(AzureFaceEval.__ENDPOINT, CognitiveServicesCredentials(AzureFaceEval.__KEY))

        # Person Groupを初期化する
        try:
            self.__face_client.person_group.delete(AzureFaceEval.__PERSON_GROUP_ID)
        except:
            if AzureFaceEval.__DEBUG_MODE is True:
                print("deleted person group id = {}".format(AzureFaceEval.__PERSON_GROUP_ID))

        self.__face_client.person_group.create(AzureFaceEval.__PERSON_GROUP_ID, AzureFaceEval.__PERSON_GROUP_ID)

    def train(self):
        """
        学習する
        """
        # 学習するためのpersonを登録する。
        for person_name in self.__person_name_list:
            # personを登録する
            person = self.__face_client.person_group_person.create(AzureFaceEval.__PERSON_GROUP_ID, person_name)

            if AzureFaceEval.__DEBUG_MODE is True:
                print('successfully register person_id = {}'.format(person.person_id))

            for train_idx in AzureFaceEval.__TRAIN_IMG_IDX:
                img_file_path = osp.join(self.__data_set_path,
                                         person_name + '/' + person_name + train_idx + AzureFaceEval.__IMG_EXT)

                if osp.exists(img_file_path) is True:
                    image = open(img_file_path, 'r+b')
                    self.__face_client.person_group_person.add_face_from_stream(AzureFaceEval.__PERSON_GROUP_ID,
                                                                                person.person_id, image)
                else:
                    print('Warning: image file path is not exist {}'.format(img_file_path))

        # 学習を実行する
        self.__face_client.person_group.train(AzureFaceEval.__PERSON_GROUP_ID)

        while True:
            training_status = self.__face_client.person_group.get_training_status(AzureFaceEval.__PERSON_GROUP_ID)
            print("Training status: {}.".format(training_status.status))
            if training_status.status is TrainingStatusType.succeeded:
                break
            elif training_status.status is TrainingStatusType.failed:
                sys.exit('Training the person group has failed.')
            time.sleep(5)

    def identify(self):
        """
        顔認識
        """
        unknown = 0
        acc = 0
        total = 0
        for person_name in self.__person_name_list:
            for identify_idx in AzureFaceEval.__IDENTIFY_IMG_IDX:
                img_file_path = osp.join(self.__data_set_path,
                                         person_name + '/' + person_name + identify_idx + AzureFaceEval.__IMG_EXT)

                total += 1
                if osp.exists(img_file_path) is True:
                    image = open(img_file_path, 'r+b')
                    faces = self.__face_client.face.detect_with_stream(image)

                    # 一つの画像には複数のfaceがある場合の対応
                    face_ids = []
                    for face in faces:
                        face_ids.append(face.face_id)
                    # Identify faces
                    results = self.__face_client.face.identify(face_ids, AzureFaceEval.__PERSON_GROUP_ID)

                    if not results:
                        print('No person identified in the person group for faces from {}.'.format(
                            osp.basename(image.name)))
                    for person in results:
                        if not person.candidates:
                            unknown = unknown + 1
                            continue
                        print('face ID {} is identified in {} with a confidence of {}.'
                              .format(person.face_id,
                                      osp.basename(image.name),
                                      person.candidates[0].confidence))  # Get topmost confidence score

                        predicted_person = self.__face_client.person_group_person.get(AzureFaceEval.__PERSON_GROUP_ID,
                                                                                      person.candidates[0].person_id)
                        print('predicted person name = {}'.format(predicted_person.name))

                        if predicted_person.name == person_name:
                            acc = acc + 1
                        time.sleep(20)

                else:
                    print('Warning: image file path is not exist {}'.format(img_file_path))

        return total, acc, unknown


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', metavar='PATH', dest='data_set')
    args = parser.parse_args()

    eval = AzureFaceEval(data_set_path=args.data_set)

    # 学習する
    eval.train()

    # 評価する
    total, acc, unknown = eval.identify()
    print('total = {}, acc = {}, unknow = {}'.format(total, acc, unknown))

