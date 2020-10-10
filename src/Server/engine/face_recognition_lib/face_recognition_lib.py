import face_recognition
from glob import glob
import os.path as osp


class FaceRecognitionLib(object):
    """
    face_recognition library を利用した顔認証検証
    """
    # クラス変数設定
    __data_set_dir = './../../dataset/japanese'      # データ・セットディレクトリ
    __known_image_idx = (1,)                        # 既存画像のインデックス
    __unknown_image_idx = (2, 3, 4, 5)              # 検証画像のインデックス
    __tolerance = 0.4                               # Recognitionの距離threshold

    def __init__(self):
        # get sub directory
        sub_dirs = glob(FaceRecognitionLib.__data_set_dir + '/*/')

        # get list of name
        self.__people = [sub_dir.split('/')[-2] for sub_dir in sub_dirs]

        # 既存画像と検証画像のファイルリストを生成する。
        known_images_path = []
        unknown_images_path = []
        for img_idx in self.__known_image_idx:
            known_images_path.extend(
                [osp.join(sub_dir, sub_dir.split('/')[-2] + str(img_idx) + '.jpg') for sub_dir in sub_dirs])

        for img_idx in self.__unknown_image_idx:
            unknown_images_path.extend(
                [osp.join(sub_dir, sub_dir.split('/')[-2] + str(img_idx) + '.jpg') for sub_dir in sub_dirs])

        self.__unknown_images_paths = unknown_images_path

        # set face encodings for known faces
        self.__known_face_encodings = self.__make_face_encodings(images_path=known_images_path)
        print('shape of known_face_encodings = ({}, {})'.format(len(self.__known_face_encodings),
                                                                len(self.__known_face_encodings[0])))

    @staticmethod
    def __make_face_encodings(images_path):
        """
        face encode情報を生成する。
        """
        face_encodings = []

        for img_path in images_path:
            img = face_recognition.load_image_file(img_path)
            face_encodings.append(face_recognition.face_encodings(img)[0])

        return face_encodings

    def recognition(self):
        """
        Recognition
        """
        unknown_face_encodings = self.__make_face_encodings(images_path=self.__unknown_images_paths)
        print('shape of unknown_face_encodings = ({}, {})'.format(len(unknown_face_encodings),
                                                                  len(unknown_face_encodings[0])))

        accuracy = 0
        wrong = 0

        for face_to_compare in self.__known_face_encodings:
            print(face_recognition.face_distance(unknown_face_encodings, face_to_compare))

        for i, unknown_face_encoding in enumerate(unknown_face_encodings):
            img_file = osp.basename(self.__unknown_images_paths[i])
            results = face_recognition.compare_faces(self.__known_face_encodings,
                                                     unknown_face_encoding,
                                                     tolerance=FaceRecognitionLib.__tolerance)

            name = "Unknown"

            for person in range(len(self.__people)):
                if results[person]:
                    name = self.__people[person]
                    break

            if name in img_file:
                accuracy += 1
            else:
                wrong += 1

            print("Found {} in the photo {}".format(name, img_file))

        print('accuracy = {}, wrong = {}'.format(accuracy, wrong))


if __name__ == "__main__":
    face_recognition_lib = FaceRecognitionLib()
    face_recognition_lib.recognition()



