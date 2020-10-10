# coding: utf-8
import pandas


class ImageLoader(object):
    def __init__(self):
        self.image_folder_path = "C:/Users/tranthanh/Desktop/vn_celeb_face_recognition/train"
        self.label_file_path = "C:/Users/tranthanh/Desktop/vn_celeb_face_recognition/train.csv"

    def csv_reader(self):
        df = pandas.read_csv(self.label_file_path)
        for value in df['label'].values:
            print(value)


if __name__ == "__main__":
    image_loader = ImageLoader()
    image_loader.csv_reader()


