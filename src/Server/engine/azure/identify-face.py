# coding: utf-8
import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
import argparse
import time
from matplotlib import pyplot as plt

KEY = os.environ['FACE_SUBSCRIPTION_KEY']
ENDPOINT = os.environ['FACE_ENDPOINT']

def addImage(name, face_client, dataset_path):
    path = dataset_path + name + '/'
    face_ids_list = []
    images = []
    for filename in os.listdir(path):
        if (filename == ".DS_Store"):
            continue
        filepath = path + filename
        print(filepath)
        if filepath != glob.glob(path + '*5.jpg')[0]:
            image = open(filepath, 'r+b')
            print('> Load image: ', filename)
            print('of', name)
            faces = face_client.face.detect_with_stream(image)
            for face in faces:
                face_ids_list.append(face.face_id)
                print(face.face_id)
            images.append(image)
        time.sleep(15)
    return face_ids_list, images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", metavar="PATH", dest="image_path")
    args = parser.parse_args()

    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
    PERSON_GROUP_ID = 'japanese'
    unknown = 0
    acc = 0
    for subdir in os.listdir(args.image_path):
        if not os.path.isdir(args.image_path) or (subdir == ".DS_Store"):
            continue
        # Detect faces
        path = args.image_path + subdir + '/'
        for filename in os.listdir(path):
            if (filename == ".DS_Store"):
                continue
            filepath = path + filename
            if filepath != glob.glob(path + '*5.jpg')[0]:
                image = open(filepath, 'r+b')
                print('> Load image: ', filename)
                print('of', subdir)
                faces = face_client.face.detect_with_stream(image)
                face_ids = []
                for face in faces:
                    face_ids.append(face.face_id)
                # Identify faces
                results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
                print('Identifying faces in {}'.format(os.path.basename(image.name)))

                if not results:
                    print('No person identified in the person group for faces from {}.'.format(os.path.basename(image.name)))
                for person in results:
                    face = face_client
                    if not person.candidates:
                        unknown = unknown + 1
                        continue
                    predicted_person = face_client.person_group_person.get(PERSON_GROUP_ID, person.candidates[0].person_id)
                    print("Predicted result: ", predicted_person.name)
                    print("Confidence: ", person.candidates[0].confidence)
                    if (predicted_person.name == subdir):
                        acc = acc + 1
                    time.sleep(20)
    print("accuracy: ", acc)
    print("unknow:", unknown)
    """
    # Visualizing
    img = Image.open(args.image_path)
    plt.imshow(img)
    title = '%s(%.3f)' %(predicted_person.name, person.candidates[0].confidence)
    plt.title(title)
    plt.show()
    """
