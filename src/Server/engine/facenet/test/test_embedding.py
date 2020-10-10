# coding: utf-8

# Add PYTHONPATH for problem with import module 
import sys
sys.path.append('/Users/admin/OmniFaceRecognition/src/server/')
import argparse
import numpy as np 
from scipy.spatial import distance
import matplotlib.pyplot as plt
from keras.models import load_model 
from facenet.extract_face import ExtractFace
filepath = list()

def convert(faces):
    # Load the FaceNet model 
    model = load_model('../model/keras-facenet/model/facenet_keras.h5', compile=False)
    print('> Loaded model')

    new = list()
    # Training dataset
    # Convert each face to an embedding
    for face in faces: 
        embed = embedding(model, face)
        new.append(embed)
    new = np.asarray(new)
    # Checking new dataset dimemsion
    return new

def embedding(model, faces):
    # Scale pixel values 
    faces = faces.astype('float32')
    # Standardize pixel value across channels (global)
    mean, std = faces.mean(), faces.std()
    faces = (faces - mean) / std 
    # Transfomr face into one sample 
    samples = np.expand_dims(faces, axis=0)
    # Make prediction to get embedding 
    Y_hat = model.predict(samples)
    return Y_hat[0]

def L2normalizer(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', metavar='PATH', dest='filepath' )
    args = parser.parse_args()

    # Extract faces using NN
    mtcnn = ExtractFace(args.filepath)

    # Load faces and annotation 
    faces, labels = mtcnn.load_dataset()

    # Embedding 
    embeds = convert(faces)

    # Normalize 
    embeds = L2normalizer(embeds)

    print(embeds)

    # indices which belong to label
    np.savez_compressed('japanese-actor.npz', embeds, labels)

