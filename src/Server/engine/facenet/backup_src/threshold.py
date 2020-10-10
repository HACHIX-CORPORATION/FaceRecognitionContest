#coding: utf-8

# Add PYTHONPATH for problem with import module 
import sys
sys.path.append('/Users/admin/OmniFaceRecognition/src/server/engine')
import argparse
import numpy as np 
from scipy.spatial import distance
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
from facenet.extract_face import ExtractFace
filepath = list()

def convert(faces):
    # Load the FaceNet model
    model = load_model('../model/keras-facenet/model/facenet_keras.h5', compile=False)
    print(model.summary())
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

def compare(label2idx, normalized_embed):
    match_distance = []
    for i in range(10):
        idx = label2idx[i]
        distances = []
        for j in range(len(idx)-1):
            for k in range(j+1, len(idx)):
                distances.append(distance.euclidean(normalized_embed[j], normalized_embed[k]))
            match_distance.extend(distances)

    unmatch_distance = []
    for i in range(10):
        ids = label2idx[i]
        distances = [] 
        for j in range(5):
            idx = np.random.randint(normalized_embed.shape[0])
            while idx in label2idx[i]:
                idx = np.random.randint(normalized_embed.shape[0])
            distances.append(distance.euclidean(normalized_embed[ids[np.random.randint(len(ids))]], normalized_embed[idx]))
        unmatch_distance.extend(distances)
    return match_distance, unmatch_distance    

class Baseline:
    def __init__(self, threshold, anchor , validation, label2idx):
        self.threshold = threshold
        self.anchor = anchor 
        self.validation = validation
        self.label2idx = label2idx
    def valid(self):
        unknown = 0 
        positive = 0 
        for i in range(len(anchor)):
            distances = []
            max_distances = []
            for j in range(1, 10):
                distances.append(np.min([distance.euclidean(anchor[i], validation[k]) for k in label2idx[j]]))
                max_distances.append(np.max([distance.euclidean(anchor[i], validation[k]) for k in label2idx[j]]))
            if all(x > bound for x in max_distances):
                unknown += 1 
            elif any(x < threshold for x in distances):
                postive += 1 
            else:
                negative += 1
        acc = positive/40
        return unknown, acc
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

    # indices which belong to label 
    label2idx = []
    for i in range(len(labels)):
        label2idx.append([i, i+1, i+2, i+3, i+4])
    
    # Compare for matching distance
    match_distance, unmatch_distance = compare(label2idx, embeds)
    _,_,_ = plt.hist(match_distance,bins=50)
    _,_,_=plt.hist(unmatch_distance,bins=50,fc=(1, 0, 0, 0.5))
    plt.show()
    ''' 
    # Split data for validation
    idx = label2idx[0]
    anchor = []
    for ids, j in enumerate(idx): 
        anchor.append(embeds[j]) 
   
    validation = []
    for j in range(1, len(label2idx)):
        idx = label2idx[j]
        for ids, k in enumerate(idx):
            validation.append(embeds[k])

    # Validation 
    threshold = 0.9 
    baseline = Baseline(0.9, anchor, validation, label2idx)
    unknown, acc = baseline.valid()
    print('unknow:', unknow)
    print('acc:', acc)
    '''
