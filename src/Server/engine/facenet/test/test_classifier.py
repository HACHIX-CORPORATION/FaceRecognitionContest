# coding: utf-8
import numpy as np
from scipy.spatial import distance
from sklearn.svm import SVC 
from random import choice 
from sklearn.preprocessing import LabelEncoder


class Baseline:
    def __init__(self, threshold, anchor , evaluateation, label2idx):
        self.threshold = threshold
        self.anchor = anchor 
        self.evaluateation = evaluateation
        self.label2idx = label2idx

    def evaluate(self):
        bound = 1.5
        unknown = 0 
        positive = 0 
        for k in range(len(evaluateation)):
            distances = []
            max_distances = []
            for i in range(len(anchor)):
                distances.append(distance.euclidean(evaluateation[k], anchor[i]))
            if (np.max(distances) > bound):
                unknown +=1
            if (np.min(distances) < threshold):
                positive +=1
                # print(np.min(distances))
        
        acc = positive/40
        return unknown, acc


class SVM: 
    def __init__(self, embeds, labels):
        self.embeds = embeds 
        self.labels = labels

    def split(self):
        # Indicalize
        label2idx = []
        n = 5
        for i in range(int(len(self.labels)/n)):
            label2idx.append([i*n, i*n+1, i*n+2, i*n+3, i*n+4])
        # Split data
        testX = []
        testY = []
        for i in range(len(label2idx)):
            idx = label2idx[i][0]
            testX.append(self.embeds[idx])
            testY.append(self.labels[idx])

        trainX = []
        trainY = []
        for i in range(len(label2idx)):
            for j in range(1, 5):
                idx = label2idx[i][j]
                trainX.append(self.embeds[idx])
                trainY.append(self.labels[idx])
        return testX, testY, trainX, trainY

    def predict(self):
        # Split
        testX, testY, trainX, trainY = self.split()
        # Convert to numpy array
        testX = np.asarray(testX)
        trainX = np.asarray(trainX)
        trainY = np.asarray(trainY)
        testY = np.asarray(testY)
        # Label encode targets
        encoder = LabelEncoder()
        encoder.fit(trainY)
        trainY = encoder.transform(trainY)
        testY = encoder.transform(testY)
        # Fit model (SVM for classifier)
        model = SVC(kernel='linear', probability=True)
        model.fit(trainX, trainY)
        # Test model on a random exa ple from the test dataset
        selection = choice([i for i in range(testX.shape[0])])
        random_embedding = testX[selection]
        random_class = testY[selection]
        random_name = encoder.inverse_transform([random_class]) 
        # Prediction for the face 
        samples = np.expand_dims(random_embedding, axis=0)
        predict_class = model.predict(samples)
        predict_prob = model.predict_proba(samples)
        # Get name of the class 
        class_index = predict_class[0]
        class_probability = predict_prob[0, class_index] * 100
        predict_names = encoder.inverse_transform(predict_class)
        print('Predicted: %s (%.3f)' %(predict_names[0], class_probability))
        print('Expected: %s' %random_name[0])


if __name__ == "__main__":
    # Load compress data (japanese data)t
    data = np.load('./compressed/japanese-actor.npz')
    embeds, labels = data['arr_0'], data['arr_1']

    print('shape of embeds = ({}, {})'.format(len(embeds), len(embeds[0])))
    
    # indices which belong to label
    label2idx = []
    n = 5
    for i in range(int(len(labels)/n)):
        label2idx.append([i*n, i*n+1, i*n+2, i*n+3, i*n+4])

    print(label2idx)
     
    # Split data for evaluateation
    anchor = []
    for i in range(len(label2idx)):
        idx = label2idx[i][0]
        anchor.append(embeds[idx])

    evaluateation = []
    for i in range(len(label2idx)):
        for j in range(1, 5):
            idx = label2idx[i][j]
            evaluateation.append(embeds[idx])

    # evaluateation 
    threshold = 0.95 
    baseline = Baseline(threshold, anchor, evaluateation, label2idx)
    unknown, acc = baseline.evaluate()
    print('unknown:', unknown)
    print('acc:', acc)
    
    '''
    # SVM classifier
    classifier = SVM(embeds, labels)
    classifier.predict()
    '''
