import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from PIL import Image
from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# instantiated naive bayes classifier according to Gaussian Distribution


def getData(filename):
    dataset = pd.read_csv(filename).as_matrix()
    print(filename, len(dataset))
    
    data = dataset[0:len(dataset),:-1]
    data_label = dataset[0:len(dataset),-1]
    
    print(len(data[0]))
    return data, data_label

# Gaussian NB
def GausNB(train,train_label,test,test_label):
    clf = GaussianNB()
    clf.fit(train,train_label)
    results = clf.predict(test)
    hit = 0.0
    for i in range(len(results)):
        if(results[i] == test_label[i]):
            hit = hit + 1.0
    accuracy = hit/float(len(test))

    return accuracy

# Bernoulli NB
def BernNB(train,train_label,test,test_label):
    clf = BernoulliNB()
    clf.fit(train,train_label)
    results = clf.predict(test)
    hit = 0.0
    for i in range(len(test)):
        if(results[i] == test_label[i]):
            hit = hit + 1.0
    accuracy = hit/float(len(test))

    return accuracy

# Random Forest classifier
def randomForrest(train,train_label,test,test_label,numTree,depth,numRows,numCols):
    
    clf = RandomForestClassifier(n_estimators= numTree, max_depth = depth) 
    clf.fit(train,train_label)
    value = test[0]
    value.shape= (numRows,numCols)
    results = clf.predict(test)
    hit = 0.0
    for i in range(len(test)):
        if(results[i] == test_label[i]):
            hit = hit + 1.0
    accuracy = hit/float(len(test))

    return accuracy

# convert unbound -> stretch and bounded 20 x 20
def boundBox(imageFile):

    stretchedBB = []
    
    for data in imageFile:
        
        # change shape to fit image formate
        image = np.asarray(data)
        image.shape = (28,28)
        
        # find image bounding box
        imageBound = Image.fromarray(image.astype('uint8'))
        imageBound = imageBound.getbbox()
        
        # find length of new bounding box dimensions
        width = imageBound[2] - imageBound[0]
        height = imageBound[3] - imageBound[1]
        
        newimage = []
        
        # loop through and extract relevent grayscale values
        for row in range(len(image)):
            for col in range(len(image)):
                if( col >= imageBound[0] and col < imageBound[2] and row >= imageBound[1] and row < imageBound[3]):
                    newimage.append(image[row][col])
        
        # convert image to numpy array
        formatedImage = np.asarray(newimage)
        formatedImage.shape = (height,width)
        
        # create image and resize to 20 x 20 stretched bound box

        stretchedImage = Image.fromarray(formatedImage.astype('uint8'))
        stretchedImage = stretchedImage.resize((20,20))
        
        stretchedImage = np.array(stretchedImage)
        stretchedImage.shape = (400,)    
        stretchedBB.append(stretchedImage)
    stretchedBB = np.array(stretchedBB)
    
    return stretchedBB
    
if __name__ == "__main__":
    
    training_set = 'mnist_train.csv'
    testing_set = 'mnist_test.csv'
  
    # unbounded data
    train, train_label = getData(training_set)
    test, test_label = getData(testing_set)
   
    # stretched and 20 x 20 bounded data 
    stretchedTrain = boundBox(train)
    stretchedTest = boundBox(test)

    print(len(train),len(stretchedTrain))
    print(len(test),len(stretchedTest))

    # classify unbounded data
    uGausAccuracy = GausNB(train,train_label,test,test_label)
    uBernAccuracy = BernNB(train,train_label,test,test_label)

    # classify stretched and bounded data
    bGausAccuracy = GausNB(stretchedTrain,train_label,stretchedTest,test_label)
    bBernAccuracy = BernNB(stretchedTrain,train_label,stretchedTest,test_label)

    # print accuracies
    print("uGaussianNB: ",uGausAccuracy)
    print("BGaussianNB: ",bGausAccuracy)
    print("uBernoulliNB: ",uBernAccuracy)
    print("bBerniulluNB: ",bBernAccuracy)
    
    numTree = [10,10,10,20,20,20,30,30,30]
    depth = [4,8,16,4,8,16,4,8,16]
    
    uNumRows = 28
    uNumCols = 28
    sNumRows = 20
    sNumCols = 20

    print("Untouched Raw Pixels")
    for i in range(len(depth)):
        accuracy = randomForrest(train, train_label, test, test_label, numTree[i], depth[i],uNumRows,uNumCols) 
        print("randomForest: ", accuracy," numTree: ", numTree[i]," depth: ", depth[i])

    print("Stretched Bounding Box")
    for i in range(len(depth)):
        accuracy = randomForrest(stretchedTrain,train_label,stretchedTest,test_label,numTree[i],depth[i],sNumRows,sNumCols) 
        print("randomForest: ",accuracy," numTree: ",numTree[i]," depth: ",depth[i])
