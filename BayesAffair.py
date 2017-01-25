# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 21:09:25 2016

@author: Matthew
"""

from numpy import *
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
BayesTrainResults = namedtuple('BayesTrainingResults', 'probCatC0 probCatC1 statsC0 statsC1 PC0 PC1')
import time

def file2matrixAffair(filename, width, limit=None):
    fr = open(filename)
    if limit==None:
        numberOfLines = len(fr.readlines())
    else:
        numberOfLines = limit
    returnMat = zeros((numberOfLines,width))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split()
        returnMat[index, :] = listFromLine[0:width]
       # print listFromLine[-3]
        if (double(listFromLine[-3])>0):
            classLabelVector.append(1)
        else:
            classLabelVector.append(0)
        index += 1
        if index==limit:
            break    
    returnMat= np.delete(returnMat, [1,8,11,12,13],axis=1) #Delete unused columns, but hold on to ID in order to subtract test enteries from main data
    returnMat = np.c_[returnMat, classLabelVector] #Append new vector adjusted to binary values
    #Create a test data matrix...
    i=0
    #Split matrix into two matrices one for true and one for false 
    for a in classLabelVector:
        if classLabelVector[i] == 0:
            break
        i+=1
    testMatTrue = returnMat[:i,:]
    testMatFalse = returnMat[i:,:]
    np.take(testMatTrue, np.random.permutation(testMatTrue.shape[0]), axis=0, out=testMatTrue)
    np.take(testMatFalse, np.random.permutation(testMatFalse.shape[0]), axis=0, out=testMatFalse)
    sizetestMat = int(len(classLabelVector))*0.25
    p = int(sizetestMat/2)
    testMat = np.r_[testMatTrue[:p,:], testMatFalse[:p,:]]
    np.take(testMat, np.random.permutation(testMat.shape[0]), axis=0, out=testMat)
    lengthT = int(testMat.shape[0])
    lengthDat = int(returnMat.shape[0])
    dataMat = returnMat
    i=0
    q=0
    #Delete Entries Reserved for testing
    for i in range(lengthT):
        for q in range(lengthDat):
            if testMat[i,0] == returnMat[q,0]:
                dataMat = np.delete(dataMat, [q], axis=0)
   
  #Create testMat for CrossValidation
  
    testMat = np.delete(testMat, [0], axis=1) #Delete labels
    dataMat = np.delete(dataMat, [0], axis=1)       
    
    testLabelVector = testMat[:,8] 
    classLabelVector = dataMat[:,8]
    
    #Create a test matrix with 50% random true and 50% random false for 25% the size of training data
    #randomize the data by column        
    return dataMat, classLabelVector, testMat, testLabelVector


def trainNB0(trainDataCategorical, numCategories, trainDataNumeric, trainClass):   
    trainClass = np.array(trainClass)
    
    # class 0 categorical data
    class0trainCat = trainDataCategorical[nonzero(trainClass == 0)]
    # class 1 categorical data
    class1trainCat = trainDataCategorical[nonzero(trainClass == 1)]
    
    
    # class 0 numeric data
    class0trainNum = trainDataNumeric[nonzero(trainClass == 0)]
    # class 1 numeric data
    class1trainNum = trainDataNumeric[nonzero(trainClass == 1)]
   
    probCategoryClass0 = zeros((class0trainCat.shape[1], max(numCategories)), dtype = int) 
    probCategoryClass1 = zeros((class1trainCat.shape[1], max(numCategories)), dtype = int)

    # count up the number of examples of each category for class 0
    for i in range(class0trainCat.shape[0]):
        for j in range(class0trainCat.shape[1]):
            for k in range(numCategories[j]):
                if class0trainCat[i,j] == k:
                    probCategoryClass0[j,k] += 1;

    # calculate categorical class 0 probabilities
    probCategoryClass0 =  divide(probCategoryClass0, float(class0trainCat.shape[0]))

    # count up the number of examples of each category for class 1
    for i in range(class1trainCat.shape[0]):
        for j in range(class1trainCat.shape[1]):
            for k in range(numCategories[j]):
                if class1trainCat[i,j] == k:
                    probCategoryClass1[j,k] += 1;

    # calculate categorical class 1 probabilities                    
    probCategoryClass1 =  divide(probCategoryClass1, float(class1trainCat.shape[0]))

    # calculate mean and standard deviation for both classes for numeric data 
    meanClass0 = mean(class0trainNum, axis = 0)
    meanClass1 = mean(class1trainNum, axis = 0)
    stdClass0 = std(class0trainNum, axis = 0, ddof = 1)
    stdClass1 = std(class1trainNum, axis = 0, ddof = 1)

    statsC0 = vstack((meanClass0, stdClass0))
    statsC1 = vstack((meanClass1, stdClass1))

    # calculate the probability of class 0 and class 1
    numDataPoints = trainClass.shape[0] ##Change to length op since it is a vector and not a matrix.
    numC0DataPoints = sum(trainClass == 0)
    numC1DataPoints = sum(trainClass == 1)

    PC0 = numC0DataPoints / float(numDataPoints)
    PC1 = numC1DataPoints / float(numDataPoints)

    result = BayesTrainResults(probCategoryClass0, probCategoryClass1, statsC0, statsC1, PC0, PC1)
    
    return result


def classifyNB(inXCat, inXNum, trainResult):

    # if the categorical input is a scalar it is not an array so shape will not work
    # therefore turn the input into an array
    if inXCat.shape == ():
        inXCat = array([inXCat])

    # if the numeric input is a scalar it is not an array so shape will not work
    # therefore turn the input into an array
    if inXNum.shape == ():
        inXNum = array([inXNum])


    PAttCatC0 = empty(inXCat.shape[0])
    PAttCatC1 = empty(inXCat.shape[0])
    
    
    # look up the probability of categorical attributes
    for i in range(inXCat.shape[0]):
        PAttCatC0[i] = trainResult.probCatC0[i, inXCat[i]]
        PAttCatC1[i] = trainResult.probCatC1[i, inXCat[i]]

    C0LogCat = sum(log(PAttCatC0))
    C1LogCat = sum(log(PAttCatC1))

   

    PAttNumC0 = empty(inXNum.shape[0])
    PAttNumC1 = empty(inXNum.shape[0])

    # calculate the probability of numeric attributes
    for i in range(inXNum.shape[0]):
        PAttNumC0[i] = (1.0/(sqrt(2.0*pi)*trainResult.statsC0[1,i]))*exp( -((inXNum[i] - trainResult.statsC0[0,i])**2)/(2.0*(trainResult.statsC0[1,i])**2.0) ) 
        PAttNumC1[i] = (1.0/(sqrt(2.0*pi)*trainResult.statsC1[1,i]))*exp( -((inXNum[i] - trainResult.statsC1[0,i])**2.0)/(2.0*(trainResult.statsC1[1,i])**2.0) ) 

    C0LogNum = sum(log(PAttNumC0))
    C1LogNum = sum(log(PAttNumC1))

    # calculate the overall probability of each class
    resultC0 = C0LogCat + C0LogNum + log(trainResult.PC0)
    resultC1 = C1LogCat + C1LogNum + log(trainResult.PC1)
    #print "%r > %r" %(resultC0, resultC1)
    # return the resulting class or -1 if there is a tie
    
    #Collect rankings for true positive for ROC
    if resultC0 == resultC1:
        return -1, resultC1
    elif resultC0 > resultC1:
        return 0, resultC1
    else:
        return 1, resultC1

def splitMatrix(matrix):
    #Separate Data Types into seperate matrices for distance calcs
    numericMat = np.c_[matrix[:,1:4], matrix[:, 6:8]] #v2, v3, v4, v7, v8
    
    
    V1 = matrix[:, 0]
    V5 = matrix[:, 4]
    V6 = matrix[:, 5]
    Yrb = matrix[:, 8]
    
    columnSize = int(V1.shape[0])
    for i in range(columnSize): #adjust values by starting from zero and ranking up by one to work with algorithm
        V1[i] = V1[i]-1 #decrease V1 entries by one to range values from 0-4
        V5[i] = V5[i]-1
        if V6[i] == 9.0:
            V6[i] = 0
        elif V6[i] == 12.0:
            V6[i] = 1
        elif V6[i] == 14.0:
            V6[i] = 2
        elif V6[i] == 16.0:
            V6[i] = 3
        elif V6[i] == 17.0:
            V6[i] = 4
        elif V6[i] == 20.0:
            V6[i] = 5
    nominalMat = np.c_[V1, V5, V6, Yrb] #v1, v5, v6, and adjusted Yrb
            
#    print V1
#    print V5
#    print V6
#    print Yrb
#    print nominalMat
   
    
    
    #Normalize Numeric&Ordinal Data
    minVals = numericMat.min(0)
    maxVals = numericMat.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(numericMat))
    m = numericMat.shape[0]
    normDataSet = numericMat - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return nominalMat, normDataSet
    
def classTest(normDataSet, nominalMat, Labels, trainResult): 
     
     m = normDataSet.shape[0]
     errorCount = 0.0
     trueProb = []
     for i in range (m):
         classifierResult, C1=  classifyNB(nominalMat[i], normDataSet[i], trainResult)
         if classifierResult == 1:
             trueProb.append(C1)
         print "The classifier came back with: %d, the real answer is: %d" % (classifierResult, Labels[i])
         if (classifierResult != Labels[i]): errorCount += 1.0
     print "The total error rate in percent is: %f" % (errorCount/float(m) * 100) 
     print "Total Accuracy rate in percent is: %f" % (100 - (errorCount/float(m) * 100))
     return trueProb

def plotROC(predStrengths, classLabels):
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for Naive Bayesian Affair Prediction')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep     
#Start Code
trainMat,trainLabels, testMat, testLabels = file2matrixAffair("AffairData\CleanRbTapeData.txt", 14) 
nomTestMat, normTestMat = splitMatrix(testMat) # nominal test matrix and normalized numeric test matrix
nomTrainMat, normTrainMat = splitMatrix(trainMat) #split for training data as well.
numCategories = [5, 4, 6, 2]

trainStart = time.clock()
trainResult = trainNB0(nomTrainMat, numCategories, normTrainMat, trainLabels)
trainStop = time.clock()
trainTime = trainStop-trainStart

testStart = time.clock()
trueProb = classTest(normTestMat, nomTestMat, testLabels, trainResult)
testStop = time.clock()
testTime = testStop - testStart

trueProb = np.asmatrix(trueProb)