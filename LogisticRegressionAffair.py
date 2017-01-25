# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 16:20:35 2016

@author: Matthew Fischer, Yury Stanev
"""

from numpy import *
import operator
import numpy as np
#Create data set function;
from numpy import *
from numpy import array
import operator
import time
import matplotlib.pyplot as plt

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def regClassify(trainMat, testMat, trainingLabels, testLabels, weightIter, numTestVecs):
    trainStart = time.clock()
    trainWeights = stocGradAscent1(array(trainMat), trainingLabels, weightIter)
    trainEnd = time.clock()
    trainTime = trainEnd-trainStart
    
    errorCount =0
    sizeTrainMat = int(len(testMat))
    strengths = []
    FP = 0
    TN = 0
    TP = 0
    FN = 0
    testStart = time.clock()
    for i in range(sizeTrainMat):
        classifyAns = classifyVector(array(testMat[i]), trainWeights)
        strengths.append(sum(array(testMat[i])*trainWeights))
        print "The classifier came back with %r, the actual answer is %r" %(classifyAns, testLabels[i])
        if int(classifyAns)!= int(testLabels[i]):
            errorCount+=1
        if (classifyAns == 1 and testLabels[i]==1):
            TP +=1
            
        elif (classifyAns == 1 and testLabels[i] == 0):
            FP +=1
            
        elif (classifyAns == 0 and testLabels[i] == 1):
            FN +=1
            
        elif (classifyAns == 0 and testLabels[i] == 0):
            TN +=1
            
    errorRate = (float(errorCount)/numTestVecs)
    print "the error rate of this test is: %f" % errorRate
    testStop = time.clock()
    testTime = testStop-testStart
    return errorRate, strengths, TP, FP, FN, TN, trainTime, testTime
    
def makeVectors(trainMat, testMat, CvMat, width): 
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    cvSet =[]; cvLabels = []
   
    sizeDatMat = int(trainMat.shape[0])
    
    sizeTestMat = int(testMat.shape[0])
   
    sizeCvMat = int(CvMat.shape[0])
    i=0
    a=0
 
    for i in range(sizeDatMat):
        currLine = trainMat[i,: ]
        lineArr =[]
        for a in range(width):
            lineArr.append(float(currLine[a]))               
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[width-1]))
    
#    print trainingLabels
      
    i=0
    a=0
    numTestVecs = 0
    for i in range(sizeTestMat):
        numTestVecs += 1.0
        currLine = testMat[i,: ]
        lineArr = []
        for a in range(width):
            lineArr.append(float(currLine[a]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[width-1]))
    
    i=0
    a=0
    for i in range(sizeCvMat):
        currLine = testMat[i,: ]
        lineArr = []
        for a in range(width):
            lineArr.append(float(currLine[a]))
        cvSet.append(lineArr)
        cvLabels.append(float(currLine[width-1]))
#    print testSet
    return trainingSet, trainingLabels, testSet, testLabels, numTestVecs, cvSet, cvLabels   

#def iterCrossValidation( cvSet, cvLabels,foldNum, errthreshold): #Allowed to set maximum error in %
#    z = 0
#    cvSetLength = int(len(cvSet))/foldNum
#  
#    errorRate =0
#   
#    while True: 
#        z=500
#        errorRateTotal =0
#        for i in range(foldNum):
#           
#            cvTrainDat1 = cvSet[(cvSetLength*(i+1)):]
#            cvTrainLab1 = cvLabels[(cvSetLength*(i+1)):]
#            cvTrainDat2 = cvSet[:(cvSetLength*i):]
#            cvTrainLab2 = cvSet[:(cvSetLength*i):]
#                            
#            cvTestDat = cvSet[(i*cvSetLength):((i+1)*cvSetLength)]
#            cvTestLab = cvLabels[(i*cvSetLength):((i+1)*cvSetLength)]
#            cvTrainDat = cvTrainDat1 + cvTrainDat2
##            print cvTrainDat
#            cvTrainLab = cvTrainLab1 + cvTrainLab2
##            print cvTrainLab
#            errorRateTotal += regClassify(cvTrainDat, cvTestDat,cvTrainLab,cvTestLab,z,cvSetLength)
#            print "z = %r" %z
#        errorRate = (errorRateTotal/foldNum)*100
#        if (errthreshold>=errorRate):
#            break
              
#    weightIter = z  
#    print "Achieved an errorRate of %r using %r iterations" %(errorRate, weightIter)
#    return weightIter

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    strengths = mat(zeros((m,1)))
    for j in range(numIter):      
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            strengths[i] = (sum(dataMatrix[randIndex]*weights))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights
    
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
    
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
    i=0
    for a in classLabelVector:
        if classLabelVector[i] == 0:
            break
        i+=1
    testMatTrue = returnMat[:i,:]
    testMatFalse = returnMat[i:,:]
    np.take(testMatTrue, np.random.permutation(testMatTrue.shape[0]), axis=0, out=testMatTrue)
    np.take(testMatFalse, np.random.permutation(testMatFalse.shape[0]), axis=0, out=testMatFalse)
    CvTestMat =  np.r_[testMatTrue[:p,:], testMatFalse[:p,:]] #crosstest matrix
    np.take(CvTestMat, np.random.permutation(CvTestMat.shape[0]), axis=0, out=CvTestMat)
    CvTestMat = np.delete(CvTestMat, [0], axis=1)
  
    testMat = np.delete(testMat, [0], axis=1) #Delete labels
    dataMat = np.delete(dataMat, [0], axis=1)       
    
    testLabelVector = testMat[:,8] 
    classLabelVector = dataMat[:,8]
    m, width = testMat.shape
    #Create a test matrix with 50% random true and 50% random false for 25% the size of training data
    #randomize the data by column        
    return dataMat, classLabelVector, testMat, testLabelVector, CvTestMat, width    
    
def cleanData(filename, filename2): ##Remove periods at the end of objects [ we are using space as a delimeter]
    infile = open(filename, 'r')
    outfile = open(filename2, 'w')
    
    data = infile.read()
    listdat = list(data)
    
   # print listdat
    i = 0
    for a in listdat:
        if listdat[i] == "." and listdat[i+1] == " ":
            listdat[i] = " "
        i+=1
    i = 0
    for a in listdat:
         outfile.write(listdat[i])   
         i+=1
   # print listdat    
def plotROC(predStrengths, classLabels, title):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1.0/float(numPosClas); xStep = 1.0/float(len(classLabels)-numPosClas)
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
    
    plt.title(title)
    ax.axis([0,1,0,1])
    plt.show()
    fig.savefig("OutputData\ROC\Regression\RegGradAsc1Iter.png")
    print "the Area Under the Curve is: ",ySum*xStep   
#Start Code Here:
dataMat, classLabelVector, testMat, testLabelVector, CvMat, width = file2matrixAffair("AffairData\CleanRbTapeData.txt", 14)
trainSet, trainLabels, testSet, testLabels, numTestVecs, cvSet, cvLabels = makeVectors(dataMat, testMat, CvMat, width)
print len(trainSet)
print len(testSet)

#weightIter = iterCrossValidation( cvSet, cvLabels,5, 5.0)
errRate, strengths, TP, FP, FN, TN, trainTime, testTime = regClassify(trainSet, testSet, trainLabels, testLabels, 1, numTestVecs)
strengths = np.asmatrix(strengths)
plotROC(strengths, testLabels, 'ROC curve for Regression Affair Prediction System iter=1')


#plotROC(strengths.T, testLabels)