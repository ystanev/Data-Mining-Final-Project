# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 01:35:12 2016

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
            classLabelVector.append(-1)
        index += 1
        if index==limit:
            break    
    returnMat= np.delete(returnMat, [1,8,11,12,13],axis=1) #Delete unused columns, but hold on to ID in order to subtract test enteries from main data
    returnMat = np.c_[returnMat, classLabelVector] #Append new vector adjusted to binary values
    #Create a test data matrix...
    i=0
    #Split matrix into two matrices one for true and one for false 
    for a in classLabelVector:
        if classLabelVector[i] == -1:
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
  
  
    testMat = np.delete(testMat, [0], axis=1) #Delete labels
    dataMat = np.delete(dataMat, [0], axis=1)       
    
    testLabelVector = testMat[:,8] 
    classLabelVector = dataMat[:,8]
    m, width = testMat.shape
    #Create a test matrix with 50% random true and 50% random false for 25% the size of training data
    #randomize the data by column        
   
    return dataMat, classLabelVector, testMat, testLabelVector, width    
    
def makeVectors(trainMat, testMat, width): 
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    
    sizeDatMat = int(trainMat.shape[0])
    sizeTestMat = int(testMat.shape[0])
    print sizeTestMat
   
    i=0
    a=0
 
    for i in range(sizeDatMat):
        currLine = trainMat[i,: ]
        lineArr =[]
        for a in range(width):
            lineArr.append(float(currLine[a]))               
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[width-1]))
#    print trainingSet
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
    
#    print testSet
    return trainingSet, trainingLabels, testSet, testLabels, numTestVecs   
    

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst   
    
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    print len(classifierArr)
    
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
       # print aggClassEst
    signArr = sign(aggClassEst)
    error, TP, FP, FN, TN = errRate(signArr, testLabels)
    return error , aggClassEst, TP, FP, FN, TN


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
def errRate(sign, testLabels):
    errRate = 0
    i=0
    TP=0
    FP=0
    FN=0
    TN=0
    size = int(len(sign))
    for i in range(size):
        print "The classifier came back with %r, the real answer is %r" %(sign[i], testLabels[i])
        if sign[i] != testLabels[i]:
            errRate+=1
        if (sign[i] == 1 and testLabels[i]==1):
            TP +=1
            
        elif (sign[i] == 1 and testLabels[i] == -1):
            FP +=1
            
        elif (sign[i] == -1 and testLabels[i] == 1):
            FN +=1
            
        elif (sign[i] == -1 and testLabels[i] == -1):
            TN +=1
    print "The total error rate in percent is: %r" %((errRate/float(len(sign)))*100)

    return errRate, TP, FP, FN, TN   
    
def plotROC(predStrengths, classLabels, title):
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
    plt.title(title)
    ax.axis([0,1,0,1])
    plt.show()
    fig.savefig("OutputData\ROC\Adaboost\AdaStump40.png")
    print "the Area Under the Curve is: ",ySum*xStep
    
#Start Code:
dataMat, classLabelVector, testMat, testLabelVector, width = file2matrixAffair("AffairData\CleanRbTapeData.txt", 14)
trainSet, trainLabels, testSet, testLabels, numTestVecs = makeVectors(dataMat, testMat, width)

trainStart = time.clock()
weakClassifierArr, aggClassEstTrain = adaBoostTrainDS(trainSet, trainLabels,40) #Develop decision stumps
trainStop = time.clock()
trainTime = trainStop - trainStart

testStart = time.clock()
sign, aggClassEstTest, TP, FP, FN, TN = adaClassify(testSet, weakClassifierArr) # Collect the sign to compare
testStop = time.clock()
testTime = testStop - testStart

 #Check the results and calculate the error rate
#plotROC(aggClassEstTrain.T, trainLabels)
plotROC(aggClassEstTest.T, testLabels,'ROC curve for AdaBoost Affair Prediction 40 Stumps')