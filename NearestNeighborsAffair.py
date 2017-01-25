# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:33:04 2016

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


start_time = time.time() ##Start Program Clock

# time in seconds 
#start = time.clock()
#end = time.clock()
#time = (end-start)

#Overflown file2matrix to limit read entries from testdata
def file2matrix(filename, width, limit=None):
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
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
        if index==limit:
            break
    return returnMat, classLabelVector
    
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
    print returnMat.shape #Return Matrix 6366 entries with 14 columns
    
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
    k = dfoldCrossValid(CvTestMat, 5) #Find best value of K through cross-validation
    testLabelVector = testMat[:,8] 
    classLabelVector = dataMat[:,8]
    
    #Create a test matrix with 50% random true and 50% random false for 25% the size of training data
    #randomize the data by column        
    return dataMat, classLabelVector, testMat, testLabelVector,k

def dfoldCrossValid(testMat, foldNum):
    i=0
    errRate = zeros((foldNum,2))
    totalerrRate=0
    
    for i in range(foldNum):
        for q in range(foldNum):
            testVal = int(testMat.shape[0]/foldNum) #Calculate the range of the testdata points
            trainDat1 = testMat[(testVal*(q+1)):,:] #Later section of train data after test indices
            trainDat2 = testMat[:(testVal*q), :]  #train dat before test dat indices
            testDat = testMat[(q*testVal):((q+1)*testVal),:] #Testdat ranges between the end of trainDat2 and the beginning of trainDat1
            trainDat = np.r_[trainDat1, trainDat2] #Append trainDat1&2
            #Create Labels
            trainDatLabels = trainDat[:,8]
            testDatLabels = testDat[:,8]
            a,b,c,d,e,f,g,h,z = classTest(trainDat, trainDatLabels, testDat, testDatLabels, ((i+1)*5)+i*15)
            #Only value required is a at this point, the rest are dummy variables
            totalerrRate += a
        errRate[i,0 ] = (totalerrRate/double(q+1))
        errRate[i,1] = ((i+1)*5+i*15)
        totalerrRate = 0
    errRatesort = errRate[errRate[:,0].argsort()] ##Sort values to select best k value
    print errRatesort
    k = errRatesort[0,1]   
    print "k = %r" %(k)
    
    return int(k)
    
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
            
        
    #data = data.replace(".", "")
#    outfile.write(listdat)
            
def cleanRBMatrix(matrix, vector):  
    matrix = np.delete(matrix, [0,1,8,11,12,13],axis=1) #Delete unused columns
    matrix = np.c_[matrix, vector] #Append new vector adjusted to binary values
    return matrix         
    
def splitMatrix(matrix):
    #Separate Data Types into seperate matrices for distance calcs
    numericMat = np.c_[matrix[:,1:4], matrix[:, 6:8]] #v2, v3, v4, v7, v8
    nominalMat = np.c_[matrix[:,0], matrix[:, 4:6], matrix[:, 8]] #v1, v5, v6, and adjust Yrb
    

    #Normalize Numeric&Ordinal Data
    minVals = numericMat.min(0)
    maxVals = numericMat.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(numericMat))
    m = numericMat.shape[0]
    normDataSet = numericMat - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return nominalMat, normDataSet
    
   
        
def classifyMixed(norminX, normDataSet,nominX,nomDataSet, labels, k):  
    #Distances Euclidean
    dataSetSize = normDataSet.shape[0]
    diffMat = tile(norminX, (dataSetSize, 1)) - normDataSet
    sqDiffMat = diffMat**2
    
    sqDistances = sqDiffMat.sum(axis=1)
    
    distances = sqDistances**0.5
    #Distances Nominal
    metric = np.array([])
    Row = int(dataSetSize)
    Column = double(len(nomDataSet[0,:]))
    for m in range(Row):
        entity = np.where((nomDataSet[m,:] != nominX))
        metric = np.append(metric,(len(entity[0])/Column))
    #Mixed Dissimalairities
    mixedD = np.mean(np.array([distances, metric]), axis=0)
   
    
    sortedDistIndices = mixedD.argsort()
    voteIlabel = [] #VoteIlabel converted to vector in order to save voters for ROC graph
    classCount={}
    for i in range(k):
        voteIlabel.append(labels[sortedDistIndices[i]])
        classCount[voteIlabel[i]] = classCount.get(voteIlabel[i], 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1), reverse=True)
    #For ROC collect NN information those who supported a and b out of k {threshold} append to vector within classtest

    Majority = (sum(float(num) == 1 for num in voteIlabel))/float(k)
   
    return sortedClassCount[0][0], Majority
    

def classTest(matrix, vector, testMat,testLabel, k):
     
     nomMat, numMat = splitMatrix(matrix)
     nomMatTest, numMatTest = splitMatrix(testMat)
     numTestVecs = testMat.shape[0]
   
     errorCount = 0.0
     TPos = 0
     FPos = 0
     TNeg = 0
     FNeg = 0

     
     TP = []
     TN = []
     kMajority = 0
     for i in range (numTestVecs):
         classifierResult,kMajority = classifyMixed(numMatTest[i, :], numMat,nomMatTest[i, :], nomMat, vector, k)
         
         print "The classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[i])
         if (classifierResult != testLabel[i]): errorCount += 1.0
         if (classifierResult == 1 and testLabel[i]==1):
             TPos +=1
             TP.append(kMajority)
         elif (classifierResult == 1 and testLabel[i] == 0):
             FPos +=1
             TN.append(kMajority)
         elif (classifierResult == 0 and testLabel[i] == 1):
             FNeg +=1
         elif (classifierResult == 0 and testLabel[i] == 0):
             TNeg +=1
     print "The total error rate is: %f for %r" % (errorCount/float(numTestVecs), k)
     errorRate = errorCount/float(numTestVecs)
     
     return ((errorRate)*100), errorCount, numTestVecs, TPos, FPos, FNeg, TNeg,TP, TN
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
    plt.title('ROC curve for AdaBoost Affair Prediction System')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep   
#start code
np.set_printoptions(3,None, None, None, True) ##set Precision of display to 3 and disable scientific notation
#cleanData("PtTapeData.txt", "CleanPtTapeData.txt") #Not enough data points just use the RB study
cleanData("AffairData\RbTapeData.txt", "AffairData\CleanRbTapeData.txt")
trainMat, trainLabel, testMat,testLabel, k = file2matrixAffair("AffairData\CleanRbTapeData.txt", 14)


#eR = errorRate, eC = errorCount
timestart = time.clock()
eR, eC, m, YY, YN, NY, NN,TP,FP = classTest(trainMat, trainLabel,testMat,testLabel, k)
timestop = time.clock()

time = timestop - timestart

#plotROC(trueCount, trainLabel) #plot ROC graph

