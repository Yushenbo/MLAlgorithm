#-*- coding:utf-8 -*-
#########################################################################
# File Name: DecisionTree.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/python3

import operator
from math import log
import decisionTreePlot as dtPlot
import threading

def createDataSet():
    '''
    DataSet fundamatal data

    Args:
        None
    Returns:
        DataSet and associated labels
    '''
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    '''
    calcShannonEnt(calculate shannon entropy)

    Args:
        dataSet
    Returns:
        reurn every kinds of ones group feature, shanno entropy expectatin
    '''

    # calculate the length of list stands for data mounts
    numEntries = len(dataSet)

    # calculate the count of labels occurance
    labelCounts = {}
    # The number of unique elements and their occurance
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # create a dic for every elements that could occur, if key not exist
        # add it, and record thir numbers of occurance
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        
    shannonEnt = 0.0
    for key in labelCounts:
        # Calculate label's possibility by the occurence times
        prob = float(labelCounts[key])/numEntries
        #Calculate shannon entropy, log 2
        shannonEnt -= prob * log(prob, 2)
    
    return shannonEnt

def splitDataSet(dataSet, index, value):
    '''
    splitDataSet

    Args:
        dataSet
        index 
        value 
    Reutrn:
        index values set of index column
    '''
    retDataSet = []
    for featVec in dataSet:
        #index column values set
        #check index value
        if featVec[index] == value:
            #chop out index used for splitting
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            #[index+1:] jump index + 1 rows, keep the remains
            #collect value in index row
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    '''

    Args:
        dataSet
    Returns:
        bestFeature
    '''

    #calculate how many columns in certain rows
    numFeatures = len(dataSet[0]) - 1

    # label shannon entropy
    baseEntropy = calcShannonEnt(dataSet)
    # Best shannon entropy gain, Best feature label
    bestInfoGain, bestFeature = 0.0, -1
    # iterate over all the feature
    for i in range(numFeatures):
        #Create a list of all the examples of this feature
        #gain i+1 feature of every entity
        featList = [example[i] for example in dataSet]
        #get a set of unique values
        uniqueVals = set(featList)
        # ceate a temporary datainfo entropy
        newEntropy = 0.0

        # travese a certain colum n value set, cal info entropy
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # gain [Info Gain]
        infoGain = baseEntropy - newEntropy
        print('infoGain', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            beatFeature = i

    return bestFeature

def majorityCnt(classList):
    '''
    majorityCnt

    Args:
        classList label column set
    Returns:
        bestFeature best feature column
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    #Revese classCount to get a revese dic
    sortedClassCount = sorted(classCount.iteriterms(),
            key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    # Get Label name
    bestFeatLabel = labels[bestFeat]
    # Initial tree
    myTree = {bestFeatLabel: {}}
    
    del(labels[bestFeat])
    # Gain best column,
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # Get remain labels
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,
            bestFeat, value), subLabels)

    return myTree

def classify(inputTree, featLabels, testVec):
    '''

    Args:
        inputTree decisionTree module
        featLabels Feature lable name associated
        testVec Input test data
    Returns:
        classLable  classfied result
    '''

    # Get tree root node key value
    firstStr = list(inputTree.keys())[0]
    print('firstStr: ', firstStr)

    secondDict = inputTree[firstStr]
    print('SecondDict: ', secondDict)

    featIndex = featLabels.index(firstStr)
    print('featIndex: ', featIndex)

    key = testVec[featIndex]
    print('key: ', key, 'testVec: ', testVec)
    valueOfFeat = list(secondDict)[key]
    print('+++', firstStr, 'xxx', secondDict,
            '---', key, '>>>', valueOfFeat)

    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat

    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def fishTest():
    #1, Create data & result label
    myDat, labels = createDataSet()

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)

    print(classify(myTree, labels, [1, 1]))

    #draw
    dtPlot.createPlot(myTree)


def ContactLensesTest():
    '''
    Desc:
        Lenses glasses test
    Args:
        None
    Returns:
        None
    '''

    fr = open('./lenses.txt')

    lenses = [inst.strip().split('\t') for inst in fr.readlines()]

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

    lensesTree = createTree(lenses, lensesLabels)

    print(lensesTree)

    #Draw pic
    dtPlot.createPlot(lensesTree)

if __name__ == '__main__':
    '''
    threads = []
    t1 = threading.Thread(target=fishTest)
    threads.append(t1)
    t2 = threading.Thread(target=ContactLensesTest)
    threads.append(t2)

    for t in threads:
        t.setDaemon(True)
        t.start
    '''
    fishTest()
    ContactLensesTest()
