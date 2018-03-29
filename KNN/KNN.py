#-*- coding:utf-8 -*-
#########################################################################
# File Name: KNN.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/env python3

import operator
from numpy import *

from os import listdir

def createDataSet():
    """
    创建数据集和标签

    调用方式
    import KNN
    group, label = KNN.createDataSet()
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']

    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    inx[1, 2, 3]
    DS = [[1, 2, 3], [1, 2, 0]]
    inX: 用于匪类的输入向量
    dataSet: d输入的训练样本集
    labels: 标签向量
    k: 选择最近邻居的数目
    注意： labels元素的数目和dataSet的行数是相同的，程序使用的欧式距离公式，

    预测数据所在分类可在输入下列命令
    KNN.classify0([0, 0], group, labels, 3)
    """
    # 距离计算
    dataSetSize = dataSet.shape[0]

    #tile 生成和训练样本对应的矩阵， 并于训练样本求差
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    
    sqDistances = sqDiffMat.sum(axis = 1)

    distances = sqDistances ** 0.5

    sortedDistIndicies = distances.argsort()


    #寻找z距离的最小的K个点
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key =
            operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]




    
