#-*- coding:utf-8 -*-
#########################################################################
# File Name: fileParse.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/env python3

from numpy import *

import operator
from os import listdir

def file2matrix(filename):
    '''
    Des:
        import training data
    parameters:
        filename: data directory
    return:
        数据矩阵 returnMat 和对应的类别 classLabelVector

    '''

    fr = open(filename)
    # get the line numbers of data
    numberOfLines = len(fr.readlines())
    #生成对应的空矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0

    for line in fr.readlines():
        #str.strip([chars]) --返回以出字符串头尾制定的字符生成的新字符串
        line = line.strip()
        #以'\t' 切割字符串
        listFromLine = line.split('\t')
        #每列属性data
        returnMat[index, :]  = listFromLine[0: 3]

        #每列的类别数据， 就是label的标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''
    归一化特征值，消除属性之间不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet, ranges和minVals即最小值与范围，并没有
    用到归一化公式:
        Y = (X - Xmin)/(Xmax - Xmin)
        其中的min 和 max
        分别是数据集中最小特征值和最大特征值。该函数可以自动将数字特征转化为到0
        到 1 的区间。
    '''
    #计算每种属性的最大值、最小值、范围
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    #极差
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    #生成与最下之之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    #将最下之差除以范围组成的矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1)) # element wise divide
    return normDataSet, ranges, minVals






if __name__ == '__main__':
    returnMat, classLabelVector = file2matrix('datingTestSet.txt')
    print('--'*40)
    print(classLabelVector)

    print('--'*40)
    print(returnMat)

