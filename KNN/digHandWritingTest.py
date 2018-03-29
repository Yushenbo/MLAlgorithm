#-*- coding:utf-8 -*-
#########################################################################
# File Name: digHandWritingTest.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/env python3

import KNN
from os import listdir
from numpy import *

def img2vector(filename):
    '''
    将图像数据转化为向量
    :param filename : 图片文件 一维我们的输入数据的图片格式是 32 * 32
    :return : 一维矩阵
    该函数将图像转换为向量: 该函数创建 1 * 1024 的numpy数组， 然后打开给定的文件
    循环读出文件的钱32行，并将美韩的头32个字符值存储在NumPy数组中，最后返回数组
    '''

    returnVect = zeros((1, 1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])

    return returnVect

def handwritingClassTest():
    # 1 Import data
    hwLabels = []
    trainingFileList = listdir('./trainingDigits') # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    # HwLabels 存储 0 ~ 9 对应的index的位置， trainingMat存放
    # 每个位置对应的图片向量
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0] # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        #将 32  * 32 的矩阵 -> 1 * 1024 的一维矩阵
        trainingMat[i, :] = img2vector('./trainingDigits/%s'%fileNameStr)

    # 2 Import Test Data
    testFileList = listdir('./testDigits') # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('./testDigits/%s'%fileNameStr)
        classifierResult = KNN.classify0(vectorUnderTest, trainingMat,
                hwLabels, 3)

        print('the classfier came back with : %d , the real answer is:%d'
                %(classifierResult, classNumStr))

        if (classifierResult != classNumStr):
            errorCount += 1.0
    print('\nThe total number of Errors is: %d'%errorCount)
    print('\nThe total number of Errors rate is: %f'
            %(errorCount /float(mTest)))

if __name__ == '__main__' :
    handwritingClassTest()

