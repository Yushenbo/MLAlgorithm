#-*- coding:utf-8 -*-
#########################################################################
# File Name: datingClassTest.py
# Author: Nichol.Shen
# mail: nichol_shen@yahoo.com
#########################################################################
#!/usr/bin/env python3

import KNN
import fileParse

def datingClassTest():
    '''
    对约会网站 的测试方法
    :return: 错误数目
    '''
    # 设置测试数据的一个比例（训练数据集的比例 = 1 - hotRatio）
    hoRatio = 0.1 #测试范围， 一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = fileParse.file2matrix('./datingTestSet.txt')
    # 归一化数据
    normMat, ranges, miuVals = fileParse.autoNorm(datingDataMat)
    # m  表示数据h的行数， 即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs: m 便是训练样本的数量
    numTestVecs = int(m * hoRatio)

    print('numTestVecs =', numTestVecs)

    errorCount = 0.0

    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = KNN.classify0(normMat[i, :], normMat[numTestVecs:m, :],
                datingLabels[numTestVecs: m], 3)
        print('||'*40)
        print('The clssifier came back with:%d, the real answer is:%d'%(classifierResult, datingLabels[i]))

        if(classifierResult != datingLabels[i]):
            errorCount += 1.0

        print('the total error rate is:%f'%(errorCount / float(numTestVecs)))
        print('errorCount is:', errorCount)
        print()


if __name__ == "__main__":
    datingClassTest()
