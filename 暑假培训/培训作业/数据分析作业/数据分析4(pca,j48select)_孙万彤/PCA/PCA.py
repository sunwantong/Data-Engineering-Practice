import pandas as pd
import numpy as np
import math as m
import random as rd
import matplotlib.pyplot as plot
import sklearn.cluster as km
import time as t

"""
    
   
"""


def getDataSource():
    sets = pd.read_csv("D://pcadataset.csv", sep=" ")
    dataSet = []
    # print(sets)
    for i in range(len(sets)):
        initDataSet = [x for x in sets.loc[i].tolist()]
        dataSet.append(initDataSet)

    dataMat = np.mat(dataSet,dtype=float)
    dataMatTrans = dataMat.T
    return dataMatTrans


#将X的每一行（代表一个属性字段）进行零均值化，也就是减去这一行的均值
def normalize(dataSet):
    # print(dataSet.shape)
    rowLen = dataSet.shape[0]
    columnLen = dataSet.shape[1]
    posOut = 0
    for row in range(rowLen):

        meanValue = np.mean(dataSet[row])

        for column in range(columnLen):
            dataSet[row, column] = dataSet[row, column]-meanValue

    return dataSet,columnLen


#计算求得协方差矩阵,并将计算结果返回
def getCovMatrix(dataSet,columnLen):
    dataTransMat = dataSet.T
    multiMatrix = dataSet*dataTransMat
    # print(dataSet)
    # print(dataTransMat)
    for i in range(multiMatrix.shape[0]):
        for j in range(multiMatrix.shape[1]):
            multiMatrix[i, j] = multiMatrix[i,j] / columnLen
    return multiMatrix


#用雅克比算法计算特征值和特征向量
def getFeatureValueWithJacobi(dataSet):
    #算法迭代的临界条件
    thresHold = 0.001
    #矩阵的行数和列数
    rowLen,columnLen = dataSet.shape
    rows,columns = 0,0
    featVectors, mediaMat = initData(columnLen, rowLen)
    # print(mediaMat)
    while True:

        columns, maxValue, rows = getMatrixValue(columnLen, columns, dataSet, rowLen, rows)

        if maxValue < thresHold:
            break

        cosAlpha,sinAlpha = getAngle(columns, dataSet, rows)

        featVectors = reNewMatrix(columnLen, columns, cosAlpha, dataSet, featVectors, mediaMat, rowLen, rows, sinAlpha)
    return dataSet,featVectors



# 在矩阵的非对角线上找到绝对值最大的元素
def getMatrixValue(columnLen, columns, dataSet, rowLen, rows):

    maxValue = 0
    for i in range(rowLen):
        for j in range(columnLen):
            if i != j:
                absValue = abs(dataSet[i, j])
                if absValue > maxValue:
                    maxValue = absValue
                    # 找到最大元素Apq  p = rows ,q = columns
                    rows = i
                    columns = j
    return columns, maxValue, rows




#第一轮迭代完成之后，对原始矩阵进行更新,
def reNewMatrix(columnLen, columns, cosAlpha, dataSet, featVectors, mediaMat, rowLen, rows, sinAlpha):
    for i in range(columnLen):

        if i != rows and i != columns:
            dataSet[rows, i] = sinAlpha * dataSet[columns, i] + cosAlpha * dataSet[rows, i]
            dataSet[columns, i] = cosAlpha * dataSet[columns, i] - sinAlpha * dataSet[rows, i]
    for j in range(rowLen):

        if j != rows and j != columns:
            dataSet[j, rows] = sinAlpha * dataSet[j, columns] + cosAlpha * dataSet[j, rows]
            dataSet[j, columns] = cosAlpha * dataSet[j, columns] - sinAlpha * dataSet[j, rows]

    # 计算特征向量
    mediaMat[rows, rows] = cosAlpha
    mediaMat[rows, columns] = -sinAlpha
    mediaMat[columns, rows] = sinAlpha
    mediaMat[columns, columns] = cosAlpha
    featVectors = featVectors * mediaMat
    return featVectors




#计算平面旋矩阵所要旋转的角度
def getAngle(columns, dataSet, rows):
    matPPValue = dataSet[rows, rows]
    matPQValue = dataSet[rows, columns]
    matQQValue = dataSet[columns, columns]
    # print(matPPValue,matPQValue,matQQValue)
    alphaValue = ((-2) * matPQValue) / (matQQValue - matPPValue)
    # 求旋转角度 反 正切函数
    # alpha = (1/2)*m.atan(alphaValue)
    alpha = m.atan(alphaValue) / 2
    sinAlpha = m.sin(alpha)
    cosAlpha = m.cos(alpha)
    sin2Alpha = m.sin(2 * alpha)
    cos2Alpha = m.cos(2 * alpha)
    dataSet[rows, rows] = matPPValue * cosAlpha * cosAlpha + \
                          matQQValue * sinAlpha * sinAlpha + 2 * matPQValue * cosAlpha * sinAlpha
    dataSet[columns, columns] = matPPValue * sinAlpha * sinAlpha + \
                                matQQValue * cosAlpha * cosAlpha - 2 * matPQValue * cosAlpha * sinAlpha
    dataSet[rows, columns] = (1/2) * (matQQValue - matPPValue) * sin2Alpha + matPQValue * cos2Alpha
    dataSet[columns, rows] = dataSet[rows, columns]
    return cosAlpha, sinAlpha




#特征向量矩阵的初始化
def initData(columnLen, rowLen):
    # 存放特征向量的矩阵
    featVectors = np.zeros((rowLen, columnLen),dtype=float)
    mediaMat = np.zeros((rowLen, columnLen),dtype=float)
    for i in range(rowLen):
        for j in range(columnLen):
            featVectors[i, j] = 1

    # 定义一个中间矩阵mediaMat,每次迭代的时候都需要更新
    for i in range(rowLen):
        for j in range(columnLen):
            mediaMat[i, j] = 1

    return featVectors, mediaMat




#选取前k个特征值(对特征值按由大到小排序)所代表的特征向量
def choosseFeatureVector(dataSetDiag,featVectors,k):
    # print("特征值:",dataSetDiag)
    # print("特征向量:",featVectors)
    featValue = dict()
    sortedFeatValue = []
    posList = []
    featureMatrix = []

    for i in range(dataSetDiag.shape[0]):
        featValue[dataSetDiag[i,i]] = i
    sortedFeatValue = sorted(featValue,reverse=True)

    # print(featValue)
    # print("a",sortedFeatValue)

    for ele in sortedFeatValue:
        # print(featValue.get(ele))
        posList.append(featValue.get(ele))

    #由大到小排好序之后的位置列表
    # print(posList)


    for i in range(k):
        featureMatrix.append(featVectors[posList[i]])  #?
    # print(np.mat(featureMatrix))

    # 返回特征向量组成的矩阵
    return np.mat(featureMatrix)




"""
 利用斜率计算
 找到最好的降维之后的维数(特征的个数)，
 找到斜率变化最大的那个点，即为最好的维数(特征的个数)
 dataSetDiag 协方差矩阵对角化之后的矩阵
"""
def getBestKValuesWithSlope(dataSetDiag):
    k = 0
    featValue = dict()
    xAndyValue = []
    slopeList = []
    for i in range(dataSetDiag.shape[0]):
        featValue[dataSetDiag[i,i]] = i
    sortedFeatValue = sorted(featValue,reverse=True)

    lens = len(sortedFeatValue)
    maxValue = sortedFeatValue[0]+1
    xArray = list(range(1,lens+1))
    # print(xArray)

    xAndyValue.append(xArray)
    xAndyValue.append(sortedFeatValue)
    xAndyValue = np.mat(xAndyValue)
    row,column = xAndyValue.shape
    # print(xAndyValue)
    # print(row,column)

    for j in range(column):
        if j + 1 <= 5:
            diffMat = xAndyValue[:, j + 1] - xAndyValue[:, j]
            slopeList.append(abs(diffMat[1,0] / diffMat[0,0]))

    #对求出来的斜率进行排序
    slopeList = sorted(slopeList,reverse=True)
    # print(slopeList)
    k += 2

    xMax = lens + 1
    yMax = maxValue
    #由于第一个元素是最大位置，所以第一个元素的位置+1（因为下表从零开始）+1 （图像第一个位置没有斜率，所以加一）即为最佳k值
    return k,xMax,yMax,xArray,sortedFeatValue


"""
 利用累计方差百分比计算最佳特征的个数
  找到最好的降维之后的维数(特征的个数)，
 若累计方差百分比大于某一个阈值，则可用前面几个特征作为降维之后的特征
"""
def getBestKValueWithPercent(dataSetDiag):
    k = 0
    featValue = dict()
    xAndyValue = []
    slopeList = []
    for i in range(dataSetDiag.shape[0]):
        featValue[dataSetDiag[i,i]] = i
    sortedFeatValue = sorted(featValue,reverse=True)

    lens = len(sortedFeatValue)
    maxValue = sortedFeatValue[0]+1
    xArray = list(range(1,lens+1))
    # print(xArray)
    thresHold = 0.8
    # print(sortedFeatValue)
    sums = np.sum(sortedFeatValue)
    pos = 1
    tempSum = sortedFeatValue[0]

    for ele in sortedFeatValue:
        temp = tempSum / sums
        if temp > thresHold:

            break
        else:
            tempSum += ele
            pos += 1

    xMax = lens + 1
    yMax = maxValue
    return pos,xMax,yMax,xArray,sortedFeatValue




#主函数
def main():

    dataMat = getDataSource()
    initDataMat = dataMat.copy()
    #dataset 均值化后的数据

    dataSet,columnLen = normalize(dataMat)
    covMatrix = getCovMatrix(dataSet,columnLen)

    # a,b = np.linalg.eig(covMatrix)
    # print("特征值",a)
    # # print("特征向量",b)

    dataSetDiag,featVectors = getFeatureValueWithJacobi(covMatrix)


    #返回最佳特征值个数

    #利用斜率计算
    # k,xMax,yMax,xArray,yArray = getBestKValuesWithSlope(dataSetDiag)

    #利用累计方差百分比计算最佳k(降维之后特征的个数)值的数值
    k, xMax, yMax, xArray, yArray = getBestKValueWithPercent(dataSetDiag)


    #求特征向量
    featValue = choosseFeatureVector(dataSetDiag,featVectors,k)

    print("最佳特征的个数为:",k)



    #测试pca的效果

    start = t.clock()
    testPerformance(initDataMat)
    end = t.clock()
    print("PCA降维前算法执行时间:",end - start)

    #降维之后的数据  特征向量乘均值化后的数据
    finalMatrix = featValue*dataSet

    #最终降维后的数据


    start1 = t.clock()
    testPerformance(finalMatrix)
    end1 = t.clock()
    print("PCA降维后算法执行时间:", end1 - start1)

    print("\n")

    print("最终降维后的数据", finalMatrix)
    plotImage(xMax, yMax, xArray, yArray)
    pass

"""
  图像  横坐标为特征的个数
       纵坐标为特征的方差的百分比
  方差比越大，则此特征越重要
"""

#用k-means算法测试降维之后的效果
def testPerformance(dataMat):
    newMat = dataMat.T.copy()
    kmeans = km.KMeans(n_clusters=2, random_state=0).fit(dataMat)
    pass


def plotImage(xMax,yMax,xArray,yArray):
    # plot.title("累计方差百分比")
    plot.xlabel("the count of feature")
    plot.ylabel("the variance of feature")
    plot.xlim(0.0, xMax)
    plot.ylim(0.0, yMax)
    plot.plot(xArray, yArray)
    plot.show()


if __name__ == "__main__":
    main()