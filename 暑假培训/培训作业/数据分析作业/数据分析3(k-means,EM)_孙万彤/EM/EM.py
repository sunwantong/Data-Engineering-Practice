import pandas as pd
import numpy as np
import math as m
import random as rd
import matplotlib.pyplot as plt
import functools as ft


"""
   EM 算法实现
   其实k-means是EM算法的一个实例
"""

def readDataset():
    sets = pd.read_csv("D://fiveSet.csv", sep=" ", names=["x1", "x2"])
    dataSet = []
    for i in range(len(sets)):
        initDataSet = [round(float(x), 3) for x in sets.loc[i].tolist()]
        dataSet.append(initDataSet)
    return dataSet


#将x和y分别加到两个不同的集合中
def getXandY(dataSet):
    xArray = []
    yArray = []
    pos = 0
    for ele in dataSet:
        xArray.append(ele[0])
        yArray.append(ele[1])
        pos += 1
    # print(xArray,yArray)
    return xArray,yArray



#计算数据集合和初始簇中心的的隶属度
def calMembershipDegree(dataSet):
    xAndYArray = []
    xArray,yArray = getXandY(dataSet)
    xAndYArray.append(xArray)
    xAndYArray.append(yArray)
    dataLen = len(dataSet)
    cluster = [[],[]]

    meanVec = [[],[]]
    num = 0

    clusterCenter = [[3.0,3.0],[4.0,10.0]]
    distance = []
    outDistance1 = []
    outDistance2 = []
    outDistance = [[],[]]
    posOut = 0
    count = 0
    while count == 0:
        cluster = [[], []]
        meanVec = [[], []]
        num += 1
        for data in dataSet:
            posIn = 0

            for ele in clusterCenter:
                # distance = []
                if data in clusterCenter:
                    if data == ele:
                        cluster[posIn].append(1)
                    else:
                        cluster[posIn].append(0)
                    posIn += 1
                else:
                    vecData = np.array(data)
                    centors = np.array(ele)
                    dist = np.sum(np.square(vecData - centors))
                    distance.append(dist)
                    # cluster[posIn].append(dist)
                    posIn += 1
            flag = data in clusterCenter
            #E步
            if flag is False:
                sums = distance[0]+distance[1]
                outDistance1.append(round(float(distance[1]/sums),2))
                outDistance2.append(round(float(distance[0] / sums),2))
                distance = []
                posOut += 1
        # print(outDistance1,outDistance2)
        outDistance[0].append(outDistance1)

        outDistance[1].append(outDistance2)
        outDistance1 = []
        outDistance2 = []
        for i in range(2):
            for x in outDistance[i]:
                cluster[i].extend(x)
        # print(cluster)
        outDistance = [[],[]]
        #M步
        for ins in range(2):
            for out in range(2):
                aa = [(cluster[ins][n]**2)*xAndYArray[out][n] for n in range(dataLen)]
                bb = [cluster[ins][m]**2 for m in range(dataLen)]
                # print(aa)
                # print(bb)
                aa = ft.reduce(add,aa)
                bb = ft.reduce(add,bb)
                result = retainDecimal(aa / bb)
                meanVec[ins].append(result)

        for i in range(2):
            vecData = np.array(meanVec[i])
            cent = np.array(clusterCenter[i])
            oDist = np.sum(np.square(vecData - cent))
            if oDist < 0.0001:  #说明两个均值接近，可不用在迭代
                count += 1
            else:
                pass
        if count >= 2:
            count = 100
        else:
            count = 0
            clusterCenter[0] = meanVec[0]
            clusterCenter[1] = meanVec[1]

    return meanVec,cluster,num

#连加函数
def add(x, y):
    return x + y


#小数保留
def retainDecimal(number):
    result = round(float(number),2)
    return result

def main():
    dataSet = readDataset()
    pos = 0
    meanVec,cluster,num = calMembershipDegree(dataSet)
    print("迭代次数为:",num)
    print("最后产生的簇中心向量为:",meanVec)
    #里边是每个样本属于该类别的概率
    print("最终形成的分类数组为:",cluster)
    for x in cluster[0]:
        if x > 0.5:
            print('样本%s' % dataSet[pos],'的类别为1')
        else:
            print('样本%s' % dataSet[pos],'类别为2')
        pos += 1

if __name__ == "__main__":
    main()