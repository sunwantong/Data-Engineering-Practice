import pandas as pd
import numpy as np
import math as m
import random as rd
import matplotlib.pyplot as plt


"""
  k-means算法
  k表示当前cluster的个数
"""

#加载本地数据集
def loadDataSet():
    sets = pd.read_csv("D://melon.csv",sep="\t",names=["x1","x2"])
    dataSet = []
    for i in range(len(sets)):
        initDataSet = [round(float(x),3) for x in sets.loc[i].tolist()]
        dataSet.append(initDataSet)
    return dataSet

#随机抽取k个向量作为均值向量进行聚类
def getRandomMeanValues(k,dataSet):
    setLens = len(dataSet)
    meanVec = []
    # 从range()里边随机抽取k个数
    ranList = rd.sample(range(setLens), k)
    for i in ranList:
        meanVec.append(dataSet[i])
    return meanVec


#k均值算法,迭代更新均值向量，直至均值向量都不更新为止
    #先找出均值向量
def calDistanaceAndIteration(dataSet,k):
    meansVec = getRandomMeanValues(k,dataSet)

    cluster = []
    for e in range(k):
        cluster.append([])

    nums = 0
    count = 0
    sortArray = []
    while count == 0:
        nums += 1
        # cluster = [[], [], []]  #每次都需要清空
        cluster = []
        for e in range(k):
            cluster.append([])

        for data in dataSet:
            min = 0
            #当前均值向量的位置
            p = 0
            for pos in range(len(meansVec)):
                vecData = np.array(data)
                vecMean = np.array(meansVec[pos])
                dist = np.sqrt(np.sum(np.square(vecData - vecMean)))
                sortArray.append(dist)

            min = sortArray[0]
            for m in range(len(sortArray)):
                if sortArray[m] < min:
                    min = sortArray[m]
                    p = m
            cluster[p].append(data)
            sortArray = []
        # 计算新的均值向量,如果每次三个均值都更新的话，count每次更新都加1，否则就是没更新，继续循环，划分新的均指向量
        pos = 0
        # print(meansVec)
        tempList = []
        for data in cluster:
            sumx1 = 0
            sumx2 = 0
            lens = len(data)
            # 计算初始的一个cluster的新的均值
            for d in data:
                sumx1 += d[0]
                sumx2 += d[1]
            meanX1, meanX2 = round(float(sumx1 / lens), 3), round(float(sumx2 / lens), 3)
            tempList.append(meanX1)
            tempList.append(meanX2)
            # print(meanX1, meanX2)
            # print(tempList)
            #不等于，表示还得继续循环
            if meansVec[pos] != tempList:
                meansVec[pos] = tempList
                count += 1  #要是变，说明有一个更新了，还需要继续循环
            else:
                pass
            pos += 1
            tempList = []
        if count != 0:
            #还需要继续循环
            count = 0
        else:  #不需要循环,给定一个较大的数值,让他直接退出
            count = 100
        # print(meansVec)
        # print(count)
    return cluster,meansVec,nums,k


#计算新的均值向量,如果每次三个均值都更新的话，count每次更新都加1，否则就是没更新，继续循环，划分新的均指向量
def getNewMeanVec(cluster,meanVec,count,meansVec):
    count += 1
    pos = 0
    # print(meansVec)
    tempList = []
    for data in cluster:
        sumx1 = 0
        sumx2 = 0
        lens = len(data)
        #计算初始的一个cluster的新的均值
        for d in data:
            sumx1 += d[0]
            sumx2 += d[1]
        meanX1,meanX2 = round(float(sumx1/lens),3),round(float(sumx2/lens),3)
        tempList.append(meanX1)
        tempList.append(meanX2)
        # print(meanX1, meanX2)
        # print(tempList)
        if meansVec[pos] != tempList:
            meansVec[pos] = tempList
            count += 1
        pos += 1
        tempList = []
    print(meansVec)
    # print(meanX1,meanX2)

def drawImage(dataSet,k,meanVec):
    xValue = []
    yValue = []
    center_x = []
    center_y = []
    for e in range(k):
        xValue.append([])
        yValue.append([])
    pos = 0

    # 画出三类数据点及聚类中心
    for data in dataSet:
        for ele in data:
            xValue[pos].append(ele[0])
            yValue[pos].append(ele[1])
        center_x.append(meanVec[pos][0])
        center_y.append(meanVec[pos][1])
        pos += 1

    plt.figure(figsize=(10, 8), dpi=80)
    pt = plt.subplot(111)
    clusterList = []
    strs = []
    colors = ['red','green','blue','yellow','black','purple','aqua','brown','darkblue']

    center = pt.scatter(center_x, center_y, s=90, c='black', marker='*')
    for e in range(k):
        #画散点图
        cluster = pt.scatter(xValue[e], yValue[e], s=90, c=colors[e])
        clusterList.append(cluster)
        e = str(e)
        strs.append(e)

    #添加均值向量
    clusterList.append(center)
    strs.append("center")

    #表明哪个是x轴，那个是y轴
    # plt.xlabel('x')
    # plt.ylabel('y')

    # pt.legend((cluster1, cluster2, cluster3,cluster4), ('0','1','2','3'), loc=1)
    pt.legend(clusterList,strs, loc=1)
    plt.show()


# k-means算法评价函数(cluster之间的最小化误差平方和),
# 绘制E-K图,找到最优k值(E-K第一个拐点处的位置)
def calEvaluateMethod(cluster,meanVec,k):
    pos = 0
    sums = 0
    for data in cluster:
        for ele in data:
            vecData = np.array(ele)
            vecMean = np.array(meanVec[pos])
            dist = np.sum(np.square(vecData - vecMean))
            # print(dist)
            # print(np.sqrt(dist),dist)
        sums += dist
        pos += 1
    sums = round(float(sums),3)
    # print(sums)
    return sums,k

# E-K曲线的绘制
def drawEkCurve(eArray,kArray):
    plt.plot(kArray,eArray)
    plt.show()


def main():
    #聚类的个数
    # 从range()里边随机抽取1个数
    k = rd.sample(range(2,10), 1)
    k = k[0]
    # print(k)
    bestK = 0
    #E,k保存的位置
    eArray = []
    kArray = []
    dataSet = loadDataSet()

    # i表示随机产生的k值
    for i in range(2,10):
        cluster, meanVec, nums, k = calDistanaceAndIteration(dataSet, i)
        sums,k = calEvaluateMethod(cluster,meanVec,i)
        eArray.append(sums)
        kArray.append(i)
    drawEkCurve(eArray,kArray)

    #根据多次执行E-K得知，k = 4的时候聚类效果是最好的
    #所以令 k = 4的时候,迭代聚类
    bestK = 4
    cluster, meanVec, nums, k = calDistanaceAndIteration(dataSet, bestK)
    print("迭代的次数为:", nums)
    drawImage(cluster,bestK,meanVec)


if __name__ == "__main__":
    main()