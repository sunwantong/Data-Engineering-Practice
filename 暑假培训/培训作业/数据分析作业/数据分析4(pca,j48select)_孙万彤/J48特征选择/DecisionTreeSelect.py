import numpy as np
import random as rd
import math as m
import sklearn.naive_bayes as NB
import time as t
from sklearn.model_selection import cross_val_score

#读取训练集
def loadDataSet():
    f = open("D://数据处理数据源//j48_train.txt",'r')
    dataSet = f.readlines()
    lables = []
    dataMat = []
    attrValue = []
    for line in dataSet:
        line = line.strip().split(",")
        lables.append(line[-1])
        dataMat.append(line[0:(len(line))])
    attrValue = dataMat[0]
    dataMat.pop(0)
    lables.pop(0)
    attrValue.pop(-1)
    return lables,attrValue,dataMat



#读取测试集
def loadTestDataSet():
    f = open("D://数据处理数据源//j48_test.txt",'r')
    dataSet = f.readlines()
    dataMat = []
    for line in dataSet:
        line = line.strip().split(",")
        dataMat.append(line)
    return dataMat




#计算数据集合熵的数值
def calAllEntryValue(dataSet=[]):

    entryValue = 0
    #dataSet是一行
    lenSet = len(dataSet)
    #定义一个字典，保存yes的个数和no的个数
    dataDict = dict([(types,dataSet.count(types)) for types in dataSet])
    for k in dataDict:
        # print(k,dataDict[k],lenSet)
        temp = float(dataDict[k]/lenSet)
        entryValue += temp*m.log(temp,2)
    entryValue = -entryValue
    finaEntryValue = round(entryValue,3)
    return finaEntryValue




#计算熵的数值
def calEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for data in dataSet:
        currentLabel = data[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
        entryValue = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        entryValue -= prob * m.log(prob,2)
    return entryValue




#计算单个属性的最大熵值，并返回，以便于进行数据集的划分
def getBestSingleEntryDataSet(dataSet):
    #获取特征个数
    attrNums = len(dataSet[0]) - 1  #去除最后一项标记类
    bestInfoGainRatio = 0.0
    baseEntropy = calEntropy(dataSet)
    # print(baseEntropy)
    bestFeaturePos = -1
    #遍历所有的特征值
    for i in range(attrNums):
        featureArray = [data[i] for data in dataSet]
        #去除重复值
        noRepeatValue = set(featureArray)

        tempEntropy = 0.0
        infoIV = 0.0
        for value in noRepeatValue:
            #拿到特征值里边取某一个值的集合(为了计算熵方便)
            subDataSet = getSubDataSet(dataSet, i, value)
            probab = len(subDataSet) / float(len(dataSet))
            tempEntropy += probab * calEntropy(subDataSet)
            infoIV -= (probab * m.log(probab, 2))
        if infoIV != 0:
            infoGainRatio = (baseEntropy - tempEntropy) / infoIV
        else:
            infoGainRatio = 0
        if infoGainRatio > bestInfoGainRatio:
            bestInfoGainRatio = infoGainRatio
            #属性在数据集中列的位置
            bestFeaturePos = i
    return bestFeaturePos




#特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def calVoteCount(classArray):
    classCount = {}
    for data in classArray:
        if data not in classCount.keys():
            classCount[data] = 0
        classCount[data] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



#获得同一属性的集合
def getSubDataSet(dataSet, pos, value):
    restDataSet = []
    for data in dataSet:
        if data[pos] == value:
            #去除pos位置的属性
            restdFeat = data[:pos]
            restdFeat.extend(data[pos+1:])
            restDataSet.append(restdFeat)
    return restDataSet



#计算当前树的深度并且返回
def getTreeDeep(myTree):
    c = str(myTree)
    count = 0
    for ele in c:
        if ele == "{":
            count += 1
    print(count)
    return count



#矩阵由字符类型转换成数值类型(便于在分类器上边进行测试)
def string2numeric(dataMat):
    row,col = dataMat.shape
    newMat = np.zeros((row,col),dtype=int)
    # print(newMat)
    # print(dataMat)
    for i in range(col):
        # print(dataMat[:,i])
        tempValue = list(removeRepeatValue(dataMat[:,i].flatten().A[0]))

        lens = len(tempValue)
        # print(tempValue)
        for j in range(row):
            if len(tempValue) == 2:
                if dataMat[j,i] == tempValue[0]:
                    newMat[j,i] = int(6)
                if dataMat[j,i] == tempValue[1]:
                    newMat[j,i] = int(8)
            else:
                if dataMat[j,i] == tempValue[0]:
                    newMat[j,i] = int(1)
                if dataMat[j,i] == tempValue[1]:
                    newMat[j,i] = int(2)
                if dataMat[j,i] == tempValue[2]:
                    newMat[j,i] = int(3)
                pass
    # print(newMat)
    return newMat



#去除矩阵中元素重复数值
def removeRepeatValue(data):
    newData = []
    for ele in data:
        if ele not in newData:
            newData.append(ele)
    return newData



#将特征集合提取出来之后在分类器上边进行测试
def testPerPerformance(dataMat):
    row,col = dataMat.shape

    gnb = NB.GaussianNB()

    # 朴素贝叶斯预测数值
    preditValue = gnb.fit(dataMat[:,0:col-1], dataMat[:,-1]).predict(dataMat[:,0:col-1])

    # 样本总数
    sampleSums = row

    # 预测错误的个数
    falsePreditNums = (dataMat[:,-1] != preditValue).sum()
    precision = (sampleSums - falsePreditNums) / sampleSums

    #五折交叉验证
    scores = cross_val_score(gnb, dataMat[:,0:col-1],dataMat[:,-1], cv=5)
    # print("分数为:",scores)
    return np.mean(scores)



#找到最佳N值的函数(N值表示决策树的层次)
def getBestN(FList, attrSubList, attrValue, dataMat, labels, lenAttrValue, testDataSet):

    # 假设选择最上面N(假设N的范围是从2到5)层节点所用到的特征作为特征子集。必须对N有一个评价
    for N in range(2, 5):
        myTree = createTree(dataMat, attrValue, attrSubList, N, lenAttrValue)
        attrSubList.clear()
        f = getEvaluateTarget(labels, myTree, testDataSet)
        FList[N] = f
    maxValue = 0
    bestN = 0
    for k, v in FList.items():
        if v > maxValue:
            maxValue = v
            bestN = k
    # print(bestN)
    # print(FList)
    return bestN


#求解降维之后的子数据集
def getChoosedDataSet(dataSet,attrValue,attrSubList):
    # print(dataSet)
    posList = []
    subDataSet = []
    for ele in attrSubList:
        posList.append(attrValue.index(ele))

    # print(posList)
    dataMat = np.mat(dataSet)
    for pos in posList:
        subDataSet.append(dataMat[:,pos])
    subDataSet.append(dataMat[:,-1])

    #根据选出来的特征子集选择subDataSet
    subDataMat = np.hstack(subDataSet)
    return subDataMat,dataMat



#算法评价函数
def getEvaluateTarget(labels, myTree, testDataSet):
    TP,FP,FN,TN = 0,0,0,0
    presion,recall,f = 0,0,0
    for testData in testDataSet:
        #预测样本类别(返回预测后的样本的类别)
        des = classify(myTree,labels,testData)
        #实际值
        sign = testData[-1]
        # print(testData,des)
        if sign == '是':
            if des == '是':
                TP += 1
            else:
                FN += 1
        if sign == '否':
            if des == '是':
                FP += 1
            else:
                TN += 1

    # print(TP,FP,FN,TN)
    # yesPresion = TP / (FP + TP)
    # yesRecall = TP / (TP + FN)
    # yesF1 = (2*yesPresion*yesRecall) / (yesPresion+yesRecall)

    noPrecision = float(TN) / (FN + TN)
    noRecall = float(TN) / (TN + FP)
    noF1 = (2*noPrecision*noRecall) / (noPrecision+noRecall)
    # print("Precision: %f" % noPrecision)
    # print("Recall: %f" % noRecall)
    # print("F1: %f\n" % noF1)
    return noF1



#创建树
def createTree(dataSet,attrValue,attrList,N,lens):

    #数据集类标记信息的集合 ['是', '是', '是', '是', '是', '否', '否', '否', '否', '否']
    classArray = [data[-1] for data in dataSet]
    if classArray.count(classArray[0]) == len(classArray):
        return classArray[0]

    if len(dataSet[0]) == 1:
        return calVoteCount(classArray)

    #bestFeat返回字段在所在行的位置
    bestFeatPos = getBestSingleEntryDataSet(dataSet)

    bestFeatLabel = attrValue[bestFeatPos]
    # print("最佳位置:", bestFeatLabel)

    attrList.append(bestFeatLabel)


    myTree = {bestFeatLabel:{}}
    subLabels = attrValue[:]
    subLabels.pop(bestFeatPos)
    # print("层级关系:",lens-len(subLabels))
    featValues = [data[bestFeatPos] for data in dataSet]
    noRepeatValue = set(featValues)

    #如果当前树的层次等于给定的N值，则返回当前所建好的树
    if lens-len(subLabels) == N:
        return myTree

    for data in noRepeatValue:
        #程序递归生成子数
        myTree[bestFeatLabel][data] = createTree(getSubDataSet(dataSet, bestFeatPos, data), subLabels,attrList,N,lens)
    # print(myTree)
    # print(N)
    return myTree  #生成的树




#遍历树从而对结果进行分类
def classify(myTree, attrLabels, testSet):
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    classLabel = ''
    secondDict = myTree[firstStr]
    pos = attrLabels.index(firstStr)
    for key in secondDict.keys():
        if testSet[pos] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],attrLabels,testSet)
            else:
                classLabel = secondDict[key]
    return classLabel




def main():
    # 加载训练集
    lables,attrValue,dataMat = loadDataSet()

    #加载测试集
    testDataSet = loadTestDataSet()

    attrSubList = []
    FList = dict()

    #根据训练集建立决策树
    lenAttrValue = len(attrValue)

    labels = ['色泽','根蒂','敲声','纹理','脐部','触感']

    #找到一个最优N解(N代表树的层次)
    bestN = getBestN(FList, attrSubList, attrValue, dataMat, labels, lenAttrValue, testDataSet)
    myTree = createTree(dataMat, attrValue, attrSubList, bestN, lenAttrValue)
    #返回特征选择之后的数据集
    subDataMat,dataMats = getChoosedDataSet(dataMat, attrValue, attrSubList)
    #用分类器进行测试

    print("选择出来的属性子集为:",attrSubList)

    print("\n")

    #降维前：

    data = string2numeric(dataMats)
    start1 = t.clock()
    precison = testPerPerformance(data)
    end1 = t.clock()
    print("降维前算法执行时间:",end1 - start1)
    print("降维前算法预测准确率",precison)

    print("\n")

    #降维后

    data = string2numeric(subDataMat)
    start = t.clock()
    precison = testPerPerformance(data)
    end = t.clock()
    print("降维后算法执行时间",end - start)
    print("降维后算法预测准确率", precison)




if __name__ == "__main__":
    main()