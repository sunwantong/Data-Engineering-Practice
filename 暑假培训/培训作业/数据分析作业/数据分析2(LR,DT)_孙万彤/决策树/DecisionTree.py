import numpy as np
import random as rd
import math as m


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
    for featVec in dataSet:
        currentLabel = featVec[-1]
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
    bestFeature = -1
    #遍历所有的特征值
    for i in range(attrNums):
        listFeature = [data[i] for data in dataSet]
        #去除重复值
        uniqueVals = set(listFeature)

        tempEntropy = 0.0
        infoIV = 0.0
        for value in uniqueVals:
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
            bestFeature = i
    return bestFeature




#特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCount(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



#获得同一属性的集合
def getSubDataSet(dataSet, pos, value):
    restDataSet = []
    for featVec in dataSet:
        if featVec[pos] == value:
            #去除pos位置的属性
            reducedFeatVec = featVec[:pos]
            reducedFeatVec.extend(featVec[pos+1:])
            restDataSet.append(reducedFeatVec)
    return restDataSet



#创建树
def createTree(dataSet,labels):
    #数据集类标记信息的集合 ['是', '是', '是', '是', '是', '否', '否', '否', '否', '否']
    classArray = [data[-1] for data in dataSet]
    if classArray.count(classArray[0]) == len(classArray):
        return classArray[0]

    if len(dataSet[0]) == 1:
        return majorityCount(classArray)

    #bestFeat返回字段在所在行的位置
    bestFeat = getBestSingleEntryDataSet(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    subLabels = labels[:]
    subLabels.pop(bestFeat)
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        #程序递归生成子数
        myTree[bestFeatLabel][value] = createTree(getSubDataSet(dataSet, bestFeat, value), subLabels)
    return myTree  #生成的树




#遍历树
def classify(inputTree, featLabels, testVec):
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    classLabel = ''
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel




def main():
    # 加载训练集
    dataSet = loadDataSet()
    #加载测试集
    testDataSet = loadTestDataSet()

    #拿到数据表第一行的各个字段
    labels = dataSet[1]
    #根据训练集建立决策树
    myTree = createTree(dataSet[2],labels)

    print("生成的决策树为:%s" % myTree)

    # labels = ['色泽','根蒂','敲声','纹理','脐部','触感']

    #对测试集进行预测得到评估方法
    p,r,f = getEvaluateTarget(labels, myTree, testDataSet)

    # getEvaluateTarget(labels, myTree, testDataSet)
    # print("yesPrecision: %f" % p)
    # print("yesRecall: %f" % r)
    # print("yesF1: %f" % f)



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
    yesPresion = TP / (FP + TP)
    yesRecall = TP / (TP + FN)
    yesF1 = (2*yesPresion*yesRecall) / (yesPresion+yesRecall)

    noPrecision = float(TN) / (FN + TN)
    noRecall = float(TN) / (TN + FP)
    noF1 = (2*noPrecision*noRecall) / (noPrecision+noRecall)
    print("Precision: %f" % noPrecision)
    print("Recall: %f" % noRecall)
    print("F1: %f\n" % noF1)
    return yesPresion,yesRecall,yesF1



if __name__ == "__main__":
    main()