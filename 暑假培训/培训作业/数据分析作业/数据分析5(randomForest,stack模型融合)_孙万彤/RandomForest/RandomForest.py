import numpy as np
import random as rd
import math as m
import pandas as pd
import sklearn.tree as tree


"""
  随机森林算法的实现
"""



#数据集和训练集的加载
def loadDataSet(trainFileName,testFileName):
    trainDataSet = pd.read_csv(trainFileName,header=None,sep=",")
    testDataSet = pd.read_csv(testFileName,header=None,sep=",")
    return trainDataSet,testDataSet



#每次随机选取sqrt(k)个特征(k代表的是特征总数)，来训练模型
def generateMultiModel(dataSet):
    #模型数目
    modelNums = 10

    # 特征总数
    totalFeature = len(dataSet.columns) - 1
    # print("数目为:",totalFeature)

    # 每次选取的特征的数目,四舍五入做个处理
    featNums = round(m.sqrt(totalFeature))

    #用矩阵存储每次选取的特征的位置
    randomFeatMat = np.zeros((modelNums,featNums),dtype=int)

    #模型列表
    modelList = []

    for i in range(modelNums):
        ranLsit = rd.sample(range(0,totalFeature),featNums)
        for j in range(featNums):
            randomFeatMat[i,j] = ranLsit[j]

    tempList = []
    #依次循环训练出10个模型
    for i in range(modelNums):
    	#保存中间结果
        tempList = []
        for j in range(featNums):
            dataSet[randomFeatMat[i,j]]
            tempList.append(dataSet[randomFeatMat[i,j]])
        selectedDataset = np.mat(tempList).T
        labelSet = np.mat(dataSet[dataSet.columns[-1]]).T
        models = decisionTreeModel(selectedDataset,labelSet)
        modelList.append(models)

    #返回每次选择的随机特征子集和该特征子集训练出来的模型集合
    return randomFeatMat,modelList



"""
 用训练生成的模型对测试集进行测试，
 若是回归，则用均值法，若是分类，则用投票法
"""

def testMultiModelPerformance(modelList,randomFeatMat,testDataSet):

    dataSetBak = testDataSet.copy()

    #去除label标签之后的数据
    dataSetMat = np.mat(testDataSet[testDataSet.columns[:-1]])

    #预测数值的列表
    predValueList = []

    # 取得每一个测试集样本的真实标记
    actualLableSet = np.mat(testDataSet[testDataSet.columns[-1]]).T

    #测试样本总数
    sampleLen = len(testDataSet)
    tempList = []
    sums = 0
    #循环所有的模型，对每一个样本进行预测，循环完之后对预测结果取一个均值
    for rows in range(sampleLen):

        sums = 0
        for i in range(len(modelList)):
            tempList = []
            for ele in range(randomFeatMat.shape[1]):
                colums = randomFeatMat[i,ele]
                tempList.append(dataSetMat[rows,colums])
            tempSet = np.mat(tempList)
            result = modelList[i].predict(tempSet)
            sums += result
        #预测结果求一个均值
        tmpValue = sums / len(modelList)
        predValueList.append(tmpValue)
    #测试集的预测结果组成的矩阵为
    finalMatrix = np.hstack((np.mat(predValueList),actualLableSet))
    return finalMatrix



#均方误差评价回归性能
def evaluateIndex(dataSet):
    sums = 0
    row,column = dataSet.shape
    for i in range(row):
    	#求解误差总和
        sums += (dataSet[i,0] - dataSet[i,1])**2
     #返回均方误差
    return sums / row



#训练决策树模型
def decisionTreeModel(dataSet,label):
    mTree = tree.DecisionTreeRegressor()
    models = mTree.fit(dataSet,label)
    return models



"""
   1.用单模型对数据集建模，然后测试，
   2.和多模型进行比较
"""
def getSingleMode(trainDataSet):
    #数据集
    dataSet = np.mat(trainDataSet[trainDataSet.columns[:-1]])
    #标签集
    labelSet = np.mat(trainDataSet[trainDataSet.columns[-1]]).T

    #用决策树建模
    models = decisionTreeModel(dataSet,labelSet)
    return models



#单模型在测试集上的测试
def testSingleModel(testDataSet,model):
    #预测标记值
    predictValue = []
    #真实标记值
    actualValue = np.mat(testDataSet[testDataSet.columns[-1]]).T

    dataSet = np.mat(testDataSet[testDataSet.columns[:-1]])
    #训练获取模型对测试集的测试结果，并将它转换为矩阵
    for row in range(dataSet.shape[0]):
        value = model.predict(dataSet[row])
        predictValue.append(value)
    a = np.mat(predictValue)
    resutlMatrix = np.hstack((a,actualValue))
    #返回预测值和真实值组成的矩阵，以便于计算均方误差
    return resutlMatrix



def main():
    # 文件路径
    trainFilePath = "D://WeKaDataSet//Train.csv"
    testFilePath = "D://WeKaDataSet//Test.csv"

    # 返回训练集和测试集
    trainDataSet, testDataSet = loadDataSet(trainFilePath, testFilePath)

    trainData = trainDataSet.copy()
    testData = testDataSet.copy()

    # 训练单模型
    singleModel = getSingleMode(trainData)
    # 对单模型性能进行测试
    resultSingleMat = testSingleModel(testData, singleModel)

    # 训练多模型(随机森林)
    randomFeatMat, modelList = generateMultiModel(trainDataSet)
    # 测试多模型(随机森林)的好坏
    resultMatrix = testMultiModelPerformance(modelList, randomFeatMat, testDataSet)
    # 对测试结果进行评价(多模型均方误差)
    mse = evaluateIndex(resultMatrix)
    # 对测试结果进行评价(单模型均方误差)
    mseSingle = evaluateIndex(resultSingleMat)

    print("单决策树均方误差为:" + str(mseSingle))
    print("随机森林均方误差为:" + str(mse))


if __name__ == "__main__":
    main()
