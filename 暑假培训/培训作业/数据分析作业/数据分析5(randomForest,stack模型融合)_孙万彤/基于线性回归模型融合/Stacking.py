import numpy as np
import random as rd
import math as m
import pandas as pd
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict

"""
 基于线性回归的模型融合算法的实现
"""

def loadDataSet(trainFilePath,testFilePath):
    #读取训练集
    trainDataSet = pd.read_csv(trainFilePath,header=None,sep=",")
    #读取测试集
    testDataSet = pd.read_csv(testFilePath, header=None, sep=",")
    return trainDataSet,testDataSet


"""
 1.用训练集训练出k个弱学习器,并且返回
 2.用训练集交叉预测,将预测结果与原来训练集特征组成，形成新的数据集
 3.这个新的数据集就是次级学习器的训练集
"""
def generateMuitiModel(trainDataSet):
    #交叉验证预测数值
    predValue = []
    #存放训练好的模型
    modelList = []
    trainDataSetBak = trainDataSet.copy()
    #取出不包括label项的所有数据
    dataSet = np.mat(trainDataSet[trainDataSet.columns[:-1]])

    #取出label这一列的数据
    labelSet = np.mat(trainDataSet[trainDataSet.columns[-1]].values.tolist()).T
    #训练出9个模型
    for i in range(3):
        model,algri = decisionTree(i+3,dataSet,labelSet)
        modelList.append(model)
        #五折交叉预测
        trainPredValue = cross_val_predict(algri, dataSet, labelSet, cv=5)
        predValue.append(trainPredValue)

        model,algri = bayesianRidge(300+i,dataSet,labelSet)
        modelList.append(model)
        #五折交叉预测
        trainPredValue = cross_val_predict(algri, dataSet, labelSet, cv=5)
        predValue.append(trainPredValue)

        model,algri = svmAlgrith((i / 10)+0.1,dataSet,labelSet)
        modelList.append(model)
        #五折交叉预测
        trainPredValue = cross_val_predict(algri, dataSet, labelSet, cv=5)
        predValue.append(trainPredValue)
    tmpMat = np.mat(predValue).T
    #次级学习器的训练集
    secondTrainData = np.hstack((dataSet,tmpMat,labelSet))
    return modelList,secondTrainData


#决策树回归
def decisionTree(maxDeep,dataSet,label):
    # 设置决策树的层次
    mTree = tree.DecisionTreeRegressor(max_depth=maxDeep)
    models = mTree.fit(dataSet,label)
    return models,mTree

#svm回归
def svmAlgrith(epsiValue,dataSet,label):
    # kernel 算法中指定要使用的内核模型
    #epsilon 惩罚项，默认值为0.1
    clf = svm.SVR(kernel="rbf",epsilon=epsiValue)
    models = clf.fit(dataSet,label)
    return models,clf


#贝叶斯岭回归
def bayesianRidge(nIterValue,dataSet,label):
    #最大迭代次数。默认值为300。
    reg = linear_model.BayesianRidge(n_iter=nIterValue)
    models = reg.fit(dataSet,label)
    return models,reg


"""
 1.通过训练好的初级学习器对测试集进行预测
 2.将预测值当做新的特征，形成新的数据集(次级学习器的测试集)
 3.新形成的数据集的标签还是原来测试集的标签
"""
def generateNewTestData(modelList,testDataSet):
    #产生除label的数据集
    dataSet = np.mat(testDataSet[testDataSet.columns[:-1]])
    #label集
    labelSet = np.mat(testDataSet[testDataSet.columns[-1]]).T
    #存放预测值的集合
    predictValue = []
    rows,columns = dataSet.shape
    for row in range(rows):
        featValue = dataSet[row]
        tempList = []
        for model in modelList:
            values = model.predict(featValue)
            tempList.append(values[0])
        predictValue.append(tempList)
    resutlMat = np.mat(predictValue)
    #原来特征+预测值+标签  形成新的数据集(是次级学习器的测试集)
    secondTestData = np.hstack((dataSet,resutlMat,labelSet))
    return secondTestData,np.mat(np.hstack((resutlMat,labelSet)))




#训练次级学习器(线性回归模型)，并且用次级测试集测试，然后返回测试结果
def generateSecondLearn(secondTrainData,secondTestData):
    #产生除label以外的数据集
    dataSet = secondTrainData[:,:-1]

    #label集
    labelSet = secondTrainData[:,-1]
    #线性回归模型
    reg = linear_model.LinearRegression()
    modelLR = reg.fit(dataSet, labelSet)

    #次级测试集的真实标记
    labset = secondTestData[:,-1]
    # 次级测试集除label以外的数据
    testData = secondTestData[:,:-1]
    #次级测试集的预测标记
    predValue = modelLR.predict(testData)

    #返回次级学习器对次级测试集的预测值和其真实值组成的矩阵
    secondResMat = np.hstack((labset,predValue))
    return secondResMat



#用均方误差评价融合之后模型的回归性能
def evaluateIndex(dataSet):
    sums = 0
    row,column = dataSet.shape
    for i in range(row):
        sums += (dataSet[i,0] - dataSet[i,1])**2
    return sums / row


#均方误差评价9个单模型的回归性能
def evalSingModeMse(dataSet):
    # 9个模型的预测结果+测试集的真实标签
    mseList = []
    rows,columns = dataSet.shape
    label = dataSet[:,-1]
    for col in range(columns-1):
        tmpValue = evaluateIndex(np.hstack((dataSet[:,col],label)))
        mseList.append(tmpValue)
    #返回均方误差的集合
    return mseList


def main():
    # 文件路径
    trainFilePath = "D://WeKaDataSet//Train.csv"
    testFilePath = "D://WeKaDataSet//Test.csv"
    # 数据集
    trainDataSet, testDataSet = loadDataSet(trainFilePath, testFilePath)
    # 返回多个模型和次级学习器的训练集
    modelList, secondTrainData = generateMuitiModel(trainDataSet)
    # 返回次级学习器的测试集 和每个模型对测试集的测试结果和真实结果的矩阵
    secondTestData, testAndAcutalMat = generateNewTestData(modelList, testDataSet)
    # 然后用新的训练集训练一个次级学习器(LR),并返回对次级测试集的测试结果
    secondResMat = generateSecondLearn(secondTrainData, secondTestData)
    # 次级学习器的均方误差
    secondMSE = evaluateIndex(secondResMat)
    # 初级多个单模型学习器的均方误差
    singleModelMSE = evalSingModeMse(testAndAcutalMat)
    print("融合线性回归模型的均方误差为:\n", secondMSE)
    print("\n\n")
    print("9个单模型的均方误差分别为:\n", np.mat(singleModelMSE).T)


if __name__ == "__main__":
    main()
