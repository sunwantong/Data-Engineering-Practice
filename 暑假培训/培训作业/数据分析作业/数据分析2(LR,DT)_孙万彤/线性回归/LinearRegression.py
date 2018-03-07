import numpy as np
import math as m
import matplotlib.pyplot as plt

"""
  线性回归
    1.最小二乘法
    2.梯度下降法
    3.x^代表x的转置矩阵的意思

"""

#读取本地数据源,然后进行数据预处理
def getDataSource():
    f = open("D://数据处理数据源//LR.txt", 'r')
    lines = f.readlines()
    train_x = []
    train_y = []
    #x矩阵的原始值，不添加1
    trainx_init = []
    trainy_init = []
    train_matri_y = []
    for line in lines:
        line = line.strip().split(",")
        #print(line)
        listTemp = []
        listTemp.append(int(line[0]))
        # print(line[0])
        trainx_init.append(int(line[0]))
        listTemp.append(int(line[1]))
        train_x.append(listTemp)


        train_y.append(int(line[2]))
        trainy_init.append(int(line[2]))
        # print(train_y)
        train_matri_y.append(train_y)
        train_y = []
    return train_matri_y,train_x,trainx_init



#求x^*x的乘积（x的转置和x的乘积）
def getXAndXMulti():
    trainSet = getDataSource()
    train_x = trainSet[1]
    train_y = trainSet[0]
    #x的转置矩阵
    transx_matrix = trans(train_x)
    transx_matrix = list(transx_matrix)

    #x和x转置矩阵的乘积
    finalMulti = getMatrixMuliply(transx_matrix,train_x)
    return list(finalMulti),transx_matrix,train_y



#x*x^的逆矩阵和x^再和y相乘
def getLrParam():
    dataSet = getXAndXMulti()
    #对乘积之后的数据源求逆矩阵（求x*x^的逆矩阵）
    sets = inverseMatrix(dataSet[0])
    trans_x = dataSet[1]
    train_y = dataSet[2]
    #x*x^的逆矩阵和x^相乘后最后和y相乘
    return getMatrixMuliply(getMatrixMuliply(sets, trans_x),train_y)




#求转置矩阵
def trans(array):
    valueMatrix = []
    spec = []
    m = 0
    for i in range(len(array[0])):
        for j in range(len(array)):
            valueMatrix.append(array[j][i])
            if len(valueMatrix) == len(array):
                spec.append(valueMatrix)
                valueMatrix = []
            m += 1
    return spec



#求矩阵和矩阵的乘积
def getMatrixMuliply(traina,trainb):
    transssb = trans(trainb)
    matrixvalue = []
    templist = []
    sum = 0
    for x in traina:
        for y in transssb:
            sum = x[0]*y[0] + x[1]*y[1]
            templist.append(sum)
            if len(templist) == len(trainb[0]):
                matrixvalue.append(templist)
                templist = []
    return matrixvalue





# 求矩阵的伴随矩阵
def getAdjointMatrix(nums):
    matLen = len(nums)
    # 放置求完之后的伴随矩阵
    adjoinMarix = []
    # 记录当前这个元素的列号
    colums = 0
    tempAdjoin = []
    if matLen > 0:
        for k in range(matLen):
            for i in range(matLen):  #
                # 记录当前这个元素的行号
                rows = i
                temp = []
                newMat = []
                for j in range(matLen):
                    if j != rows:  # 把同一行去掉
                        # newMat.append(nums[j])
                        for m in range(len(nums[j])):
                            if m != colums:  # 把同一列去掉
                                temp.append(nums[j][m])
                                if len(temp) == matLen - 1:  # 因为求代数余子式，所以减1,如果求伴随，则不减1
                                    newMat.append(temp)
                                    temp = []
                # 再次递归求代数余子式的值
                yuZiShi = ((-1) ** ((i + 1) + (colums + 1))) * getDetValue(newMat)

                tempAdjoin.append(yuZiShi)
                if len(tempAdjoin) == matLen:
                    adjoinMarix.append(tempAdjoin)
                    tempAdjoin = []
            colums += 1
        return adjoinMarix  # [15, -7, -2, -7, 3, 1, -2, 1, 0]





#求行列式的值,默认按第一列展开
def getDetValue(nums):
    result = isinstance(nums, int)
    if result: return nums
    # print(nums[0])
    matLen = len(nums)
    if matLen == 1:
        if len(nums[0]) == 1:
            return nums[0][0]
        else:
            return nums[0]
    sum = 0.0
    #程序递归出口
    if matLen == 2:
        return nums[0][0]*nums[1][1] - nums[0][1]*nums[1][0]
    if matLen > 2:

        for i in range(matLen):  #按第一列展开
            #记录当前这个元素的行号和列号
            colums = 0
            rows = i
            newMat = []
            temp = []
            #这个for循环就是为了生成余子式
            for j in range(matLen):
                if j != rows:
                    #newMat.append(nums[j])
                    for m in range(len(nums[j])):
                        if m != colums:
                            temp.append(nums[j][m])
                            if len(temp) == matLen - 1:
                                newMat.append(temp)
                                temp = []
            # print(newMat)
            sum += nums[i][0]*((-1)**((i+1)+1))*getDetValue(newMat)
        return sum


#求逆矩阵(利用行变换求逆矩阵 返回求得的逆矩阵)
def inverseMatrix(trainSet=[]):
    # print(trainSet)
    # trainSet = plusParam(trainSet)
    #求得行列式的数值
    hangValue = getDetValue(trainSet)
    #判断行列式是否可逆,如果行列式的值不为零则可逆
    if hangValue == 0.0:
        return
    #求伴随矩阵
    banSuiSet = getAdjointMatrix(trainSet)
    #计算逆矩阵
    banSuiLen = len(banSuiSet)
    for i in range(banSuiLen):
        for j in range(banSuiLen):
            temp = banSuiSet[i][j] / hangValue
            banSuiSet[i][j] = float(temp)
    # print(banSuiSet)
    #banSuiSet已经除去行列式的数值了
    inverMatrix = banSuiSet

    #返回逆矩阵
    return inverMatrix




def drawImages():
    #x和y的实际数值
    w = getLrParam()
    nomimal = getDataSource()
    real_x = nomimal[1]
    real_y = nomimal[0]

    real_x_init = nomimal[2]
    #y是m行1列
    predit_y = getMatrixMuliply(real_x,w)
    a = float(w[0][0])
    b = float(w[1][0])
    a = round(a,4)
    b = round(b,3)
    print("最终的回归直线为:y = %s*x + %s" % (a,b))

    # 画回归直线y = w0*x+w1
    #真实值
    plt.plot(real_x_init, real_y,"*")
    #预测数值
    plt.plot(real_x_init,predit_y,c='m')
    plt.show()






def main():
    print(drawImages())


if __name__ == "__main__":
    main()






