import numpy as np
import random as rd
import math as m


"""
  1.数值型数列均值，方差，以及分位数的计算
  2.实现2种噪声数据过滤和缺失值补全方法
  3.实现2种数据离散化、数据数值化、 数据归一化方法
  
       
"""

def cal_mean(*nums):
    sum = 0
    for i in nums:
        sum += i
    mean = sum / len(nums)
    return mean


def cal_var(*nums):
    mean = cal_mean(*nums)
    sumvar = 0
    for j in nums:
        sumvar += (j - mean) ** 2
    var = sumvar / len(nums)
    return var

#计算分位数(四分位)
def cal_quantile(n,*nums):
    #nums = range(1,21)
    #分位数的个数
    #n = 3

    #从列表上里边随机抽取ran_count个数
    ran_count = 10
    if ran_count < len(nums):
        samlist = rd.sample(list(nums), ran_count)
    else:
        return
    #对随机数进行排序
    sortlist = np.sort(samlist)
    lens = len(sortlist)
    #每一个区间的长度
    part = (sortlist[lens - 1] - sortlist[0]) / n
    i = 1
    x = 0
    list_quantile = []
    x += sortlist[0] + part
    list_quantile.append(x)
    for i in range(1,n-1):
        x += part
        list_quantile.append(x)
    return list_quantile



#主函数
if __name__ == "__main__":
    nums = range(1, 21)
    # print(cal_mean(*nums))
    # print(cal_var(*nums))
    print(cal_quantile(3,*nums))