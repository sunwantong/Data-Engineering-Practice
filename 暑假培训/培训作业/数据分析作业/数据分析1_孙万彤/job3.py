from com.sun.tong.task1.job1 import *


"""
  实现2种数据离散化、数据数值化、数据归一化方法
"""

# 数据归一化的方法：
"""
一、min-max标准化（Min-Max Normalization）

也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1]之间。转换函数如下：

    x_new = (x - min)/(max - min)
二、Z-score标准化方法

这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
   经过处理的数据符合标准正态分布，即均值为0，标准差为1，转化函数为：

   x_new = (x - mean) / variance
"""

#z-score标准化方法
def data_noraml1():
    nums = range(1, 21)
    mean = cal_mean(*nums)
    variance = cal_var(*nums)
    new_data = []
    for i in nums:
        value = (i - mean) / variance
        new_data.append(round(value, 2))
    return new_data

#min-max标准化方法
def data_noraml2():
    nums = range(1, 21)
    mean = cal_mean(*nums)
    variance = cal_var(*nums)
    min = np.max(nums)
    max = np.min(nums)
    max_min = max - min
    new_data = []
    for i in nums:
        value = (i - min) / max_min
        realValue = float(value)
        new_data.append(round(realValue, 3))
    return new_data


# 数据离散化的方法
"""
  数据离散化的方法(连续型数据离散化)
    1.等箱法  每个箱子的宽度一样
    2.等频法  每个箱子里边的数量是一样的
"""


def equal_bins():
    nums = [14,9,11,5,6,12,13,2,13,19,18,20,22]
    sorted_nums = sorted(nums)
    lens = len(sorted_nums)
    # n 表示每个箱子的宽度
    n = 3
    # 箱子的数量
    count_bin = (sorted_nums[lens - 1] - sorted_nums[0]) / n
    real_countbin = int(count_bin + 1)
    i = 0
    var = 0
    m = 0
    list = []
    for x in range(real_countbin):
        if var <= lens - 1:
            m = sorted_nums[var] + n
            while var <= lens - 1 and sorted_nums[var] <= m:
                list.append(sorted_nums[var])
                var += 1
            print(list)
            list = []




#round  四舍五入的意思
def equal_frequency():
    nums = [17,81,21,44,78,98,94,56,56,78,94,45,1,23,29,78,26]
    lens = len(nums)
    #假定每个箱子里边的数量是4个
    n = 4
    sortnums = sorted(nums)
    #箱子数量  判断余数是否大于零点五，以便于决定加一后四舍五入或者直接四舍五入
    bin_num = lens / n
    remain = lens % n
    if remain > 0:
        bin_num += 1
    i = 0
    for x in range(int(bin_num)):
        print(sortnums[i:i+n])
        print("第%d个箱子的类标记为:%.2f" % (x+1,np.mean(sortnums[i:i+4])))
        i += n


#数据数值化
"""
 数据数值化
  1.one-hot编码
  2.排序编码
"""


#one-hot编码
def data2numerical1():
    nums = []
    i = 0
    m = 0
    weather = ["rainy","sunny","snowy","windy","cold"]
    for value in weather:  #控制行
        while m < len(weather):  #控制列，
            if value == weather[i]:
                nums.append(1)
            else:
                nums.append(0)
            m += 1
            i += 1
        m = 0
        i = 0
        print(value+"的one-hot编码如下:%s" % nums)
        nums = []

    
#排序编码
def data2numerical2():
    #基本的类型
    ball_basic = ["small","median","big","large"]
    #数字标记集合
    ball_type = []
    balls = ["small","median","big","large","small","median","large","big"]
    i = 1
    loc = 0
    nlist = []
    for value in ball_basic:
        if i <= len(ball_basic):
            ball_type.append(i)
            i += 1
    for x in balls:
        for m in range(len(ball_basic)):
            if x == ball_basic[m]:
                nlist.append(ball_type[m])
    return nlist

#主函数
if __name__ == "__main__":
    # print(equal_frequency())
    # print(equal_bins())

    # print(data_noraml1())
    print(data_noraml2())


    # print(data2numerical1())
    # print(data2numerical2())