from com.sun.tong.task1.job1 import *



"""
  实现两种相似度计算方法
   1.数值型(欧氏距离)  

   2.标称型 𝑑(𝒙,𝒙′)=(𝑝−𝑚)/𝑝
"""
#.数值型相似度实现(欧式距离)
def numercial_type():
    nums_one = [7,87,41,32,17,8,97,45,12,31,48,4]
    nums_two = [13,6,9,7,54,8,41,23,1,64,54,5]
    #对数据做一个归一化处理
    normalData_one = data_noraml2(*nums_one)
    normalData_two = data_noraml2(*nums_two)
    lens = len(nums_one)
    dis = [normalData_one[i]-normalData_two[i] for i in range(lens)]
    sum = 0
    for x in dis:
        sum += x**2
    return round(m.sqrt(sum),2)

#.标称型相似度实现 𝑑(𝒙,𝒙′)=(𝑝−𝑚)/𝑝
def nominal_type():
    type_one = [8,4,6,51,23,12,31]
    type_two = [31,8,41,23,12,3,4]
    lens = len(type_one)
    sim_counts = 0
    for i in type_one:
        for j in type_two:
            if i == j:
                sim_counts += 1
    dis = (lens - sim_counts) / lens
    return round(dis,2)

#min-max标准化方法
def data_noraml2(*nums):
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

#主函数
if __name__ == "__main__":
    print(numercial_type())
    # print(nominal_type())


















