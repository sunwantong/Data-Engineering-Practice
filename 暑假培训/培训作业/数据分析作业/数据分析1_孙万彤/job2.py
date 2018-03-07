from com.sun.tong.task1.job1 import *


"""
   实现2种噪声数据过滤 
     1.等频装箱平滑
     2.等宽装箱平滑
   缺失值补全方法
     1.均值补全
     2.启发式补全
"""


#等宽平滑
def equal_bins():
    nums = [7,9,11,5,6,12,3,2,13,14,18,20,22]
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
    # [2, 3, 5, 6, 7, 9, 11, 12]

    for x in range(real_countbin):
        if var <= lens - 1:
            m = sorted_nums[var] + n
            while var <= lens - 1 and sorted_nums[var] <= m:
                list.append(sorted_nums[var])
                var += 1
            print(list,"中位数光滑之后为:%d" % np.median(list))
            list = []




#等频平滑
def equal_frequency():
    nums = [47,81,21,74,78,98,94,56,56,78,94,65,1,23,29,78,26]
    lens = len(nums)
    #假定每个箱子里边的数量是4个
    n = 5
    sortnums = sorted(nums)
    #箱子数量  判断余数是否大于零点五，以便于决定加一后四舍五入或者直接四舍五入
    bin_num = lens / n
    remain = lens % n
    if remain > 0:
        bin_num += 1
    i = 0
    for x in range(int(bin_num)):
        print(sortnums[i:i+n])
        print("第%d个箱子的均值光滑后的数值为:%.2f" % (x+1,np.mean(sortnums[i:i+4])))
        i += n

#均值补全
def missvalue_process1():
    #0表示缺失值,用均值代替缺失数值
    nums = [84,45,12,37,89,45,13,21,0,65,48,7]
    mean = cal_mean(*nums)
    nums[8] = round(mean,2)
    return nums

#启发式的补全
def missvalue_process2():
    #1代表男生 0代表女生  最后两个数值缺失
    sex_height = [[0,1,0,1,0,1,0,1],[161,175,155,174,163,180,0,0]]
    #首先求出有数值的男生身高
    sexs = sex_height[0]
    heights = sex_height[1]
    location_man = []
    location_women = []
    for i in range(len(sexs) - 2):
        if sexs[i] == 1:
            location_man.append(i)
        if sexs[i] == 0:
            location_women.append(i)
    #print("男的为%s,女的为:%s" % (location_man,location_women))
    #统计男生身高
    hei_man = 0
    hei_women = 0
    for j in location_man:
        hei_man += heights[j]
    for m in location_women:
        hei_women += heights[m]
    #男生平均身高，保留2位小数
    man_mean = round(hei_man / len(location_man), 2)
    #女生平均身高
    women_mean = round(hei_women / len(location_women),2)
    heights[7] = man_mean
    heights[6] = women_mean
    return heights



#主函数
if __name__ == "__main__":
    # print(missvalue_process1())
    # print(missvalue_process2())
    print(equal_bins())
    # print(equal_frequency())