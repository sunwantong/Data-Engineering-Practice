from com.sun.shopPos.config import *
import pandas as pd
import numpy as np
import os as os


def load_data_set():
    shop_info_df = pd.read_csv(ccf_shop_info,sep=",")
    user_shop_df = pd.read_csv(ccf_user_shop_behavior,sep=",")
    test_set = pd.read_csv(ccf_evaluation_public_set,sep=",")
    return shop_info_df,user_shop_df,test_set


def labeled_data_process(test_info_df, shop_info_df):
    mall_list = shop_info_df['mall_id'].drop_duplicates()
    mall_list = list(mall_list)
    outer = []
    inner = []
    i = 0
    for mall in mall_list:
        i += 1
        shop_info = shop_info_df[shop_info_df.mall_id == mall].reset_index(drop=True)
        test = test_info_df[test_info_df.mall_id == mall].reset_index(drop=True)

        shop_id_list = shop_info['shop_id'].drop_duplicates()
        shop_id_list = list(shop_id_list)

        j = 0
        for line in test.values:
            for shop_id in shop_id_list:
                if j == 0:
                    inner.append(1)
                    inner.append(line[0])
                    inner.append(line[1])
                    inner.append(shop_id)
                    inner.append(line[2])
                    inner.append(line[3])
                    inner.append(line[4])
                    inner.append(line[5])
                    inner.append(line[-1])
                    outer.append(inner)
                    inner = []
                else:
                    inner.append(0)
                    inner.append(line[0])
                    inner.append(line[1])
                    inner.append(shop_id)
                    inner.append(line[2])
                    inner.append(line[3])
                    inner.append(line[4])
                    inner.append(line[5])
                    inner.append(line[-1])
                    outer.append(inner)
                    inner = []
                j += 1
            j = 0

        i = str(i)
        column = ['label','row_id', 'user_id', 'shop_id','mall_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos']
        pd.DataFrame(outer, columns=column).to_csv("data/test/test"+i+".csv", index=None)
        i = int(i)
        print(i)
        outer = []
        # break



def get_max_signal(s):
    s = s.split(";")
    string = ''
    signal = {}
    top_max_signal = []
    for ele in s:
        ele = str(ele)
        s = ele.split("|")
        signal[s[0]] = s[1]
    # reverse = true,降序,反之
    wifi_infos = sorted(signal.items(), key=lambda x: x[1], reverse=False)
    wifi = wifi_infos[0:4]
    # print(wifi)
    str_len = len(wifi)
    i = 0
    for x in wifi:
        if i == str_len-1:
            string = string + x[0]
        else:
            string = string + x[0] + ";"
        i += 1
    return string

def get_max_joined_wifi_signal(s):
    s = s.split(":")
    signal = []
    for ele in s:
        ele = int(ele)
        signal.append(ele)
    sorted_signal = sorted(signal,reverse=True)
    print(sorted_signal[0:4])
    return sorted_signal[0:4]


def get_max_signal_wifi_id(s):
    s = s.split(";")
    signal = {}
    for ele in s:
        ele = str(ele)
        s = ele.split("|")
        signal[s[0]] = s[1]
    # reverse = true,降序,反之
    wifi_infos = sorted(signal.items(), key=lambda x: x[1], reverse=False)
    return wifi_infos[0][0]

def filter_labeled_data(filepath,i):
    path = 'data/test/'
    labeled_data = pd.read_csv(path+filepath, sep=",")
    filter_rule = pd.read_csv(ccf_filter_rule, sep=",")

    labeled_data['max_wifi_id'] = labeled_data['wifi_infos'].apply(get_max_signal_wifi_id)
    labeled_data['xx'] = labeled_data['shop_id'] + ":" + labeled_data['max_wifi_id']


    filter_rule['yy'] = filter_rule['shop_id'] + ":" + filter_rule['mwifi_id']

    filter_rule = filter_rule[filter_rule.counts > 30]
    aa = set(list(labeled_data['xx'])) & (set(list(filter_rule['yy'])))

    shop_id_list = []
    for ele in aa:
        s = ele.split(":")
        shop_id_list.append(s[0])

    labeled_data = labeled_data[labeled_data['shop_id'].isin(shop_id_list)]

    i = str(i)
    labeled_data.to_csv("data/test_filter/test"+i+".csv",index=None)
    i = int(i)
    print(i)



#遍历文件进行过滤
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    i = 0
    for allDir in pathDir:
        i += 1
        filter_labeled_data(allDir,i)
        # break


def main():
    # shop_info_df, user_shop_df, test_set = load_data_set()
    # labeled_data_process(test_set,shop_info_df)
    eachFile('data/test')


if __name__ == '__main__':
    main()