from com.sun.shopPos.config import *
import pandas as pd
import numpy as np
import os as os


def load_data_set():
    shop_info_df = pd.read_csv(ccf_shop_info,sep=",")
    user_shop_df = pd.read_csv(ccf_user_shop_behavior,sep=",")
    test_set = pd.read_csv(ccf_evaluation_public_set,sep=",")
    return shop_info_df,user_shop_df,test_set


def labeled_data_process(shop_info_df, user_shop_df):
    user_shop_mall_set = pd.merge(user_shop_df, shop_info_df[['shop_id', 'mall_id']], on="shop_id", how="left")

    mall_list = shop_info_df['mall_id'].drop_duplicates()
    mall_list = list(mall_list)
    outer = []
    inner = []
    i = 0
    for mall in mall_list:
        i += 1
        train = user_shop_mall_set[user_shop_mall_set.mall_id == mall].reset_index(drop=True)

        train['label'] = -10
        shop_id_list = train['shop_id'].drop_duplicates()

        shop_id_list = list(shop_id_list)

        for line in train.values:
            for shop_id in shop_id_list:
                if line[1] == shop_id:
                    line[-1] = 1
                    inner.append(line[0])
                    inner.append(line[1])
                    inner.append(line[2])
                    inner.append(line[3])
                    inner.append(line[4])
                    inner.append(line[5])
                    inner.append(line[-2])
                    inner.append(line[-1])
                else:
                    line[-1] = 0
                    a = line[1]
                    a = shop_id
                    inner.append(line[0])
                    inner.append(a)
                    inner.append(line[2])
                    inner.append(line[3])
                    inner.append(line[4])
                    inner.append(line[5])
                    inner.append(line[-2])
                    inner.append(line[-1])
                outer.append(inner)
                inner = []
        i = str(i)
        column = ['user_id', 'shop_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos', 'mall_id','label']
        # a = pd.DataFrame(outer, columns=column)
        # print(a['label'].value_counts())
        pd.DataFrame(outer,columns=column).to_csv("data/train/train"+i+".csv",index=None)
        i = int(i)
        print(i)
        outer = []
        # break


def get_virtual_max_signal(s):
    s = s.split(";")
    signal = {}
    for ele in s:
        ele = str(ele)
        s = ele.split("|")
        signal[s[0]] = s[1]
    # reverse = true,降序,反之
    wifi_infos = sorted(signal.items(), key=lambda x: x[1], reverse=False)
    # print(wifi_infos)
    return wifi_infos[0][0]

def get_max_joined_wifi_signal(s):
    s = s.split(":")
    signal = []
    for ele in s:
        ele = int(ele)
        signal.append(ele)
    sorted_signal = sorted(signal,reverse=True)
    return sorted_signal[0]

#过滤打完标之后的数据
def filter_labeled_data(filepath,i):
    path = 'data/train/'
    labeled_data = pd.read_csv(path+filepath,sep=",")

    labeled_data['max_wifi_id'] = labeled_data['wifi_infos'].apply(get_virtual_max_signal)
    labeled_data['labeled_join'] = labeled_data['shop_id'] + ":" + labeled_data['max_wifi_id']
    data1 = labeled_data[labeled_data.label == 1]
    data2 = labeled_data[labeled_data.label == 0]

    filter_rule = pd.read_csv(ccf_filter_rule, sep=",")
    filter_rule['filt_joined'] = filter_rule['shop_id'] + ":" + filter_rule['mwifi_id']
    filter_rule = filter_rule[filter_rule.counts > 100]

    intersetctons = set(list(data2['labeled_join'])) & (set(list(filter_rule['filt_joined'])))

    shop_id_list = []
    for ele in intersetctons:
        s = ele.split(":")
        shop_id_list.append(s[0])

    data2 = data2[data2['shop_id'].isin(shop_id_list)]
    train = data1.append(data2)

    # a = train['label'].value_counts()
    # print(a)
    #
    i = str(i)
    train.to_csv("data/train_filter/train"+i+".csv", index=None)
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
#
def main():
    # shop_info_df, user_shop_df, test_set = load_data_set()
    # labeled_data_process(shop_info_df,user_shop_df)
    eachFile('data/train')




if __name__ == '__main__':
    main()