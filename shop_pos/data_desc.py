#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/1 0001 10:34
# @Author  : Sun
# @File    : data_desc.py
from com.sun.shopPos.config import *
import pandas as pd






#分割测试集
def split_evaluate_set(evaluation_public):
    evaluate_A = evaluation_public[(evaluation_public.time_stamp > "2017-09-01 00:00")&
                      (evaluation_public.time_stamp < "2017-09-08 00:00")]
    a = evaluate_A['time_stamp'].min()
    b = evaluate_A['time_stamp'].max()
    print(a,b)
    evaluate_A.to_csv("data/a_rank.csv",index=None)



#找到最强wifi的id,和shop_id相关联
def get_max_signal(s):
    s = s.split(";")
    signal = {}
    for ele in s:
        ele = str(ele)
        s = ele.split("|")
        signal[s[0]] = s[1]

    #reverse = true,降序,反之
    wifi_infos = sorted(signal.items(),key=lambda x:x[1],reverse=False)
    # print(wifi_infos)
    return wifi_infos[0][0]



#wifi和商铺的关联规则
def get_wifi_with_shop(user_shop_behavior):
    user_shop_behavior['max_wifi_id'] = user_shop_behavior["wifi_infos"].apply(get_max_signal)

    return user_shop_behavior

def load_data_set():
    shop_info_df = pd.read_csv(ccf_shop_info, sep=",")
    user_shop_behavior_df = pd.read_csv(ccf_user_shop_behavior, sep=",")
    evaluation_public = pd.read_csv(ccf_evaluation_public_set,sep=",")

    return shop_info_df,user_shop_behavior_df,evaluation_public


def get_eval_wifis(s):
    s = s.split(";")
    signal = {}
    for ele in s:
        ele = str(ele)
        s = ele.split("|")
        signal[s[0]] = s[1]

    # reverse = true,降序,反之
    wifi_infos = sorted(signal.items(), key=lambda x: x[1], reverse=False)
    return wifi_infos[0][0]

def pred_evaluate_set(user_shop_behavior_related):
    evaluate_A = pd.read_csv(ccf_evaluate_A, sep=",")
    evaluate_A = evaluate_A[['row_id','wifi_infos']]
    evaluate_A['max_wifi_id_eval'] = evaluate_A['wifi_infos'].apply(get_eval_wifis)
    evaluate_A = evaluate_A[['row_id','max_wifi_id_eval']]

    # result = pd.concat([evaluate_A,user_shop_behavior_related],axis=1)
    # result.to_csv("d://new.csv",index=None)
    # result[""]
    row_shop_infos = {}

    for evals in evaluate_A.values:   # row_id,max_wifi_eval
        for real in user_shop_behavior_related.values:  # shop_id,max_wifi
            if evals[1] == real[1]:
                row_shop_infos[evals[0]] = real[0]

    df = pd.DataFrame(list(row_shop_infos.items()), columns=['row_id', 'shop_id'])
    df.to_csv("d://aa.csv")


def main():
    shop_info_set,user_shop_behavior_set,evaluation_public = load_data_set()
    # split_evaluate_set(evaluation_public)
    user_shop_behavior_related = get_wifi_with_shop(user_shop_behavior_set)
    print(user_shop_behavior_related['shop_id'].drop_duplicates().count())
    user_shop_behavior_related = user_shop_behavior_related[['shop_id','max_wifi_id','user_id']]
    user_shop_behavior_related.to_csv("data/test3.csv",index=None)
    # pred_evaluate_set(user_shop_behavior_related)
    # a = user_shop_behavior_related.groupby(['shop_id','max_wifi_id']).agg("count").reset_index()
    # a = a[['shop_id','max_wifi_id']]
    # a = a.groupby(['shop_id']).agg('count').reset_index()
    # print(len(a))
    # a = a[a.max_wifi_id == 1]
    # print(len(a))
    # a.to_csv("data/test1.csv",index=None)
    # print(a)




if __name__ == '__main__':
    main()