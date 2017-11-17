import pandas as pd
from com.sun.shopPos.config import *
import numpy as np



def load_data_set():
    shop_info_df = pd.read_csv(ccf_shop_info, sep=",")
    user_shop_behavior_df = pd.read_csv(ccf_user_shop_behavior, sep=",")
    evaluate_public = pd.read_csv(ccf_evaluation_public_set, sep=",")
    return user_shop_behavior_df,shop_info_df,evaluate_public

def get_max_signal(s):
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

def main():
    user_shop_info,shop_info,test = load_data_set()
    user_shop_info['mwifi_id'] = user_shop_info['wifi_infos'].apply(get_max_signal)
    user_shop_info = user_shop_info[['shop_id','mwifi_id','user_id']]
    user_shop_info = user_shop_info.groupby(['shop_id','mwifi_id']).agg('count').reset_index()
    user_shop_info = user_shop_info.sort_values(by='user_id',ascending=False)
    user_shop_info.rename(columns={'user_id': 'counts'}, inplace=True)
    print(len(user_shop_info))
    user_shop_info.to_csv("d://aa.csv",index=None)

if __name__ == '__main__':
    main()