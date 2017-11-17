from com.sun.shopPos.config import *
import pandas as pd
from collections import defaultdict

def get_infos(s):
    s = s.split(";")
    for ele in s:
        ele = str(ele)
        s = ele.split("|")
        if s[2] == "true":
            return s[0]
        else:
            return -1

def get_all_infos(s):
    s = s.split("_")
    # print(s)
    return s[1]


def main():
    shop_info_df = pd.read_csv(ccf_shop_info,sep=",")
    user_shop_behavior_df = pd.read_csv(ccf_user_shop_behavior, sep=",")
    evaluate_public = pd.read_csv(ccf_evaluation_public_set, sep=",")



    max_time = evaluate_public['time_stamp'].max()
    min_time = evaluate_public['time_stamp'].min()

    a = list(set(evaluate_public['user_id'])&set(user_shop_behavior_df['user_id']))
    print(len(a))
    # print(shop_info_df['mall_id'].drop_duplicates().count())
    print(evaluate_public['user_id'].drop_duplicates().count())
    # print(max_time,min_time)
    # user_shop_mall_set = pd.merge(user_shop_behavior_df, shop_info_df[['shop_id', 'mall_id']], on="shop_id", how="left")
    # evaluate_public = evaluate_public[evaluate_public.mall_id == 'm_690']
    # evaluate_public.to_csv('d:/bb.csv',index=None)
    # a = evaluate_public.sort_values(by='row_id',ascending=True)
    # a.to_csv("d://aa.csv",index=None)
    # print(a)
    # print(shop_info_df)
    # print(len(set(shop_info_df['shop_id'])))

    # result4 = list(set(evaluate_public['mall_id']).intersection(set(shop_info_df['mall_id'])))
    # print(len(result4))
    # wifi_infos(user_shop_behavior_df)
    # shop_info_df = shop_info_df[['mall_id','shop_id','price']]
    # a = shop_info_df.groupby(['mall_id','shop_id']).agg("count").reset_index()
    # c = a[a.price > 1]
    # print(c)
    # r714608esult1 = shop_info_df["shop_id"].drop_duplicates().count()
    # result2 = user_shop_behavior_df["shop_id"].drop_duplicates().count()
    #
    # result6 = user_shop_behavior_df["wifi_infos"]
    #
    # result3 = shop_info_df["mall_id"].drop_duplicates().count()
    #
    # result4 = list(set(shop_info_df['shop_id']).intersection(set(user_shop_behavior_df['shop_id'])))
    # print(len(result4))
    # # print(result2)
    # # print(len(result3))
    # user_shop_behavior_df['infos'] = user_shop_behavior_df["wifi_infos"].apply(get_infos)
    # user_shop_behavior_df['other'] = user_shop_behavior_df["shop_id"].apply(get_all_infos)
    # print(user_shop_behavior_df['other'])
    #
    # result4 = list(set(user_shop_behavior_df['other']).intersection(set(user_shop_behavior_df['infos'])))
    # # a = user_shop_behavior_df[user_shop_behavior_df.infos != -1]
    # # a['infos'].to_csv("c://aa.csv",index=None)
    #
    # print(result4)


def split_evaluate_set(user_shop_hehavior):
    pass


#构造wifi相关规则
def wifi_infos(user_shop_hehavior):
    wifi_to_shops = defaultdict(lambda: defaultdict(lambda: 0))
    for line in user_shop_hehavior.values:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')], key=lambda x: int(x[1]), reverse=True)[0]
        # wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1
        print(wifi)
    # print(wifi)

if __name__ == '__main__':
    main()
