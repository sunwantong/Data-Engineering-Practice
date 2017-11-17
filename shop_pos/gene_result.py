import pandas as pd
import numpy as np
import os as os




def load_file(allDir,filepath):
    data = pd.read_csv(filepath+allDir,sep=",")
    return data

def eachFile(filepath):
    pathDir = os.listdir(filepath)
    i = 0
    res_list = []
    for allDir in pathDir:
        i += 1

        data = load_file(allDir,filepath)
        res = process_data(data)
        res_list.append(res)

        print(i)
    return res_list




def process_data(data):
    def get_max_prob_shop_id(x):
        x = x.sort_values(by='prob', ascending=False)
        x = x.reset_index(drop=True)

        return x.loc[0, 'shop_id']

    data = data[['row_id','user_id','prob','shop_id']]
    data = data.groupby(['row_id','user_id']).apply(get_max_prob_shop_id).reset_index()
    data.columns = ['row_id','user_id','shop_id']
    return data


def main():
    res_list = eachFile('data/final_test/')
    result = pd.concat(res_list)
    result.to_csv("d:/result_init.csv",index=None)


def process_result():
    data = pd.read_csv('d:/result_init.csv', sep=",")
    data = data.sort_values(by='row_id', ascending=True)
    data = data[['row_id', 'shop_id']]
    print(len(data))
    data.to_csv("d://result.csv", index=None)

if __name__ == '__main__':
    # main()
    process_result()
