from com.sun.shopPos.config import *
from com.sun.shopPos.gene_labeled_data_train import *
import pandas as pd
import numpy as np
import xgboost as xgb



#加载数据
def load_file(all_dir,filepath):
    data = pd.read_csv(filepath+all_dir,sep=",")
    return data



#传入的数据为每一个训练集或训练集
def get_shop_feature(data,test_set):

    #该shop发生交易行为的次数
    d1 = data[['shop_id','price']]
    d1 = d1.groupby(['shop_id']).agg('count').reset_index()
    d1.rename(columns={'price':'shop_trade_count'},inplace=True)

    #该shop发生交易行为时价格的均值
    d2 = data[['shop_id','price']]
    d2 = d2.groupby(['shop_id']).agg('mean').reset_index()
    d1.rename(columns={'price': 'mean_of_price'}, inplace=True)

    #
    # d3 =
    return data

def get_wifi_feature():
    pass


def get_wifi_with_shop_feature():
    pass


#遍历文件进行过滤
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    i = 0
    mode_list = []
    for allDir in pathDir:
        i += 1
        data = load_file(allDir,filepath)
        model = xgb_model(data)
        mode_list.append(model)
        p该wifi被连接的次数rint("模型",i)
    return mode_list

def xgb_model(train_set):
    # print(train_set.info())

    train_set = train_set[['label','latitude','longitude']]
    train_y = train_set.label
    train_x = train_set.drop(['label'],axis=1)

    xgb_train = xgb.DMatrix(train_x, label=train_y)

    params = {  'booster':'gbtree',
                'objective': 'binary:logistic',  #二分类的问题
                # 'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
                'max_depth':5,  # 构建树的深度，越大越容易过拟合
                # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                'subsample':0.7,  # 随机采样训练样本
                'colsample_bytree':0.7,  # 生成树时进行的列采样
                'min_child_weight':3,
                # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
                #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
                #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
                'silent':0,  #设置成1则没有运行信息输出，最好是设置为0.
                'eta': 0.05,  # 如同学习率
                'seed':1000,
                'nthread':7,  # cpu 线程数
                'eval_metric': 'error'  # 评价方式
              }

    plst = list(params.items())
    num_rounds = 30  # 迭代次数
    model = xgb.train(plst, xgb_train, num_rounds)
    return model


def eachFile_test(filepath,modelist):
    pathDir = os.listdir(filepath)
    i = 0
    for allDir in pathDir:
        model = modelist[i]
        data = load_file(allDir, filepath)
        data1 = data
        data = data[['latitude','longitude']]
        xgb_test = xgb.DMatrix(data)
        pred_value = model.predict(xgb_test)
        i += 1
        a = pd.DataFrame(pred_value,columns=['prob'])
        res = pd.concat([data1[['row_id','user_id','shop_id','mall_id']],a],axis=1)
        i = str(i)
        res.to_csv("data/final_test/test"+i+".csv",index=None)
        i = int(i)
        print("测试集",i)

def main():
    # shop_info_df, user_shop_df, test_set = load_data_set()
    mode_list = eachFile('data/train_filter/')
    eachFile_test('data/test_filter/',mode_list)

if __name__ == '__main__':
    main()