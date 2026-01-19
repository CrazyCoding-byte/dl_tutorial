import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmoid, softmax  # 激活函数


# 读取数据
def get_data():
    # 1.从文件加载数据
    data = pd.read_csv("../data/train.csv")
    # 2.划分数据集
    x = data.drop("label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    #3.特征工程:归一化
    scaler=MinMaxScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.fit_transform(x_test)
    return x_test,y_test