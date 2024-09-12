import pandas as pd
import numpy as np


# 加载数据集
filename = "./BEGAN/data.csv"

data = pd.read_csv(filename)


# 离群值分析
def outlier_detect(data, threshold=10):
    outliers = pd.DataFrame()
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outliers = pd.concat([outliers, col_outliers])
    return outliers


outliers = outlier_detect(data)
# 删除离群值
data = data.drop(outliers.index)
data = data.dropna()


# 单次编码
def one_hot_encode(data, column):
    dummies = pd.get_dummies(data[column], prefix=column)
    data = pd.concat([data, dummies], axis=1)
    data = data.drop(column, axis=1, inplace=True)
    return data


categorical_columns = ["proto", "service", "state"]

for column in categorical_columns:
    data = one_hot_encode(data, column)


# 特征缩放
def feature_scaling(data):
    for column in data.select_dtypes(include=np.number).columns:
        min_value = data[column].min()
        max_value = data[column].max()
        data[column] = (data[column] - min_value) / (max_value - min_value)
    return data


data = feature_scaling(data)

# 将data转为numpy数组
data = data.to_numpy()
print(data)
