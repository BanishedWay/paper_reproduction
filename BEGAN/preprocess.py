import pandas as pd
from utils.basic_funcs import outlier_detect, one_hot_encode, normalize

# 加载数据集
dir_name = "./Data/UNSW15"

train_data = pd.read_csv(dir_name + "/UNSW_NB15_training-set.csv")
test_data = pd.read_csv(dir_name + "/UNSW_NB15_testing-set.csv")

combine_data = pd.concat([train_data, test_data])

# 删去id、label列
combine_data.drop("id", axis=1, inplace=True)
combine_data.drop("label", axis=1, inplace=True)

# 离群值分析
combine_data = outlier_detect(combine_data)

# one-hot编码
categorical_columns = ["proto", "service", "state"]
for column in categorical_columns:
    combine_data = one_hot_encode(combine_data, column)

# 将attack_cat列暂存
tmp = combine_data["attack_cat"]
combine_data.drop("attack_cat", axis=1, inplace=True)

# 数据归一化
combine_data = normalize(combine_data)

# 将bool型数据转换为int型，其余类型不变
for column in combine_data.columns:
    if combine_data[column].dtype == "bool":
        combine_data[column] = combine_data[column].astype(int)

# 将attack_cat转为class名称
combine_data["class"] = tmp
