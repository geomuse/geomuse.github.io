#%%
import pandas as pd
import matplotlib.pyplot as pt
from matplotlib import style
import numpy as np
from sklearn.model_selection import train_test_split
url = '/home/geo/.cache/kagglehub/datasets/laotse/credit-risk-dataset/versions/1/credit_risk_dataset.csv'
df = pd.read_csv(url)
df
#%%

def train_test_split_3_row(X,y,test_size=0.2,test_size_val=0.25,random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # 再从剩余数据中切验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=test_size_val,  # 0.25 × 0.8 = 0.2
        stratify=y_temp,
        random_state=random_state
    )

    print("训练集:", X_train.shape)
    print("验证集:", X_val.shape)
    print("测试集:", X_test.shape)
    return X_train , X_val , X_test , y_train , y_val , y_test

#%%
import seaborn as sns
import matplotlib.pyplot as plt 

plt.figure(figsize=(6,4)) 
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()
#%%
for columns_name in df.columns : 
    print(f'{columns_name} : {df[columns_name].isna().sum()}')
#%%
df.describe()
#%%
style.use('ggplot')
for columns_name in df.columns : 
    if type(df[columns_name][0]) in (np.float64,np.int64) :
        pt.boxplot(df[columns_name])
        pt.show()
    else : 
        print('not number.')
# %%
