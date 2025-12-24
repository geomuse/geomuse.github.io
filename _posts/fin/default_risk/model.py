#%%
import pandas as pd

df = pd.read_csv('/home/geo/.cache/kagglehub/datasets/laotse/credit-risk-dataset/versions/1/credit_risk_dataset.csv')
print(df)

print(df.columns)
#%%
df['loan_grade']