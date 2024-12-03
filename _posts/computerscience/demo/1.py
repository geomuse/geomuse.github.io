#%%
import pandas as pd
default = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv')

print(default.head())
#%%
default.info()
# %%
default['Default'].value_counts()