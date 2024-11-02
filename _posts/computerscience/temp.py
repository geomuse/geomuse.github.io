#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
# 生成一些随机数据
data = sns.load_dataset("tips")["total_bill"]

# 绘制分布图
sns.histplot(data)
# sns.distplot(data)
sns.kdeplot(data,color="black")
# sns.lineplot(data)
# sns.kdeplot(data)
plt.show()

# %%
sns.histplot(data, color="black")
plt.title("Total Bill Distribution", fontsize=18, fontweight='bold')
plt.xlabel("Total Bill", fontsize=14, fontweight='bold')
plt.ylabel("Frequency", fontsize=14, fontweight='bold')
plt.show()

# %%
sns.set_style("whitegrid")
data = sns.load_dataset("tips")["total_bill"]
sns.distplot(data, color="black")
pt.show()
# %%
