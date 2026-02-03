import seaborn as sns
import matplotlib.pyplot as pt
import pandas as pd
import numpy as np
pt.style.use('ggplot')

# df = sns.load_dataset("tips")
# print(df.head())

# # sns.lineplot(data=df, x="total_bill", y="tip")
# # sns.barplot(data=df, x="day", y="total_bill")

# # sns.barplot(
# #     data=df,
# #     x="day",
# #     y="total_bill",
# #     hue="sex"
# # )
# # sns.set_theme(style="whitegrid")
# # sns.pointplot(data=df, x="day", y="tip")

# # sns.countplot(data=df, x="day")
# # sns.histplot(data=df, x="total_bill")
# # sns.histplot(data=df, x="total_bill", bins=20)
# # sns.kdeplot(data=df, x="total_bill")
# # sns.histplot(data=df, x="total_bill", kde=True)
# # sns.boxplot(data=df, x="day", y="total_bill")

# corr = df.corr(numeric_only=True)
# print(corr)
# mask = np.triu(corr)
# num_df = df.select_dtypes(include="number")

# df_male = df[df['sex'] == 'Male']
# sns.heatmap(df_male.corr(numeric_only=True))

# # sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
# # sns.violinplot(data=df, x="day", y="total_bill")



# palette = ["lightgray", "lightgray", "red", "lightgray"]

# sns.barplot(
#     x=["A","B","C","D"],
#     y=[10,12,30,11],
#     palette=palette
# )

# pt.savefig('/home/geo/Downloads/geo/_posts/visualization/output.png')
# print("图片已生成在 Downloads 目录下，请双击查看！")
# pt.close()

import plotly.express as px
import pandas as pd

df = px.data.iris()

fig = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    size="petal_length",
    title="鸢尾花特征关系"
)

fig.show()
