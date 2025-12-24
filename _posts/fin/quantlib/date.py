import QuantLib as ql

# 设置评估日
today = ql.Date().todaysDate()
ql.Settings.instance().evaluationDate = today
print("Today:", today)  # 输出格式：Date(3, 12, 2025)

# 创建指定日期
d = ql.Date(25, 12, 2025)  # 25 Dec 2025
print("Specific Date:", d)

# 日期加减
d2 = d + 10  # 加 10 天
print("10 days later:", d2)
d3 = d - 5   # 减 5 天
print("5 days earlier:", d3)

from QuantLib import Following

cal = ql.TARGET()
date = ql.Date(1, 1, 2025)  # 元旦，假日

adjusted = cal.adjust(date, Following)
print("Adjusted:", adjusted)