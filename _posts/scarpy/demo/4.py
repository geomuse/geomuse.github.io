import re

# 定义字符串和正则表达式
text = "用户ID: 12345, 订单号: A6789"
pattern = r'(\d+)'

# 匹配数字
matches = re.findall(pattern, text)
print("所有匹配项:", matches)

# 替换数字为 "#"
replaced_text = re.sub(pattern, '#', text)
print("替换结果:", replaced_text)

# 分割字符串
split_text = re.split(r'\s+', text)
print("分割结果:", split_text)
