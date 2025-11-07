import re

def basic_text_cleaning(text):
    """执行基础文本清理操作：大小写、数字、符号、标点、空白。"""
    
    # 步骤 1: 大小写统一 (仅对英文有效)
    # 将所有字符转换为小写，实现标准化
    text = text.lower()
    
    # 步骤 2: 移除特殊符号和标点符号
    # 使用正则表达式，匹配并替换所有非字母、非数字、非中文、非空白字符
    # [^a-zA-Z0-9\s\u4e00-\u9fa5] 匹配所有不属于这些类别的字符
    # \u4e00-\u9fa5 是中文汉字的 Unicode 范围
    text = re.sub(r'[^a-zA-Z0-9\s\u4e00-\u9fa5]', ' ', text)
    
    # 步骤 3: 数字处理 (替换为占位符 <NUM>)
    # 替换所有连续的数字为占位符，如果需要保留数字特征，可以跳过此步骤
    # text = re.sub(r'\d+', ' <NUM> ', text)

    # 步骤 4: 统一处理多余的空白字符
    # 将多个连续的空白字符（空格、换行、Tab）替换为单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- 演示清理 ---
raw_text = """
我是 杰米 AI，今天的日期是 2025/11/07。
WOW! Text Mining 很有用, 价格是 $1999.00！ 
你知道吗…… (符号) *&^%^。 
请检查我的邮件地址：Gemini@google.com。
"""

print("\n--- 原始文本 ---")
print(raw_text)

cleaned_text = basic_text_cleaning(raw_text)

print("\n--- 清理后的文本 ---")
print(cleaned_text)

print("\n--- 清理效果分解 ---")
print(f"**原始文本:** {raw_text.strip().splitlines()[0]}")
print(f"**小写/符号移除后:** {basic_text_cleaning(raw_text)}")