# import requests

# url = "https://investor.tsmc.com/sites/ir/annual-report/2024/2024%20Annual%20Report-C.pdf"

# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "Accept": "application/pdf"
# }

# response = requests.get(url, headers=headers)

# if "Enable JavaScript" in response.text:
#     print("仍然被封锁，需要进阶方法。")
# else:
#     with open("tsmc_report.pdf", "wb") as f:
        # raw_text = f.write(response.content)
import fitz # PyMuPDF
from opencc import OpenCC
from snownlp import SnowNLP

# 改为读取本地文件
doc = fitz.open("2024_Annual_Report-C.pdf") 
full_text = ""
for page in doc:
    full_text += page.get_text()

# import jieba

# # 載入自定義財務辭典（建議加入如：CoWoS, 先進製程, 毛利率等詞）
# jieba.add_word("CoWoS")
# jieba.add_word("先進製程")

# def preprocess_text(text):
#     # 分詞
#     words = jieba.lcut(text)
#     # 過濾長度小於 2 的詞或空白
#     cleaned_words = [w for w in words if len(w) > 1 and w.strip()]
#     return cleaned_words

# words_list = preprocess_text(full_text)

# t2s 代表 Traditional to Simplified (繁轉簡)
cc = OpenCC('t2s')

# 假設 raw_text 是你從台積電 PDF 讀取的繁體內容
simplified_text = cc.convert(full_text)

print("轉換前:", full_text[:20])
print("轉換後:", simplified_text[:20])

# 由於年報太長，直接分析全文容易造成數值溢位或精度損失 (0.00)
# 我們改用繁轉簡後的文字，並將其拆分為句子，計算平均情緒分
s = SnowNLP(simplified_text)
sentences = s.sentences

sentiment_scores = []
for sentence in sentences:
    if sentence.strip():
        sentiment_scores.append(SnowNLP(sentence).sentiments)

if sentiment_scores:
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    print(f"AI 模型判斷情緒分 (平均值): {avg_sentiment:.4f}")
else:
    print("未偵測到有效文字進行情緒分析")