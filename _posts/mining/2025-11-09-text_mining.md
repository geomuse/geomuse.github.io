---
layout: post
title:  python文字探勘 分词(Tokenization)
date:   2025-11-09 09:01:00 +0800
tags: 
    - python
    - mining
image: 10.jpg
---

```py
import nltk
# 只需要运行一次
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
```

```py
import nltk
from nltk.tokenize import word_tokenize

# 示例文本
text = "NLTK is a powerful library for NLP. It's often used for academic research."

# 分词
nltk_tokens = word_tokenize(text)

print("--- NLTK 分词结果 ---")
print(nltk_tokens)
# 预期输出: ['NLTK', 'is', 'a', 'powerful', 'library', 'for', 'NLP', '.', 'It', "'s", 'often', 'used', 'for', 'academic', 'research', '.']
```