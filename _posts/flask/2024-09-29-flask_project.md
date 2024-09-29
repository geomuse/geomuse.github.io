---
layout: post
title:  flask news 
date:   2024-09-29 09:24:29 +0800
categories:
    - flask
    - project
---

### 新闻

主要是个人阅读使用

因为不想只站定具有主观的新闻媒体,所以收集后进行阅读

英文,中文等都可以经过翻译后阅读

相似度删除 : 把文章相似的删除具有应该怎么完成? text mining 处理

翻译 : 有套件简单处理

### 预定计划

- 相似度删除

```py
def fetch_similarity(texts):
    # 向量化文本
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()

    # 计算余弦相似度
    cosine_sim = cosine_similarity(vectors)

    # 构建 DataFrame 并标记重复
    df = pd.DataFrame({'text': texts})
    df['is_duplicate'] = False

    # 设置相似度阈值
    threshold = 0.6

    # 标记重复项
    for i in range(len(cosine_sim)):
        for j in range(i + 1, len(cosine_sim)):
            if cosine_sim[i][j] > threshold:
                df.loc[j, 'is_duplicate'] = True

    # 保留唯一的文章
    unique_texts_df = df[df['is_duplicate'] == False].reset_index(drop=True)

    # 输出去重后的结果
    # print(unique_texts_df[['text']])
    return unique_texts_df[['text']]
```

- 多网址汇入

```py
@app.route('/')
def fetch_news():
    news = news_ltn()
    news_ = news_ettoday()
    news = news + news_
    # return news
    return render_template('news.html', news_list=news)
```

- 翻译成原句

- 设定主题新闻,如俄乌战争

设定主题后 -> `俄乌战争`