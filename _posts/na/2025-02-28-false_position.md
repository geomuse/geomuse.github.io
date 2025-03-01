---
layout: post
title:  false position
date:   2025-02-28 11:24:29 +0800
categories:
    - python
    - na
---

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

```py
def false_position(po,p1,tol,no,f):
    i = 2
    qo = f(po)
    q1 = f(p1)
    while i <= no : 
        p = p1 - q1*(p1-po)/(q1-qo)
        yield p
        if abs(p-p1) < tol :
            return 
        i+=1
        q=f(p)
        if q*q1 < 0 : 
            po = p1 
            qo = q1
        p1=p
        q1=q
f = lambda x : x**2 - x - 2

for value in false_position(0,5,1e-10,500,f):
    print(value)
```