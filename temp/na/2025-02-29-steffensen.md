---
layout: post
title:  steffensen’s
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
def steffensen(po,tol,no,f):
    i = 1
    while i <= no : 
        p1 = f(po)
        p2 = f(p1)
        p = po - (p1-po)**2 / (p2-2*p1+po)
        yield p
        if abs(p-po) < tol :
            return 
        i+=1
        po = p

g = lambda x : (10/(x+4))**0.5

for value in steffensen(10,1e-10,500,g):
    print(value) 
```