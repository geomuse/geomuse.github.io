---
layout: post
title:  fixed point
date:   2025-02-25 11:24:29 +0800
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

```python
def fixed_point(po,tol,no,f):
    i = 1
    while i <= no :
        p = f(po)
        yield p
        if abs(p-po) < tol :
            return  
        i+=1
        po=p
```