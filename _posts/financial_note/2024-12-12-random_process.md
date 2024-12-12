---
layout: post
title:  建构基本的随机过程
date:   2024-12-12 11:24:29 +0800
categories: 
    - financial
    - processing
---

<!-- 可使用下述程式碼把markdown格式轉成word

```
pandoc -o output.docx -f markdown -t docx input.md
``` -->

<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
  };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>


金融中的随机过程公式涉及各种数学模型和随机过程的描述。这些公式用于定量化金融现象，通常包括价格波动、利率变化和衍生品定价等。以下是一些主要的随机过程及其公式分类：

### 1. **布朗运动（Brownian Motion）**
- 定义：
  $$
  W(t) \sim \mathcal{N}(0, t), \quad W(0) = 0
  $$
  - $W(t)$：标准布朗运动
  - $\mathcal{N}(0, t)$：均值为 0，方差为 $t$ 的正态分布
- 性质：
  - 连续性：轨迹几乎处处连续。
  - 增量独立性：对任意 $0 \leq t_1 < t_2 \leq T$，$W(t_2) - W(t_1)$ 服从正态分布。

### 2. **几何布朗运动（Geometric Brownian Motion, GBM）**
- 公式：
  $$
  dS_t = \mu S_t dt + \sigma S_t dW_t
  $$
  - $S_t$：资产价格
  - $\mu$：漂移率（期望回报率）
  - $\sigma$：波动率
  - $W_t$：标准布朗运动
- 解：
  $$
  S_t = S_0 \exp \left( \left(\mu - \frac{\sigma^2}{2} \right)t + \sigma W_t \right)
  $$

### 3. **均值回复过程（Mean Reverting Process, Ornstein-Uhlenbeck Process）**
- 公式：
  $$
  dX_t = \theta (\mu - X_t) dt + \sigma dW_t
  $$
  - $\theta$：均值回复速度
  - $\mu$：长期均值
  - $\sigma$：波动率
  - $W_t$：标准布朗运动
- 应用：利率（Vasicek 模型）和波动率（Heston 模型）建模。

### 4. **跳跃扩散过程（Jump Diffusion Process, Merton Model）**
- 公式：
  $$
  dS_t = \mu S_t dt + \sigma S_t dW_t + J_t S_t
  $$
  - $J_t$：泊松跳跃过程
  - 泊松分布：
    $$
    P(N(t) = k) = \frac{(\lambda t)^k}{k!} e^{-\lambda t}
    $$
  - 跳跃幅度：
    $$
    J_t = \exp(Y) - 1, \quad Y \sim \mathcal{N}(\mu_J, \sigma_J^2)
    $$

### 5. **Cox-Ingersoll-Ross (CIR) 模型**
- 用于利率建模：
  $$
  dr_t = \theta (\mu - r_t) dt + \sigma \sqrt{r_t} dW_t
  $$
  - $r_t$：短期利率
  - $\theta$：均值回复速度
  - $\mu$：长期均值
  - $\sigma$：波动率

### 6. **Heston 模型**
- 波动率服从随机过程：
  $$
  dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t
  $$
  - $v_t$：资产波动率
  - $\kappa$：均值回复速度
  - $\theta$：长期波动率均值
  - $\xi$：波动率的波动率
  - $W_t$：布朗运动（通常与资产价格过程相关联）

### 7. **随机利率模型**
#### a. **Ho-Lee 模型**
$$
dr_t = \theta(t) dt + \sigma dW_t
$$

#### b. **Hull-White 模型**
$$
dr_t = (\theta(t) - \alpha r_t) dt + \sigma dW_t
$$

### 8. **马尔科夫过程**
- 定义：
  $$
  P(X_{t+h} | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+h} | X_t)
  $$
  - 马尔科夫性：未来状态只与当前状态相关，与过去无关。

### 9. **其他过程**
#### a. **Bachelier 模型**
$$
dS_t = \mu dt + \sigma dW_t
$$

#### b. **双重跳跃模型**
结合跳跃扩散和随机波动率：
$$
dS_t = \mu S_t dt + \sigma(S_t) dW_t + J_t S_t
$$

#### c. **极端风险模型**
如基于 Lévy 过程或重尾分布模型。