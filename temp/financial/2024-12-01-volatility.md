---
layout: post
title : 基于市场报价计算隐含波动率 
date : 2024-12-01 11:24:29 +0800
categories: 
    - financial
    - volatility
---

在金融衍生品中，使用碟式 (Butterfly)、ATM (At-the-Money) 和风险逆转 (Risk Reversal, RR) 数据来计算隐含波动率是一种常见的方法

```py
def calculate_volatility(atm_vol, risk_reversal, butterfly):
    # ATM volatility
    sigma_atm = atm_vol
    
    # Calculate 25 Delta Call and Put volatility
    sigma_25_delta_call = sigma_atm + (risk_reversal / 2) + butterfly
    sigma_25_delta_put = sigma_atm - (risk_reversal / 2) + butterfly
    
    return sigma_25_delta_call, sigma_25_delta_put

# 示例数据
atm_vol = 0.20  # ATM volatility (20%)
risk_reversal = 0.02  # Risk reversal (2%)
butterfly = 0.01  # Butterfly (1%)

# 计算隐含波动率
sigma_25_call, sigma_25_put = calculate_volatility(atm_vol, risk_reversal, butterfly)

print(f"25 Delta Call Volatility: {sigma_25_call:.2%}")
print(f"25 Delta Put Volatility: {sigma_25_put:.2%}")
```

**25 Delta Put 波动率**是隐含波动率的一个特定值，表示期权交易市场中 Delta 为 -25% 的看跌期权的隐含波动率。以下是详细解释：

---

### 1. **什么是 Delta？**
- **Delta**是期权的希腊字母之一，表示期权价格相对于标的资产价格变化的敏感度。
- Delta 的取值范围通常在 -1 到 1 之间：
  - **看跌期权 (Put)** 的 Delta 为负值。
  - **看涨期权 (Call)** 的 Delta 为正值。
  
对于**25 Delta Put**：
- 它是指 Delta 为 -25%（-0.25）的看跌期权。
- 这类期权的价格变动相当于标的资产价格变动的 25%。

---

### 2. **隐含波动率的意义**
隐含波动率是从期权市场价格反推出的波动率，代表市场对标的资产未来价格波动的预期。

**25 Delta Put 波动率**的含义：
- 它反映了市场对标的资产下跌方向的波动性预期。
- 通常用来衡量市场的看跌情绪。如果波动率较高，说明市场对标的资产未来下跌方向的波动性预期更大。

---

### 3. **与市场偏斜 (Skew) 的关系**
- 在隐含波动率曲面中，**Put 波动率**和**Call 波动率**的差异反映了市场对标的资产价格分布的看法。
- **波动率偏斜 (Skew)** 是 25 Delta Put 和 25 Delta Call 波动率之间的差异，反映了市场对资产价格尾部风险（如大幅下跌）的担忧。
- 如果**25 Delta Put 波动率**显著高于 ATM 波动率：
  - 表明市场更关注资产价格下跌的风险。

---

### 4. **计算公式回顾**
从碟式 (Butterfly)、风险逆转 (Risk Reversal)、ATM 波动率计算 25 Delta Put 波动率：
\[
\sigma_{25\Delta \text{Put}} = \sigma_{ATM} - \frac{\text{Risk Reversal}}{2} + \text{Butterfly}
\]

- **ATM 波动率** $\sigma_{ATM}$：市场最接近当前标的价格的期权隐含波动率。
- **风险逆转** $\sigma_{RR}$：反映市场对看涨和看跌方向偏好的差异。
- **碟式波动率** $\sigma_{Butterfly}$：测量波动率曲线的对称性。

---

### 5. **实际意义**
- **风险管理**：
  - 投资者可以通过观察 25 Delta Put 波动率来判断市场对下跌风险的关注程度。
  - 高波动率可能暗示投资者对未来的避险需求较高。
  
- **定价与交易**：
  - 25 Delta Put 波动率常用于定价尾部风险期权。
  - 在波动率套利交易中，交易员会利用 25 Delta 波动率的差异。

---

### 6. **应用案例**
如果市场数据如下：
- ATM 波动率：20% $(\sigma_{ATM} = 0.20$)
- 风险逆转：2% $(\sigma_{RR} = 0.02 $)
- 碟式波动率：1% $(\sigma_{Butterfly} = 0.01$)

计算 25 Delta Put 波动率：
\[
\sigma_{25\Delta \text{Put}} = 0.20 - \frac{0.02}{2} + 0.01 = 0.20 - 0.01 + 0.01 = 0.20
\]

解释：
- **隐含波动率为 20%**，表明市场预期标的资产在 Delta -25% 的情况下，下跌方向的波动性为 20%

### references

https://chatgpt.com/c/674bc389-e51c-800f-bebc-cd7d43ad0731