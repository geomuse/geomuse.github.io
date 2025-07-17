以下是障碍期权的8种形态在Black-Scholes模型下的完整定价公式（连续监测假设）。公式基于标准金融工程文献（如Haug, 2007），并已简化表达。

---

### **通用参数定义**
- $ S_0 $: 标的资产现价
- $ K $: 执行价
- $ H $: 障碍水平
- $ T $: 到期时间
- $ r $: 无风险利率
- $ q $: 股息率
- $ \sigma $: 波动率
- $ \lambda = \dfrac{r - q + \sigma^2/2}{\sigma^2} $
- $ d_1 = \dfrac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}} $
- $ d_2 = d_1 - \sigma\sqrt{T} $
- $ N(\cdot) $: 标准正态累积分布函数

辅助变量：  
$$
\begin{aligned}
d_3 &= \dfrac{\ln(H^2 / (S_0 K)) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \\
d_4 &= d_3 - \sigma\sqrt{T}, \\
d_5 &= \dfrac{\ln(S_0 / H)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}, \\
d_6 &= d_5 - \sigma\sqrt{T}, \\
d_7 &= \dfrac{\ln(H / S_0)}{\sigma\sqrt{T}} + \lambda\sigma\sqrt{T}, \\
d_8 &= d_7 - \sigma\sqrt{T}.
\end{aligned}
$$

---

### **1. 向下敲出看涨期权 (Down-and-Out Call, DOC)**
- **条件**: $ H < S_0 $ 且 $ H \leq K $
- **定价公式**:
$$
V_{\text{DOC}} = S_0 e^{-qT} \left[ N(d_1) - \left( \frac{H}{S_0} \right)^{2\lambda} N(d_3) \right] - K e^{-rT} \left[ N(d_2) - \left( \frac{H}{S_0} \right)^{2\lambda-2} N(d_4) \right]
$$

---

### **2. 向上敲出看涨期权 (Up-and-Out Call, UOC)**
- **条件**: $ H > S_0 $ 且 $ H \geq K $
- **定价公式**:
$$
V_{\text{UOC}} = S_0 e^{-qT} \left[ N(d_1) - N(d_5) - \left( \frac{H}{S_0} \right)^{2\lambda} \left( N(-d_8) - N(-d_6) \right) \right] - K e^{-rT} \left[ N(d_2) - N(d_6) - \left( \frac{H}{S_0} \right)^{2\lambda-2} \left( N(-d_7) - N(-d_5) \right) \right]
$$

---

### **3. 向下敲出看跌期权 (Down-and-Out Put, DOP)**
- **条件**: $ H < S_0 $ 且 $ H \leq K $
- **定价公式**:
$$
V_{\text{DOP}} = K e^{-rT} \left[ N(-d_2) - \left( \frac{H}{S_0} \right)^{2\lambda-2} N(-d_4) \right] - S_0 e^{-qT} \left[ N(-d_1) - \left( \frac{H}{S_0} \right)^{2\lambda} N(-d_3) \right]
$$

---

### **4. 向上敲出看跌期权 (Up-and-Out Put, UOP)**
- **条件**: $ H > S_0 $ 且 $ H \geq K $
- **定价公式**:
$$
V_{\text{UOP}} = K e^{-rT} \left[ N(-d_2) - N(-d_6) - \left( \frac{H}{S_0} \right)^{2\lambda-2} \left( N(d_7) - N(d_5) \right) \right] - S_0 e^{-qT} \left[ N(-d_1) - N(-d_5) - \left( \frac{H}{S_0} \right)^{2\lambda} \left( N(d_8) - N(d_6) \right) \right]
$$

---

### **5. 向下敲入看涨期权 (Down-and-In Call, DIC)**
- **条件**: $ H < S_0 $
- **定价公式** (通过敲出期权推导):
$$
V_{\text{DIC}} = \underbrace{S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)}_{\text{普通看涨}} - V_{\text{DOC}}
$$

---

### **6. 向上敲入看涨期权 (Up-and-In Call, UIC)**
- **条件**: $ H > S_0 $
- **定价公式**:
$$
V_{\text{UIC}} = \underbrace{S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)}_{\text{普通看涨}} - V_{\text{UOC}}
$$

---

### **7. 向下敲入看跌期权 (Down-and-In Put, DIP)**
- **条件**: $ H < S_0 $
- **定价公式**:
$$
V_{\text{DIP}} = \underbrace{K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)}_{\text{普通看跌}} - V_{\text{DOP}}
$$

---

### **8. 向上敲入看跌期权 (Up-and-In Put, UIP)**
- **条件**: $ H > S_0 $
- **定价公式**:
$$
V_{\text{UIP}} = \underbrace{K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)}_{\text{普通看跌}} - V_{\text{UOP}}
$$

---

### **关键说明**
1. **障碍关系**:
   - 敲入期权价格 = 普通期权价格 - 对应敲出期权价格
   - 当 $ H > K $ 或 $ H < K $ 时，部分公式需调整（详见条件）
2. **回扣条款**:
   若触发障碍返还金额 $ R $，需增加回扣项：
   $$
   V_{\text{rebate}} = R e^{-rT} \left[ N(\pm d_{\text{barrier}}) + \left( \frac{H}{S_0} \right)^{2\lambda \pm 1} N(\pm d_{\text{reflected}}) \right]
   $$
3. **离散监测**:
   需使用蒙特卡洛模拟或修正公式（如Broadie-Glasserman近似）
