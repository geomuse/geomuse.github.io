---
layout: post
title:  fe bond 进阶
date:   2024-10-08 11:24:29 +0800
categories: 
    - financial 
    - python
---

---

## **1. 政府债券（固定利率债券）**

### **介绍**

政府债券通常被视为无风险债券，其定价主要考虑利率风险。定价方法是将未来的现金流（定期的利息支付和到期时的本金偿还）按照市场利率进行贴现。

### **定价方法**

- **贴现现金流模型（Discounted Cash Flow, DCF）**：
  - 计算每一期的现金流，包括利息和本金。
  - 使用对应期限的市场利率（即零息利率曲线）对现金流进行贴现。
  - 将所有贴现后的现金流相加，得到债券的现值。

### **代码示例**

```python
import numpy as np

def fixed_rate_bond_price(face_value, coupon_rate, maturity, market_rates, frequency=1):
    """
    计算固定利率债券的价格。

    参数：
    - face_value: 债券面值
    - coupon_rate: 年化票面利率（如5%输入0.05）
    - maturity: 到期时间（以年为单位）
    - market_rates: 市场利率列表，对应每个现金流的贴现率（年化）
    - frequency: 每年的付息次数（默认1次）

    返回：
    - bond_price: 债券价格
    """
    # 计算每期现金流
    coupon = face_value * coupon_rate / frequency
    total_periods = int(maturity * frequency)
    cash_flows = np.full(total_periods, coupon)
    cash_flows[-1] += face_value  # 最后一期加上本金

    # 计算每期的贴现因子
    discount_factors = np.array([
        1 / (1 + market_rates[i]/frequency) ** (i+1)
        for i in range(total_periods)
    ])

    # 计算债券价格
    bond_price = np.sum(cash_flows * discount_factors)
    return bond_price

# 参数示例
face_value = 1000  # 面值
coupon_rate = 0.05  # 票面利率5%
maturity = 5  # 5年到期
frequency = 1  # 每年付息一次

# 假设市场利率曲线（年化）
market_rates = np.array([0.03, 0.032, 0.035, 0.037, 0.04])

price = fixed_rate_bond_price(face_value, coupon_rate, maturity, market_rates, frequency)
print(f"债券价格：{price:.2f}")
```

### **代码解释**

- **现金流计算**：每期的利息支付等于`面值 * 票面利率 / 付息频率`。最后一期需要加上本金。
- **贴现因子计算**：使用每期对应的市场利率计算贴现因子，考虑了复利效果。
- **债券价格计算**：将每期现金流乘以对应的贴现因子，然后求和。

---

## **2. 零息债券**

### **介绍**

零息债券不支付定期利息，以折价发行，到期时以面值偿还。定价时，只需将面值按照市场利率贴现。

### **定价方法**

- **现值公式**：
  \[
  P = \frac{F}{(1 + r)^n}
  \]
  其中：
  - \( P \) 是债券价格。
  - \( F \) 是面值。
  - \( r \) 是市场利率。
  - \( n \) 是到期时间（年）。

### **代码示例**

```python
def zero_coupon_bond_price(face_value, market_rate, maturity):
    """
    计算零息债券的价格。

    参数：
    - face_value: 债券面值
    - market_rate: 市场利率（年化）
    - maturity: 到期时间（以年为单位）

    返回：
    - bond_price: 债券价格
    """
    bond_price = face_value / (1 + market_rate) ** maturity
    return bond_price

# 参数示例
face_value = 1000
market_rate = 0.05  # 市场利率5%
maturity = 5  # 5年到期

price = zero_coupon_bond_price(face_value, market_rate, maturity)
print(f"零息债券价格：{price:.2f}")
```

### **代码解释**

- 使用简单的现值公式，将未来的面值按照市场利率进行贴现，得到债券的当前价格。

---

## **3. 浮动利率债券**

### **介绍**

浮动利率债券的利息支付根据市场利率（如LIBOR或SHIBOR）定期调整。定价需要预测未来的利率。

### **定价方法**

- **预期现金流贴现**：
  - 预测每期的利率（通常使用远期利率）。
  - 计算每期的预期利息支付。
  - 使用对应的市场利率贴现每期现金流。

### **代码示例**

```python
def floating_rate_bond_price(face_value, spread, maturity, market_rates, forward_rates, frequency=1):
    """
    计算浮动利率债券的价格。

    参数：
    - face_value: 债券面值
    - spread: 利差（固定加在浮动利率上的部分）
    - maturity: 到期时间（以年为单位）
    - market_rates: 市场贴现率列表
    - forward_rates: 预期未来利率列表
    - frequency: 每年的付息次数（默认1次）

    返回：
    - bond_price: 债券价格
    """
    total_periods = int(maturity * frequency)
    # 计算每期现金流（预期利率 + 利差）
    cash_flows = np.array([
        face_value * (forward_rates[i]/frequency + spread/frequency)
        for i in range(total_periods)
    ])
    cash_flows[-1] += face_value  # 最后一期加上本金

    # 计算贴现因子
    discount_factors = np.array([
        1 / (1 + market_rates[i]/frequency) ** (i+1)
        for i in range(total_periods)
    ])

    bond_price = np.sum(cash_flows * discount_factors)
    return bond_price

# 参数示例
face_value = 1000
spread = 0.002  # 20个基点的利差
maturity = 3
frequency = 2  # 每年付息两次

# 市场贴现率和预期未来利率（半年期）
market_rates = np.array([0.015, 0.017, 0.018, 0.02, 0.022, 0.025])
forward_rates = np.array([0.016, 0.018, 0.019, 0.021, 0.023, 0.026])

price = floating_rate_bond_price(face_value, spread, maturity, market_rates, forward_rates, frequency)
print(f"浮动利率债券价格：{price:.2f}")
```

### **代码解释**

- **现金流计算**：每期的利息等于`面值 * （预期利率 + 利差） / 付息频率`。
- **贴现因子计算**：使用市场贴现率计算每期的贴现因子。
- **债券价格计算**：贴现所有现金流并求和。

---

## **4. 可转换债券**

### **介绍**

可转换债券可以在特定条件下转换为发行公司的股票，其定价需要考虑债券部分和期权部分的价值。

### **定价方法**

- **可转换债券价值 = 纯债券价值 + 转股期权价值**
- 使用期权定价模型（如Black-Scholes模型）计算转股期权的价值。

### **代码示例**

```python
import numpy as np
from scipy.stats import norm

def convertible_bond_price(face_value, coupon_rate, maturity, market_rate, conversion_ratio, stock_price, volatility, risk_free_rate):
    """
    计算可转换债券的价格。

    参数：
    - face_value: 债券面值
    - coupon_rate: 票面利率
    - maturity: 到期时间（以年为单位）
    - market_rate: 市场利率（债券部分的贴现率）
    - conversion_ratio: 转换比率（每张债券可转换的股票数量）
    - stock_price: 当前股票价格
    - volatility: 股票的年化波动率
    - risk_free_rate: 无风险利率（用于期权定价）

    返回：
    - bond_price: 可转换债券价格
    """
    # 计算纯债券价值
    coupon = face_value * coupon_rate
    cash_flows = np.full(int(maturity), coupon)
    cash_flows[-1] += face_value  # 加上本金
    discount_factors = np.array([
        1 / (1 + market_rate) ** (i+1)
        for i in range(int(maturity))
    ])
    bond_value = np.sum(cash_flows * discount_factors)

    # 使用Black-Scholes模型计算期权价值
    S = stock_price
    K = face_value / conversion_ratio
    T = maturity
    sigma = volatility
    r = risk_free_rate

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    option_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    option_value *= conversion_ratio  # 考虑转换比率

    # 可转换债券价格
    convertible_bond_price = bond_value + option_value
    return convertible_bond_price

# 参数示例
face_value = 1000
coupon_rate = 0.04
maturity = 5
market_rate = 0.05
conversion_ratio = 20  # 每张债券可转换为20股股票
stock_price = 45
volatility = 0.3
risk_free_rate = 0.03

price = convertible_bond_price(face_value, coupon_rate, maturity, market_rate, conversion_ratio, stock_price, volatility, risk_free_rate)
print(f"可转换债券价格：{price:.2f}")
```

### **代码解释**

- **纯债券价值**：按照固定利率债券的方法，使用市场利率贴现未来的现金流。
- **期权价值**：使用Black-Scholes模型计算转换期权的价值，注意行权价为`面值 / 转换比率`。
- **合计**：将纯债券价值和期权价值相加，得到可转换债券的价格。

**注意**：可转换债券的定价较为复杂，实际中可能需要考虑股息、提前赎回条款、波动率微笑等因素。

---

## **5. 可赎回债券**

### **介绍**

可赎回债券允许发行人在特定条件下提前赎回债券。投资者面临被提前赎回的风险，因此债券价值会受到影响。

### **定价方法**

- **可赎回债券价值 = 普通债券价值 - 发行人期权价值**
- 需使用期权定价模型（如二叉树模型）估计发行人赎回期权的价值。

### **代码示例**

```python
import numpy as np

def callable_bond_price(face_value, coupon_rate, maturity, market_rate, call_price, call_time, volatility, frequency=1):
    """
    计算可赎回债券的价格。

    参数：
    - face_value: 债券面值
    - coupon_rate: 票面利率
    - maturity: 到期时间（以年为单位）
    - market_rate: 市场利率
    - call_price: 赎回价格
    - call_time: 可赎回时间（以年为单位）
    - volatility: 利率的年化波动率
    - frequency: 每年的付息次数

    返回：
    - bond_price: 可赎回债券价格
    """
    # 时间步长
    steps = int(maturity * frequency)
    dt = maturity / steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1 / up
    p = (np.exp(market_rate * dt) - down) / (up - down)

    # 构建利率树
    rates = np.zeros((steps + 1, steps + 1))
    rates[0, 0] = market_rate
    for i in range(1, steps + 1):
        rates[0:i, i] = rates[0:i, i-1] * up
        rates[i, i] = rates[i-1, i-1] * down

    # 计算现金流贴现因子
    discount_factors = np.exp(-rates * dt)

    # 初始化债券价值矩阵
    bond_values = np.zeros((steps + 1, steps + 1))

    # 计算终端节点的债券价值
    for i in range(steps + 1):
        bond_values[i, steps] = face_value + face_value * coupon_rate / frequency

    # 反向递推债券价值
    for j in reversed(range(steps)):
        for i in range(j + 1):
            # 现金流
            cash_flow = face_value * coupon_rate / frequency
            # 贴现后的预期价值
            expected_value = (p * bond_values[i, j+1] + (1 - p) * bond_values[i+1, j+1]) * discount_factors[i, j]
            # 判断是否可赎回
            if (j+1)*dt >= call_time:
                bond_values[i, j] = min(expected_value + cash_flow, call_price)
            else:
                bond_values[i, j] = expected_value + cash_flow

    return bond_values[0, 0]

# 参数示例
face_value = 1000
coupon_rate = 0.06
maturity = 5
market_rate = 0.05
call_price = 1020  # 赎回价格
call_time = 2  # 2年后可赎回
volatility = 0.2
frequency = 1

price = callable_bond_price(face_value, coupon_rate, maturity, market_rate, call_price, call_time, volatility, frequency)
print(f"可赎回债券价格：{price:.2f}")
```

### **代码解释**

- **利率树构建**：使用二叉树模型模拟利率的可能路径。
- **贴现因子计算**：根据利率树计算每个节点的贴现因子。
- **债券价值计算**：从终端节点开始，反向递推债券价值，考虑现金流和赎回选项。
- **赎回判断**：如果在可赎回期内，债券价值为预期价值和赎回价格的最小值。

**注意**：实际中，利率模型可能更为复杂，如Hull-White模型等。

---

## **6. 资产支持证券（ABS）**

### **介绍**

资产支持证券的现金流来自于一组基础资产，如房屋按揭贷款。定价需要对基础资产的现金流进行建模，考虑提前偿还、违约等因素。

### **定价方法**

- **蒙特卡罗模拟**：
  - 模拟基础资产池的现金流。
  - 考虑提前偿还率、违约率等因素。
  - 将预期现金流按照适当的折现率贴现。

### **代码示例**

```python
import numpy as np

def abs_price(principal, interest_rate, maturity, prepayment_rate, default_rate, discount_rate, simulations):
    """
    计算简单的资产支持证券价格。

    参数：
    - principal: 总本金
    - interest_rate: 资产池的平均利率
    - maturity: 资产池的平均期限（以年为单位）
    - prepayment_rate: 提前偿还率
    - default_rate: 违约率
    - discount_rate: 贴现率
    - simulations: 模拟次数

    返回：
    - abs_price: ABS的价格
    """
    total_cash_flows = np.zeros(maturity)
    for _ in range(simulations):
        remaining_principal = principal
        cash_flows = []
        for t in range(int(maturity)):
            # 计算利息
            interest = remaining_principal * interest_rate
            # 计算正常还款
            scheduled_payment = principal / maturity
            # 计算提前还款
            prepayment = remaining_principal * prepayment_rate
            # 计算违约损失
            default = remaining_principal * default_rate
            # 总现金流
            cash_flow = interest + scheduled_payment + prepayment - default
            cash_flows.append(cash_flow)
            # 更新剩余本金
            remaining_principal -= (scheduled_payment + prepayment + default)
            if remaining_principal <= 0:
                break
        # 将现金流累加
        for i in range(len(cash_flows)):
            total_cash_flows[i] += cash_flows[i]

    # 计算平均现金流
    avg_cash_flows = total_cash_flows / simulations

    # 贴现现金流
    discount_factors = np.array([
        1 / (1 + discount_rate) ** (i+1)
        for i in range(len(avg_cash_flows))
    ])
    abs_price = np.sum(avg_cash_flows * discount_factors)
    return abs_price

# 参数示例
principal = 1000000
interest_rate = 0.05
maturity = 5
prepayment_rate = 0.02
default_rate = 0.01
discount_rate = 0.06
simulations = 1000

price = abs_price(principal, interest_rate, maturity, prepayment_rate, default_rate, discount_rate, simulations)
print(f"资产支持证券价格：{price:.2f}")
```

### **代码解释**

- **现金流模拟**：对每次模拟，逐期计算利息、正常还款、提前还款和违约损失。
- **累计现金流**：将所有模拟的现金流累加，计算平均值。
- **贴现现金流**：使用贴现率将平均现金流贴现到现值。

**注意**：ABS的定价非常复杂，实际中需要精细的现金流建模，考虑更多的风险因素和结构化分层。

---

## **总结**

- **市场数据的重要性**：所有定价模型都依赖于市场数据，如利率曲线、波动率、信用利差等。数据的准确性直接影响定价结果。
- **模型假设**：定价模型基于一系列假设，需确保这些假设在实际情况中适用。
- **复杂债券的定价**：对于包含期权或特殊条款的债券，如可转换债券、可赎回债券，定价更为复杂，可能需要使用数值方法（如蒙特卡罗模拟、二叉树模型）。

---

**提示**：上述代码示例为简化模型，用于演示债券定价的基本原理。实际应用中，可能需要使用更加复杂和精确的模型，并考虑市场细节和风险因素。建议在实际投资和定价时，参考专业金融软件和咨询专业人士。