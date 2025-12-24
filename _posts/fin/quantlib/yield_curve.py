#%%
import QuantLib as ql

today = ql.Date(3, 12, 2025)
ql.Settings.instance().evaluationDate = today

day_count = ql.Actual360()

# 平坦利率 4%
flat_rate = ql.FlatForward(today, 0.04, day_count, ql.Compounded, ql.Annual)
curve = ql.YieldTermStructureHandle(flat_rate)

# 折现因子（1年）
print("1 yr discount factor:", curve.discount(today + 365))
print("1 yr discount factor:", curve.discount(today + ql.Period(1,ql.Years)))

#%%
discount_factor = []
for _ in range(1,1000):
    discount_factor.append(curve.discount(today + ql.Period(_,ql.Days)))

import matplotlib.pyplot as pt
pt.plot(discount_factor)
pt.show()
#%%
# 零利率
# zero_rate = curve.zeroRate(1.0, day_count, ql.Compounded, ql.Annual)
zero_rate = curve.zeroRate(1.0, ql.Compounded, ql.Annual, True)
print("Zero rate (1yr):", zero_rate.rate())

# 远期利率
fwd_rate = curve.forwardRate(0.5, 1.0, ql.Simple, ql.Annual, True)
# fwd_rate = curve.forwardRate(0.5, 1.0, day_count, ql.Simple)
print("Forward rate (0.5→1.0):", fwd_rate.rate())
# %%
dates = [
    ql.Date(3,12,2025), # 今天
    ql.Date(3,6,2026), # 6M
    ql.Date(3,12,2026), # 1Y
    ql.Date(3,12,2027), # 2Y
    ql.Date(3,12,2028), # 3Y
]

rates = [0.0, 0.03, 0.035, 0.04, 0.045]  # 从 today 开始，第一点必须对应利率 0

zero_curve = ql.ZeroCurve(dates, rates, day_count)
curve = ql.YieldTermStructureHandle(zero_curve)

print("Discount 2 years:", curve.discount(today + ql.Period(2,ql.Years)))
print("Zero Rate 2 years:", curve.zeroRate(2.0,ql.Compounded, ql.Annual).rate())
# %%
import numpy as np
zero_rates = []
times = []
for _ in np.arange(0,3.04,0.01):
    times.append(_)
    zero_rates.append(curve.zeroRate(_,ql.Compounded, ql.Annual).rate())

pt.plot([0,0.5,1,2,3],rates)
pt.plot(times,zero_rates)
pt.show()
# %%