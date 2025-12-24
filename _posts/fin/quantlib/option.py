import QuantLib as ql

# 日期和日历
today = ql.Date().todaysDate()
ql.Settings.instance().evaluationDate = today

# 欧式期权参数
expiry = today + ql.Period(6, ql.Months)
strike = 100
option_type = ql.Option.Call  # 看涨
payoff = ql.PlainVanillaPayoff(option_type, strike)
exercise = ql.EuropeanExercise(expiry)

# 创建期权
option = ql.VanillaOption(payoff, exercise)
# 市场参数
spot = 100
rate = 0.03
vol = 0.2

# 构建定价过程
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, ql.Actual360()))
flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), vol, ql.Actual360()))
bsm_process = ql.BlackScholesMertonProcess(spot_handle, flat_ts, flat_ts, flat_vol_ts)

# 定价引擎
engine = ql.AnalyticEuropeanEngine(bsm_process)
option.setPricingEngine(engine)

# 计算期权价格
price = option.NPV()
print("Option Price:", price)

# 二叉树引擎
time_steps = 10000
engine = ql.BinomialVanillaEngine(bsm_process, "crr", time_steps)
option.setPricingEngine(engine)
price = option.NPV()
print("Option Price (Binomial Tree):", price)

engine = ql.MCEuropeanEngine(bsm_process, "PseudoRandom", timeSteps=1, requiredSamples=100000)
option.setPricingEngine(engine)
price = option.NPV()
print("Option Price (Monte Carlo):", price)