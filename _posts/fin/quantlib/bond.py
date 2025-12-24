import QuantLib as ql

# 1. 设置评估日
today = ql.Date(4, 12, 2025)
ql.Settings.instance().evaluationDate = today # 设置评估日

# 2. 债券参数
issue_date = ql.Date(3, 12, 2020) # 发行日期
maturity_date = ql.Date(3, 12, 2030) # 到期日期
tenor = ql.Period(ql.Semiannual)  # 每半年付息
calendar = ql.TARGET() # 日历
business_convention = ql.Following # 业务约定
day_count = ql.Actual360() # 日期计数
coupon_rate = [0.03]  # 年利率 3%
face_value = 1000 # 面值

# 3. 生成付息日期
schedule = ql.Schedule(issue_date, maturity_date, tenor, calendar,
                       business_convention, business_convention,
                       ql.DateGeneration.Backward, False) # 生成付息日期

# 4. 创建固定利率债券
bond = ql.FixedRateBond(0, face_value, schedule, coupon_rate, day_count) # 创建固定利率债券

# 5. 使用贴现曲线定价
flat_rate = ql.FlatForward(today, 0.04, day_count, ql.Compounded, ql.Semiannual) # 创建贴现曲线（复利频率应与付息频率匹配）
discount_curve = ql.YieldTermStructureHandle(flat_rate) # 创建贴现曲线
bond_engine = ql.DiscountingBondEngine(discount_curve) # 创建贴现引擎
bond.setPricingEngine(bond_engine)

# 6. 计算价格和收益率
clean_price = bond.cleanPrice() # 计算不含应计利息的价格
dirty_price = bond.dirtyPrice() # 计算含应计利息的价格          

print("Clean Price:", clean_price*face_value/100) # 显示百分比价格
print("Dirty Price:", dirty_price*face_value/100)
print("cashflows:", [ _.amount() for _ in bond.cashflows()] )