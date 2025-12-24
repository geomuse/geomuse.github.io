import QuantLib as ql

# 日期设置
issue_date = ql.Date(1, 1, 2025)
maturity_date = ql.Date(1, 1, 2030)
calendar = ql.TARGET()
tenor = ql.Period(6, ql.Months)

# 生成付息日程
schedule = ql.Schedule(issue_date, maturity_date, tenor, calendar,
                       ql.ModifiedFollowing, ql.ModifiedFollowing,
                       ql.DateGeneration.Backward, False)

# 创建债券
face_value = 1000
coupon_rate = 0.03
bond = ql.FixedRateBond(1, face_value, schedule, [coupon_rate], ql.ActualActual())
print("Bond created successfully")