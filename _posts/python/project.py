import math 

height = float(input())
weight = float(input())

bmi = weight / math.pow(height,2)

def match_bmi(n):
    match n :
        case _ if n < 18.5 : return "过轻"
        case _ if 18.5 <= n < 24.9 : return "正常"
        case _ if 25 <= n < 29.9 : return "超重"
        case _ if n >= 30 : return "肥胖"

print(match_bmi(bmi))