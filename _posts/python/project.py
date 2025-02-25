import random

def match_num(num,count,real_num):
    match num :
        case _ if num > real_num : 
            return "太大了！"
        case _ if num < real_num:
            return "太小了！"
        case real_num : 
            return f"猜对了,{count}"
print(real_num := random.randint(1, 100))        
count = 0
while True : 
    num = float(input())
    print(match_num(num,count,real_num))
    count+=1