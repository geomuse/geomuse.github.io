import math 

def match_even_or_odd(num):
    match num :
        case _ if num % 2 == 0 : return "even"
        case _ : return "odd"

print(match_even_or_odd(7))