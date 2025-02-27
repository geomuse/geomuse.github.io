def find_max(num1,num2,num3):
    r = num1 
    z = num1 , num2 , num3
    for num in z :
        if num > r :
            r = num
    return r
    
print(find_max(5,12,90))