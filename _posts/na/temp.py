def bisection(a,b,tol,no,f):
    fa = f(a)
    i = 0
    while i <= no :
        p = a + (b-a)/2
        fp = f(p)
        yield p
        if fp == 0 or (b-a)/2 < tol :
            return  
        i+=1 
        if fa*fp > 0 :
            a=p 
            fa=fp
        else :
            b=p

def fixed_point(po,tol,no,f):
    i = 1
    while i <= no :
        p = f(po)
        yield p
        if abs(p-po) < tol :
            return  
        i+=1
        po=p

def newtion(po,tol,no,f,fp):
    i = 1
    while i <= no : 
        p = po - f(po)/fp(po)
        yield p
        if abs(p-po) < tol :
            return 
        i+=1
        po = p

def secant(po,p1,tol,no,f):
    i = 2
    qo = f(po)
    q1 = f(p1)
    while i <= no :
        p = p1 - q1*(p1-po)/(q1-qo)
        yield p
        if abs(p-p1) < tol :
            return 
        i+=1
        po=p1; qo=q1 ; p1=p ; q1=f(p)

def false_position(po,p1,tol,no,f):
    i = 2
    qo = f(po)
    q1 = f(p1)
    while i <= no : 
        p = p1 - q1*(p1-po)/(q1-qo)
        yield p
        if abs(p-p1) < tol :
            return 
        i+=1
        q=f(p)
        if q*q1 < 0 : 
            po = p1 
            qo = q1
        p1=p
        q1=q
f = lambda x : x**2 - x - 2

for value in false_position(0,5,1e-10,500,f):
    print(value) 