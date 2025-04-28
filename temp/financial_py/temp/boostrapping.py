import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from matplotlib import style
style.use('ggplot')

y = lambda t , tau1 : (1-np.exp(-t/tau1))/(t/tau1)

def nelson_and_siegel_model(beta0,beta1,beta2,tau1,t):    
    return beta0 + beta1*y(t,tau1) + beta2*(y(t,tau1)-np.exp(-t/tau1))

def svensson(beta0,beta1,beta2,beta3,tau1,tau2,t):
    return beta0 + beta1*y(t,tau1) + beta2*(y(t,tau1)-np.exp(-t/tau1)) + beta3*(y(t,tau2)-np.exp(-t/tau2))

if __name__ == '__main__' :

    result = nelson_and_siegel_model(2/100,-1/100,1/100,2,10)

    # t = np.arange(0,10+0.25,0.25)
    # result = nelson_and_siegel_model(2/100,-1/100,1/100,2,t)

    data = pd.DataFrame({
        'index' : ["A","B","C","D","E"] , 
        'tenor' : [2,5,10,20,30] ,
        'coupon rate' : [0.75,1,1.375,2.1250,2]/100 ,
        # 'maturity date' : ['2013-02-14']
    })

    print(data)