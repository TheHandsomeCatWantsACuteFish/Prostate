import pandas as pd
import numpy as np
import statsmodels.api as sm
def ols(path):

    idata = pd.read_csv(path)
    #standardlize
    data = (idata-idata.mean())/(idata.std())
    data2 = idata.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    y = idata.iloc[:, 8]
    x = data2.iloc[:, 0:8]

    x2 = data2.iloc[:,0:2]

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print(model.summary())

    #x2 = sm.add_constant(x2)
    #model2 = sm.OLS(y,x2).fit()
    #print(model2.summary())


    return

if __name__ == '__main__':
    path = './train.csv'
    ols(path)
