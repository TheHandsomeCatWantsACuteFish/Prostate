import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools
from sklearn.linear_model import LinearRegression
def bsbset(path):

    idata = pd.read_csv(path)
    itdata = pd.read_csv('./test.csv')
    #standardlize
    data = (idata-idata.mean())/(idata.std())
    data2 = idata.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    data3 = itdata.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    y = idata.iloc[:, 8]
    x_train = data.iloc[:, 0:8]
    x_test = data3.iloc[:,0:8]
    y_test = itdata.iloc[:,8]

    best_e = 10000000000
    print(x_train)
    print(range(x_train.shape[1]+1))
    #for k in range(1, x_train.shape[1] + 1):
        # Loop over all possible subsets of size k
    for subset in itertools.combinations([0,1,2,3,4,5,6,7], 2):
        x = x_train.iloc[:,list(subset)]
        x_t = x_test.iloc[:,list(subset)]
        print(x)

        model = LinearRegression().fit(x,y)
        prediction = model.predict(x_t)
        mae = np.mean(np.abs(y_test-prediction))
        print(mae)
        if mae < best_e:
            best_e = mae
            bestsubset = subset

    print(bestsubset)
    #linreg_model = LinearRegression(normalize=True).fit(X_train[:, subset], y_train)
    #linreg_prediction = linreg_model.predict(X_test[:, subset])
    #linreg_mae = np.mean(np.abs(y_test - linreg_prediction))
    #results = results.append(pd.DataFrame([{'num_features': k,
    #                                                'features': subset,
    #                                                'MAE': linreg_mae}]))
    x = x_train.iloc[:, list(bestsubset)]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    print(model.summary())

    model = LinearRegression().fit(x, y)
    prediction = model.predict(x)
    mae = np.mean(np.abs(y - prediction))
    print(mae)
    return

if __name__ == '__main__':
    path = './train.csv'
    bsbset(path)
