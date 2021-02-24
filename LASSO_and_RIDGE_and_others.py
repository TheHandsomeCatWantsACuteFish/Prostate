import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.metrics import mean_squared_error
import math


def others(path):

    idata = pd.read_csv(path)
    itdata = pd.read_csv('./test.csv')
    data3 = itdata.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

    y_test = itdata.iloc[:,8]
    #standardlize
    data = (idata-idata.mean())/(idata.std())
    data2 = idata.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
    y = idata.iloc[:, 8]
    x = idata.iloc[:,0:8]
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x))

    lassocv = LassoCV()
    lassocv.fit(x,y)
    alpha_l = lassocv.alpha_
    #print(alpha_l)

    ridgecv = RidgeCV()
    ridgecv.fit(x,y)
    alpha_r = ridgecv.alpha_
    #print(alpha_r)

    lasso2 = LassoLarsIC()
    lasso2.fit(x,y)
    #print(lasso2.coef_)
    #print(lasso2.intercept_)
    #x_t = data3.iloc[:,0:8]
    #prediction = lasso2.predict(x_t)
    #mae = np.mean(np.abs(y_test - prediction))
    #print(mae)

    #lasso regression
    lasso = Lasso(alpha = 0.206)
    lasso.fit(x,y)

    #ridge regression
    ridge = Ridge(alpha = 25)
    ridge.fit(x, y)
    #print(ridge.coef_)
    #print(ridge.intercept_)
    #x_t = data3.iloc[:,0:8]
    #prediction = ridge.predict(x_t)
    #mae = np.mean(np.abs(y_test - prediction))
    #print(mae)

    lr = LinearRegression()
    lr.fit(x, y)

    pca = PCA(n_components=7)
    pcax = pca.fit_transform(x)
    pca.fit(x)
    com = np.mat(pca.components_)

    lrpca = LinearRegression()
    lrpca.fit(pcax,y)
    pcacoef = np.mat(lrpca.coef_)
    #print(lrpca.coef_,lrpca.intercept_)
    real_coef = pcacoef * com
    #print(real_coef)
    #print(lrpca.intercept_)

    pls = PLSRegression(n_components=2)
    pls.fit(x, y)
    x_t = data3.iloc[:,0:8]
    prediction = pls.predict(x_t)
    mae = np.mean(np.abs(y_test - prediction[0]))
    std = np.sqrt(np.abs(y_test - prediction[0] - np.mean(y_test - prediction[0]))/30)
    print(mae,std)
    print(pls.coef_.T)
    pls_intercept = pls.y_mean_ - np.dot(pls.x_mean_ , pls.coef_)
    print(pls_intercept)






    return


if __name__ == '__main__':
    path = './train.csv'
    others(path)
