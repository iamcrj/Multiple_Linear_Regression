# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 03:11:37 2018

@author: chirag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_X=LabelEncoder()
X[:,3]=label_X.fit_transform(X[:,3])

onehot=OneHotEncoder(categorical_features=[3])
X=onehot.fit_transform(X).toarray()

#Avoiding Trap of Dummy Variables
X=X[:,1:]

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)



import statsmodels.formula.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,4]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X=X[:,[0,1,4]]

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred_bw=regressor.predict(X_test)




