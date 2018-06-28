import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error as mse
import utils
import grade_model
import statsmodels.api as sm

def make_qqplot_boulder():
    boulders, routes = grade_model.import_data()
    boulder = grade_model.drop_columns_grades(boulders)
    X_train1, y_train1, X_hold, y_hold = grade_model.split_data_grades(boulder,'usa_boulders')
    X_train2, X_test, y_train2, y_test = train_test_split(X_train1, y_train1)
    standardizer = utils.XyScaler()
    standardizer.fit(X_train2,y_train2)
    X_train_std, y_train_std = standardizer.transform(X_train2, y_train2)
    X_test_std, y_test_std = standardizer.transform(X_test, y_test)
    X_hold_std, y_hold_std = standardizer.transform(X_hold,y_hold)
    ridge = RidgeCV(alphas = np.logspace(-2,4,num=250),cv=10)
    ridge.fit(X_train_std,y_train_std)
    residuals = y_hold_std - ridge.predict(X_hold_std)
    sm.graphics.qqplot(residuals)
    plt.savefig('images/boulders_qqplot.png',fit=True,line='45')
    plt.show()
    plt.clf()

def make_qqplot_route():
    boulders, routes = grade_model.import_data()
    route = grade_model.drop_columns_grades(routes)
    X_train1, y_train1, X_hold, y_hold = grade_model.split_data_grades(route,'usa_routes')
    X_train2, X_test, y_train2, y_test = train_test_split(X_train1, y_train1)
    standardizer = utils.XyScaler()
    standardizer.fit(X_train2,y_train2)
    X_train_std, y_train_std = standardizer.transform(X_train2, y_train2)
    X_test_std, y_test_std = standardizer.transform(X_test, y_test)
    X_hold_std, y_hold_std = standardizer.transform(X_hold,y_hold)
    ridge = RidgeCV(alphas = np.logspace(-2,4,num=250),cv=10)
    ridge.fit(X_train_std,y_train_std)
    residuals = y_hold_std - ridge.predict(X_hold_std)
    sm.graphics.qqplot(residuals)
    plt.savefig('images/routes_qqplot.png',fit=True,line='45')
    plt.show()
