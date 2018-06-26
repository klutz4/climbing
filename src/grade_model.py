import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt

boulders = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')

#drop columns that are either object types or correllated to another column
cols = ['Unnamed: 0' ,'notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','grade_id']
for df in [routes,boulders]:
    for col in cols:
        df.drop(col,axis=1,inplace=True)

boulder_y = boulders.pop('usa_boulders')
boulder_X = boulders
routes_y= routes.pop('usa_routes')
routes_X = routes

def run_ridge_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    ridge = RidgeCV()
    ridge.fit(X_train,y_train)
    ridge_score = ridge.score(X_test,y_test)
    return ridge, ridge_score

def run_lasso_model(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        lasso = LassoCV()
        lasso.fit(X_train,y_train)
        y_pred = lasso.predict(X_test)
        plt.scatter(y_pred,y_test)
        plt.show()
        lasso_score = lasso.score(X_test,y_test)
        return lasso, lasso_score

def get_coefs(model,X):
    df = pd.DataFrame(model.coef_)
    df['coef_names'] = X.columns
    return df

def plot_model_predictions(y_true,y_pred,filename):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    ax = sns.lmplot(x=y_pred,y=y_true,color='green',alpha=0.5)
    plt.savefig(filename)
    plt.clf()

# def main():
boulder_ridge, boulder_ridge_score = run_ridge_model(boulder_X,boulder_y)
boulder_lasso, boulder_lasso_score = run_lasso_model(boulder_X,boulder_y)
# routes_ridge, routes_ridge_score = run_ridge_model(routes_X,routes_y)
# routes_lasso, routes_lasso_score = run_lasso_model(routes_X,routes_y)
