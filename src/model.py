import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import utils

boulders = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')

def get_grade_datasets():
    cols = ['Unnamed: 0' ,'notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','grade_id','climb_name']
    for df in [routes,boulders]:
        for col in cols:
            df.drop(col,axis=1,inplace=True)

    boulder_y = boulders.pop('usa_boulders').values
    boulder_X = boulders.values
    routes_y= routes.pop('usa_routes').values
    routes_X = routes.values
    return boulder_X, boulder_y, routes_X, routes_y

def get_ability_datasets():
    cols = ['Unnamed: 0' ,'notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','climb_name']
    for df in [routes,boulders]:
        for col in cols:
            df.drop(col,axis=1,inplace=True)

    route_ability_y = routes.usa_routes[routes['ascent_date_year'] == 2017].values
    route_ability_X = routes[routes['ascent_date_year'] != 2017].sample(6504)
    route_ability_X = route_ability_X.values
    boulder_ability_y = boulders.usa_boulders[boulders['ascent_date_year'] == 2017].values
    boulder_ability_X = boulders[boulders['ascent_date_year'] != 2017].sample(7200)
    boulder_ability_X = boulder_ability_X.values
    return boulder_ability_X, boulder_ability_y, route_ability_X, route_ability_y


def run_ridge_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    standardizer = utils.XyScaler()
    standardizer.fit(X_train,y_train)
    X_train_std, y_train_std = standardizer.transform(X_train, y_train)
    X_test_std, y_test_std = standardizer.transform(X_test, y_test)
    ridge = RidgeCV(alphas = np.logspace(-2,4,num=250),cv=10)
    ridge.fit(X_train_std,y_train_std)
    y_hats_std = ridge.predict(X_test_std)
    X_test, y_hats = standardizer.inverse_transform(X_test_std,y_hats_std)
    ridge_score = ridge.score(X_test_std,y_test_std)
    return ridge, ridge_score, y_hats, y_test

def run_lasso_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lasso = LassoCV(alphas = np.logspace(-2,4,num=250),cv=10)
    lasso.fit(X_train,y_train)
    y_hats = lasso.predict(X_test)
    lasso_score = lasso.score(X_test,y_test)
    return lasso, lasso_score

def get_coefs(model,X):
    df = pd.DataFrame(model.coef_)
    df['coef_names'] = X.columns
    return df

def plot_model_predictions(y_true,y_pred,filename):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    ax =plt.scatter(x=y_pred,y=y_true,color='green',alpha=0.5)
    ax = plt.plot(x=y_pred,y=y_true,color='green',alpha=0.5)
    plt.savefig(filename)
    plt.clf()

def find_error_scores(X,y,model_function,n):
    r2_scores = []
    rss_scores = []
    for i in range(n+1):
        model, score, y_hat, y_true = model_function(X,y)
        r2_scores.append(score)
        rss_scores.append(mean_squared_error(y_true,y_hat))
    return max(r2_scores),min(rss_scores)

# def main():
# boulder_ridge, boulder_ridge_score, y_hats, y_test = run_ridge_model(boulder_X,boulder_y)
# boulder_lasso, boulder_lasso_score = run_lasso_model(boulder_X,boulder_y)
# routes_ridge, routes_ridge_score, y_hats, y_test = run_ridge_model(routes_X,routes_y)
# routes_lasso, routes_lasso_score = run_lasso_model(routes_X,routes_y)
