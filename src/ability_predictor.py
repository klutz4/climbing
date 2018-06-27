import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import utils
from src.model import get_boulder_models, get_route_models, test_model_on_hold

boulders = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')

def drop_columns_ability(df):
    cols = ['Unnamed: 0' ,'notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','climb_name']
    for col in cols:
        df.drop(col,axis=1,inplace=True)
    return df

def split_data_ability(df,col):
    #sample num = 7200 for boulders, 6504 for routes
    train, test = train_test_split(df, test_size=.25)
    y_train = train[col][train['ascent_date_year'] == 2017].values
    X_train = train[train['ascent_date_year'] != 2017].sample(len(y_train))
    X_train = X_train.values
    y_hold = test[col][test['ascent_date_year'] == 2017].values
    X_hold = test[test['ascent_date_year'] != 2017].sample(len(y_hold))
    X_hold = X_hold.values
    return X_train, y_train, X_hold, y_hold

def main_ability_boulder():
    boulder = drop_columns_ability(boulders)
    X_train, y_train, X_hold, y_hold = split_data_ability(boulder,'usa_boulders')
    boulder_ridge, boulder_lasso = get_boulder_models(X_train,y_train)
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Ridge,boulder_ridge.alpha_,'Final Ridge Prediction for Boulder Ability','images/ridge_model_boulder_ability.png')
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Lasso,boulder_lasso.alpha_,'Final Lasso Prediction for Boulder Ability','images/lasso_model_boulder_ability.png')

def main_ability_route():
    route = drop_columns_ability(routes)
    X_train, y_train, X_hold, y_hold = split_data_ability(route,'usa_routes')
    route_ridge, route_lasso = get_route_models(X_train,y_train)
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Ridge,route_ridge.alpha_,'Final Ridge Prediction for Route Ability','images/ridge_model_route_ability.png')
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Lasso,route_lasso.alpha_,'Final Lasso Prediction for Route Ability','images/lasso_model_route_ability.png')
