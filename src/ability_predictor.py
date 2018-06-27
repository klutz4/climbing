import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import model

boulders = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')

def drop_columns_ability():
    cols = ['Unnamed: 0' ,'notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','climb_name']
    for df in [routes,boulders]:
        for col in cols:
            df.drop(col,axis=1,inplace=True)
    return df

def split_data_ability(df,col, sample_num):
    #sample num = 7200 for boulders, 6504 for routes
    train, test = train_test_split(df, test_size=.25)
    y_train = train[col][train['ascent_date_year'] == 2017].values
    X_train = train[train['ascent_date_year'] != 2017].sample(sample_num)
    X_train = X_train.values
    y_hold = test[col][test['ascent_date_year'] == 2017].values
    X_hold = test[test_size['ascent_date_year'] != 2017].sample(sample_num)
    X_hold = X_hold.values
    return X_train, y_train, X_hold, y_hold

# routes_y = routes.usa_routes[routes['ascent_date_year'] == 2017]
# routes_X = routes[routes['ascent_date_year'] != 2017].sample(6504)
# boulders_y = boulders.usa_boulders[boulders['ascent_date_year'] == 2017]
# boulders_X = boulders[boulders['ascent_date_year'] != 2017].sample(7200)

X_train, y_train, X_hold, y_hold = split_data_ability(boulders,'usa_boulders',7200)
