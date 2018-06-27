import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import matplotlib.pyplot as plt
import grade_model

boulders = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')

#drop columns that are either object types or correllated to another column
cols = ['Unnamed: 0' ,'notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','grade_id','climb_name']
for df in [routes,boulders]:
    for col in cols:
        df.drop(col,axis=1,inplace=True)



routes_y = routes.usa_routes[routes['ascent_date_year'] == 2017]
routes_X = routes[routes['ascent_date_year'] != 2017].sample(6504)
boulders_y = boulders.usa_boulders[boulders['ascent_date_year'] == 2017]
boulders_X = boulders[boulders['ascent_date_year'] != 2017].sample(7200)

def rss(y, y_hat):
    return np.mean((y - y_hat)**2)

def find_error_scores(X,y,model_function,n):
    r2_scores = []
    rss_scores = []
    for i in range(n+1):
        model, score = model_function(X,y)
        r2_scores.append(score)
        y_hat = model.predict(X)
        rss_scores.append(rss(y,y_hat))
    return max(r2_scores),min(rss_scores)
