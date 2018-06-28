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

import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 18.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'small',
    'legend.fontsize'     : 'medium',
})

boulders = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')

def drop_columns_grades(df):
    cols = ['Unnamed: 0','notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','grade_id','climb_name','ascent_date_day','ascent_date_month','chipped']
    for col in cols:
        df.drop(col,axis=1,inplace=True)
    df = df.astype('float64')
    return df

def split_data_grades(df,target_column):
    train, test = train_test_split(df, test_size=.25)
    X_train, y_train = train.drop(target_column, axis=1).values, train[target_column].values
    X_hold, y_hold = test.drop(target_column, axis=1).values, test[target_column].values
    return X_train, y_train, X_hold, y_hold

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
    ridge_score = r2_score(y_test_std,y_hats_std)
    return ridge, ridge_score, y_hats, y_test, X_test

def run_lasso_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    standardizer = utils.XyScaler()
    standardizer.fit(X_train,y_train)
    X_train_std, y_train_std = standardizer.transform(X_train, y_train)
    X_test_std, y_test_std = standardizer.transform(X_test, y_test)

    lasso = LassoCV(alphas = np.logspace(-2,4,num=250),cv=10)
    lasso.fit(X_train_std,y_train_std)
    y_hats_std = lasso.predict(X_test_std)
    X_test, y_hats = standardizer.inverse_transform(X_test_std,y_hats_std)
    lasso_score = r2_score(y_test_std,y_hats_std)
    return lasso, lasso_score, y_hats, y_test, X_test

def get_boulder_models(X_train,y_train):
    boulder_ridge, boulder_ridge_score, ridge_y_hats, ridge_y_true, ridge_X_test = run_ridge_model(X_train,y_train)
    boulder_lasso, boulder_lasso_score,lasso_y_hats, lasso_y_true, lasso_X_test = run_lasso_model(X_train,y_train)
    print('The Ridge R2 score is {}'.format(boulder_ridge_score))
    print('The optimal ridge alpha is {}'.format(boulder_ridge.alpha_))
    print('The Lasso R2 score is {}'.format(boulder_lasso_score))
    print('The optimal lasso alpha is {}'.format(boulder_lasso.alpha_))
    return boulder_ridge, boulder_lasso

def get_route_models(X_train,y_train):
    route_ridge, route_ridge_score, ridge_y_hats, ridge_y_true, ridge_X_test = run_ridge_model(X_train,y_train)
    route_lasso, route_lasso_score,lasso_y_hats, lasso_y_true, lasso_X_test = run_lasso_model(X_train,y_train)
    print('The Ridge R2 score is {}'.format(route_ridge_score))
    print('The optimal ridge alpha is {}'.format(route_ridge.alpha_))
    print('The Lasso R2 score is {}'.format(route_lasso_score))
    print('The optimal lasso alpha is {}'.format(route_lasso.alpha_))
    return route_ridge, route_lasso

def test_model_on_hold(X_train,y_train,X_hold,y_hold,model,alpha,title,filename):
    standardizer = utils.XyScaler()
    standardizer.fit(X_train,y_train)
    X_train_std, y_train_std = standardizer.transform(X_train, y_train)
    X_hold_std, y_hold_std = standardizer.transform(X_hold, y_hold)
    final_model = model(alpha)
    final_model.fit(X_train_std,y_train_std)
    y_pred_std = final_model.predict(X_hold_std)
    X_hold, y_pred = standardizer.inverse_transform(X_hold,y_pred_std)
    plot_model_predictions(y_hold, y_pred, title, filename)
    final_score = final_model.score(X_hold_std,y_hold_std)
    final_mse = mse(y_hold,y_pred)
    print('Final R2 score: {}'.format(final_score))
    print('Final RMSE: {}'.format(np.sqrt(final_mse)))
    return final_score, final_mse

def get_coefs(model,X):
    df = pd.DataFrame(model.coef_)
    df['coef_names'] = X.columns
    return df

def find_error_scores(df,target_column,model_function,n):
    r2_scores = []
    rss_scores = []
    df2 = drop_columns_grades(df)
    X_train, y_train, X_hold, y_hold = split_data_grades(df2,target_column)
    for i in range(n+1):
        model, score, y_hat, y_true, y_test = model_function(X_train,y_train)
        r2_scores.append(score)
        mse_scores.append(mse(y_true,y_hat))
    return max(r2_scores),min(mse_scores)

def make_scatter_plots(df,target):
    all_columns = df.columns
    for col in all_columns:
        df.plot(kind='scatter', y=target, x=col, edgecolor='none', figsize=(12, 5))
        plt.xlabel(col)
        plt.ylabel('Grade')
        plt.show()

def plot_mse(model, X_train, y_train, X_test, y_test,title,filename):
    train_errors = []
    test_errors = []
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    for alpha in model.alphas:
        train_errors.append(mse(y_train, train_pred))
        test_errors.append(mse(y_test,test_pred))

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(np.log10(model.alphas), train_errors)
    ax.plot(np.log10(model.alphas), test_errors)
    ax.axvline(np.log10(model.alpha_), color='grey')
    ax.set_title(title)
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("MSE")
    plt.savefig(filename)


def plot_model_predictions(y_true,y_pred,title,filename):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    ax.set_title(title)
    ax = plt.scatter(x=y_pred,y=y_true,color='green',alpha=0.5)
    plt.savefig(filename)
    plt.clf()

def main_boulder():
    boulder = drop_columns_grades(boulders)
    X_train, y_train, X_hold, y_hold = split_data_grades(boulder,'usa_boulders')
    boulder_ridge, boulder_lasso = get_boulder_models(X_train, y_train)
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Ridge,boulder_ridge.alpha_,'Final Ridge Prediction for Boulder grades','images/ridge_model_boulder_final.png')
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Lasso,boulder_lasso.alpha_,'Final Lasso Prediction for Boulder grades','images/lasso_model_boulder_final.png')
    boulder_coefs = get_coefs(boulder_lasso,boulder.drop('usa_boulders',axis=1))
    print(boulder_coefs.sort_values(0))

def main_route():
    route = drop_columns_grades(routes)
    X_train, y_train, X_hold, y_hold = split_data_grades(route,'usa_routes')
    route_ridge, route_lasso = get_route_models(X_train,y_train)
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Ridge,route_ridge.alpha_,'Final Ridge Prediction for Route grades','images/ridge_model_route_final.png')
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Lasso,route_lasso.alpha_,'Final Lasso Prediction for Route grades','images/lasso_model_route_final.png')
    route_coefs = get_coefs(route_lasso,route.drop('usa_routes',axis=1))
    print(route_coefs.sort_values(0))

def plot_predictions_and_mse():
    plot_mse(route_ridge, X_train, y_train, ridge_X_test, ridge_y_true,"Ridge Regression Train and Test MSE" , 'images/route_ridge_MSE.png')
    plot_mse(route_lasso, X_train, y_train, lasso_X_test, lasso_y_true,"Lasso Regression Train and Test MSE" , 'images/route_lasso_MSE.png')
    plot_model_predictions(ridge_y_true,ridge_y_hats,'Ridge Prediction for Route grades','images/ridge_model_route.png')
    plot_model_predictions(lasso_y_true,lasso_y_hats,'Lasso Prediction for Route Grades','images/lasso_model_route.png')
    plot_model_predictions(ridge_y_true,ridge_y_hats,'Ridge Prediction for Boulder grades','images/ridge_model_boulder.png')
    plot_model_predictions(lasso_y_true,lasso_y_hats,'Lasso Prediction for Boulder Grades','images/lasso_model_boulder.png')
    plot_mse(boulder_ridge, X_train, y_train, ridge_X_test, ridge_y_true,"Ridge Regression Train and Test MSE" , 'images/boulder_ridge_MSE.png')
    plot_mse(boulder_lasso, X_train, y_train, lasso_X_test, lasso_y_true,"Lasso Regression Train and Test MSE" , 'images/boulder_lasso_MSE.png')

def restrict_boulder():
    boulders = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')

    boulders_restrict = boulders.copy()
    boulders_restrict = boulders_restrict.loc[boulders_restrict['usa_boulders'] >= 5]
    boulder_restrict = drop_columns_grades(boulders_restrict)
    X_train, y_train, X_hold, y_hold = split_data_grades(boulder_restrict,'usa_boulders')
    boulder_ridge, boulder_lasso = get_boulder_models(X_train, y_train)
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Ridge,boulder_ridge.alpha_,'Final Ridge Prediction for Boulder grades','images/ridge_model_boulder_restrict.png')
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Lasso,boulder_lasso.alpha_,'Final Lasso Prediction for Boulder grades','images/lasso_model_boulder_restrict.png')
    boulder_coefs = get_coefs(boulder_lasso,boulders_restrict.drop('usa_boulders',axis=1))
    print(boulder_coefs.sort_values(0))

def restrict_routes():
    routes = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')

    routes_restrict = routes.copy()
    routes_restrict = routes_restrict.loc[routes_restrict['usa_routes'] >= 5.11]
    route_restrict = drop_columns_grades(routes_restrict)
    X_train, y_train, X_hold, y_hold = split_data_grades(route_restrict,'usa_routes')
    route_ridge, route_lasso = get_route_models(X_train, y_train)
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Ridge,route_ridge.alpha_,'Final Ridge Prediction for Route grades','images/ridge_model_route_restrict.png')
    test_model_on_hold(X_train,y_train,X_hold,y_hold,Lasso,route_lasso.alpha_,'Final Lasso Prediction for Route grades','images/lasso_model_route_restrict.png')
    route_coefs = get_coefs(route_lasso,routes_restrict.drop('usa_routes',axis=1))
    print(route_coefs.sort_values(0))
