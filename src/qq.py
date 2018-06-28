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


boulders, routes = grade_model.import_data()
boulder = grade_model.drop_columns_grades(boulders)
X_train, y_train, X_hold, y_hold = grade_model.split_data_grades(boulder,'usa_boulders')
boulder_ridge, boulder_lasso = grade_model.get_boulder_models(X_train, y_train)
residuals = y_hold - boulder_ridge.predict(X_hold)
sm.graphics.qqplot(residuals)
plt.savefig('images/boulders_qqplot.png')
plt.show()
