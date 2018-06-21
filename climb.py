import pandas as pd
import numpy as np


ascents = pd.read_csv('data/ascent.csv')
grades = pd.read_csv('data/grades.csv')
users = pd.read_csv('data/user.csv')
method = pd.read_csv('data/method.csv')

users = users[(users['country'] == 'USA')]
for col in ['first_name','last_name','presentation','interests', 'anonymous']:
    users.drop(col,axis=1,inplace=True)

users.replace(['NaN'], np.nan, inplace = True)
for col in ['birth', 'height', 'weight']:
    users[col].dropna(inplace = True)

users.rename(index=str, columns= {'country':'birth_country', 'city':'birth_city'}, inplace=True)
users = users[(users['height'] != 0.0) & (users['weight'] != 0.0)]

grades = grades[['id', 'usa_routes', 'usa_boulders']]

method.rename(index=str, columns= {'name':'method'}, inplace=True)

ascents.rename(index=str, columns={'name':'climb_name', 'country' : 'climb_country'}, inplace=True)
ascents.drop('description',axis=1,inplace=True)

ag= ascents.merge(grades, how = 'left', left_on = 'grade_id', right_on = 'id')
agm = ag.merge(method, how = 'left', left_on = 'method_id', right_on = 'id')
climb_data = agm.merge(users, how = 'left', left_on = 'user_id', right_on = 'id')

climb_data.drop(['id_x','id_y'],axis=1,inplace=True)

for col in ['birth', 'height', 'weight']:
    climb_data[col].dropna(inplace=True)
