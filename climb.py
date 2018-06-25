import pandas as pd
import datetime as DT
import numpy as np
import matplotlib.pyplot as plt

ascents = pd.read_csv('/Users/Kelly/galvanize/capstones/climbing_data/ascent.csv')
grades = pd.read_csv('data/grades.csv')
users = pd.read_csv('data/user.csv')
method = pd.read_csv('data/method.csv')

pd.get_option("display.max_columns")
pd.set_option("display.max_columns", 100)

users = users[(users['country'] == 'USA')]
for col in ['first_name','last_name','presentation','interests', 'anonymous']:
    users.drop(col,axis=1,inplace=True)

users.rename(index=str, columns= {'country':'birth_country', 'city':'birth_city'}, inplace=True)

grades = grades[['id', 'usa_routes', 'usa_boulders']]

method.rename(index=str, columns= {'name':'method'}, inplace=True)

ascents.rename(index=str, columns={'name':'climb_name', 'country' : 'climb_country'}, inplace=True)
ascents.drop('description',axis=1,inplace=True)

ag= ascents.merge(grades, how = 'left', left_on = 'grade_id', right_on = 'id')
agm = ag.merge(method, how = 'left', left_on = 'method_id', right_on = 'id')
climb_data = agm.merge(users, how = 'left', left_on = 'user_id', right_on = 'id')

climb_data.drop(['id_x','id_y'],axis=1,inplace=True)
fill_nan_cols = ['birth', 'height', 'weight','usa_routes','usa_boulders']
climb_data[fill_nan_cols] = climb_data[fill_nan_cols].fillna(0)
climb_data = climb_data[(climb_data['height'] != 0.0) & (climb_data['weight'] != 0.0) & (climb_data['birth'] != 0.0) & (climb_data['usa_routes'] != 0.0) & (climb_data['usa_boulders'] != 0.0) ]

climb = climb_data.copy()
#sex: 0 = male, 1 = female
#climb_type: 0 = routes, 1 = boulders

#remove rows without a starting year
climb = climb[climb['started'] != 0]

#remove deactivated users
climb = climb[climb['deactivated'] != 1]

#remove toprope ascents
climb = climb[(climb['method'] != 'Toprope')]

#remove route grades 3/4, 5.1, 5.3 and boulder grade VB
grades = ['3/4','5.1','5.3']
for grade in grades:
    climb = climb[(climb['usa_routes'] != grade)]

climb = climb[(climb['usa_boulders'] != 'VB')]

#remove best_area, worst_area, guide_area, climb_try, repeat, yellow_id (?), user_recommended, comment (use notes instead)
drop_cols = ['best_area','worst_area','guide_area','climb_try','repeat','yellow_id','user_recommended','comment','last_year','competitions']
climb.drop(drop_cols, axis=1, inplace = True)

#convert kf to lbs for weight, cm to in for height
climb['weight'] = climb['weight'] * 2.20462
climb['height'] = climb['height'] / 2.54

#convert birth to datetime, add new column with age
climb.birth = climb.birth.str.replace('-','')
climb.birth = pd.to_datetime(climb['birth'], format='%Y%m%d')
now = pd.Timestamp(DT.datetime.now())
climb['birth'] = climb['birth'].where(climb['birth'] < now, climb['birth'] -  np.timedelta64(100, 'Y'))
climb['age'] = (now - climb['birth']).astype('<m8[Y]')

#convert columns with dates into datetime
climb.rec_date = pd.to_datetime(climb.rec_date, unit='s')
climb.date = pd.to_datetime(climb.date, unit='s')
climb.project_ascent_date = pd.to_datetime(climb.project_ascent_date, unit='s')

#need to separate datetime columns into something useful - drop the timestamp?
#started column only has year
#make sponsored (1 or 0) column
#convert usa_boulders to all numbers (remove V) - need to deal with /
for i in range(0,19):
    climb.usa_boulders.replace('V{}'.format(i), '{}'.format(i), inplace=True)

#convert usa_routes to all numbers
grade_dict = {'a':1,'b':2,'c':3,'d':4}
