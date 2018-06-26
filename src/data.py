import pandas as pd
import datetime as DT
import numpy as np
import matplotlib.pyplot as plt

climb_data = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/climb_data.csv')

climb = climb_data.copy()
#sex: 0 = male, 1 = female
#climb_type: 0 = routes, 1 = boulders

#remove rows without a starting year
years = [0,1901,1914,1960,2020]
for year in years:
    climb = climb[climb['started'] != year]

#remove deactivated users
climb = climb[climb['deactivated'] != 1]

#remove toprope ascents
climb = climb[(climb['method'] != 'Toprope')]

#remove route grades 3/4, 5.1, 5.3 and boulder grade VB
r_grades = ['3/4','5.1','5.3']
for grade in r_grades:
    climb = climb[(climb['usa_routes'] != grade)]

#remove columns that aren't needed
drop_cols = ['best_area','worst_area','guide_area','climb_try','repeat','yellow_id','user_recommended','comment','last_year','competitions','raw_notes','shorthand','exclude_from_ranking','score','total_score','deactivated','occupation','birth_city','birth_country','rec_date','project_ascent_date']
climb.drop(drop_cols, axis=1, inplace = True)

#lower case applied to all strings in df
strings = ['notes','crag','sector','climb_country','method','climb_name']
for col in strings:
    climb[col] = climb[col].astype(str).str.lower()

#convert columns to numeric
cols = ['rating', 'chipped', 'height', 'weight','user_id', 'grade_id', 'method_id', 'climb_type','sex','started', 'year','crag_id', 'sector_id']
for col in cols:
    climb[col] = pd.to_numeric(climb[col])

#convert kg to lbs for weight, cm to in for height
climb['weight'] = round(climb['weight'] * 2.20462, 3)
climb['height'] = round(climb['height'] / 2.54, 3)
climb = climb[(climb['height'] > 57) & (climb['height'] < 84)]

#convert birth to datetime, add new column with age
climb.birth = climb.birth.str.replace('-','')
climb.birth = pd.to_datetime(climb['birth'], format='%Y%m%d')
now = pd.Timestamp(DT.datetime.now())
millennium = pd.Timestamp('1/1/2000')
climb['birth'] = climb['birth'].where(climb['birth'] < now, climb['birth'] -  np.timedelta64(100, 'Y'))
climb['current_age'] = (now - climb['birth']).astype('<m8[Y]')
climb = climb[(climb['current_age'] > 15)]

#convert columns with dates into datetime and add ascent age column
climb.date = pd.to_datetime(climb.date, unit='s')
climb['ascent_date'] = climb['date']
# climb['date'] = climb['date'].apply(lambda x: x.date())
climb['ascent_date'] = climb['ascent_date'].where(climb['ascent_date'] < now, climb['ascent_date'] -  np.timedelta64(100, 'Y'))
climb['ascent_age'] = (climb['ascent_date'] - climb['birth']).astype('<m8[Y]')
climb = climb[(climb['ascent_age'] > 15)]

dates = ['date','birth']
for col in dates:
    climb[col] = climb[col].astype('datetime64')

#split dates into month, day
climb['ascent_date_year'] = climb.ascent_date.dt.year
climb['ascent_date_month'] = climb.ascent_date.dt.month
climb['ascent_date_day'] = climb.ascent_date.dt.day

#add time to complete proj column
climb['time_to_send'] = (climb['ascent_date_year'] - climb['started'])

#make sponsored (1 or 0) column
sponsors = ['sponsor1','sponsor2','sponsor3']
for col in sponsors:
    climb[col].replace('nan',0, inplace=True)

for col in sponsors:
    for item in climb[col].unique():
        if item != 0:
            climb[col].replace(item,1,inplace=True)

climb['sponsored'] = climb[['sponsor1','sponsor2','sponsor3']].max(axis=1)
climb.drop(sponsors, axis=1, inplace=True)

climb.drop('Unnamed: 0',axis=1,inplace=True)

#create routes df
routes = climb.copy()
routes = routes.loc[climb['climb_type'] == 0]
routes.drop('usa_boulders',axis=1, inplace=True)

#convert usa_routes to all numbers
grade_dict = {'a':1,'b':2,'c':3,'d':4}
for i in range(10,16):
    for key in grade_dict:
        routes.usa_routes.replace('5.{}{}'.format(i,key),'5.{}{}'.format(i,grade_dict.get(key)),inplace=True)
for i in range(4,10):
        routes.usa_routes.replace('5.{}'.format(i),'5.0{}'.format(i),inplace=True)
routes['usa_routes'] = pd.to_numeric(routes['usa_routes'])

dates = ['ascent_date','birth']
for col in dates:
    routes[col] = routes[col].astype('datetime64')

#create boulders df
boulders = climb.copy()
boulders = boulders.loc[climb['climb_type'] == 1]
boulders.drop('usa_routes',axis=1, inplace=True)

b_grades = ['VB','V8/9','V5/V6','V4/V5','V3/4']
for grade in b_grades:
    boulders = boulders[(boulders['usa_boulders'] != grade)]

#convert usa_boulders to all numbers (remove V) - need to deal with /
for i in range(0,19):
    boulders.usa_boulders.replace('V{}'.format(i), '{}'.format(i), inplace=True)
boulders['usa_boulders'] = pd.to_numeric(boulders['usa_boulders'])

dates = ['ascent_date','birth']
for col in dates:
    boulders[col] = boulders[col].astype('datetime64')

boulders.to_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes.to_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')
