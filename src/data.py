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

b_grades = ['VB','V8/9','V5/V6','V4/V5','V3/4']
for grade in b_grades:
    climb = climb[(climb['usa_boulders'] != grade)]

#remove columns that aren't needed
drop_cols = ['best_area','worst_area','guide_area','climb_try','repeat','yellow_id','user_recommended','comment','last_year','competitions','raw_notes','shorthand','exclude_from_ranking','score','total_score','deactivated','occupation','birth_city','birth_country']
climb.drop(drop_cols, axis=1, inplace = True)

#convert kg to lbs for weight, cm to in for height
climb['weight'] = round(climb['weight'] * 2.20462,3)
climb['height'] = round(climb['height'] / 2.54,3)

#convert birth to datetime, add new column with age
climb.birth = climb.birth.str.replace('-','')
climb.birth = pd.to_datetime(climb['birth'], format='%Y%m%d')
now = pd.Timestamp(DT.datetime.now())
climb['birth'] = climb['birth'].where(climb['birth'] < now, climb['birth'] -  np.timedelta64(100, 'Y'))
climb['age'] = (now - climb['birth']).astype('<m8[Y]')

#convert columns with dates into datetime
climb.rec_date = pd.to_datetime(climb.rec_date, unit='s')
climb['rec_date'] = climb['rec_date'].apply(lambda x: x.date())
climb.date = pd.to_datetime(climb.date, unit='s')
climb['date'] = climb['date'].apply(lambda x: x.date())
climb.project_ascent_date = pd.to_datetime(climb.project_ascent_date, unit='s')
climb['project_ascent_date'] = climb['project_ascent_date'].apply(lambda x: x.date())

#lower case applied to all strings in df
climb = pd.concat([climb[col].astype(str).str.lower() for col in climb.columns],axis=1)

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

cols = ['rating', 'chipped', 'height', 'weight', 'age','user_id', 'grade_id', 'method_id', 'climb_type','sex','started', 'year','crag_id', 'sector_id']
for col in cols:
    climb[col] = pd.to_numeric(climb[col])

#split df into routes and boulders
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

boulders = climb.copy()
boulders = boulders.loc[climb['climb_type'] == 1]
boulders.drop('usa_routes',axis=1, inplace=True)

#convert usa_boulders to all numbers (remove V) - need to deal with /
for i in range(0,19):
    boulders.usa_boulders.replace('v{}'.format(i), '{}'.format(i), inplace=True)

boulders.to_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes.to_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')
