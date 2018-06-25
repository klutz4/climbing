import pandas as pd

ascents = pd.read_csv('/Users/Kelly/galvanize/capstones/climbing_data/ascent.csv')
grades = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/grades.csv')
users = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/user.csv')
method = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/method.csv')

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

climb_data.to_csv('/Users/Kelly/galvanize/capstones/mod1/data/climb_data.csv')
