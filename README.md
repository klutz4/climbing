# Analyzing a Climbing Logbook



### Motivation and Background

As an avid climber, common questions we all have are:  
How does my height/weight/age affect how hard I can climb?  
Is this different for routes vs boulders?  
How long will it take me to progress to a certain level?  
Is this route/boulder graded correctly?

<img src='images/IMG_0326.png' width=400> <img src='images/IMG_0292.png' width=400>  
Left: V4, Right: Also V4??

As such, I've set out to see if I can answer some of these questions using data from climbers.

For the data, I'm only looking at routes graded 5.4 - 5.15a that were not a Top rope ascent and boulders graded V0 - V16 completed by climbers from the USA.

I have two ultimate goals:
1. Find a model that predicts which grade a route/boulder should have.
2. Find a model that predicts how hard users will climb the next year.

### Data Cleaning

This data comes from the site 8a.nu, where climbers can keep a log of which boulders and routes they have completed and when. The data came in the form of a sqlite database with 4 different tables: ascents, grades, methods and users. I converted the SQL tables to CSVs and then imported each to a `pandas` dataframe using the `import.py` script. The `clean_data.py`was used to clean the final `pandas` dataframe for use in EDA and modeling.

The importing and cleaning steps included:
* Merging the four dataframes into one.
* Removing users born outside of the United States.
* Dropping any rows with missing values in the birth, height, weight, usa_routes and usa_boulders columns.
* Removing deactivated users, users without a start date, route grades under 5.4, ascents labeled as Toprope.
* Dropping unwanted columns.
* Converting height and weight to inches and lbs, respectively.
* Converting birth and ascent_date to datetime format, and adding current_age and ascent_age columns.
* Adding a time_to_send column.
* Combining sponsor1, sponsor2, and sponsor3 into one sponsored columns with values 0 (not sponsored), 1 (sponsored).
* Splitting the dataframe into two: one for routes and one for boulders.

### EDA

<img src='images/boulders_per_gender.png'>
<img src='images/routes_per_gender.png'>  

<img src='images/boulders_per_age.png'>
<img src='images/routes_per_age.png'>  

<img src='images/boulders_per_height.png'>
<img src='images/routes_per_height.png'>  

<img src='images/boulders_per_weight.png'>
<img src='images/routes_per_weight.png'>  

<img src='images/boulders_send_time.png'>
<img src='images/routes_send_time.png'>  

#### Boulders
|Grade|Average ascent age|Grade|   Average ascent age
|-----|------------------|-----|------------------|
|V0|27.39 | V9 | 25.21|
|V1|26.12 | V10 | 24.59|
|V2|25.95| V11|24.18| 
| V3|25.73| V12|24.08|
| V4|25.64| V13|24.35|
| V5|25.56| V14|24.99|
| V6|25.76| V15|24.96|
| V7|25.46| V16|29.00|
| V8|26.30|

The overall average ascent age is 25.60.  

 #### Routes
|Grade|Average ascent age|Grade|   Average ascent age
|-----|------------------|-----|------------------|
|5.4|26.48| 5.12a|28.17|
|5.5|30.80| 5.12b|28.56|
|5.6|30.43| 5.12c|29.02| 
|5.7|27.82| 5.12d|28.85|
|5.8|28.26| 5.13a|28.47|
|5.9|28.45| 5.13b|28.17|
|5.10a|28.24| 5.13c|27.60|
|5.10b|28.36| 5.13d|26.78|
|5.10c|28.73| 5.14a|26.13|
|5.10d|28.73| 5.14b|25.14|
|5.11a|29.02| 5.14c|25.12|
|5.11b|29.21| 5.14d|25.25| 
|5.11d|29.14| 5.15a|25.70|

The overall average ascent age is 27.95.  

### Feature Engineering

Before modeling the data to predict the difficulty of a climb, I dropped any columns in the list below, since the column was either highly correlated with the grade (target column) or another column that remains in each dataframe.  

`['notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','grade_id','climb_name','ascent_date_day','ascent_date_month','chipped']`

Similarly, I dropped these columns before modeling the data to predict how hard users climbed in 2017.

`['notes','climb_type','crag','sector','climb_country','method','birth','ascent_date','date','climb_name']`

### Modeling the Difficulty of a Climb

I chose to use Ridge and Lasso to fit a model to my data using the sklearn's `RidgeCV` and `LassoCV` with alphas = np.logspace(-2,4,num=250) and cv=10. I first split my data into training and testing with a test size = 0.25, then standardized the training data, employed either `RidgeCV` or `LassoCV`, and compared the R2 scores.

Using the model with the best R2 score (Ridge), I fit a Ridge model with the optimal alpha from my training sets to the unseen data and plotted the predicted values vs. the true values.

<img src='images/ridge_model_boulder_final.png'>

Final R2-score: 0.2789  
Final RMSE: 2.4600  

<img src='images/ridge_model_route_final.png'> 

Final R2-score: 0.2667  
Final RMSE: 0.0133  
(The predicted range for route grades is 5.085 - 5.135)</br>

### Modeling How Hard Users Climbed in 2017

I followed the same process and code for modeling how hard users climbed in 2017 as with modeling the difficulty of a climb, with the added caveat that there were a limited number of rows with data for 2017. This forced the size of the feature matrix to shrink since a sample was taken from the full feature matrix to match the length of the target. 

Below are the results of the final models.

<img src='images/ridge_model_boulder_ability.png'>

Final R2-score: -0.0020  
Final RMSE: 2.9644  

<img src='images/ridge_model_route_ability.png'> 

Final R2-score: -0.0032  
Final RMSE: 0.0134  

### Results

### Future Work

### References

This data was downloaded from Kaggle, courtesy of David Cohen.
