# Commonn processing
import pandas as pd
import numpy as np
from collections import Counter

# Visualization
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns

# NLP
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import string
import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from langdetect import detect
  
# Datetime
from datetime import date
import datetime
import calendar
import holidays

# predictive model
import sklearn.metrics as metrics
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics
from sklearn import datasets
from sklearn import cross_validation
from sklearn import linear_model
import statsmodels.formula.api as smf

# statistics
import scipy.stats

# Configuration
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 505)

################################################################################
# Read data
################################################################################
listing_df = pd.read_csv('D:\\MyProjects\\01_Airbnb\\London_listings.csv')


listing_df.shape
listing_df.head(5)
listing_df.describe()
# unique values for all fields
listing_df.apply(lambda x:len(x.unique()))
# count unique values of a variable
listing_df.property_type.value_counts()
listing_df.room_type.value_counts()
# Datatype of a data frame
listing_df.info()
listing_df.columns


calendar_df =pd.read_csv("D:\\MyProjects\\01_Airbnb\\London_calendar.csv")
calendar_df.shape
calendar_df.head()
calendar_df.describe()
calendar_df.apply(lambda x:len(x.unique()))


review_df =pd.read_csv("D:\\MyProjects\\01_Airbnb\\London_reviews.csv")
review_df.shape
review_df.head()
review_df.describe()
review_df.apply(lambda x:len(x.unique()))


################################################################################
# Common function
################################################################################

def Price_Distribution(df):
    sns.set(style="whitegrid")
    
    # boxplot
    plt.figure(figsize=(20, 4))
    plt.subplot(121)
    sns.boxplot(x=df)
    plt.title('Price Distribution - all price')    
    plt.subplot(122)
    sns.set(style="whitegrid")
    sns.boxplot(x=df[df<501])
    plt.xticks(np.arange(0, 500, step=25))
    plt.title('Price Distribution - Price from 0 to 500')    

    # Density plot
    plt.figure(figsize=(20, 8))
    plt.subplot(121)
    sns.distplot(df, hist = True, kde = True,kde_kws = {'shade': True, 'linewidth': 3})
    plt.title('Price Distribution - all price')
    plt.subplot(122)
    sns.distplot(df[df<501], hist = True, kde = True,kde_kws = {'shade': True, 'linewidth': 3})
    plt.xticks(np.arange(0, 500, step=25))
    plt.title('Price Distribution - Price from 0 to 500')


################################################################################
# Feature Screening
################################################################################
# check missing data:
    # (1) show number of rows having missing data for each feature
    # (2) calculate percentage of null values for each column
listing_df.apply(lambda x:sum(x.isnull()))
total = listing_df.isnull().sum().sort_values(ascending=False)
percent = (listing_df.isnull().sum()/listing_df.isnull().count()).sort_values(ascending=True)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.to_csv('D:\\MyProjects\\01_Airbnb\\missing_data.csv')



################################################################################
# ALL features after removing missing data and incommon sense
################################################################################

df = listing_df[[
        'has_availability',
        'availability_30',
        'availability_60',
        'availability_90',
        'availability_365',
        'number_of_reviews',
        'street',
        'instant_bookable',
        'is_business_travel_ready',
        'cancellation_policy',
        'require_guest_profile_picture',
        'require_guest_phone_verification',
        'maximum_nights',
        'minimum_nights',
        'extra_people',
        'guests_included',
        'price',
        'amenities',
        'bed_type',
        'accommodates',
        'room_type',
        'property_type',
        'is_location_exact',
        'longitude',
        'latitude',
        'calculated_host_listings_count',
        'neighbourhood_cleansed',
        'id',
        'host_listings_count',
        'host_identity_verified',
        'host_total_listings_count',
        'host_is_superhost',
        'host_since',
        'host_name',
        'host_has_profile_pic',
        'name',
        'bedrooms',
        'beds',
        'bathrooms',
        'host_location',
        'market',
        'description',
        'zipcode',
        'summary',
        'neighbourhood',
        'host_neighbourhood',
        'reviews_per_month',
        'review_scores_rating',
        'review_scores_cleanliness',
        'review_scores_communication',
        'review_scores_accuracy',
        'review_scores_checkin',
        'review_scores_location',
        'review_scores_value',
        'cleaning_fee',
        'space'
        ]]
df.shape
df.dtypes


################################################################################
# Data preprocessing: cleaning and transfoming data
################################################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
price: convert price to number
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# price: convert text to number
    # (1) replace ', ' to '' 
    # (2) remove character '$' in price
# , # (3) convert string to float
df['price'] = df['price'].apply(lambda x:float(x[1:].replace(',','')))
# exclude the listings with 0 for price,beds,bedrooms,accomodates etc
df = df[df.price  > 0]
df.shape

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
extra_people: convert extra_people to number
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
df['extra_people'] = df['extra_people'].apply(lambda x:float(x[1:].replace(',','')))
df['extra_people'][0:10] 
df.shape

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
cleaning_fee: convert cleaning_fee to number
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# dealing with NAN values
df['cleaning_fee'].fillna('$0.00', inplace=True)
df['cleaning_fee'] = df['cleaning_fee'].apply(lambda x:float(x[1:].replace(',','')))
df['cleaning_fee'][0:10] 
df.shape


################################################################################
# Examine numerical variables
################################################################################
df.dtypes

#----------------------------------------------------------------------
# List of numerical features
#----------------------------------------------------------------------
df_bk = df
df = df_bk

df = df[[
        'availability_30',
        'availability_60',
        'availability_90',
        'availability_365',
        'number_of_reviews',
        'maximum_nights',
        'minimum_nights',
        'guests_included',
        'price',
        'accommodates',
        'longitude',
        'latitude',
        'calculated_host_listings_count',
        'host_listings_count',
        'host_total_listings_count',
        'bedrooms',
        'beds',
        'bathrooms',
        'reviews_per_month',
        'review_scores_rating',
        'review_scores_cleanliness',
        'review_scores_communication',
        'review_scores_accuracy',
        'review_scores_checkin',
        'review_scores_location',
        'review_scores_value',
        'extra_people',
        'cleaning_fee'
        ]]

df.shape
#exclude the listings with 0 for price,beds,bedrooms,accomodates etc
df = df[df.price  > 0]
df = df[df.accommodates  > 0]
df = df[df.bathrooms >0]
df = df[df.bedrooms > 0]
df = df[df.beds > 0]

df.shape
df.dtypes
df.head()

# check missing values
df.apply(lambda x:sum(x.isnull()))
# show rows having NAN values
df.loc[df.host_listings_count.isnull(),'host_listings_count']
df = df[~np.isnan(df.host_listings_count)]
df.loc[df.host_total_listings_count.isnull(),'host_listings_count']
df = df[~np.isnan(df.host_total_listings_count)]
df.loc[df.reviews_per_month.isnull(),'reviews_per_month']
df = df[~np.isnan(df.reviews_per_month)]
df.loc[df.review_scores_rating.isnull(),'review_scores_rating']
df = df[~np.isnan(df.review_scores_rating)]
df.loc[df.review_scores_cleanliness.isnull(),'review_scores_cleanliness']
df = df[~np.isnan(df.review_scores_cleanliness)]
df.loc[df.review_scores_communication.isnull(),'review_scores_communication']
df = df[~np.isnan(df.review_scores_communication)]
df.loc[df.review_scores_accuracy.isnull(),'review_scores_accuracy']
df = df[~np.isnan(df.review_scores_accuracy)]
df.loc[df.review_scores_checkin.isnull(),'review_scores_checkin']
df = df[~np.isnan(df.review_scores_checkin)]
df.loc[df.review_scores_location.isnull(),'review_scores_location']
df = df[~np.isnan(df.review_scores_location)]
df.loc[df.review_scores_value.isnull(),'review_scores_value']
df = df[~np.isnan(df.review_scores_value)]

# count unique values of a variable
df.host_listings_count.value_counts()



################################################################################
# Investigating correlation between input variables and responses
################################################################################
# distribution of numerical variables
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(15,20))
plt.savefig("D:\\MyProjects\\01_Airbnb\\attribute_histogram_plots")
plt.show()

#correlation matrix
corrmat = df.corr(method = 'spearman')
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cmap='RdYlGn_r',vmax=1.0,vmin=-1.0,linewidths=2.5, square=True);

'''
# pair plot
features = [f for f in list(df.columns.values) if f not in ['price']]
from pandas.tools.plotting import scatter_matrix
scatter_matrix(df[features], figsize=(12, 8))
plt.savefig('D:\\MyProjects\\01_Airbnb\\matrix.png')
sns.pairplot(df, x_vars=df.columns, y_vars='price', size=7, aspect=0.7, kind='reg')
'''

################################################################################
# Split data for training model
################################################################################
df.apply(lambda x:sum(x.isnull()))


# split data into training and test set
split_data= df.drop(['price'],axis=1)
split_data.head()
X_train, X_test, Y_train, Y_test=cross_validation.train_test_split(split_data,df.price, test_size=0.3,train_size = 0.7,random_state=99)
X_train.shape
X_test.shape

train_df = pd.concat((X_train,Y_train),axis = 1)






################################################################################
# LINEAR REGRESSION
################################################################################
# Create formula
features = [f for f in list(df.columns.values) if f not in ['price']]
formula = f'price ~ {"+".join(features)}'

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
### STATSMODELS ###
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# create a fitted model
lm1 = smf.ols(formula = formula,data = train_df).fit()
# print the coefficients
lm1.params
# print a summary of the fitted model
lm1.summary()
# calculate r-sauare
lm1.rsquared

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
### SCIKIT-LEARN ###
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Choose a class of model
from sklearn.linear_model import LinearRegression
# Choose model hyperparameters
lm2 = LinearRegression()

# Fit the model to the train data set
lm2.fit(X_train,Y_train)
# Finding the intercept and coefficients
print(lm2.intercept_)
print(lm2.coef_)

# Model output: pair the feature names with the coefficients
list(zip(train_df.columns, lm2.coef_))

# Making the predictions on test set
Y_pred = lm2.predict(X_test)
# Comparing the prediction with actuals
test_pred_df = pd.DataFrame({'actual': Y_test, 'predicted': np.round(Y_pred, 2) })
print(test_pred_df[0:10])

# Evaluating Model Accuracy
# MAE: 37.59
MAE = np.round(metrics.mean_absolute_error(Y_test, Y_pred),2)
# MSE
MSE = np.round(metrics.mean_squared_error(Y_test, Y_pred),2)
# RMSE: 102.63
RMSE = np.round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2)
# R squared
Rsquared = np.round(metrics.r2_score(Y_test, Y_pred),2)

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
linreg = LinearRegression()
cv_scores = cross_val_score(linreg, X_train, Y_train, scoring = 'r2', cv = 10)
print(cv_scores)
print( "Average r2 score: ", np.round( np.mean( cv_scores ), 2 ) )
print( "Standard deviation in r2 score: ", np.round( np.std( cv_scores ), 2) )


################################################################################
# REGULARIZATION
################################################################################
# Ridge regression
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=[0.1, 0.5, 1.0,5.0, 10.0,15,20], cv=3)
reg.fit(X_train, Y_train)       
reg.alpha_     

from sklearn.linear_model import Ridge
clf = Ridge(alpha=10.0)
clf.fit(X_train, Y_train) 
Y_pred = clf.predict(X_test)
clf.coef_
clf.intercept_ 
# RMSE:
RMSE = np.round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2)
# R squared
Rsquared = np.round(metrics.r2_score(Y_test, Y_pred),2)



# Lasso Regression
from sklearn import linear_model
reg = linear_model.LassoCV(alphas=[0.1, 0.5, 1.0,5.0, 10.0,15,20], cv=3)
reg.fit(X_train, Y_train)       
reg.alpha_     

reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train, Y_train) 
Y_pred = reg.predict(X_test)
# RMSE: 102.77
RMSE = np.round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2)
# R squared: 0.29
Rsquared = np.round(metrics.r2_score(Y_test, Y_pred),2)



################################################################################
# RANDOM FOREST
################################################################################
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators = 100, random_state=42)
forest_reg.fit(X_train, Y_train)
Y_pred = forest_reg.predict(X_test)

#----------------------------------------------------------------------
# Evaluating Model Accuracy
#----------------------------------------------------------------------
# R-squared
print('Random Forest R squared": %.4f' % forest_reg.score(X_test, Y_test))
# Calculate the absolute errors
errors = abs(Y_pred - Y_test)
# MAE (Mean absolute error)
np.round(metrics.mean_absolute_error(Y_test, Y_pred),2)
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / Y_test)
print(mape)
# MSE
forest_mse = np.round(metrics.mean_squared_error(Y_test, Y_pred),2)
# RMSE: 110.1400
forest_rmse = np.round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2)
print('Random Forest RMSE: %.4f' % forest_rmse)


#----------------------------------------------------------------------
# Variable Importance
#----------------------------------------------------------------------
feature_labels = X_train.columns

# Get numerical feature importance
importances = list(forest_reg.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_labels, importances)]
# Sort the feature importance by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importance 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

importance = forest_reg.feature_importances_
feature_indexes_by_importance = importance.argsort()
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))

# Visualizations
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_labels, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
plt.show()


################################################################################
# GRADIENT BOOSTING
################################################################################
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
model = ensemble.GradientBoostingRegressor()
model.fit(X_train, Y_train)

print('Gradient Boosting R squared": %.4f' % model.score(X_test, Y_test))

Y_pred = model.predict(X_test)
boosting_mse = np.round(metrics.mean_squared_error(Y_test, Y_pred),2)
# RMSE: 104.2634
boosting_rmse = np.sqrt(boosting_mse)
print('Gradient Boosting RMSE: %.4f' % boosting_rmse)

feature_labels = X_train.columns

# Get numerical feature importance
importances = list(model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_labels, importances)]
# Sort the feature importance by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importance 
[print('{:50} Importance: {}'.format(*pair)) for pair in feature_importances];





############################################################
# P11: Prepare variables
############################################################
#exclude the listings with 0 for price,beds,bedrooms,accomodates etc
df = listing_df[listing_df.bathrooms >0]
df = listing_df[listing_df.bedrooms > 0]
df = listing_df[listing_df.beds > 0]
df = listing_df[listing_df.price  > 0]
df = listing_df[listing_df.review_scores_rating  > 0]
df = listing_df[listing_df.accommodates  > 0]
df.shape

# convert categorical columns to numeric values
room_type = pd.get_dummies(listing_df.room_type).astype(int)
room_type.head()

# drop original columns and replace them with indicator columns
df = df[['bathrooms','bedrooms','beds','price','number_of_reviews','accommodates','review_scores_rating']]
df = pd.concat((df,  room_type[['Private room','Shared room']]), axis = 1)
df.head()
df.shape
df.columns

#correlation matrix
corrmat = df.corr(method = 'spearman')
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cmap='RdYlGn_r',vmax=1.0,vmin=-1.0,linewidths=2.5, square=True);

import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.savefig("D:\\MyProjects\\01_Airbnb\\attribute_histogram_plots")
plt.show()


from pandas.tools.plotting import scatter_matrix
attributes = ['bathrooms','bedrooms','beds','price','number_of_reviews','accommodates','review_scores_rating']
scatter_matrix(df[attributes], figsize=(12, 8))
plt.savefig('D:\\MyProjects\\01_Airbnb\\matrix.png')

sns.pairplot(df, x_vars=split_data.columns, y_vars='price', size=7, aspect=0.7, kind='reg')


# split data into training and test set
split_data= df.drop(['price'],axis=1)
X_train, X_test, Y_train, Y_test=cross_validation.train_test_split(split_data,df.price, test_size=0.3,train_size = 0.7,random_state=99)
X_train.shape
X_test.shape

train_df = pd.concat((X_train,Y_train),axis = 1)
train_df.columns = ['bathrooms', 'bedrooms', 'beds', 'number_of_reviews', 'accommodates', 'review_scores_rating', 'Private_room', 'Shared_room','price']
split_data.head()

################################################################################
# LINEAR REGRESSION
################################################################################
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
### STATSMODELS ###
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# create a fitted model
lm1 = smf.ols(formula = 'price ~ bathrooms+bedrooms+beds+number_of_reviews+accommodates+review_scores_rating+Private_room+Shared_room',data = train_df).fit()
# print the coefficients
lm1.params
# print a summary of the fitted model
lm1.summary()
# calculate r-sauare
lm1.rsquared

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
### SCIKIT-LEARN ###
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# select input variables
feature_cols = ['bathrooms', 'bedrooms', 'beds', 'number_of_reviews', 'accommodates', 'review_scores_rating', 'Private_room', 'Shared_room']
X_train = train_df[feature_cols]
X_train.shape
X_train.head()

# Choose a class of model
from sklearn.linear_model import LinearRegression
# Choose model hyperparameters
lm2 = LinearRegression()

# Fit the model to the train data set
lm2.fit(X_train,Y_train)
# Finding the intercept and coefficients
print(lm2.intercept_)
print(lm2.coef_)

# Model output: pair the feature names with the coefficients
list(zip(train_df.columns, lm2.coef_))

# Making the predictions on test set
Y_pred = lm2.predict(X_test)
# Comparing the prediction with actuals
test_pred_df = pd.DataFrame({'actual': Y_test, 'predicted': np.round(Y_pred, 2) })
print(test_pred_df[0:10])

# Evaluating Model Accuracy
# MAE
MAE = np.round(metrics.mean_absolute_error(Y_test, Y_pred),2)
# MSE
MSE = np.round(metrics.mean_squared_error(Y_test, Y_pred),2)
# RMSE: 106.13
RMSE = np.round(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)),2)
# R squared
Rsquared = np.round(metrics.r2_score(Y_test, Y_pred),2)

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
linreg = LinearRegression()
cv_scores = cross_val_score(linreg, X_train, Y_train, scoring = 'r2', cv = 10)
print(cv_scores)
print( "Average r2 score: ", np.round( np.mean( cv_scores ), 2 ) )
print( "Standard deviation in r2 score: ", np.round( np.std( cv_scores ), 2) )




#######################################################################
# PCA
#######################################################################
    
X_data = df[['bathrooms','bedrooms','beds','price','number_of_reviews','accommodates','review_scores_rating']] 
    
from sklearn.decomposition import PCA  # 1. Choose the model class
PCA_model = PCA(n_components=2)            # 2. Instantiate the model with hyperparameters
PCA_model.fit(X_data)                      # 3. Fit to data. Notice y is not specified!
X_2D = PCA_model.transform(X_data)         # 4. Transform the data to two dimensions

PCA_df = pd.DataFrame()
PCA_df['PCA1'] = X_2D[:, 0]
PCA_df['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", data=PCA_df, fit_reg=False);
plt.show()









#----------------------------------------------------------------------
# Add categorical variables
#----------------------------------------------------------------------

#this will create 4 columns namely flexible, moderate, strict, super_strict_30 which are its categories
cancel_policy = pd.get_dummies(listing_df.cancellation_policy).astype(int)
cancel_policy.head()

# Similiarly converting remaining categorial column,instant_booking  into indiactor variables
instant_booking = pd.get_dummies(listing_df.instant_bookable, prefix = 'instant_booking').astype(int)
instant_booking.head()

property_type = pd.get_dummies(listing_df.property_type).astype(int)
property_type.head()

neibourhood_type = pd.get_dummies(listing_df.neighbourhood_cleansed).astype(int)
neibourhood_type.head()











#----------------------------------------------------------------------
# add categorical variables
#----------------------------------------------------------------------
df = listing_df
df = df[[
        'calculated_host_listings_count',
        'longitude',
        'accommodates',
        'price',
        'number_of_reviews',
        'availability_365',
        'bedrooms',
        'beds',
        'bathrooms',
        'review_scores_rating',
        'reviews_per_month',
        'room_type',
        'neighbourhood_cleansed'
        ]]
#exclude the listings with 0 for price,beds,bedrooms,accomodates etc
df = df[df.accommodates  > 0]
df = df[df.price  > 0]
df = df[df.bathrooms >0]
df = df[df.bedrooms > 0]
df = df[df.beds > 0]
df = df[df.review_scores_rating > 0]


df.apply(lambda x:sum(x.isnull()))
df.shape


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outlet
var_mod = ['room_type','neighbourhood_cleansed']
for i in var_mod:
    df[i] = le.fit_transform(df[i])
    
df.head(1)   

#One Hot Coding:
df = pd.get_dummies(df, columns=['room_type','neighbourhood_cleansed',])

df.shape
df.columns






X_df = df.drop(['price'],axis=1)
X_df.head()

#correlation matrix
corrmat = df.corr(method = 'pearson')
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cmap='RdYlGn_r',vmax=1.0,vmin=-1.0,linewidths=2.5, square=True);

from pandas.plotting.scatter_matrix import scatter_matrix
scatter_matrix(df, figsize=(12, 8))


# split data into training and test set
X_df= df.drop(['price'],axis=1)
X_train, X_test, Y_train, Y_test=cross_validation.train_test_split(X_df,df.price, test_size=0.3,train_size = 0.7,random_state=99)
X_train.shape
X_test.shape

train_df = pd.concat((X_train,Y_train),axis = 1)
train_df.columns 
train_df.head()








