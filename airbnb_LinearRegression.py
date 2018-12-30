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
# Data preprocessing: cleaning and transfoming data
################################################################################
# check missing data:
    # (1) show number of rows having missing data for each feature
    # (2) calculate percentage of null values for each column
listing_df.apply(lambda x:sum(x.isnull()))
total = listing_df.isnull().sum().sort_values(ascending=False)
percent = (listing_df.isnull().sum()/listing_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# dealing with NAN values
# listing_df.fillna(0, inplace=True)

# price: convert text to number
    # (1) replace ', ' to '' 
    # (2) remove character '$' in price
# , # (3) convert string to float
listing_df['price'] = listing_df['price'].apply(lambda x:float(x[1:].replace(',','')))

# exclude the listings with 0 for price,beds,bedrooms,accomodates etc
listing_df = listing_df[listing_df.price  > 0]
listing_df.shape



################################################################################
# EXAMINE VARIABLE
################################################################################
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
price
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# distribution plot
Price_Distribution(listing_df['price'])


################################################################################
# More features
################################################################################

df = listing_df[[
        'id',
        'neighbourhood_cleansed',
        'calculated_host_listings_count',
        'latitude',
        'longitude',
        'property_type',
        'room_type',
        'accommodates',
        'bed_type',
        'amenities',
        'price',
        'minimum_nights',
        'maximum_nights',
        'cancellation_policy',
        'instant_bookable',
        'street',
        'number_of_reviews',
        'availability_365',
        'availability_90',
        'availability_60',
        'availability_30',
        'has_availability',
        'host_since',
        'host_is_superhost',
        'host_total_listings_count',
        'bedrooms',
        'beds',
        'bathrooms',
        'description',
        'zipcode',
        'summary'
        ]]

#----------------------------------------------------------------------
# Examine numerical variables
#----------------------------------------------------------------------
df = listing_df
df = df[[
        'calculated_host_listings_count',
        'latitude',
        'longitude',
        'accommodates',
        'price',
        'minimum_nights',
        'maximum_nights',
        'number_of_reviews',
        'availability_365',
        'bedrooms',
        'beds',
        'bathrooms',
        'review_scores_rating',
        'reviews_per_month'
        ]]


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
        'reviews_per_month'
        ]]


#exclude the listings with 0 for price,beds,bedrooms,accomodates etc
df = df[df.accommodates  > 0]
df = df[df.price  > 0]
df = df[df.bathrooms >0]
df = df[df.bedrooms > 0]
df = df[df.beds > 0]
df = df[df.review_scores_rating > 0]


# add categorical variables
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




################################################################################
# LINEAR REGRESSION
################################################################################
features = [f for f in list(df.columns.values)
                if f not in ['price']]

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
# MAE
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
# CLASSIFICATION
############################################################
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
        'host_is_superhost'
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

X_df = df.drop(['host_is_superhost'],axis=1)

# Visualize data
# sns.pairplot(df,hue='host_is_superhost',size=1.5)


# split the data into a training set and a testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X_df,df.host_is_superhost, test_size=0.3,train_size = 0.7,random_state=1)
X_train.shape
X_test.shape

# choose a class of model
from sklearn.naive_bayes import GaussianNB 

# instantiate model
model = GaussianNB()

# fit model to data
model.fit(X_train, Y_train)                  

# predict on new data
Y_pred = model.predict(X_test)              

# Evaluate the model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Summary of the predictions made by the classifier
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
#  see the fraction of predicted labels that match their true value
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))



#######################################################################
# Logistic Regression
#######################################################################

# split the data into a training set and a testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X_df,df.host_is_superhost, test_size=0.3,train_size = 0.7,random_state=1)
X_train.shape
X_test.shape

# Choose a model
from sklearn.linear_model import LogisticRegression

# Initialize model and choose hyperparameters
classifier = LogisticRegression()

# Fit the model on training data set
classifier.fit(X_train, Y_train)

# Predict the model on test set
Y_pred = classifier.predict(X_test)

# Evaluate the model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Summary of the predictions made by the classifier
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(Y_pred,Y_test))


#######################################################################
# KNN
#######################################################################
# Choose a class of model
from sklearn.neighbors import KNeighborsClassifier
# Choose model hyperparameters
model = KNeighborsClassifier(n_neighbors=3)
# split the data into training set and test seet
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X_df,df.host_is_superhost, test_size=0.3,train_size = 0.7,random_state=1)

# fit the model on one set of data
model.fit(X_train, Y_train)
# Predict labels for test data
Y_predict = model.predict(X_test)
# evaluate the model on the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_predict))
# K-fold cross-validation
from sklearn.model_selection import cross_val_score
print('K-fold: ',cross_val_score(model, X_df, df.host_is_superhost, cv=10))


#----------------------------------------------------------------------
# Randomforest
#----------------------------------------------------------------------
from sklearn.ensemble.forest import RandomForestRegressor,\
    RandomForestClassifier
from sklearn import cross_validation, metrics

clf = RandomForestClassifier (n_estimators=20, max_features=9, min_samples_split=5)
clf.fit(X_train, Y_train)
print (pd.crosstab(Y_train, clf.predict(X_train), rownames=["Actual"]))
print (np.sum(Y_train==clf.predict(X_train)) / float(len(X_train)))
print (np.sum(Y_test==clf.predict(X_test)) / float(len(X_test)))








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
















