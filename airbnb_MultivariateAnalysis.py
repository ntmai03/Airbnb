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
from sklearn import ensemble

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
listing_df = listing_df[['id','name','summary','longitude','latitude','space','description','instant_bookable','neighborhood_overview','neighbourhood_cleansed','host_id','host_name','host_since',
                 'host_response_time','street', 'zipcode','review_scores_rating','property_type','room_type','accommodates','bathrooms','bedrooms','beds','reviews_per_month','amenities','cancellation_policy','number_of_reviews','price']]
listing_df.shape
listing_df.head()
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
# Data preprocessing: cleaning and transfoming data
################################################################################
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Listing
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# missing data:
    # (1) show number of rows having missing data for each feature
    # (2) replacing NaN values with 0
listing_df.apply(lambda x:sum(x.isnull()))
listing_df.fillna(0, inplace=True)

# text to number: 
    # (1) replace ', ' to '' 
    # (2) remove character '$' in price
# , # (3) convert string to float
listing_df['price'] = listing_df['price'].apply(lambda x:float(x[1:].replace(',','')))

# exclude the listings with 0 for price,beds,bedrooms,accomodates etc
listing_df = listing_df[listing_df.price  > 0]
listing_df.shape

review_df = review_df.dropna()
review_df.shape



############################################################
# EXAMINE VARIABLE
############################################################
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
target
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#histogram and normal probability plot
sns.distplot(listing_df['price'])
fig = plt.figure()
res = stats.probplot(listing_df['price'], plot=plt)
#skewness and kurtosis
print("Skewness: %f" % listing_df['price'].skew())
print("Kurtosis: %f" % listing_df['price'].kurt())


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
check out missing data
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
total = listing_df.isnull().sum().sort_values(ascending=False)
percent = (listing_df.isnull().sum()/listing_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)




'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Correlation
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#correlation matrix
corrmat = listing_df.corr(method = 'spearman')
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);





############################################################
# P11: PREDICTIVE MODELS
############################################################
#exclude the listings with 0 for price,beds,bedrooms,accomodates etc
listing_df = listing_df[listing_df.bathrooms >0]
listing_df = listing_df[listing_df.bedrooms > 0]
listing_df = listing_df[listing_df.beds > 0]
listing_df = listing_df[listing_df.price  > 0]
listing_df = listing_df[listing_df.review_scores_rating  > 0]
listing_df = listing_df[listing_df.accommodates  > 0]
listing_df.shape

for i,r in listing_df.iterrows():
    listing_df.set_value(i,'year',r['host_since'][:4])
    listing_df.set_value(i,'month',r['host_since'][5:7])
    listing_df.set_value(i,'day',r['host_since'][8:])
listing_df.head(1)

# this graph shows count of host registered everyday by resampling the data for each day
plt.figure(figsize=(18,10))
join_dates = pd.to_datetime(listing_df['host_since']).value_counts().resample('D').mean().fillna(0)
join_dates.plot()
plt.xlabel('year')
plt.ylabel('number of hosts')

plt.figure(figsize=(16,10))
year_df = listing_df.groupby(['year']).id.count()
year_df = year_df.reset_index()
year_df = year_df.rename(columns={'id':'Number of hosts'})
ax = sns.barplot(x='year', y="Number of hosts", data=year_df,palette="Blues_d")
plt.ylabel('Number of hosts')

#plotting for all months for each year
yearMonth=listing_df.groupby(['year','month']).id.count()
yearMonth=yearMonth.reset_index()
yearMonth=yearMonth.sort_values(['year','month'],ascending=[1,1])
yearMonth=yearMonth.rename(columns={"id":"Number of hosts"})
sns.factorplot(data=yearMonth, x="month", y="Number of hosts",col="year",col_wrap=3)

plt.figure(figsize=(10,8))
pd.to_datetime(listing_df['host_since']).dt.dayofweek.value_counts().sort_index().plot(kind='bar')
plt.xlabel("Days")
plt.ylabel("Number of hosts")

#this will create 4 columns namely flexible, moderate, strict, super_strict_30 which are its categories
cancel_policy = pd.get_dummies(listing_df.cancellation_policy).astype(int)
cancel_policy.head()

# Similiarly converting remaining categorial column,instant_booking  into indiactor variables
instant_booking = pd.get_dummies(listing_df.instant_bookable, prefix = 'instant_booking').astype(int)
instant_booking.head()

room_type = pd.get_dummies(listing_df.room_type).astype(int)
room_type.head()

# drop original columns and replace them with indicator columns
listing_df1 = listing_df[['bathrooms','bedrooms','beds','price','number_of_reviews','accommodates','review_scores_rating']]
listing_df1 = pd.concat((listing_df1, cancel_policy, instant_booking, room_type), axis = 1)
listing_df1.head()
listing_df1.shape
listing_df1.columns

# splitting the amenities list to count the number of amenities
'''
amenities_list = []

for element in inputDF.amenities:
    element = element[1:]
    element = element[:-1]
    x = element.split()
    amenities_list.append(len(x))

inputDF.amenities = amenities_list
'''
split_data= listing_df1.drop(['price'],axis=1)
train1,test1,train2,test2=cross_validation.train_test_split(split_data,listing_df1.price, test_size=0.4,train_size = 0.6,random_state=13)

# Lets analyze if linear regression can predict the prices accurately
# mean of prices
mean = np.mean(listing_df1.price)
# standard deviation to compare 
std = np.std(listing_df1.price)
# linear regression testing
linear_reg = linear_model.LinearRegression()
linear_reg.fit(train1, train2)
linear_reg_error = metrics.median_absolute_error(test2, linear_reg.predict(test1))
print ("Linear Regression: " + str(linear_reg_error))

























