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
# PART 6: Time Series Analysis
############################################################

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SEASONAL PATTERN OF PRICES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#Extracting prices from the table
calendar_df['price'] = calendar_df['price'].apply(lambda x:float(x[1:].replace(',','')))

#separating date column into day month and year
calendar_df.head()
calendar_df['Year'],calendar_df['Month'],calendar_df['Day'] = calendar_df['date'].str.split('-',2).str

# create year-month df by averagePrice
year_df = calendar_df.groupby(['Year','Month']).price.mean()
year_df = year_df.reset_index()
year_df = year_df.rename(columns={'price':'average_Price'})
year_df['year-Month'] = year_df['Year'].map(str) + "-" + year_df['Month'].map(str)

# visualizing the trend of year/Month and average prices of the listing
objects = year_df['year-Month']
y_pos = year_df['average_Price']
year_df.plot(kind='bar', 
           x='year-Month',
           y='average_Price',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London Airbnb prices trend over months', 
           legend = False)
plt.ylabel('Average Price')

#getting day name from date data and holidays
calendar_df.fillna(0, inplace=True)
calendar_df.shape
uk_holidays = holidays.UK()
calendar_df['day_Name']='default'
calendar_df['holiday']='False'
calendar_df['uk_holidays_name']='working'
for index,row in calendar_df.iterrows():
    sdate = datetime.date(int(row['Year']),int(row['Month']),int(row['Day']))
    vall=date(int(row['Year']),int(row['Month']),int(row['Day'])) in uk_holidays
    calendar_df.set_value(index,'day_Name',calendar.day_name[sdate.weekday()])
    calendar_df.set_value(index,'holiday',vall)
    calendar_df.set_value(index,'uk_holidays_name',uk_holidays.get(sdate))
calendar_df.head()
calendar_df.to_csv('D:\\MyProjects\\01_Airbnb\\preprocessed_London_calendar.csv')

#calculating Average price for each day
day_df = calendar_df.groupby('day_Name').price.mean()
day_df = day_df.reset_index()
day_df['day_num'] = 0
for index,row in day_df.iterrows():
    if row['day_Name']=='Monday':
        day_df.set_value(index,'day_num',1)
    if row['day_Name']=='Tuesday':
        day_df.set_value(index,'day_num',2)
    if row['day_Name']=='Wednesday':
        day_df.set_value(index,'day_num',3)
    if row['day_Name']=='Thursday':
        day_df.set_value(index,'day_num',4)
    if row['day_Name']=='Friday':
        day_df.set_value(index,'day_num',5)
    if row['day_Name']=='Saturday':
        day_df.set_value(index,'day_num',6)
    if row['day_Name']=='Sunday':
        day_df.set_value(index,'day_num',7)
day_df = day_df.sort_values('day_num',ascending=[1])
day_df = day_df.rename(columns={'price':'Average_Price'})
day_df

#checking which holiday has maximum listings
holiday_df = calendar_df.groupby('uk_holidays_name').listing_id.count()
holiday_df = holiday_df.reset_index()
holiday_df = holiday_df .sort_values('listing_id',ascending=[0])
holiday_df
plt.figure(figsize=(15,7))
ax = sns.barplot("uk_holidays_name", y="listing_id", data=holiday_df,palette="Blues_d")
ax.set(xlabel='UK Holidays Name', ylabel='Total Number of Listings')

#checking which holiday has maximum prices
holidayPrice_df = calendar_df.groupby('uk_holidays_name').price.mean()
holidayPrice_df = holidayPrice_df.reset_index()
holidayPrice_df = holidayPrice_df.sort_values('price',ascending=[0])
holidayPrice_df.head(10)
plt.figure(figsize=(15,8))
ax = sns.barplot("uk_holidays_name", y="price", data=holidayPrice_df,palette="Blues_d")
ax.set(xlabel='US Holidays Name', ylabel='Average Price')

# merge holiday listings and holiday price
merger = pd.merge(holiday_df,holidayPrice_df,on='uk_holidays_name')
merger = merger.rename(columns={'listing_id':'number_Of_Listings'})
merger = merger.rename(columns={'price':'average_Price'})
merger

#analyzing data of July and August
checkHolidays_df = calendar_df[(calendar_df['Year'] == '2019') & ((calendar_df['Month'] == '07') |(calendar_df['Month'] == '08'))]
checkHolidays_df = checkHolidays_df.groupby(['Year','Month','holiday']).price.mean()
checkHolidays_df = checkHolidays_df.reset_index()
checkHolidays_df.head()

# analyzing data from date 5th of October to date 13th of October which includes both long weekend and normal workdays
July_df = calendar_df[(calendar_df['Year'] == '2019') & (calendar_df['Month'] == '07')]
July_df = July_df.groupby('Day').price.mean()
July_df = July_df.reset_index()
July_df = July_df.sort_values('Day',ascending=[1])
July_df = July_df.rename(columns={'price':'Average_Price'})
July_df['date'] = July_df['Day'].apply(lambda x:datetime.date(2019,7,int(x)))
July_df['day_Name'] = July_df['date'].apply(lambda x:calendar.day_name[x.weekday()])
July_df.head(10)

x=July_df['Day'].tolist()
y=July_df['Average_Price'].tolist()
plt.figure(figsize=(15,8))
plt.plot(x,y, 'ro-')
plt.ylabel('Average Price')
plt.xlabel('Days')
plt.show()


















