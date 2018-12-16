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

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
calendar
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#replacing NaN values with 0
calendar_df.fillna(0, inplace=True)
calendar_df = calendar_df[calendar_df.price != 0]
calendar_df.shape
calendar_df.head()


################################################################################
# Host Analysis
################################################################################

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Analyzing and plotting the number of listings based on their room type
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Categorizing differernt listings based on room_type
roomtype = pd.DataFrame(listing_df.groupby('room_type').id.count().sort_values())
roomtype.columns = ['room_type']
roomtype.plot.pie(y = 'room_type', 
                 colormap = 'Blues_r', 
                 figsize=(10,10), 
                 fontsize = 20, autopct = '%.2f',
                 legend = False,
                 title = 'Room Type Distribution')

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Analyzing and plotting the number of listings based on their property type
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Categorizing differernt listings based on property_type
propertytype = pd.DataFrame(listing_df.groupby('property_type').id.count())
propertytype.columns = ['Number_Of_Listings']
propertytype['property_type'] = propertytype.index.values
propertytype.plot(kind='bar', 
           x='property_type',
           y='Number_Of_Listings',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London Property Type Frequency', 
           legend = False)
plt.ylabel('property type')
plt.ylabel('Number of listings')

#Plotting the same on a heatMap
plt.figure(figsize=(12,12))
sns.heatmap(listing_df.groupby([
        'property_type', 'room_type']).price.mean().unstack(),annot=True, fmt=".0f")

# HeatMap for variation of prices with number of bedrooms for listings
plt.figure(figsize=(12,12))
sns.heatmap(listing_df.groupby([
        'neighbourhood_cleansed', 'bedrooms']).price.mean().unstack(),annot=True, fmt=".0f")

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Analyzing and plotting word cloud for summary
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
summary_df = listing_df[['summary','price']]
summary_df = summary_df[pd.notnull(summary_df['summary'])]
summary_df = summary_df[summary_df['summary']!=0]
summary_df = summary_df.sort_values('price',ascending=[0])
top100_df = summary_df.head(100)
words=''
for index,row in top100_df.iterrows():
    words += row['summary']
string_punctuation = string.punctuation
ignoreChar=['\r','\n','',' ',"'s"]
nums=['0','1','2','3','4','5','6','7','8','9']
summary_data=nltk.word_tokenize(words)
words_only = [l.lower() for l in summary_data if l not in string_punctuation if l not in ignoreChar if l not in nums]
filtered_data=[word for word in words_only if word not in stopwords.words('english')] 
wnl = nltk.WordNetLemmatizer() 
final_data=[wnl.lemmatize(data) for data in filtered_data]
final_words=' '.join(final_data)
final_words[:50]
wordcloud = WordCloud(width = 1000, height = 700).generate(final_words)
plt.figure(figsize=(18,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Analyzing what amenities costs more ? top price
amenities_df = listing_df[['amenities','price','id',]]
amenitiesDFTopper = amenities_df.sort_values('price',ascending=[0])
amenitiesDFtop = amenitiesDFTopper.head(30)
allemenities = ''
for index,row in amenitiesDFtop.iterrows():
    p = re.sub('[^a-zA-Z]+',' ', row['amenities'])
    allemenities+=p
allemenities_data=nltk.word_tokenize(allemenities)
filtered_data=[word for word in allemenities_data if word not in stopwords.words('english')] 
wnl = nltk.WordNetLemmatizer() 
allemenities_data=[wnl.lemmatize(data) for data in filtered_data]
allemenities_words=' '.join(allemenities_data)
wordcloud = WordCloud(width = 1000, height = 700).generate(allemenities_words)
plt.figure(figsize=(18,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#Analyzing what amenities costs more ? bottom price
amenitiesDFbott =amenities_df.sort_values('price',ascending=[1])
amenitiesDFbottom=amenitiesDFbott.head(30)
allemenitiesb = ''
for index,row in amenitiesDFbottom.iterrows():
    p = re.sub('[^a-zA-Z]+',' ', row['amenities'])
    allemenitiesb+=p
allemenities_datab=nltk.word_tokenize(allemenitiesb)
filtered_datab=[word for word in allemenities_datab if word not in stopwords.words('english')] 
wnl = nltk.WordNetLemmatizer() 
allemenities_datab=[wnl.lemmatize(data) for data in filtered_datab]
allemenities_wordsb=' '.join(allemenities_datab)
wordcloud = WordCloud(width = 1000, height = 700).generate(allemenities_wordsb)
plt.figure(figsize=(18,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


################################################################################
# P8: Investment/ Business opportunities
################################################################################
# Average prices for each type of listing
avgPrice_df = listing_df.groupby('room_type').price.mean()
avgPrice_df = avgPrice_df.reset_index()
avgPrice_df = avgPrice_df.rename(columns={'price':'average_Price'})
avgPrice_df
home = listing_df[(listing_df.room_type == 'Entire home/apt')]
private = listing_df[(listing_df.room_type == 'Private room')]
shared = listing_df[(listing_df.room_type == 'Shared room')]

# grouping neighbourhood by number of listings
neighbourhood_df = listing_df.groupby('neighbourhood_cleansed').id.count()
neighbourhood_df = neighbourhood_df.reset_index()
neighbourhood_df = neighbourhood_df.rename(columns={'id':'Number_Of_Listings'})
neighbourhood_df = neighbourhood_df.sort_values('Number_Of_Listings',ascending=[0])
neighbourhood_df.head()
# grouping neighbourhood by average price of listings
neighbourhoodPrice_df = home.groupby('neighbourhood_cleansed').price.mean()
neighbourhoodPrice_df = neighbourhoodPrice_df.reset_index()
neighbourhoodPrice_df = neighbourhoodPrice_df.rename(columns={'price':'Average_Price'})
neighbourhoodPrice_df = neighbourhoodPrice_df.sort_values('Average_Price',ascending=[0])
neighbourhoodPrice_df.head()
#Merging above two dataframes
merge_df =pd.merge(neighbourhood_df,neighbourhoodPrice_df,on='neighbourhood_cleansed')
merge_df.head()

# Now lets visualize average price of these listings on the basis of neighbourhood where room type is entire apartment
objects = neighbourhoodPrice_df['neighbourhood_cleansed']
y_pos = neighbourhoodPrice_df['Average_Price']
neighbourhoodPrice_df.plot(kind='bar', 
           x='neighbourhood_cleansed',
           y='Average_Price',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'Boston Neighborhood Average price', 
           legend = False)
plt.ylabel('Average Price')

# Let's look at the differences between the words used in listings for Airbnb locations in different Boston neighborhoods
uniqueNeighborhoods = listing_df.neighbourhood_cleansed.unique().tolist()
#function to clean the data and compile a list of most common words
def cleanData(neighbrhood_name,descrip):
    p = re.sub('[^a-zA-Z]+',' ', descrip)
    cmn_words=['The','I','Boston','room']
    descrip_data=nltk.word_tokenize(p)
    filtered_data=[word.lower() for word in descrip_data if word not in cmn_words if word not in stopwords.words('english')] 
    wnl = nltk.WordNetLemmatizer() 
    counts=Counter([wnl.lemmatize(data) for data in filtered_data])
    commn_words=[]
    for w in counts.most_common(5):
        commn_words.append(w[0])
    return ' '.join(commn_words)
summ={}
for n in uniqueNeighborhoods:
    text=''
    for index,row in listing_df.iterrows():
        if row['neighbourhood_cleansed']==n:
            if ((row['description']!=0) & (row['space']!=0) & (row['neighborhood_overview']!=0)):
                text =text+row['description']+row['space']+row['neighborhood_overview']
    summ[n]=text
final_df_neighbrhood=pd.DataFrame(columns=['neighborhood','top 5 words in description'])
for a in summ.items():
    top5words=cleanData(a[0],a[1])
    final_df_neighbrhood=final_df_neighbrhood.append(pd.Series([a[0],top5words],index=['neighborhood','top 5 words in description']),ignore_index=True)

final_df_neighbrhood.to_csv('top_words_description.csv')
final_df_neighbrhood.head()



















