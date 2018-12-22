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
summary: Analyzing and plotting word cloud for summary
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


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
longitude + latitude
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
longlat_df = listing_df[['description','longitude','latitude']]


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
space
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
space_df = listing_df[['space','price']]
space_df.head(20)

space_df = space_df[pd.notnull(space_df['space'])]
space_df = space_df[space_df['space']!=0]
space_df = space_df.sort_values('price',ascending=[0])
top100_df = space_df.head(100)
words=''
for index,row in top100_df.iterrows():
    words += row['space']
string_punctuation = string.punctuation
ignoreChar=['\r','\n','',' ',"'s"]
nums=['0','1','2','3','4','5','6','7','8','9']
space_data=nltk.word_tokenize(words)
words_only = [l.lower() for l in space_data if l not in string_punctuation if l not in ignoreChar if l not in nums]
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



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
description: Analyzing and plotting word cloud for description
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
description_df = listing_df[['description','price']]
description_df = description_df[pd.notnull(description_df['description'])]
description_df = description_df[description_df['description']!=0]
description_df = description_df.sort_values('price',ascending=[0])
top100_df = description_df.head(100)
words=''
for index,row in top100_df.iterrows():
    words += row['description']
string_punctuation = string.punctuation
ignoreChar=['\r','\n','',' ',"'s"]
nums=['0','1','2','3','4','5','6','7','8','9']
description_data=nltk.word_tokenize(words)
words_only = [l.lower() for l in description_data if l not in string_punctuation if l not in ignoreChar if l not in nums]
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

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
instant_bookable
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
instant_bookable_df = listing_df[['instant_bookable','price']]
instant_bookable_df.head(20)
listing_df['instant_bookable'].unique()


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
neighborhood_overview
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
neighborhood_overview_df = listing_df[['neighborhood_overview','price']]
neighborhood_overview_df.head(20)

neighborhood_overview_df = neighborhood_overview_df[pd.notnull(neighborhood_overview_df['neighborhood_overview'])]
neighborhood_overview_df = neighborhood_overview_df[neighborhood_overview_df['neighborhood_overview_df']!=0]
neighborhood_overview_df = neighborhood_overview_df.sort_values('price',ascending=[0])
top100_df = neighborhood_overview_df.head(100)
words=''
for index,row in top100_df.iterrows():
    words += row['neighborhood_overview']
string_punctuation = string.punctuation
ignoreChar=['\r','\n','',' ',"'s"]
nums=['0','1','2','3','4','5','6','7','8','9']
neighborhood_overview_data=nltk.word_tokenize(words)
words_only = [l.lower() for l in neighborhood_overview_data if l not in string_punctuation if l not in ignoreChar if l not in nums]
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



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
neighbourhood_cleansed
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# grouping neighbourhood by number of listings
neighbourhood_df = listing_df.groupby('neighbourhood_cleansed').id.count()
neighbourhood_df = neighbourhood_df.reset_index()
neighbourhood_df = neighbourhood_df.rename(columns={'id':'Number_Of_Listings'})
neighbourhood_df = neighbourhood_df.sort_values('Number_Of_Listings',ascending=[0])
neighbourhood_df.head()
neighbourhood_df.plot(kind='bar', 
           x='neighbourhood_cleansed',
           y='Number_Of_Listings',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London neighbourhood Type Frequency', 
           legend = False)
plt.xlabel('neighbourhood type')
plt.ylabel('Number of listings')


# grouping neighbourhood by average price of listings
neighbourhoodPrice_df = listing_df.groupby('neighbourhood_cleansed').price.mean()
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
           title = 'London Neighborhood Average price', 
           legend = False)
plt.xlabel('neighbourhood type')
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

final_df_neighbrhood = pd.read_csv('top_words_description.csv')
final_df_neighbrhood.head()


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
host_id + host_name
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
hostname_df = listing_df[['host_id','host_name','price']]
hostname_df.head(20)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
host_since
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
host_since_df = listing_df[['host_since','price']]
host_since_df.head(20)

listing_df = listing_df[listing_df.host_since  != 0]
listing_df.shape
listing_df['year'] = 0
listing_df['month'] = 0
listing_df['day'] = 0

for i,r in listing_df.iterrows():
    listing_df.set_value(i,'year',r['host_since'][:4])
    listing_df.set_value(i,'month',r['host_since'][5:7])
    listing_df.set_value(i,'day',r['host_since'][8:])
listing_df.head(1)
listing_df['host_since'][0:10]

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

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
host_response_time
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
host_response_time_df = listing_df[['host_response_time','price']]
host_response_time_df.head(20)


host_response_time = pd.DataFrame(listing_df.groupby('host_response_time').id.count().sort_values())
host_response_time.columns = ['NumOfListings']
host_response_time.plot.pie(y = 'NumOfListings',
                 colormap = 'Blues_r', 
                 figsize=(10,10), 
                 fontsize = 20, autopct = '%.2f',
                 legend = False,
                 title = 'host_response_time Distribution')

# Average prices for each type of listing
host_response_time_df = listing_df.groupby('host_response_time').price.mean()
host_response_time_df = host_response_time_df.reset_index()
host_response_time_df = host_response_time_df.rename(columns={'price':'average_Price'})
host_response_time_df

# Now lets visualize average price of these listings on the basis of neighbourhood where room type is entire apartment
objects = host_response_time_df['host_response_time']
y_pos = host_response_time_df['average_Price']
host_response_time_df.plot(kind='bar', 
           x='host_response_time',
           y='average_Price',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London host_response_time Average price', 
           legend = False)
plt.xlabel('host_response_time')
plt.ylabel('Average Price')


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
street
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
street_df = listing_df[['street','price']]
street_df.head(20)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
zipcode
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
zipcode_df = listing_df[['zipcode','price']]
zipcode_df.head(20)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
review_scores_rating
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
review_scores_rating_df = listing_df[['review_scores_rating','price']]
review_scores_rating_df.head(20)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
property_type: Categorizing differernt listings based on property_type
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
propertytype = pd.DataFrame(listing_df.groupby('property_type').id.count())
propertytype.columns = ['Number_Of_Listings']
propertytype = propertytype.reset_index()
propertytype.plot(kind='bar', 
           x='property_type',
           y='Number_Of_Listings',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London Property Type Frequency', 
           legend = False)
plt.xlabel('property type')
plt.ylabel('Number of listings')

# Average prices for each type of listing
propertytype_df = listing_df.groupby('property_type').price.mean()
propertytype_df = propertytype_df.reset_index()
propertytype_df = propertytype_df.rename(columns={'price':'average_Price'})
propertytype_df = propertytype_df.sort_values('average_Price',ascending=[0])
propertytype_df

# Now lets visualize average price of these listings on the basis of neighbourhood where room type is entire apartment
objects = propertytype_df['property_type']
y_pos = host_response_time_df['average_Price']
propertytype_df
propertytype_df.plot(kind='bar', 
           x='property_type',
           y='average_Price',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London property_type Average price', 
           legend = False)
plt.xlabel('property_type')
plt.ylabel('Average Price')



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
room_type: Categorizing differernt listings based on room_type
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
roomtype = pd.DataFrame(listing_df.groupby('room_type').id.count().sort_values())
roomtype.columns = ['NumOfListings']
roomtype.to_csv('D:\\MyProjects\\01_Airbnb\\common_info.csv')
roomtype.plot.pie(y = 'NumOfListings',
                 colormap = 'Blues_r', 
                 figsize=(10,10), 
                 fontsize = 20, autopct = '%.2f',
                 legend = False,
                 title = 'Room Type Distribution')

# Average prices for each type of listing
avgPrice_df = listing_df.groupby('room_type').price.mean()
avgPrice_df = avgPrice_df.reset_index()
avgPrice_df = avgPrice_df.rename(columns={'price':'average_Price'})
avgPrice_df



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
accommodates: Categorizing differernt listings based on property_type
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
accommodates_df = listing_df[['accommodates','price']]
accommodates_df.head(20)

accommodates = pd.DataFrame(listing_df.groupby('accommodates').id.count())
accommodates.columns = ['Number_Of_Listings']
accommodates = accommodates.reset_index()
accommodates = accommodates.sort_values('Number_Of_Listings',ascending=[0])
accommodates.plot(kind='bar', 
           x='accommodates',
           y='Number_Of_Listings',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London Property Type Frequency', 
           legend = False)
plt.xlabel('accommodates')
plt.ylabel('Number of listings')

# Average prices for each type of listing
accommodates_df = listing_df.groupby('accommodates').price.mean()
accommodates_df = accommodates_df.reset_index()
accommodates_df = accommodates_df.rename(columns={'price':'average_Price'})
accommodates_df = accommodates_df.sort_values('average_Price',ascending=[0])
accommodates_df

# Now lets visualize average price of these listings on the basis of neighbourhood where room type is entire apartment
objects = accommodates_df['accommodates']
y_pos = accommodates_df['average_Price']
accommodates_df
accommodates_df.plot(kind='bar', 
           x='accommodates',
           y='average_Price',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London accommodates Average price', 
           legend = False)
plt.xlabel('accommodates')
plt.ylabel('Average Price')





'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
bathrooms: Categorizing differernt listings based on property_type
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
bathrooms_df = listing_df[['bathrooms','price']]
bathrooms_df.head(20)

bathrooms = pd.DataFrame(listing_df.groupby('bathrooms').id.count())
bathrooms.columns = ['Number_Of_Listings']
bathrooms = bathrooms.reset_index()
bathrooms = bathrooms.sort_values('Number_Of_Listings',ascending=[0])
bathrooms.plot(kind='bar', 
           x='bathrooms',
           y='Number_Of_Listings',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London Property Type Frequency', 
           legend = False)
plt.xlabel('bathrooms')
plt.ylabel('Number of listings')

# Average prices for each type of listing
bathrooms_df = listing_df.groupby('bathrooms').price.mean()
bathrooms_df = bathrooms_df.reset_index()
bathrooms_df = bathrooms_df.rename(columns={'price':'average_Price'})
bathrooms_df = bathrooms_df.sort_values('average_Price',ascending=[0])
bathrooms_df

# Now lets visualize average price of these listings on the basis of neighbourhood where room type is entire apartment
objects = bathrooms_df['bathrooms']
y_pos = bathrooms_df['average_Price']
bathrooms_df
bathrooms_df.plot(kind='bar', 
           x='bathrooms',
           y='average_Price',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London accommodates Average price', 
           legend = False)
plt.xlabel('bathrooms')
plt.ylabel('Average Price')

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
bedrooms: Categorizing differernt listings based on property_type
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
bedrooms_df = listing_df[['bedrooms','price']]
bedrooms_df.head(20)

bedrooms = pd.DataFrame(listing_df.groupby('bedrooms').id.count())
bedrooms.columns = ['Number_Of_Listings']
bedrooms = bedrooms.reset_index()
bedrooms = bedrooms.sort_values('Number_Of_Listings',ascending=[0])
bedrooms.plot(kind='bar', 
           x='bedrooms',
           y='Number_Of_Listings',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London Property Type Frequency', 
           legend = False)
plt.xlabel('bedrooms')
plt.ylabel('Number of listings')


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
beds
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
reviews_per_month
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
reviews_per_month_df = listing_df[['reviews_per_month','price']]
reviews_per_month_df.head(20)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
amenities: 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Analyzing what amenities costs more ? top price
amenities_df = listing_df[['amenities','price','id']]
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


# splitting the amenities list to count the number of amenities
amenities_list = []

for element in listing_df.amenities:
    element = element[1:]
    element = element[:-1]
    x = element.split()
    amenities_list.append(len(x))

listing_df['num_of_amenities'] = amenities_list

amenities_df = listing_df[['num_of_amenities','price']]
amenities_df.head(20)

amenities = pd.DataFrame(listing_df.groupby('num_of_amenities').id.count())
amenities.columns = ['Number_Of_Listings']
amenities = amenities.reset_index()
amenities = amenities.sort_values('Number_Of_Listings',ascending=[0])
amenities = amenities.loc[amenities.Number_Of_Listings > 100,:]

amenities.plot(kind='bar', 
           x='num_of_amenities',
           y='Number_Of_Listings',
           color = '#66c2ff', 
           figsize =(18,8), 
           title = 'London Property Type Frequency', 
           legend = False)
plt.xlabel('amenities')
plt.ylabel('Number of listings')

# Average prices for each type of listing
amenities_df = listing_df.groupby('num_of_amenities').price.mean()
amenities_df = amenities_df.reset_index()
amenities_df = amenities_df.rename(columns={'price':'average_Price'})
amenities_df = amenities_df.sort_values('average_Price',ascending=[0])
amenities_df

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
cancellation_policy
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
cancellation_policy_df = listing_df[['cancellation_policy','price']]
cancellation_policy_df.head(20)
listing_df['cancellation_policy'].unique()

cancellation_policy = pd.DataFrame(listing_df.groupby('cancellation_policy').id.count().sort_values())
cancellation_policy.columns = ['NumOfListings']
cancellation_policy.plot.pie(y = 'NumOfListings',
                 colormap = 'Blues_r', 
                 figsize=(10,10), 
                 fontsize = 20, autopct = '%.2f',
                 legend = False,
                 title = 'cancellation_policy Distribution')


# Average prices for each type of listing
cancellation_policy_df = listing_df.groupby('cancellation_policy').price.mean()
cancellation_policy_df = cancellation_policy_df.reset_index()
cancellation_policy_df = cancellation_policy_df.rename(columns={'price':'average_Price'})
cancellation_policy_df = cancellation_policy_df.sort_values('average_Price',ascending=[0])
cancellation_policy_df

# Now lets visualize average price of these listings on the basis of neighbourhood where room type is entire apartment
objects = cancellation_policy_df['cancellation_policy']
y_pos = cancellation_policy_df['average_Price']
cancellation_policy_df
cancellation_policy_df.plot(kind='bar', 
           x='cancellation_policy',
           y='average_Price',
           color = '#66c2ff', 
           figsize =(15,8), 
           title = 'London cancellation_policy Average price', 
           legend = False)
plt.xlabel('cancellation_policy')
plt.ylabel('Average Price')






'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
number_of_reviews
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
number_of_reviews_df = listing_df[['number_of_reviews','price']]
number_of_reviews_df.head(20)



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Price 
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Price distribution

plt.figure(figsize=(10, 6))
listing_df.price.plot.hist()
plt.title('Price distribution - (All price)') 

plt.figure(figsize=(10, 6))
listing_df.price[listing_df.price< 1001].plot.hist()
plt.title('Price distribution - ($0 - $1000)') 

data = listing_df.price[listing_df.price< 201]
binwidth = 10
plt.figure(figsize=(10, 6))
plt.hist(data,bins=range(min(data), max(data) + binwidth, binwidth))
plt.title('Price distribution - ($0 - $200)') 
plt.xticks(range(min(data), max(data) + binwidth, binwidth))

data = listing_df.price[listing_df.price< 101]
binwidth = 1
plt.figure(figsize=(10, 6))
plt.hist(data,bins=range(min(data), max(data) + binwidth, binwidth))
plt.title('Price distribution - ($0 - $100)') 
plt.xticks(range(min(data), max(data) + binwidth, binwidth))

#Plotting the same on a heatMap
listing_df.groupby(['property_type', 'room_type']).price.mean().unstack()
plt.figure(figsize=(12,12))
sns.heatmap(listing_df.groupby(['property_type', 'room_type']).price.mean().unstack(),annot=True, fmt=".0f")

# HeatMap for variation of prices with number of bedrooms for listings
listing_df.groupby(['neighbourhood_cleansed', 'bedrooms']).price.mean().unstack()
plt.figure(figsize=(12,12))
sns.heatmap(listing_df.groupby(['neighbourhood_cleansed', 'bedrooms']).price.mean().unstack(),annot=True, fmt=".0f")

#correlation matrix
corrmat = listing_df.corr(method = 'spearman')
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);



############################################################
# PART 6: Time Series Analysis
############################################################

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
SEASONAL PATTERN OF PRICES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
listing_df.loc[listing_df.id == 9554,'price']
calendar_df.loc[calendar_df.listing_id == 9554,:]

#Extracting prices from the table
calendar_df['price'] = calendar_df['price'].apply(lambda x:float(x[1:].replace(',','')))

calendar_df['price'] = calendar_df['price'].apply(lambda x:x[1:])

calendar_df.shape

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


############################################################
# P7: NLP Analysis
############################################################

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Sentiment Analysis: Where to Invest a Property in BOSTON to get maximum returns from Airbnb?
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#assigning polarity score to each comment and adding the same to dataframe
sid = SentimentIntensityAnalyzer()
review_df['polarity_value']="Default"
review_df['neg']=0.0
review_df['pos']=0.0
review_df['neu']=0.0
review_df['compound']=0.0
for index,row in review_df.iterrows():
    ss = sid.polarity_scores(row['comments'])
    review_df.set_value(index,'polarity_value',ss)
    review_df.set_value(index,'neg',ss['neg'])
    review_df.set_value(index,'pos',ss['pos'])
    review_df.set_value(index,'neu',ss['neu'])
    review_df.set_value(index,'compound',ss['compound'])
review_df.head()
review_df.to_csv('D:\\MyProjects\\01_Airbnb\\London_pority_Values_reviews.csv')
review_df = pd.read_csv('D:\\MyProjects\\01_Airbnb\\London_pority_Values_reviews.csv')

# Our dataframe consists of reviews in different language as well.Hence removing the comments which are not in english
def detect_lang(sente):
    sente=str(sente)
    try:
        return detect(sente)
    except:
        return "None"
for index,row in review_df.iterrows():
    lang=detect_lang(row['comments'])
    review_df.set_value(index,'language',lang)    
#taking rows whose language is English
EngReviews_df = review_df[review_df.language=='en']
EngReviews_df = review_df

# Lets visualize polarity score for positivity
polar_df = EngReviews_df[['pos']]
polar_df = polar_df.groupby(pd.cut(polar_df["pos"], np.arange(0, 1.1, 0.1))).count()
polar_df = polar_df.rename(columns={'pos':'count_of_Comments'})
polar_df = polar_df.reset_index()
polar_df = polar_df.rename(columns={'pos':'range_i'})
for i,r in polar_df.iterrows():
    polar_df.set_value(i,'RANGE',float(str(r['range_i'])[1:4].replace(',',''))) 
    polar_df.set_value(i,'Sentiment','positive')
del polar_df['range_i']
polar_df.head()

# Lets visualize polarity score for negativity
negpolar_df = EngReviews_df[['neg']]
negpolar_df = negpolar_df.groupby(pd.cut(negpolar_df["neg"], np.arange(0, 1.1, 0.1))).count()
negpolar_df = negpolar_df.rename(columns={'neg':'count_of_Comments'})
negpolar_df = negpolar_df.reset_index()
negpolar_df = negpolar_df.rename(columns={'neg':'range_i'})
for i,r in negpolar_df.iterrows():
    negpolar_df.set_value(i,'RANGE',float(str(r['range_i'])[1:4].replace(',',''))) 
    negpolar_df.set_value(i,'Sentiment','negative')
del negpolar_df['range_i']
for i,r in negpolar_df.iterrows():
    polar_df = polar_df.append(pd.Series([r[0],r[1],r[2]],index=['count_of_Comments','RANGE','Sentiment']),ignore_index=True)    
negpolar_df.head()

# Lets visualize polarity score for neutrality
neupolar_df = EngReviews_df[['neu']]
neupolar_df = neupolar_df.groupby(pd.cut(neupolar_df["neu"], np.arange(0, 1.0, 0.1))).count()
neupolar_df = neupolar_df.rename(columns={'neu':'count_of_Comments'})
neupolar_df = neupolar_df.reset_index()
neupolar_df = neupolar_df.rename(columns={'neu':'range_i'})
for i,r in neupolar_df.iterrows():
    neupolar_df.set_value(i,'RANGE',float(str(r['range_i'])[1:4].replace(',',''))) 
    neupolar_df.set_value(i,'Sentiment','neutrl')
del neupolar_df['range_i']

for i,r in neupolar_df.iterrows():
    polar_df = polar_df.append(pd.Series([r[0],r[1],r[2]],index=['count_of_Comments','RANGE','Sentiment']),ignore_index=True)   
neupolar_df.head()

# Visualization 
plt.figure(figsize=(10,10))
sns.factorplot(data=polar_df, x="RANGE", y="count_of_Comments",col="Sentiment")

listing_df2 = listing_df[['number_of_reviews','price','review_scores_rating']]
price_review = listing_df2[['number_of_reviews', 'price']].sort_values(by = 'price')
price_review.plot(x = 'price', 
                  y = 'number_of_reviews', 
                  style = 'o',
                  figsize =(12,8),
                  legend = False,
                  title = 'Reviews based on Price')
plt.xlabel("price")
plt.ylabel("Number of reviews")

words=''
for index,row in EngReviews_df.iterrows():
    words += row['comments']

reviews_data=nltk.word_tokenize(words)
string_punctuation = string.punctuation
ignoreChar=['\r','\n','',' ',"'s",'!',',',]
nums=['0','1','2','3','4','5','6','7','8','9']
cachedStopWords = stopwords.words("english")
cachedStopWords.append('the')

words_only = [''.join(c for c in s if c not in string.punctuation if c not in nums if c not in ignoreChar) for s in reviews_data]
words_only = [s for s in words_only if s]
comments_filtered_data=' '.join(words_only)
comments_filtered_data = ' '.join([word.lower() for word in comments_filtered_data.split() if word not in cachedStopWords])

wordcloud = WordCloud(width = 1000, height = 700).generate(comments_filtered_data)
plt.figure(figsize=(18,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


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

#this will create 4 columns namely flexible, moderate, strict, super_strict_30 which are its categories
cancel_policy = pd.get_dummies(listing_df.cancellation_policy).astype(int)
cancel_policy.head()

# Similiarly converting remaining categorial column,instant_booking  into indiactor variables
instant_booking = pd.get_dummies(listing_df.instant_bookable, prefix = 'instant_booking').astype(int)
instant_booking.head()

room_type = pd.get_dummies(listing_df.room_type).astype(int)
room_type.head()

property_type = pd.get_dummies(listing_df.property_type).astype(int)
property_type.head()

neibourhood_type = pd.get_dummies(listing_df.neighbourhood_cleansed).astype(int)
neibourhood_type.head()

# drop original columns and replace them with indicator columns
listing_df1 = listing_df[['bathrooms','bedrooms','beds','price','number_of_reviews','accommodates','review_scores_rating']]
listing_df1 = pd.concat((listing_df1, cancel_policy, instant_booking, room_type,property_type,neibourhood_type), axis = 1)
listing_df1.head()
listing_df1.shape
listing_df1.columns

listing_df1 = listing_df[['price']]
listing_df1 = pd.concat((listing_df1, room_type,property_type), axis = 1)

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

listing_df1.head()

#----------------------------------------------------------------------
# Evaluating Model Accuracy
#----------------------------------------------------------------------
# Calculating MSE
from sklearn import metrics
mse = np.sqrt(metrics.mean_squared_error(test2,linear_reg.predict(test1)))
print('MSE: ',round(mse,2))
# Calculating R squared
print('R squared: ',metrics.r2_score(test2,linear_reg.predict(test1)))
# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
linreg = LinearRegression()
cv_scores = cross_val_score(linreg, X_train, Y_train, scoring = 'r2', cv = 10)
print(cv_scores)
print( "Average r2 score: ", np.round( np.mean( cv_scores ), 2 ) )
print( "Standard deviation in r2 score: ", np.round( np.std( cv_scores ), 2) )
# Understanding residuals
residuals = Y_test - Y_pred
sns.jointplot(Y,residuals,size = 6)
plt.show()
sns.distplot( residuals )
plt.show()




################################################################################
# P10: Others
################################################################################
# Geographical Clusters to find out which area in Boston has maximum listings on Airbnb
# seggregating each type of property
home = listing_df[(listing_df.room_type == 'Entire home/apt')]
private = listing_df[(listing_df.room_type == 'Private room')]
shared = listing_df[(listing_df.room_type == 'Shared room')]
location_home = home[['latitude', 'longitude']]
location_private = private[['latitude', 'longitude']]
location_shared = shared[['latitude', 'longitude']]

from bokeh.io import push_notebook, show, output_notebook
import os

from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)
output_notebook()
map_options = GMapOptions(lat=42.3318, lng=-71.1212, map_type="roadmap", zoom=11)

plot = GMapPlot(
    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options,width=1200, height=750
)
plot.title.text = "Boston"

plot.api_key = os.environ['data_download_key']
source = ColumnDataSource(
    data=dict(
        lat_home=location_home['latitude'],
        lon_home=location_home['longitude'],
        lat_private=location_private['latitude'],
        lon_private=location_private['longitude'],
        lat_shared=location_shared['latitude'],
        lon_shared=location_shared['longitude']
    )
)

circle_home = Circle(x="lon_home", y="lat_home", size=4, fill_color="blue", fill_alpha=0.9, line_color=None)
circle_private =  Circle(x="lon_private", y="lat_private", size=4, fill_color="red", fill_alpha=0.9, line_color=None)
circle_shared =  Circle(x="lon_shared", y="lat_shared", size=4, fill_color="#006600", fill_alpha=0.9, line_color=None)

plot.add_glyph(source, circle_home)
plot.add_glyph(source, circle_private)
plot.add_glyph(source, circle_shared)
plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
show(plot)
