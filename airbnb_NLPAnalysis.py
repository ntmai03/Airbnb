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


















