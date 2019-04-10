
# coding: utf-8

# In[321]:


###############################################################################
#Import package
###############################################################################

#Import linear algebra and data manipulation
import numpy as np
import pandas as pd

#Import plotting packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors

#Import common utilities
import math
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# statsmodels
import pylab
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels as statm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols

#Import machine learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

# Accuracy Evaluation
from sklearn.model_selection import train_test_split #split
from sklearn.metrics import r2_score, mean_squared_error #metrics
import sklearn.metrics as metrics
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix 

# NLP
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import string
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# # Load Data

# In[5]:


###############################################################################
#Load data
###############################################################################
# listings
listings_df = pd.read_csv('C:\\DataScience\\00_AnalysisProjects\\AirbnbAnalysis\\Data\\seattle_listings.csv', sep=',')
listings_df.shape   # (3818, 92)
# calendar
calendar_df = pd.read_csv('C:\\DataScience\\00_AnalysisProjects\\AirbnbAnalysis\\Data\\seattle_calendar.csv')
calendar_df.shape   # (1393570, 4)
# reviews
review_df = pd.read_csv('C:\\DataScience\\00_AnalysisProjects\\AirbnbAnalysis\\Data\\seattle_reviews.csv')
review_df.shape   # (84849, 6)


# # Configuration

# In[7]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 505)


# # View data structure and content

# ### Data type of all colums

# In[9]:


listings_df.info()


# ### Summary stats on each of the numeric columns

# In[10]:


listings_df.describe()


# ### Columns

# In[6]:


# All column names
listings_df.columns.values


# In[12]:


# Print all column names
for col in listings_df.columns.values:
    print(col)


# # Data Frame

# ### Drop columns

# In[39]:


tmp_df = listings_df
col_list = tmp_df.columns.values
print('Num of columns of original dataframe: ' + str(col_list.size))

remove_list = ['id', 'listing_url', 'scrape_id', 'last_scraped', 'name',       
              'interaction', 'house_rules', 'thumbnail_url', 'medium_url',
              'picture_url', 'xl_picture_url', 'host_id', 'host_url',
              'host_thumbnail_url', 'host_picture_url',       
              'host_has_profile_pic',
              'calendar_last_scraped', 
              'first_review', 'last_review',
              'requires_license', 'license', 'jurisdiction_names',
              'is_business_travel_ready',
              'require_guest_profile_picture',
              'require_guest_phone_verification']

col_list = [e for e in col_list if e not in remove_list]
tmp_df = tmp_df.loc[:,col_list]
print('Num of columns of remaining dataframe: ' + str(tmp_df.shape[1]))


# ### Handling missing values

# # Common Functions

# ### Check missing data

# In[44]:


# Type 1
def Check_Missing_Data1(df):    
    # count all missing values of each column
    total = df.isnull().sum().sort_values(ascending=False)
    # calculate percentage of null values for each column
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

Check_Missing_Data1(listings_df)


# In[52]:


# Type 2
def Check_Missing_Data2(df):   
    col_analysis = []
    for column in df.columns:
        numNulls = len(df[column][df[column].isnull()])
        totalLength = len(df[column])
        dict1 = {'Name':column,'DataType':df[column].dtype, 'NumberOfNulls':numNulls, 'PercentageNulls':numNulls*100.0/totalLength}
        col_analysis.append(dict1)
    col_anal_df = pd.DataFrame(col_analysis)[['Name', 'DataType','NumberOfNulls','PercentageNulls']].sort_values(by='PercentageNulls', ascending=False)
    return col_anal_df

col_anal_df = Check_Missing_Data2(listings_df)
col_anal_df


# In[53]:


# remove the data inputs with percentage of missing values larger than 30%
useful_cols = col_anal_df[col_anal_df.PercentageNulls < 30]
print('List of Predictors and their respective percentages of missing values')
display(useful_cols)


# In[55]:


useful_cols.Name.values
tmp_df = listings_df.loc[:,useful_cols.Name.values]
tmp_df.shape


# In[109]:


def Normalize_Data(X_data):    
    for col in X_data.columns.values:
        X_data[col] = X_data[col].astype(float)
        mean = np.mean(X_data[col])
        std = np.std(X_data[col])
        X_data[col] = X_data[col].apply(lambda x: (x - mean) / std)
    return X_data

review_scores_columns = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                     'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                    'review_scores_value']
tmp_df = listings_df[review_scores_columns]

for column in review_scores_columns:
    tmp_df[column].fillna(tmp_df[column].mean(), inplace = True)# fill NaN values with 0

X_data = Normalize_Data(tmp_df)
X_data.head()


# In[283]:


def Feature_Frequency(df, feature):
    feature_frequency = pd.DataFrame(df.groupby(feature).id.count())
    feature_frequency.columns = ['Number_Of_Listings']
    feature_frequency = feature_frequency.reset_index()
    feature_frequency = feature_frequency.sort_values('Number_Of_Listings',ascending=[0])
    
    return feature_frequency


def Feature_Price(df, feature):
    feature_price = df.groupby(feature).price.mean()
    feature_price = feature_price.reset_index()
    feature_price = feature_price.rename(columns={'price':'average_Price'})
    feature_price = feature_price.sort_values('average_Price',ascending=[0])
    
    return feature_price


def Frequency_Plot(df,feature,title):     
    feature_frequency_df = Feature_Frequency(df, feature)
    feature_price_df = Feature_Price(df, feature)
    plt.style.use('fivethirtyeight')      
    feature_frequency_df.plot(kind='bar', x=feature, y='Number_Of_Listings',                      title = title + ' ' + feature + ' frequency', legend = False)
    plt.xlabel(feature)
    plt.ylabel('Frequency')    
    plt.figure(figsize=(7, 5))
    feature_price_df.plot(kind='bar', x=feature, y='average_Price',                       title = title + ' ' + feature + ' average price', legend = False)
    plt.xlabel(feature)
    plt.ylabel('Average Price')    


# In[284]:


def Text_Preprocessing(df,field_name):
    words_bag = ''
    for index,row in df.iterrows():
        # Convert ',' to ' ' for each row
        p = re.sub('[^a-zA-Z]+',' ', row[field_name])
        words_bag+=p
    # split a paragraph to list of word
    words_data=nltk.word_tokenize(words_bag)
    # only keep Enlisht words
    filtered_words_data=[word for word in words_data if word not in stopwords.words('english')] 
    wnl = nltk.WordNetLemmatizer() 
    words_data=[wnl.lemmatize(data) for data in filtered_words_data]
    # join all words together
    processed_words=' '.join(words_data)
    
    return processed_words

def Text_Processing2(df,field_name):
    words=''
    for index,row in df.iterrows():
        words += row[field_name]
    string_punctuation = string.punctuation
    ignoreChar=['\r','\n','',' ',"'s"]
    nums=['0','1','2','3','4','5','6','7','8','9']
    words_data=nltk.word_tokenize(words)
    words_only = [l.lower() for l in words_data if l not in string_punctuation if l not in ignoreChar if l not in nums]
    filtered_data=[word for word in words_only if word not in stopwords.words('english')] 
    wnl = nltk.WordNetLemmatizer() 
    final_data=[wnl.lemmatize(data) for data in filtered_data]
    final_words=' '.join(final_data)    
    return final_words


def Wordcloud_Visualization(txt):
    wordcloud = WordCloud(width = 500, height = 400).generate(txt)
    plt.figure(figsize=(7,6))
    plt.imshow(wordcloud)
    plt.axis("off")    


def count_words(df):
    word_list = []
    for element in df:
        # element = element[1:-1]
        x = element.split()
        word_list.append(len(x))  
    
    return word_list


# # Numerical features

# ### Select all numerical features in the dataset

# In[60]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64','uint8']
numericdata = listings_df.select_dtypes(include=numerics)
numericdata.shape      


# ### Impute missing values 

# In[69]:


# Impute Missing values: Use Mean for Real values Columns
imp = Imputer(missing_values='NaN', strategy='mean', axis=0) 
# numericdata[numericdata.columns.values] = imp.fit_transform(numericdata[numericdata.columns.values])
# Check_Missing_Data(numericdata)


# In[112]:


#fill missing values for review scores columns
review_scores_columns = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                     'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                    'review_scores_value']
tmp_df = listings_df[review_scores_columns]
for column in review_scores_columns:
    tmp_df[column].fillna(tmp_df[column].mean(), inplace = True)

Check_Missing_Data2(tmp_df)


# ### Correlation plot

# In[355]:


cols = ['number_of_reviews','host_listings_count','accommodates',
       'bathrooms','bedrooms','beds','price']


# In[376]:


# heatmap
plt.figure(figsize=(8, 8))
corrs = np.corrcoef(listings_df[cols].values.T)
sns.set(font_scale=1)
hm=sns.heatmap(corrs, cbar = True, annot=True, square = True, fmt = '.2f',
              yticklabels = cols, xticklabels = cols)


# In[375]:


plt.figure(figsize=(8, 8))
sns.heatmap(listings_df[cols].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="BuGn_r", 
linecolor='w',annot=True)
plt.show()


# ### Pair plot

# In[353]:


sns.set(style = 'whitegrid',context = 'notebook')
sns.pairplot(listings_df.loc[listings_df.price < 1000, cols],size=3)


# ### accommodates

# In[363]:


accommodates_df = listings_df[['accommodates','price','id']]
accommodates_df.head(20)
# Accommodates frequency
Frequency_Plot(accommodates_df,'accommodates','London')


# ### bathrooms

# In[208]:


listings_df['bathrooms'] = listings_df['bathrooms'].fillna(listings_df['bathrooms'].mode()[0])


# In[362]:


bathrooms_df = listings_df[['bathrooms','price','id']]
bathrooms_df.head()
Frequency_Plot(bathrooms_df,'bathrooms','London')


# ### bedrooms

# In[209]:


listings_df['bedrooms'] = listings_df['bedrooms'].fillna(listings_df['bedrooms'].mode()[0])


# In[358]:


bedrooms_df = listings_df[['bedrooms','price','id']]
bedrooms_df.head()
Frequency_Plot(bedrooms_df,'bedrooms','London')


# In[359]:


plt.scatter(bedrooms_df['price'],bedrooms_df['bedrooms'])
plt.ylabel('bedrooms')
plt.xlabel('Listing price in $')
plt.title('No. of bedrooms vs price')


# In[361]:


bedrooms_df.loc[bedrooms_df.price <= 1000,:].pivot(columns = 'bedrooms',values = 'price').plot.hist(stacked = True,bins=25)
plt.xlabel('Listing price in $')
plt.show()


# ### beds

# In[211]:


listings_df['beds'] =listings_df['beds'].fillna(listings_df['beds'].mode()[0])


# ### cleaning_fee

# In[30]:


# View data
listings_df['cleaning_fee'].head()


# In[32]:


# convert from string to float
listings_df['cleaning_fee'] = (listings_df['cleaning_fee'].str.replace(r'[^-+\d.]', '').astype(float))
listings_df['cleaning_fee'] = df['cleaning_fee'].str.replace("[$, ]", "").astype("float")
listings_df['cleaning_fee'].head()


# ### extra_people

# In[33]:


# View data
listings_df['extra_people'].head()


# In[34]:


# convert from string to float
listings_df['extra_people'] = (listings_df['extra_people'].str.replace(r'[^-+\d.]', '').astype(float))
listings_df['extra_people'] = df['extra_people'].str.replace("[$, ]", "").astype("float")
listings_df['extra_people'].head()


# ### host_listings_count

# In[207]:


#fill missing values with mean value for host_listings_count
listings_df['host_listings_count'].fillna(listings_df['host_listings_count'].mean(), inplace = True)


# ### host_response_rate

# In[206]:


#convert host_response_rate into number and fill in missing values, drop the original column
listings_df['host_response_rate_num'] = listings_df['host_response_rate'].astype(str)
listings_df['host_response_rate_num'] = listings_df['host_response_rate_num'].str.replace("%", "").astype("float")
listings_df['host_response_rate_num'].fillna(listings_df['host_response_rate_num'].mean(), inplace = True)


# ### price

# In[13]:


# View data
listings_df['price'].head()


# In[48]:


# data type
listings_df['price'].dtype


# In[20]:


# Convert price from string to float
listings_df['price'] = (listings_df['price'].str.replace(r'[^-+\d.]', '').astype(float))
listings_df['price'] = df['price'].str.replace("[$, ]", "").astype("float")
listings_df['price'].head()


# In[21]:


# Check num or rows having price = 0 and <> 0
print(listings_df[listings_df.price  == 0].shape)
print(listings_df[listings_df.price  > 0].shape)


# In[23]:


# remove rows with price = 0
listings_df = listings_df.loc[listings_df.price > 0,:] 
listings_df[listings_df.price  == 0].shape


# In[29]:


# transfrom price to log price
listings_df['log_price']  = listings_df['price'].apply(lambda x: math.log(x))
listings_df['log_price'].head()


# In[110]:


#find minimum, maximum and average price for listing
min_price = listings_df['price'].min()
max_price = listings_df['price'].max()
mean_price = listings_df['price'].mean()

print('Minimum price per listing is %d$.' % (min_price))
print('Maximum price per listing is %d$' % (max_price))
print('Average price per listing is %d$.' % (mean_price))


# In[364]:


# Price distribution
NumData_Distribution1(listings_df['price'],'Price distribution - (All price)')


# In[366]:


NumData_Distribution1(listings_df.price[listings_df.price<= 200],'Price distribution - ($0 - $200)')


# In[368]:


# Log Price distribution
listings_df['log_price']  = listings_df['price'].apply(lambda x: math.log(x))
NormalDistribution_Check(listings_df['log_price'],'Log Price distribution')


# In[371]:


# histogram
listings_df['price'].hist(bins=50)
plt.ylabel('Count')
plt.xlabel('Listing price')
plt.title('Histogram of listing prices')
plt.show()


# ### reviews

# In[357]:


number_of_reviews_df = listings_df[['number_of_reviews','price','id']]
number_of_reviews_df.head()

plt.scatter(number_of_reviews_df['number_of_reviews'],number_of_reviews_df['price'])
plt.ylabel('Listing price in $')
plt.xlabel('No. of reviews')
plt.title('No. of reviews vs price')


# In[356]:



review_scores_rating_df = listings_df[['review_scores_rating','price']]
review_scores_rating_df.head()

reviews_per_month_df = listings_df[['reviews_per_month','price']]
reviews_per_month_df.head()


# # Categorical features

# ### Select all categorical features in the dataset

# In[72]:


categorical = ['object']
categoricaldata = listings_df.select_dtypes(include=categorical)
categoricaldata.shape     


# ### List unique values and num of rows

# In[84]:


categoricaldata['bed_type'].value_counts()


# ### Check missing data

# In[86]:


Check_Missing_Data1(categoricaldata)


# ### Handling missing data

# In[79]:


# remove NaN values
categoricaldata[categoricaldata['summary'].isnull()].head()[['name','summary']]


# In[80]:


categoricaldata = categoricaldata[~categoricaldata['summary'].isnull()]
categoricaldata[categoricaldata['summary'].isnull()].head()[['name','summary']]


# In[91]:


# Use Highest frequency for categorical columns
categoricaldata[categoricaldata['property_type'].isnull()].head()[['name','property_type']]


# In[93]:


categoricaldata['property_type'] = categoricaldata['property_type'].fillna(categoricaldata['property_type'].mode()[0])
categoricaldata[categoricaldata['property_type'].isnull()].head()[['name','property_type']]


# In[101]:


# remove NaN values
categoricaldata[categoricaldata['host_location'].isnull()].head()[['name','host_location']]
categoricaldata[categoricaldata['host_neighbourhood'].isnull()].head()[['name','host_neighbourhood']]


# In[103]:


#Use Highest frequency for categorical columns
categorical_value_cols = ['host_neighbourhood','host_location']
categoricaldata[categorical_value_cols] = categoricaldata[categorical_value_cols].apply(lambda x:x.fillna(x.mode()[0]))


# In[105]:


categoricaldata[categoricaldata['host_location'].isnull()].head()[['name','host_location']]
categoricaldata[categoricaldata['host_neighbourhood'].isnull()].head()[['name','host_neighbourhood']]


# ### cancellation_policy

# In[351]:


listings_df['cancellation_policy'].unique()
cancellation_policy_df = listings_df[['cancellation_policy','price','id']]
Frequency_Plot(cancellation_policy_df,'cancellation_policy','London')


# ### host_id + host_name

# In[349]:


hostname_df = listings_df[['host_id','host_name','price']]
hostname_df.head()


# ### instant_bookable

# In[347]:


instant_bookable_df = listings_df[['instant_bookable','price','id']]
listings_df['instant_bookable'].unique()
Frequency_Plot(instant_bookable_df,'instant_bookable','London')


# ### property_type

# In[343]:


propertytype_df = listings_df[['property_type','price','id']]
Frequency_Plot(propertytype_df,'property_type','London')


# In[346]:


#Plotting the same on a heatMap
listings_df.groupby(['property_type', 'room_type']).price.mean().unstack()
plt.figure(figsize=(12,12))
sns.heatmap(listings_df.groupby(['property_type', 'room_type']).price.mean().unstack(),annot=True, fmt=".0f")


# ### room_type

# In[336]:


roomtype_df = listings_df[['room_type','price','id']]
Frequency_Plot(roomtype_df,'room_type','London')


# In[340]:


roomtype = pd.DataFrame(listings_df.groupby('room_type').id.count().sort_values())
roomtype.columns = ['NumOfListings']
roomtype.plot.pie(y = 'NumOfListings',
                 colormap = 'Blues_r', 
                 figsize=(10,10), 
                 fontsize = 20, autopct = '%.2f',
                 legend = False,
                 title = 'Room Type Distribution')
plt.show()


# In[342]:


roomtype_df.loc[roomtype_df.price <= 1000,:].pivot(columns = 'room_type',values = 'price').plot.hist(stacked = True,bins=25)
plt.xlabel('Listing price in $')
plt.show()


# ### zipcode

# In[ ]:


zipcode_df = listings_df[['zipcode','price']]
zipcode_df.head()


# In[380]:


listings_df = listings_df[pd.notnull(listings_df['zipcode'])]
listings_df = listings_df[listings_df['zipcode']!= ' ']

for index,row in listings_df.iterrows():
    zipcode = row['zipcode']
    zipcode = nltk.word_tokenize(zipcode)[0].upper()[0:3]
    listings_df.loc[index,'new_zipcode'] = zipcode


# In[382]:



listings_df.new_zipcode.head(10)


# # Text features

# ### amenities

# In[221]:


col = listings_df.amenities
t = col[0]
t.replace('[', '').replace("'", '').replace("]", '').replace('"', '').replace('{', '').replace('}', '').split(',')


# In[223]:


values_dict = {}
for unique_value in col.unique(): 
    for value in unique_value.replace('[', '').replace("'", '').replace("]", '').replace('"', '').replace('{', '').replace('}', '').split(','):
        if value in values_dict:
            values_dict[value] = values_dict[value] + 1
        else:
            values_dict[value] = 0            
values_sorted = sorted(values_dict.items(), key=lambda kv: kv[1], reverse = True)
values_sorted


# In[227]:


def get_val_from_list(row, column_name, value):
    ''' Fill in dummy column for values '''
    val = 0.0
    try:
        vals = row[column_name].replace('[', '').replace("'", '').replace("]", '').replace('"', '').replace('{', '').replace('}', '').split(',')
        if value in vals:
            val = 1.0
    except:
        val = 0.0
    return val

tmp_df =  listings_df[['id','amenities']]
# split into columns
for value in values_sorted[: 15]:
    tmp_df['amenities_' + value[0]] = tmp_df.apply(lambda row: get_val_from_list(row, 'amenities', value[0]),axis=1)
tmp_df.head()


# In[216]:


def get_val_from_list(row, column_name, value):
    ''' Fill in dummy column for values '''
    val = 0.0
    try:
        vals = row[column_name].replace('[', '').replace("'", '').replace("]", '').replace('"', '').replace('{', '').replace('}', '').split(',')
        if value in vals:
            val = 1.0
    except:
        val = 0.0
    return val

def split_list_into_columns(df, column_name, max_dummies_num = 10):
    ''' Split values in columns, which contain lists (for example, amenities) '''
    
    # get dictionary of unique values in lists across dataset rows
    values_dict = {}

    for unique_value in df[column_name].unique(): 
        for value in unique_value.replace('[', '').replace("'", '').replace("]", '').replace('"', '').replace('{', '').replace('}', '').split(','):
            if value in values_dict:
                values_dict[value] = values_dict[value] + 1
            else:
                values_dict[value] = 0
                
    values_sorted = sorted(values_dict.items(), key=lambda kv: kv[1], reverse = True)
      
    # split into columns
    for value in values_sorted[: max_dummies_num]:
        df[column_name + '_' + value[0]] = df.apply(lambda row: get_val_from_list(row, column_name, value[0]),axis=1)
        
    return


# In[219]:


tmp_df =  listings_df[['id','amenities']]
#split amenities into dummy columns and drop the original column
split_list_into_columns(tmp_df, 'amenities')

tmp_df.head()
tmp_df.columns.values


# In[333]:


# Analyzing what amenities costs more ? top price
amenities_df = listings_df[['amenities','price','id']]
amenitiesDFTopper = amenities_df.sort_values('price',ascending=[0])
amenitiesDFtop = amenitiesDFTopper.head(500)
amenitiesDFtop.tail(20)
allemenities_words = Text_Preprocessing(amenitiesDFtop,'amenities')    
Wordcloud_Visualization(allemenities_words)


# In[334]:


#Analyzing what amenities costs more ? bottom price
amenitiesDFbott =amenities_df.sort_values('price',ascending=[1])
amenitiesDFbottom=amenitiesDFbott.head(500)
amenitiesDFbottom.head(20)
allemenities_words = Text_Preprocessing(amenitiesDFbottom,'amenities')    
Wordcloud_Visualization(allemenities_words)


# In[335]:


# splitting the amenities list to count the number of amenities
listings_df['num_of_amenities'] = count_words(listings_df.amenities)
listings_df['num_of_amenities'].head()
amenities_df = listings_df[['num_of_amenities','price','id']]
Frequency_Plot(amenities_df,'num_of_amenities','London')


# ### description

# In[332]:


description_df = listings_df[['description','price']]
description_df = description_df[pd.notnull(description_df['description'])]
description_df = description_df[description_df['description']!=0]
description_df = description_df.sort_values('price',ascending=[0])
top_des_df = description_df.head(500)
top_des_df.tail(20)
final_words = Text_Processing2(top_des_df,'description')
Wordcloud_Visualization(final_words)


# ### neighborhood_overview

# In[330]:


neighborhood_overview_df = listings_df[['neighborhood_overview','price']]
neighborhood_overview_df = neighborhood_overview_df[pd.notnull(neighborhood_overview_df['neighborhood_overview'])]
neighborhood_overview_df = neighborhood_overview_df.sort_values('price',ascending=[0])
top_nbh_df = neighborhood_overview_df.head(500)
final_words = Text_Processing2(top_nbh_df,'neighborhood_overview')
Wordcloud_Visualization(final_words)


# ### neighbourhood_group_cleansed

# In[169]:


#get list of neighbourhoods
neighbourhoods = listings_df['neighbourhood_group_cleansed'].unique()
neighbourhoods


# In[156]:


#merge datasets
df = listings_df
df = df.rename(index=str, columns={"id": "listing_id"})
df = pd.merge(calendar_df, df, on = 'listing_id')
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].apply(lambda x:x.to_pydatetime())
df['year'] = df['date'].apply(lambda x:x.year)
df['year'] = df['year'].astype(int)
df['month'] = df['date'].apply(lambda x:x.month)
df['month'] = df['month'].astype(int)
df = df.drop(columns = ['date'])
#remove rows where price_x == nan, because price will be used as response column
df = df.dropna(subset=['price_x'])
#convert price to number and drop the price_x column
df['price'] = df['price_x'].astype(str)
df['price'] = df['price'].str.replace("[$, ]", "").astype("float")
df = df.drop(columns = ['price_x'])


# In[177]:


#get prices by month and neighbourhood
price_by_month_neighbourhood = df.groupby(['month','neighbourhood_group_cleansed']).mean().reset_index()
price_by_month_neighbourhood


# In[180]:



#plot prices for each neighbourhood
fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111)

for neighbourhood in neighbourhoods:
    ax.plot(price_by_month_neighbourhood[price_by_month_neighbourhood['neighbourhood_group_cleansed'] == neighbourhood]['month'],
             price_by_month_neighbourhood[price_by_month_neighbourhood['neighbourhood_group_cleansed'] == neighbourhood]['price'],
             label = neighbourhood)
    
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.ylabel('Average price, $')
plt.xlabel('Month')
plt.title('Average price for neighbourhood, $')
plt.show()


# In[312]:


neighbourhood_df = listings_df[['neighbourhood_cleansed','price','id']]
neighbourhood_df.head()


# In[313]:


Feature_Frequency(neighbourhood_df,'neighbourhood_cleansed')


# In[314]:


Feature_Price(neighbourhood_df,'neighbourhood_cleansed')


# In[317]:


# Let's look at the differences between the words used in listings for Airbnb locations in different Boston neighborhoods
uniqueNeighborhoods = listings_df.neighbourhood_cleansed.unique().tolist()
uniqueNeighborhoods


# In[318]:


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


# In[322]:


summ={}
for n in uniqueNeighborhoods:
    text=''
    for index,row in listings_df.iterrows():
        if row['neighbourhood_cleansed']==n:
            if (pd.notnull(row['description']) & pd.notnull(row['space']) & pd.notnull(row['neighborhood_overview'])):
                text =text+row['description']+row['space']+row['neighborhood_overview']
    summ[n]=text
final_df_neighbrhood=pd.DataFrame(columns=['neighborhood','top 5 words in description'])
for a in summ.items():
    top5words=cleanData(a[0],a[1])
    final_df_neighbrhood=final_df_neighbrhood.append(pd.Series([a[0],top5words],index=['neighborhood','top 5 words in description']),ignore_index=True)

final_df_neighbrhood.head()


# ### street

# In[306]:


street_df = listings_df[['street','price']]
street_df.head()


# ### space

# In[310]:


space_df = listings_df[['space','price']]
space_df.head(20)

space_df = space_df[pd.notnull(space_df['space'])]
space_df = space_df[space_df['space']!=0]
space_df = space_df.sort_values('price',ascending=[0])
top100_df = space_df.head(100)
final_words = Text_Processing2(top100_df,'space')
Wordcloud_Visualization(final_words)


# ### summary

# In[308]:


# summary: Analyzing and plotting word cloud for summary
summary_df = listings_df[['summary','price']]
summary_df = summary_df[pd.notnull(summary_df['summary'])]
summary_df = summary_df[summary_df['summary']!=0]
summary_df = summary_df.sort_values('price',ascending=[0])
top100_df = summary_df.head(100)
final_words = Text_Processing2(top100_df,'summary')
Wordcloud_Visualization(final_words)


# # Datetime features

# ### host_response_time

# In[281]:


host_response_time_df = listings_df[['host_response_time','price','id']]
host_response_time_df.head(10)


# In[285]:


Feature_Frequency(host_response_time_df,'host_response_time')


# In[286]:


Feature_Price(host_response_time_df,'host_response_time')


# ### host_since

# In[204]:


# convert host_since date into number and fill in missing values, drop the original column
listings_df['host_since'] = pd.to_datetime(listings_df['host_since'])
listings_df['host_since'] = listings_df['host_since'].apply(lambda x:x.to_pydatetime())
listings_df['host_since_year'] = listings_df['host_since'].apply(lambda x:x.year)
listings_df['host_since_year'].fillna(listings_df['host_since_year'].mean(), inplace = True)
listings_df['host_since_year'] = listings_df['host_since_year'].astype(int)


# In[288]:


listings_df['host_since_month'] = listings_df['host_since'].apply(lambda x:x.month)
listings_df['host_since_month'].fillna(listings_df['host_since_month'].mean(), inplace = True)
listings_df['host_since_month'] = listings_df['host_since_month'].astype(int)

listings_df['host_since_day'] = listings_df['host_since'].apply(lambda x:x.day)
listings_df['host_since_day'].fillna(listings_df['host_since_day'].mean(), inplace = True)
listings_df['host_since_day'] = listings_df['host_since_day'].astype(int)


# In[291]:


# this graph shows count of host registered everyday by resampling the data for each day
join_dates = pd.to_datetime(listings_df['host_since']).value_counts().resample('M').mean().fillna(0)
plt.figure(figsize=(18,10))
join_dates.plot()
plt.xlabel('year')
plt.ylabel('number of hosts')
plt.show()


# In[299]:


plt.figure(figsize=(16,10))
year_df = listings_df.groupby(['host_since_year']).id.count()
year_df = year_df.reset_index()
year_df = year_df.rename(columns={'id':'Number of hosts'})
ax = sns.barplot(x='host_since_year', y="Number of hosts", data=year_df,palette="Blues_d")
plt.ylabel('Number of hosts')
plt.show()


# In[300]:


#plotting for all months for each year
yearMonth=listings_df.groupby(['host_since_year','host_since_month']).id.count()
yearMonth=yearMonth.reset_index()
yearMonth=yearMonth.sort_values(['host_since_year','host_since_month'],ascending=[1,1])
yearMonth=yearMonth.rename(columns={"id":"Number of hosts"})
sns.factorplot(data=yearMonth, x="host_since_month", y="Number of hosts",col="host_since_year",col_wrap=3)
plt.show()


# In[302]:


plt.figure(figsize=(10,8))
pd.to_datetime(listings_df['host_since']).dt.dayofweek.value_counts().sort_index().plot(kind='bar')
plt.xlabel("Days")
plt.ylabel("Number of hosts")
plt.show()


# # Calendar

# In[231]:


calendar_df.shape


# In[114]:


# convert price to number 
calendar_df['price'] = calendar_df['price'].str.replace("[$, ]", "").astype("float")
calendar_df.head()


# In[230]:


# Only consider data for which there is data
nan_adjusted = calendar_df[np.isfinite(calendar_df['price'])]
#Still left with a reasonable amount of observations 
nan_adjusted.shape  


# In[120]:


# remove NaN rows for a subset of columns
calendar_df = calendar_df.dropna(subset=['price'])


# In[122]:


# remove NaN rows for all columns
calendar_df = calendar_df.dropna()


# In[127]:


# Histogram for price distribution
mean_price_for_calendar = calendar_df.groupby('listing_id').mean()['price']
mean_price_for_calendar


# In[128]:


#plot
plt.figure(figsize=(15,7))
plt.hist(mean_price_for_calendar, bins=50)
plt.xticks(np.arange(0, 1700, step=100))
plt.ylabel('Number of listings')
plt.xlabel('Price, $')
plt.title('Number of listings depending on price')
plt.show()


# In[130]:


# extract year, month for each date
calendar_df['date'] = pd.to_datetime(calendar_df['date'])
calendar_df['date'] = calendar_df['date'].apply(lambda x:x.to_pydatetime())
calendar_df['year'] = calendar_df['date'].apply(lambda x:x.year)
calendar_df['year'] = calendar_df['year'].astype(int)
calendar_df['month'] = calendar_df['date'].apply(lambda x:x.month)
calendar_df['month'] = calendar_df['month'].astype(int)

calendar_df.head()


# In[134]:


number_of_listings_by_month = calendar_df.groupby('month').listing_id.count()
number_of_listings_by_month


# In[144]:


# Line plot
plt.figure(figsize=(10,5))
plt.plot(number_of_listings_by_month)
plt.xticks(np.arange(1, 13, step=1))
plt.ylabel('Number of listings')
plt.xlabel('Month')
plt.title('Number of listings per month, 2016')
plt.show()


# In[143]:


# bar plot
plt.figure(figsize=(10,5))
number_of_listings_by_month.plot(kind='bar', x=number_of_listings_by_month.index.values, y=number_of_listings_by_month.values, legend = False)
plt.ylabel('Number of listings')
plt.xlabel('Month') 
plt.title('Number of listings per month, 2016')
plt.show()


# In[146]:


#find average price by month
average_price_by_month = calendar_df.groupby(['month'])['price'].mean()

#plot
plt.figure(figsize=(10,5))
plt.plot(average_price_by_month)
plt.ylabel('Average price, $')
plt.xlabel('Month')
plt.title('Average price')
plt.show()


# In[147]:


# bar plot
plt.figure(figsize=(10,5))
average_price_by_month.plot(kind='bar', x=average_price_by_month.index.values, y=average_price_by_month.values, legend = False)
plt.ylabel('Average price, $')
plt.xlabel('Month')
plt.title('Average price')
plt.show()


# In[229]:


calendar_df['log_price']  = calendar_df['price'].apply(lambda x: math.log(x))
calendar_df.head(10)


# In[236]:


def NumData_Distribution1(df, title):     
    plt.figure(figsize=(16, 5))
    plt.subplot(121)
    sns.boxplot(x=df)
    plt.title(title)    
    plt.subplot(122)
    sns.distplot(df, hist = True, kde = True,kde_kws = {'shade': True, 'linewidth': 3})
    plt.title(title)

def NormalDistribution_Check(df, title):
    plt.figure(figsize=(16, 5)) 
    plt.subplot(121)    
    sns.distplot(df, hist = True, kde = True,kde_kws = {'shade': True, 'linewidth': 3})
    plt.title(title)    
    plt.subplot(122)
    stats.probplot(df, plot=plt)
    plt.title(title)
    #skewness and kurtosis
    print("Skewness: %f" % df.skew())
    print("Kurtosis: %f" % df.kurt())


# In[237]:


# Price distribution
NumData_Distribution1(average_price_df['price'],'Price distribution - (All price)')


# In[238]:


NumData_Distribution1(average_price_df.price[average_price_df.price<= 200],'Price distribution - ($0 - $200)')


# In[239]:


# Log Price distribution
NormalDistribution_Check(average_logprice_df['log_price'],'Log Price distribution')


# In[248]:


# Only consider data for which there is data
nan_adjusted = calendar_df[np.isfinite(calendar_df['log_price'])]
#Still left with a reasonable amount of observations 
nan_adjusted.shape  # (9468074, 5)
# Group by listing_id, then find each listing id's average price
grouped = nan_adjusted.groupby(['listing_id'], as_index = False)


# In[250]:


print("number of unique properties:", len(nan_adjusted['listing_id'].unique()))
# Let's sort to make things easier on pattern matching algorithm
nan_adjusted.sort_values(['listing_id'], inplace=True)
nan_adjusted.reset_index(inplace=True)
nan_adjusted.head(10)


# In[251]:


mean_df = grouped.mean()
mean_df['listing_id']
mean_df.head()


# In[259]:


average_price_df = grouped['price'].mean()
average_price_df.head()


# In[258]:


average_logprice_df = grouped['log_price'].mean()
average_logprice_df.head()


# In[260]:


# Examine Distribution of average price by listing_id
fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 8))
ax1.hist(average_price_df['price'], color='red', bins=40, alpha=0.5)
ax1.set_title('distribution of mean price by listing_id')
ax2.hist(average_logprice_df['log_price'], color='blue', bins=40, alpha=0.5)
ax2.set_title('distribution of the mean of log(price) by listing_id')
plt.tight_layout()
plt.show()


# In[261]:


average_price_df = average_price_df.merge(average_logprice_df,how='left', left_on='listing_id', right_on='listing_id')
average_price_df.columns = ['listing_id','avg_price','avg_log_price']
average_price_df.head()


# In[262]:


# Initialize an empty column for the price differences 
nan_adjusted['price_diff'] = 0
nan_adjusted = nan_adjusted.merge(average_price_df,how='left', left_on='listing_id', right_on='listing_id')
nan_adjusted['price_diff'] = nan_adjusted['price'] - nan_adjusted['avg_price']
nan_adjusted['date'] = pd.to_datetime(nan_adjusted['date'])
nan_adjusted.head()


# In[263]:


# Groupby datetime, get average price difference
final = nan_adjusted.groupby(['date'], as_index = False)
final_sums = final['price_diff'].mean()
print ("Number of days average price difference was positive:", sum(final_sums['price_diff']>0))


# In[266]:


# Examine overall price difference trend
fig, ax = plt.subplots(1, 1, figsize=(65, 20))
final_sums.plot(kind='bar', x= 'date', y='price_diff', ax=ax)
ax.set_xticklabels(final_sums['date'])
ax.set_xlabel('Date')
ax.set_ylabel('Avg Price Difference')
ax.set_title('Days When List Price Exceeded Average List Price for Individual AirBnbs')
plt.show()


# In[267]:


# Subset the data for easier viewing 
fall = final_sums[(final_sums['date'] > '2016-09-22') & (final_sums['date'] < '2016-12-22')] 
winter = final_sums[(final_sums['date'] > '2016-12-21') | (final_sums['date'] < '2016-03-20')] 
spring = final_sums[(final_sums['date'] > '2016-03-19') & (final_sums['date'] < '2016-06-21')] 
summer = final_sums[(final_sums['date'] > '2016-06-20') & (final_sums['date'] < '2016-09-23')]


# In[270]:


# WINTER 
fig, ax = plt.subplots(1, 1, figsize=(55, 15))
winter.plot(kind='bar', x= 'date', y='price_diff',  ax=ax)
ax.set_xticklabels(winter['date'])
ax.set_xlabel('Date')
ax.set_ylabel('Avg Price Difference')
ax.set_title('Average Price Difference in Winter')
plt.show()


# In[273]:


# SUMMER 
fig, ax = plt.subplots(1, 1, figsize=(55, 15))
summer.plot(kind='bar', x= 'date', y='price_diff',  ax=ax)
ax.set_xticklabels(summer['date'])
ax.set_xlabel('Date')
ax.set_ylabel('Avg Price Difference')
ax.set_title('Average Price Difference in Summer')
plt.show()


# In[275]:


#FALL
fig, ax = plt.subplots(1, 1, figsize=(55, 15))
fall.plot(kind='bar', x= 'date', y='price_diff',  ax=ax)
ax.set_xticklabels(fall['date'])
ax.set_xlabel('Date')
ax.set_ylabel('Avg Price Difference')
ax.set_title('Average Price Difference in Fall')
plt.show()


# In[278]:


#FALL
fig, ax = plt.subplots(1, 1, figsize=(55, 15))
spring.plot(kind='bar', x= 'date', y='price_diff',  ax=ax)
ax.set_xticklabels(fall['date'])
ax.set_xlabel('Date')
ax.set_ylabel('Avg Price Difference')
ax.set_title('Average Price Difference in Spring')
plt.show()

