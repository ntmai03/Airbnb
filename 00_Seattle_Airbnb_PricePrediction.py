'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
https://www.kaggle.com/aleksandradeis/airbnb-seattle-reservation-prices-analysis
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'###############################################################################
'Import package
' You should consider upgrading via the 'python -m pip install --upgrade pip' command.
'###############################################################################
#Import linear algebra and data manipulation
import numpy as np
import pandas as pd

#Import plotting packages
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Import machine learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost

from sklearn.model_selection import train_test_split #split
from sklearn.metrics import r2_score, mean_squared_error #metrics


'###############################################################################
'Load data
'###############################################################################
# listings
listings_df = pd.read_csv('C:\\DataScience\\00_AnalysisProjects\\AirbnbAnalysis\\Data\\seattle_listings.csv', sep=',')
listings_df.shape   # (3818, 92)
# calendar
calendar_df = pd.read_csv('C:\\DataScience\\00_AnalysisProjects\\AirbnbAnalysis\\Data\\seattle_calendar.csv')
calendar_df.shape   # (1393570, 4)
# reviews
review_df = pd.read_csv('C:\\DataScience\\00_AnalysisProjects\\AirbnbAnalysis\\Data\\seattle_reviews.csv')
review_df.shape   # (84849, 6)

'###############################################################################
'View data structure and content
'###############################################################################
listings_df.columns.values


'###############################################################################
'Helper functions for dataset cleaning
'###############################################################################
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
values_dict = {}
col = listing_df.amenities
t = col[0]
t.replace('[', '').replace("'", '').replace("]", '').replace('"', '').replace('{', '').replace('}', '').split(',')
values_dict[t] = values_dict[t] + 1


values_dict = {}
for unique_value in col.unique(): 
    for value in unique_value.replace('[', '').replace("'", '').replace("]", '').replace('"', '').replace('{', '').replace('}', '').split(','):
        if value in values_dict:
            values_dict[value] = values_dict[value] + 1
        else:
            values_dict[value] = 0            
values_sorted = sorted(values_dict.items(), key=lambda kv: kv[1], reverse = True)

# split into columns
for value in values_sorted[: 20]:
    df[column_name + '_' + value[0]] = df.apply(lambda row: get_val_from_list(row, column_name, value[0]),axis=1)


def get_extra_people_fee(row):
    ''' Return 1 when the is fee for exatra people '''
    if row['extra_people'] == '$0.00':
        return 0.0
    else:
        return 1.0

# apply functions above to clean dataset
df = listings_df
df.shape

#drop the irrelevant columns
columns_to_drop = ['host_id','host_location','host_acceptance_rate','host_neighbourhood',
                   'host_total_listings_count', 'weekly_price', 'monthly_price',
                   'security_deposit', 'cleaning_fee', 'calendar_updated',
                   'listing_url','last_scraped' ,'scrape_id', 'name', 'summary', 'space', 'description',
                   'experiences_offered', 'street', 'neighbourhood', 'neighbourhood_cleansed', 'zipcode',
                   'neighborhood_overview', 'notes', 'transit', 'thumbnail_url', 'medium_url', 'picture_url',
                   'xl_picture_url', 'host_url', 'host_name', 'host_about', 'host_thumbnail_url', 'host_picture_url',
                   'city', 'state', 'market', 'smart_location', 'country_code', 'country', 'latitude', 'longitude',
                   'is_location_exact', 'square_feet', 'has_availability', 'availability_30',
                   'availability_60', 'availability_90', 'availability_365', 'calendar_last_scraped',
                  'first_review', 'last_review', 'requires_license', 'license', 'jurisdiction_names', 
                  'reviews_per_month']
df = df.drop(columns = columns_to_drop)
df.columns.values
# convert price to number 
df['price'] = df['price'].str.replace("[$, ]", "").astype("float")

# convert host_since date into number and fill in missing values, drop the original column
df['host_since'] = pd.to_datetime(df['host_since'])
df['host_since'] = df['host_since'].apply(lambda x:x.to_pydatetime())
df['host_since_year'] = df['host_since'].apply(lambda x:x.year)
df['host_since_year'].fillna(df['host_since_year'].mean(), inplace = True)
df['host_since_year'] = df['host_since_year'].astype(int)
df = df.drop(columns = ['host_since'])

#convert host_response_rate into number and fill in missing values, drop the original column
df['host_response_rate_num'] = df['host_response_rate'].astype(str)
df['host_response_rate_num'] = df['host_response_rate_num'].str.replace("%", "").astype("float")
df['host_response_rate_num'].fillna(df['host_response_rate_num'].mean(), inplace = True)

df['host_response_rate_buckets'] = pd.qcut(df['host_response_rate_num'], 5, labels=False, duplicates = 'drop')
df = df.drop(columns = ['host_response_rate', 'host_response_rate_num'])   

#fill missing values with mean value for host_listings_count
df['host_listings_count'].fillna(df['host_listings_count'].mean(), inplace = True)

#split host_verifications into dummy columns and drop the original column
split_list_into_columns(df, 'host_verifications')
df = df.drop(columns = ['host_verifications'])

#fill in missing values for bathrooms, bedrooms and beds with mode
df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].mode()[0])
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].mode()[0])
df['beds'] = df['beds'].fillna(df['beds'].mode()[0])

#split amenities into dummy columns and drop the original column
split_list_into_columns(df, 'amenities')
df = df.drop(columns = ['amenities'])
df.columns.values

df.amenities_Kitchen

#turn extra people fee into binary column (1 - if fee for extra people is charged, 0 - otherwise)
df['extra_people_fee'] = df.apply(lambda row: get_extra_people_fee(row),axis=1)
df = df.drop(columns = ['extra_people'])

#fill missing values for review scores columns
review_scores_columns = ['review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
                     'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
                    'review_scores_value']
for column in review_scores_columns:
    df[column].fillna(df[column].mean(), inplace = True)



############ MACHINE LEARNING
#turn categorical columns into dummies
cat_columns = list(df.select_dtypes(include=['object']).columns)
    
for col in  cat_columns:
    df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], prefix=col, prefix_sep='_',drop_first=True, dummy_na=True)], axis=1)
df.shape
calendar_df.head()

#merge datasets
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

#drop listing_id and year columns
df = df.drop(columns = ['listing_id', 'year','available'])


#find percentage of missing values for each column
listings_missing_df = df.isnull().mean()*100
#filter out only columns, which have missing values
listings_columns_with_nan = listings_missing_df[listings_missing_df > 0]

df.shape
df.columns.values

col_analysis = []
for column in df.columns:
    numNulls = len(df[column][df[column].isnull()])
    totalLength = len(df[column])
    dict1 = {'Name':column,'DataType':df[column].dtype, 'NumberOfNulls':numNulls, 'PercentageNulls':numNulls*100.0/totalLength}
    col_analysis.append(dict1)
    print('Name: ' + column + 'DataType: ' + str(df[column].dtype))

col_anal_df = pd.DataFrame(col_analysis)[['Name', 'DataType','NumberOfNulls','PercentageNulls']].sort_values(by='PercentageNulls', ascending=False)
cat_columns = list(df.select_dtypes(include=['Timestamp']).columns)

#prepare train and test datasets for modelling
TEST_SIZE = 0.3
RAND_STATE = 42

X = df.drop(columns = 'price')
y = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state=RAND_STATE)
#train RF regressor model
forest = RandomForestRegressor(n_estimators=100, 
                               criterion='mse', 
                               random_state=RAND_STATE, 
                               n_jobs=-1)
forest.fit(X_train, y_train.squeeze())

#calculate scores for the model
y_train_preds = forest.predict(X_train)
y_test_preds = forest.predict(X_test)

print('Random Forest MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_preds),
        mean_squared_error(y_test, y_test_preds)))
print('Random Forest R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_preds),
        r2_score(y_test, y_test_preds)))


#get feature importances from the model
headers = ["name", "score"]
values = sorted(zip(X_train.columns, forest.feature_importances_), key=lambda x: x[1] * -1)
forest_feature_importances = pd.DataFrame(values, columns = headers)
forest_feature_importances = forest_feature_importances.sort_values(by = ['score'], ascending = False)

features = forest_feature_importances['name'][:15]
y_pos = np.arange(len(features))
scores = forest_feature_importances['score'][:15]

#plot feature importances
plt.figure(figsize=(10,5))
plt.bar(y_pos, scores, align='center', alpha=0.5)
plt.xticks(y_pos, features, rotation='vertical')
plt.ylabel('Score')
plt.xlabel('Features')
plt.title('Feature importances (Random Forest)')

plt.savefig('feature importances RF.png')
 
plt.show()

