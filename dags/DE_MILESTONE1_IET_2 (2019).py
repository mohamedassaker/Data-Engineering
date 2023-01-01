#!/usr/bin/env python
# coding: utf-8

# # Milestone 1 - EDA and Preprocessing data 

# - Load dataset
# - Explore the dataset and ask atleast 5 questions to give you a better understanding of the data provided to you. 
# - Visualise the answer to these 5 questions.
# - Cleaing the data
# - Observe missing data and comment on why you believe it is missing(MCAR,MAR or MNAR) 
# - Observe duplicate data
# - Observe outliers
# - After observing outliers,missing data and duplicates, handle any unclean data.
# - With every change you are making to the data you need to comment on why you used this technique and how has it affected the data(by both showing the change in the data i.e change in number of rows/columns,change in distrubution, etc and commenting on it).
# - Data transformation and feature engineering
# - Add a new column named 'Week number' and discretisize the data into weeks according to the dates.Tip: Change the datatype of the date feature to datetime type instead of object.
# - Encode any categorical feature(s) and comment on why you used this technique and how the data has changed.
# - Identify feature(s) which need normalisation and show your reasoning.Then choose a technique to normalise the feature(s) and comment on why you chose this technique.
# - Add atleast two more columns which adds more info to the dataset by evaluating specific feature(s). I.E( Column indicating whether the accident was on a weekend or not). 
# - For any imputation with arbitrary values or encoding done, you have to store what the value imputed or encoded represents in a new csv file. I.e if you impute a missing value with -1 or 100 you must have a csv file illustrating what -1 and 100 means. Or for instance, if you encode cities with 1,2,3,4,etc what each number represents must be shown in the new csv file.
# - Load the new dataset into a csv file.
# - **Extremely Important note** - Your code should be as generic as possible and not hard-coded and be able to work with various datasets. Any hard-coded solutions will be severely penalised.
# - Bonus: Load the dataset as a parquet file instead of a csv file(Parquet file is a compressed file format).

# # 1 - Extraction

# Import libraries & CSV file

# In[229]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from scipy import stats
# For min_max scaling
from sklearn.preprocessing import MinMaxScaler

# For z-score scaling
from sklearn.preprocessing import StandardScaler

# For Box-Cox Normalization
from scipy import stats

# For Label Encoding
from sklearn import preprocessing

# 'C://Users/BETER/Desktop/9th Material/Data Engineering/mile1/2019_Accidents_UK.csv'
file = 'E://University/Semester 9/Data Engineering (NETW 908)/Project/2019_Accidents_UK.csv'
df = pd.read_csv(file, index_col='accident_index', low_memory=0)


# Displaying head of the dataframe

# In[230]:


df.head()


# In[231]:


df.var()


# In[232]:


df.describe()


# # 2- EDA

# Is there a relation between number of vehicles and number of casualties?    
# number_of_vehicles VS. number_of_casualties

# In[233]:


df_new = df.copy()
age_counts = df_new.groupby(['number_of_vehicles'])['number_of_casualties'].sum() #returns a series of each age and the corresponding number of individuals who survived
plt.xlabel('number_of_vehicles')
plt.ylabel('number_of_casualties')
plt.title('number_of_vehicles VS. number_of_casualties')
plt.scatter(age_counts.index,age_counts)
plt.show()


# What are number of occurence for each number of casualities?     
# Number of casualties vs Count of casualties

# In[234]:


# df_sorted_date = df.copy()
# df_sorted_date = df_sorted_date.sort_values(by=['date'], ascending=True)

class_series = df['number_of_casualties'].value_counts()
class_series
bar_plot = plt.bar(class_series.index, class_series)
plt.locator_params(integer=True)
plt.xlabel('Number of casualties')
plt.ylabel('Count of casualties')
plt.show()


# How many accidents occur each day?     
# Number of accidents that occured in each day

# In[235]:


df_weekdays = df.copy()
weekday_series = df_weekdays.day_of_week.value_counts()
weekday_series

bar_plot = plt.bar(weekday_series.index, weekday_series)
plt.locator_params(integer=True)
plt.xlabel('Days of weeks')
plt.xticks(rotation = 90)
plt.ylabel('Count of accidents')
plt.show()


# Distribution of Vehicle numbers

# In[236]:


sns.kdeplot(df.number_of_vehicles)
plt.title('Dist. of Vehicle numbers')
plt.xlabel('Number of Vehicle')
plt.ylabel('Density of distribution')
plt.show()


# Dist. of Casualities numbers

# In[237]:


sns.kdeplot(df.number_of_casualties)
plt.title('Dist. of Casualities numbers')
plt.xlabel('Number of Casulaties')
plt.ylabel('Density of distribution')
plt.show()


# Dist. of Casualities numbers & vehicles number aren't normally distributed

# Distribution of week day is normally distributed

# In[238]:


sns.kdeplot(df['day_of_week'].value_counts())
plt.title('Dist. of week day')
plt.ylabel('Density of distribution')
plt.show()


# What are the top 5 local authority districts where accidents occured? 

# In[239]:


class_series = df['local_authority_district'].value_counts().head()
class_series
bar_plot = plt.bar(class_series.index, class_series)
plt.locator_params(integer=True)
plt.xlabel('Names of local_authority_district')
plt.xticks(rotation = 90)
plt.ylabel('Count of local_authority_district')
plt.show()


# What are the top 5 dates when accidents occured?     
# 04/12/2019 was Friday. Brexit election.         
# 29/11/2019 was Friday. London Bridge Stabbing.          
# 20/09/2019 was Friday. Massive Climate Strike.         

# In[240]:


class_series = df['date'].value_counts().head()
class_series
bar_plot = plt.bar(class_series.index, class_series)
plt.locator_params(integer=True)
plt.xlabel('date')
plt.xticks(rotation = 90)
plt.ylabel('Count of date')
plt.show()


# # 3 - Cleaning Data

# In[241]:


df.accident_reference.value_counts()


# accident_reference is unqiue for each row, so it doesn't add any new info       
# It can be used as index, but we already used accident_index as index    
# As a result, it should be dropped

# In[242]:


df = df.drop('accident_reference', axis=1)
df.head()


# ## Observing Missing and duplicate Data

# Get number of missing data in each column

# In[243]:


# Calculating sum of null entries and percentage of null entries
sum_null = df.isnull().sum()
perc_null = df.isnull().sum() / len(df)
perc_null_mean = df.isnull().mean()*100

#Sum Of Missing Values
sum_null


# Percentage of missing data in each column

# In[244]:


perc_null_mean


# location_easting_osgr       
# location_northing_osgr           
# longitude               
# latitude
# 
# Missing Completly At Random because the % of missing is only 0.028% 
# There is no releation between missing data and others

# In[245]:


len(df)


# Second_road_number is not missing at random  
# If second_road_class is "-1" the second_road_number is missing

# road_type   
# weather_conditions
#  
# missing at random as it is only 1.82% of data

# Get number of duplicates in dataframe

# In[246]:


df.duplicated().sum()


# There is 1 duplicate

# In[247]:


df = df.drop_duplicates()


# In[248]:


df.duplicated().sum()


# Duplicate is now dropped

# ## Handling Missing data

# Drop rows with missing data in as they are MCAR so it won't affect the data  
# 'location_easting_osgr',    
# 'location_northing_osgr',   
# 'longitude',    
# 'latitude',     
# 'road_type',    
# 'weather_conditions'

# In[249]:


df_dropped = df.dropna(axis='index', subset=['location_easting_osgr','location_northing_osgr','longitude','latitude','road_type','weather_conditions'])


# Calculating sum of null entries and percentage of null entries after dropping these rows

# In[250]:


# Calculating sum of null entries and percentage of null entries
sum_null = df_dropped.isnull().sum()
perc_null = df_dropped.isnull().sum() / len(df)
perc_null_mean = df_dropped.isnull().mean()*100
perc_null_mean


# Calculate variance of dataframe (after previous dropping rows) to know which arbitary value is suitable for "second_road_number" column       

# In[251]:


df_dropped = df_dropped.replace('first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ', int(0))
df_dropped.second_road_number = df_dropped.second_road_number.astype(float)
df_dropped = df_dropped.replace(r'',np.NaN)
df_dropped.var()


# Replace missing data second_road_number with arbitary value (-1)      
# -1 is best option as it doesn't change variance as 9999 changes

# In[252]:


df_dropped = df_dropped.replace(np.NaN, -1)
df_dropped.second_road_number.unique()


# Calculate original dataframe variance

# In[253]:


df.var()


# Calculating variance of second_road_number after replacement

# In[254]:


df_dropped.var()


# Compare between variance different of -1 and 9999

# 9999 variance change

# In[255]:


2.330686e+07 - 1.469751e+06


# -1 variance change

# In[256]:


8.922991e+05 - 1.469751e+06


# ## Findings and conclusions

# Observing data after handling missing data

# In[257]:


df_dropped.describe()


# In[258]:


df_dropped.var()


# Dist. of Longitude before and after handling data

# In[259]:


sns.kdeplot(df_dropped.longitude)
sns.kdeplot(df.longitude)
plt.title('Dist. of Longitude')
plt.show()


# Dist. of Latitude before and after handling data

# In[260]:


sns.kdeplot(df_dropped.latitude)
sns.kdeplot(df.latitude)
plt.title('Dist. of Latitude')
plt.show()


# Dist. of location_easting_osgr before and after handling data

# In[261]:


sns.kdeplot(df_dropped.location_easting_osgr)
sns.kdeplot(df.location_easting_osgr)
plt.title('Dist. of location_easting_osgr')
plt.show()


# Dist. of location_northing_osgr before and after handling data
# 

# In[262]:


sns.kdeplot(df_dropped.location_northing_osgr)
sns.kdeplot(df.location_northing_osgr)
plt.title('Dist. of location_northing_osgr')
plt.show()


# Number of rows before handling data

# In[263]:


len(df)


# Number of rows after handling data

# In[264]:


len(df_dropped)


# Distributions aren't affected with dropping small % of data

# ## Observing outliers

# In[265]:


df_dropped.describe()


# Function to draw boxplot to observe outliers

# In[266]:


def draw_box_plot(df):
    plt.boxplot(df)
    plt.show()
    sns.kdeplot(df)
    plt.show()


# Outliers of number_of_vehicles

# In[267]:


draw_box_plot(df_dropped.number_of_vehicles)


# Outliers of number_of_casualties

# In[268]:


draw_box_plot(df_dropped.number_of_casualties)


# In[269]:


draw_box_plot(df_dropped.second_road_number)


# In[270]:


df_dropped.number_of_vehicles.skew(),df_dropped.number_of_casualties.skew(),df_dropped.second_road_number.skew()


# ## Handling outliers

# function to handle outliers using z-score

# In[271]:


def zscore(df_org,df, n):
    z_number_of_vehicles = np.abs(stats.zscore(df))
    filtered_entries_number_of_vehicles = z_number_of_vehicles < n
    df_zscore_filter = df_org.copy()
    return df_zscore_filter[filtered_entries_number_of_vehicles]


# Using z-score to handle outliers

# In[272]:


df_zscore_filter = zscore(df_dropped, df_dropped.number_of_vehicles,3)
df_zscore_filter.describe()


# In[273]:


df_dropped.describe()


# In[274]:


print(df_dropped.shape)
print(df_zscore_filter.shape)


# Outliers of number_of_vehicles are handled

# In[275]:


df_zscore_filter = zscore(df_zscore_filter,df_zscore_filter.number_of_casualties,3)
df_zscore_filter.describe()


# In[276]:


df_dropped.describe()


# In[277]:


print(df_dropped.shape)
print(df_zscore_filter.shape)


# Outliers of number_of_casualties are handled

# In[278]:


df_zscore_filter = zscore(df_zscore_filter,df_zscore_filter.second_road_number,2)
df_zscore_filter.describe()


# In[279]:


df_dropped.describe()


# In[280]:


print(df_dropped.shape)
print(df_zscore_filter.shape)


# Outliers of second_road_number aren't fully handled

# ## Findings and conclusions

# Observe dataframe after outliers are handled

# In[281]:


df_zscore_filter.describe()


# In[282]:


draw_box_plot(df_zscore_filter.number_of_vehicles)


# In[283]:


draw_box_plot(df_zscore_filter.number_of_casualties)


# In[284]:


draw_box_plot(df_zscore_filter.second_road_number)


# In[285]:


df_dropped.describe()


# In[286]:


df_zscore_filter.describe()


# Outliers are handled except for second_road_number which still contains outliers

# # 4 - Data transformation

# ## 4.1 - Discretization

# In[287]:


df_zscore_filter.describe()


# Qcut method to discretize

# In[288]:


def discrete(string,n):
    number_of_vehicles_disccretised, intervals = pd.qcut(
    df_zscore_filter[string], n, labels=None, retbins=True, precision=3, duplicates='drop')
    df_disccretised = pd.concat([number_of_vehicles_disccretised, df_zscore_filter[string]], axis=1)
    df_disccretised.columns = [f'{string}_disc',string]
    return df_disccretised


# Convert date to datetime format

# In[289]:


df_zscore_filter.date = pd.to_datetime(df_zscore_filter.date)
df_zscore_filter.date


# In[290]:


df_disccretised = discrete('date',52)
df_disccretised


# Saving disccritised week number in a new column

# In[291]:


df_zscore_filter["Week_number"] = df_disccretised.date_disc
df_zscore_filter


# ## 4.11 - Findings and conclusions

# The dates are disccritised in to range of weeks which can be used later to get number of week

# ## 4.2 - Encoding

# In[292]:


df_zscore_filter


# Use label encoding to encode categorical attributes   
# Encode categorical data as numbers without expanding dataframe size which is faster to process.

# Function to encode categorical attributes in a given dataset as numbers

# In[293]:


def number_encode_features(df):
    result = df.copy() # take a copy of the dataframe
    result['Week_number'] = preprocessing.LabelEncoder().fit_transform(result['Week_number'])
#     return result
    for column in result.columns:
        if result.dtypes[column] == object: # if attribute is categorical
            # Apply LabelEncoder method to attribute
            # fit will infer the number of numerical values needed by counting the number of categories
            # then transform will replace each category with its numerical counterpart
            if column != 'accident_reference':
                result[column] = preprocessing.LabelEncoder().fit_transform(result[column].astype(str))
    return result


# In[294]:


# Apply function defined above to income dataset
encoded_data = number_encode_features(df_zscore_filter)

# Display last 5 records in transformed dataset to verify numerical transformation
encoded_data


# Keys for encoded and arbitrary data

# In[295]:


df_key = pd.DataFrame({})
i= range(len(df_zscore_filter.columns))
for column in df_zscore_filter.columns:
        if df_zscore_filter.dtypes[column] == object:
            if column != 'accident_reference':
                df_key[column] = pd.Series(df_zscore_filter[column].unique().tolist())
                df_key[f'{column}_key'] = pd.Series(encoded_data[column].unique().tolist())
                
df_key['Week_number'] = pd.Series(df_zscore_filter['Week_number'].unique().tolist())
df_key['Week_number_key'] = pd.Series(encoded_data['Week_number'].unique().tolist())

df_key['second_road_number'] = pd.Series(df_zscore_filter.second_road_number.unique().tolist())
df_key['second_road_number_key'] = pd.Series(encoded_data.second_road_number.unique().tolist())

for i in range(len(df_key['second_road_number'])):
    if df_key['second_road_number'][i] == -1:
        df_key['second_road_number_key'][i] = "Arbitrary value"
    elif df_key['second_road_number'][i] == 0:
        df_key['second_road_number_key'][i] = 'first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero '
    else:
        df_key['second_road_number'][i] = ""
        df_key['second_road_number_key'][i] = ""
df_key = df_key.replace(np.nan, '', regex=True)
df_key


# ## 4.22 - Findings and conlcusions

# Data is encoded but there is a new problem since it uses number sequencing. The problem using the number is that they introduce relation/comparison between them. It gives weight to each column.

# In[296]:


encoded_data.describe()


# In[297]:


encoded_data.var()


# ## 4.3 - Normalisation 

# In[298]:


def normalize(df):
    # Get the index of all positive pledges (Box-Cox only takes postive values)
    index_of_positive_pledges = df > 0

    # get only positive pledges (using their indexes)
    positive_pledges = df.loc[index_of_positive_pledges]

    # normalize the pledges (w/ Box-Cox)
    normalized_pledges = stats.boxcox(positive_pledges)[0]
    # plot both together to compare
    fig, ax=plt.subplots(1,2)
    sns.distplot(positive_pledges, ax=ax[0])
    ax[0].set_title("Original Data")
    sns.distplot(normalized_pledges, ax=ax[1])
    ax[1].set_title("Normalized data")


# Longitude is skewed. Therefore, we need to normalize it.

# In[299]:


normalize(encoded_data.longitude)


# Latitude is skewed. Therefore, we need to normalize it.

# In[300]:


normalize(encoded_data.latitude)


# second_road_number is skewed. Therefore, we need to normalize it.

# In[301]:


normalize(encoded_data.second_road_number)


# ## 4.31 - Findings and conclusions

# Graphs became more normally dist. after normalization.

# ## 4.4 - Adding more columns

# In[302]:


encoded_data["Is_weekend"] = encoded_data.date.dt.dayofweek > 4
encoded_data["Is_weekend"]


# In[303]:


encoded_data["Is_friday"] = encoded_data.date.dt.dayofweek == 4
encoded_data["Is_friday"]


# In[304]:


encoded_data["Is_junction_control_automated"] = encoded_data.junction_control == 'Auto traffic signal'
encoded_data["Is_junction_control_automated"]


# ## 4.41 - Findings and concluisons

# More data is observed which helps in analyzing it

# ## 4.5 - Csv file for lookup

# In[206]:


# df_key.to_csv('E://University/Semester 9/Data Engineering (NETW 908)/Project/Accidents_UK_2019_df_encode_key.csv',index=False)


# ## 5- Exporting the dataframe to a csv file or parquet

# Parquet file doesn't work on dataframes containing strings, so it willn't work on df_key

# In[207]:


# encoded_data.to_parquet('E://University/Semester 9/Data Engineering (NETW 908)/Project/Accidents_UK_2019_cleaned_and_encoded.parquet', engine='fastparquet')


# In[208]:


# encoded_data.to_csv('E://University/Semester 9/Data Engineering (NETW 908)/Project/Accidents_UK_2019_cleaned_and_encoded.csv',index=True)


# In[209]:


# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily

def getWeather(day, month, long, lat):
    start = datetime(2019, month, day)
    end = datetime(2019, month, day)    

    vancouver = Point(lat, long, 70)
    data = Daily(vancouver, start, end)
    data = data.fetch()
    return data


# In[380]:


encoded_data.head()


# In[417]:


# result = pd.concat([df1, df4], ignore_index=True, sort=False)
details = pd.DataFrame()
details_df = pd.DataFrame()
details_df['longitude'] = encoded_data.longitude
details_df['latitude'] = encoded_data.latitude
details_df['date'] = encoded_data.date

details_df['tavg'] = ''
details_df['tmin'] = ''
details_df['tmax'] = ''
details_df['wdir'] = ''
details_df['wspd'] = ''
for i in range(len(encoded_data)):
    details = getWeather(encoded_data.date[i].day,encoded_data.date[i].month,encoded_data.longitude[i],encoded_data.latitude[i])
    if(len(details['tavg']) >= 1):
        details_df.tavg[i] = details['tavg'][0]
    if(len(details['tmin']) >= 1):
        details_df.tmin[i] = details['tmin'][0]
    if(len(details['tmax']) >= 1):
        details_df.tmax[i] = details['tmax'][0]
    if(len(details['wdir']) >= 1):
        details_df.wdir[i] = details['wdir'][0]
    if(len(details['wspd']) >= 1):
        details_df.wspd[i] = details['wspd'][0]

details_df


# In[ ]:


detail_copy = details_df.copy()
data_copy = encoded_data.copy()
len(detail_copy)


# In[ ]:


len(data_copy)


# In[ ]:


detail_copy = details_df.copy()
data_copy = encoded_data.copy()
data_copy['temp_avg'] = ''
data_copy['temp_min'] = ''
data_copy['temp_max'] = ''
data_copy['wind_dir'] = ''
data_copy['wind_speed'] = ''

# for i in range(len(detail_copy)):
for i in range(len(detail_copy)):
    data_copy['temp_avg'].iloc[i] = detail_copy['tavg'].iloc[i]
    data_copy['temp_min'].iloc[i] = detail_copy['tmin'].iloc[i]
    data_copy['temp_max'].iloc[i] = detail_copy['tmax'].iloc[i]
    data_copy['wind_dir'].iloc[i] = detail_copy['wdir'].iloc[i]
    data_copy['wind_speed'].iloc[i] = detail_copy['wspd'].iloc[i]

data_copy

