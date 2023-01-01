#!/usr/bin/env python
# coding: utf-8

# # Milestone 2 - Identify an additional feature 

# In[94]:


file = 'E://University/Semester 9/Data Engineering (NETW 908)/Project/Accidents_UK_2019_new_columns.csv'
dfdf = pd.read_csv(file, index_col=0, low_memory=0)


# In[95]:


dfdf


# In[96]:


df_test = dfdf.copy()
df_test1 = encoded_data.copy()
df_test.columns = ['temp_avg', 'temp_min', 'temp_max', 'wind_dir', 'wind_speed', 'local_authority_district', 'date']
df_test.date = pd.to_datetime(df_test.date)
df_test.head()


# In[97]:


result = df_test1.reset_index().merge(df_test, how="left", on=["date", "local_authority_district"]).set_index('accident_index')
dfdf = result
dfdf


# In[98]:


dfdf.temp_avg = dfdf.temp_avg.replace(r'',np.NaN)
dfdf.temp_min = dfdf.temp_min.replace(r'',np.NaN)
dfdf.temp_max = dfdf.temp_max.replace(r'',np.NaN)
dfdf.wind_dir = dfdf.wind_dir.replace(r'',np.NaN)
dfdf.wind_speed = dfdf.wind_speed.replace(r'',np.NaN)

sum_null = dfdf.isnull().sum()
perc_null = dfdf.isnull().sum() / len(df)
perc_null_mean = dfdf.isnull().mean()*100
perc_null_mean.tail(7)


# Data is missing at random & it is numerical

# Use End of tail impuatation to handle missing data

# In[99]:


import seaborn

def end_of_tail_imputation(data, dfdf):
    eod_value = dfdf[data] .mean() +3*dfdf[data].std()
    dfdf[data].fillna(value=eod_value, inplace = True)


# In[100]:


end_of_tail_imputation('temp_avg', dfdf)
end_of_tail_imputation('temp_min', dfdf)
end_of_tail_imputation('temp_max', dfdf)
end_of_tail_imputation('wind_dir', dfdf)
end_of_tail_imputation('wind_speed', dfdf)


# In[101]:


# Calculating sum of null entries and percentage of null entries
sum_null = dfdf.isnull().sum()
perc_null = dfdf.isnull().sum() / len(df)
perc_null_mean = dfdf.isnull().mean()*100
perc_null_mean.tail(7)


# Missing Data are handled

# Scaling data

# In[102]:


normalize(dfdf.temp_avg)


# In[103]:


normalize(dfdf.temp_max)


# In[104]:


normalize(dfdf.wind_dir)


# In[105]:


normalize(dfdf.wind_speed)


# In[106]:


normalize(dfdf.temp_min)


# Ask/Analyse 2 questions related to the feature extracted and visualise your findings

# What are top 5 wind speeds that might caused accidents?

# In[107]:


dfdf['wind_speed'].value_counts()


# In[108]:


def draw_histogram(dfdf, data):
    class_series = dfdf[data].value_counts().head().astype(str)
    class_series
    bar_plot = plt.bar(class_series.index, class_series)
    plt.locator_params(integer=True)
    plt.xlabel(data)
    plt.ylabel(f'Count of {data}')
    plt.show()


# In[109]:


draw_histogram(dfdf, 'wind_speed')


# What are top 5 avg. temp. that might have affected driving?

# In[110]:


dfdf['temp_avg'].value_counts()


# In[111]:


draw_histogram(dfdf, 'temp_avg')


# What is the relation between average of temperature and accidents count?

# In[112]:


age_counts = dfdf.temp_avg.value_counts()
plt.xlabel('Temp')
plt.ylabel('Accidents Count')
plt.title('Accidents Count VS. Temp')
plt.scatter(age_counts.index,age_counts)
plt.show()


# What is the relation between windspeed and accidents count?

# In[113]:


age_counts = dfdf.wind_speed.value_counts()
plt.xlabel('wind_speed')
plt.ylabel('Accidents Count')
plt.title('Accidents Count VS. wind_speed')
plt.scatter(age_counts.index,age_counts)
plt.show()

