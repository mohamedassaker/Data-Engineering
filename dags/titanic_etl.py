from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import pandas as pd
import numpy as np
# For Label Encoding
from sklearn import preprocessing
import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn import preprocessing

dataset = 'titanic.csv'

def milestone1(df):
    df = pd.read_csv(df, index_col='accident_index', low_memory=0)

    df_new = df.copy()
    age_counts = df_new.groupby(['number_of_vehicles'])['number_of_casualties'].sum() #returns a series of each age and the corresponding number of individuals who survived

    class_series = df['number_of_casualties'].value_counts()
    class_series

    df_weekdays = df.copy()
    weekday_series = df_weekdays.day_of_week.value_counts()
    weekday_series

    class_series = df['local_authority_district'].value_counts().head()
    class_series

    class_series = df['date'].value_counts().head()
    class_series

    df.accident_reference.value_counts()

    df = df.drop('accident_reference', axis=1)
    df.head()

    sum_null = df.isnull().sum()
    perc_null = df.isnull().sum() / len(df)
    perc_null_mean = df.isnull().mean()*100

    sum_null

    perc_null_mean
    len(df)

    df.duplicated().sum()

    df = df.drop_duplicates()

    df.duplicated().sum()

    df_dropped = df.dropna(axis='index', subset=['location_easting_osgr','location_northing_osgr','longitude','latitude','road_type','weather_conditions'])

    sum_null = df_dropped.isnull().sum()
    perc_null = df_dropped.isnull().sum() / len(df)
    perc_null_mean = df_dropped.isnull().mean()*100
    perc_null_mean

    df_dropped = df_dropped.replace('first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ', int(0))
    df_dropped.second_road_number = df_dropped.second_road_number.astype(float)
    df_dropped = df_dropped.replace(r'',np.NaN)

    df_dropped = df_dropped.replace(np.NaN, -1)
    df_dropped.second_road_number.unique()

    df_dropped.number_of_vehicles.skew(),df_dropped.number_of_casualties.skew(),df_dropped.second_road_number.skew()

    def zscore(df_org,df, n):
        z_number_of_vehicles = np.abs(stats.zscore(df))
        filtered_entries_number_of_vehicles = z_number_of_vehicles < n
        df_zscore_filter = df_org.copy()
        return df_zscore_filter[filtered_entries_number_of_vehicles]

    df_zscore_filter = zscore(df_dropped, df_dropped.number_of_vehicles,3)

    df_zscore_filter = zscore(df_zscore_filter,df_zscore_filter.number_of_casualties,3)

    df_zscore_filter = zscore(df_zscore_filter,df_zscore_filter.second_road_number,2)

    def discrete(string,n):
        number_of_vehicles_disccretised, intervals = pd.qcut(
        df_zscore_filter[string], n, labels=None, retbins=True, precision=3, duplicates='drop')
        df_disccretised = pd.concat([number_of_vehicles_disccretised, df_zscore_filter[string]], axis=1)
        df_disccretised.columns = [f'{string}_disc',string]
        return df_disccretised

    df_zscore_filter.date = pd.to_datetime(df_zscore_filter.date)
    df_zscore_filter.date

    df_disccretised = discrete('date',52)
    df_disccretised

    df_zscore_filter["Week_number"] = df_disccretised.date_disc
    df_zscore_filter

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

    encoded_data = number_encode_features(df_zscore_filter)

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
    df_key.to_csv('/opt/airflow/data/output/Accidents_UK_2019_key.csv',index=False)

    encoded_data["Is_weekend"] = encoded_data.date.dt.dayofweek > 4
    encoded_data["Is_weekend"]

    encoded_data["Is_friday"] = encoded_data.date.dt.dayofweek == 4
    encoded_data["Is_friday"]

    encoded_data["Is_junction_control_automated"] = encoded_data.junction_control == 'Auto traffic signal'
    encoded_data["Is_junction_control_automated"]

    encoded_data.to_csv('/opt/airflow/data/output/Accidents_UK_2019_cleaned_data.csv',index=False)

    # return encoded_data

def milestone2(dfdf, df):

    df = pd.read_csv(df, index_col='accident_index', low_memory=0)

    df_new = df.copy()
    age_counts = df_new.groupby(['number_of_vehicles'])['number_of_casualties'].sum() #returns a series of each age and the corresponding number of individuals who survived

    class_series = df['number_of_casualties'].value_counts()
    class_series

    df_weekdays = df.copy()
    weekday_series = df_weekdays.day_of_week.value_counts()
    weekday_series

    class_series = df['local_authority_district'].value_counts().head()
    class_series

    class_series = df['date'].value_counts().head()
    class_series

    df.accident_reference.value_counts()

    df = df.drop('accident_reference', axis=1)
    df.head()

    sum_null = df.isnull().sum()
    perc_null = df.isnull().sum() / len(df)
    perc_null_mean = df.isnull().mean()*100

    sum_null

    perc_null_mean
    len(df)

    df.duplicated().sum()

    df = df.drop_duplicates()

    df.duplicated().sum()

    df_dropped = df.dropna(axis='index', subset=['location_easting_osgr','location_northing_osgr','longitude','latitude','road_type','weather_conditions'])

    sum_null = df_dropped.isnull().sum()
    perc_null = df_dropped.isnull().sum() / len(df)
    perc_null_mean = df_dropped.isnull().mean()*100
    perc_null_mean

    df_dropped = df_dropped.replace('first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ', int(0))
    df_dropped.second_road_number = df_dropped.second_road_number.astype(float)
    df_dropped = df_dropped.replace(r'',np.NaN)

    df_dropped = df_dropped.replace(np.NaN, -1)
    df_dropped.second_road_number.unique()

    df_dropped.number_of_vehicles.skew(),df_dropped.number_of_casualties.skew(),df_dropped.second_road_number.skew()

    def zscore(df_org,df, n):
        z_number_of_vehicles = np.abs(stats.zscore(df))
        filtered_entries_number_of_vehicles = z_number_of_vehicles < n
        df_zscore_filter = df_org.copy()
        return df_zscore_filter[filtered_entries_number_of_vehicles]

    df_zscore_filter = zscore(df_dropped, df_dropped.number_of_vehicles,3)

    df_zscore_filter = zscore(df_zscore_filter,df_zscore_filter.number_of_casualties,3)

    df_zscore_filter = zscore(df_zscore_filter,df_zscore_filter.second_road_number,2)

    def discrete(string,n):
        number_of_vehicles_disccretised, intervals = pd.qcut(
        df_zscore_filter[string], n, labels=None, retbins=True, precision=3, duplicates='drop')
        df_disccretised = pd.concat([number_of_vehicles_disccretised, df_zscore_filter[string]], axis=1)
        df_disccretised.columns = [f'{string}_disc',string]
        return df_disccretised

    df_zscore_filter.date = pd.to_datetime(df_zscore_filter.date)
    df_zscore_filter.date

    df_disccretised = discrete('date',52)
    df_disccretised

    df_zscore_filter["Week_number"] = df_disccretised.date_disc
    df_zscore_filter

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

    encoded_data = number_encode_features(df_zscore_filter)

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

    encoded_data["Is_weekend"] = encoded_data.date.dt.dayofweek > 4
    encoded_data["Is_weekend"]

    encoded_data["Is_friday"] = encoded_data.date.dt.dayofweek == 4
    encoded_data["Is_friday"]

    encoded_data["Is_junction_control_automated"] = encoded_data.junction_control == 'Auto traffic signal'
    encoded_data["Is_junction_control_automated"]

    dfdf = pd.read_csv(dfdf, index_col=0, low_memory=0)

    dfdf

    df_test = dfdf.copy()
    df_test1 = encoded_data
    df_test.columns = ['temp_avg', 'temp_min', 'temp_max', 'wind_dir', 'wind_speed', 'local_authority_district', 'date']
    df_test.date = pd.to_datetime(df_test.date)
    df_test.head()

    result = df_test1.reset_index().merge(df_test, how="left", on=["date", "local_authority_district"]).set_index('accident_index')
    dfdf = result
    dfdf

    dfdf.temp_avg = dfdf.temp_avg.replace(r'',np.NaN)
    dfdf.temp_min = dfdf.temp_min.replace(r'',np.NaN)
    dfdf.temp_max = dfdf.temp_max.replace(r'',np.NaN)
    dfdf.wind_dir = dfdf.wind_dir.replace(r'',np.NaN)
    dfdf.wind_speed = dfdf.wind_speed.replace(r'',np.NaN)

    def end_of_tail_imputation(data, dfdf):
        eod_value = dfdf[data] .mean() +3*dfdf[data].std()
        dfdf[data].fillna(value=eod_value, inplace = True)


    def normalize(df):
            # Get the index of all positive pledges (Box-Cox only takes postive values)
            index_of_positive_pledges = df > 0

            # get only positive pledges (using their indexes)
            positive_pledges = df.loc[index_of_positive_pledges]

            # normalize the pledges (w/ Box-Cox)
            normalized_pledges = stats.boxcox(positive_pledges)[0]

    end_of_tail_imputation('temp_avg', dfdf)
    end_of_tail_imputation('temp_min', dfdf)
    end_of_tail_imputation('temp_max', dfdf)
    end_of_tail_imputation('wind_dir', dfdf)
    end_of_tail_imputation('wind_speed', dfdf)

    normalize(dfdf.temp_avg)
    normalize(dfdf.temp_max)
    normalize(dfdf.wind_dir)
    normalize(dfdf.wind_speed)
    normalize(dfdf.temp_min)

    dfdf['wind_speed'].value_counts()
    dfdf['temp_avg'].value_counts()

    age_counts = dfdf.temp_avg.value_counts()
    age_counts = dfdf.wind_speed.value_counts()

    dfdf.to_csv('/opt/airflow/data/output/Accidents_UK_2019_new_data_added.csv',index=False)

    # return dfdf

def extract_clean(filename):
    df = pd.read_csv(filename)
    df = clean_missing(df)
    df.to_csv('/opt/airflow/data/titanic_clean.csv',index=False)
    print('loaded after cleaning succesfully')

def encode_load(filename):
    df = pd.read_csv(filename)
    df = encoding(df)
    try:
        df.to_csv('/opt/airflow/data/titanic_transformed.csv',index=False, mode='x')
        print('loaded after cleaning succesfully')
    except FileExistsError:
        print('file already exists')

def clean_missing(df):
    df = impute_mean(df,'Age')
    df = impute_arbitrary(df,'Cabin','Missing')
    df = cca(df,'Embarked')
    return df

def impute_arbitrary(df,col,arbitrary_value):
    df[col] = df[col].fillna(arbitrary_value)
    return df

def impute_mean(df,col):
    df[col] = df[col].fillna(df[col].mean())
    return df

def impute_median(df,col):
    df[col] = df[col].fillna(df[col].mean())
    return df

def cca(df,col):
    return df.dropna(subset=[col])

def encoding(df):
    df = one_hot_encoding(df,'Embarked')
    df = label_encoding(df,'Cabin')
    return df

def one_hot_encoding(df,col):
    to_encode = df[[col]]
    encoded = pd.get_dummies(to_encode)
    df = pd.concat([df,encoded],axis=1)
    return df

def label_encoding(df,col):
    df[col] = preprocessing.LabelEncoder().fit_transform(df[col])
    return df

def load_to_csv(df,filename):
    df.to_csv(filename,index=False)

def create_dashboard(filename):
    df = pd.read_csv(filename)
    app = dash.Dash()
    app.layout = html.Div(
    children=[
        html.H1(children="2019 Accidents dataset",),
        html.P(
            children="Number of casualties vs Count of casualties 2019 Accidents dataset",
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": df['number_of_casualties'].value_counts().index,
                        "y": df['number_of_casualties'].value_counts(),
                        "type": "lines",
                    },
                ],
                "layout": {"title": "Number of casualties vs Count of casualties"},
            },
        )
    ]
)
    app.run_server(host='0.0.0.0')
    print('dashboard is successful and running on port 8001')

def load_to_postgres(filename, filename1): 
    df = pd.read_csv(filename)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/accidents_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'UK_Accidents_2019',con = engine,if_exists='replace')

    df = pd.read_csv(filename1)
    engine = create_engine('postgresql://root:root@pgdatabase:5432/accidents_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'lookup_table',con = engine,if_exists='replace')

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    '2019_Accidents_UK_etl_pipeline',
    default_args=default_args,
    description='2019_Accidents_UK etl pipeline',
)
with DAG(
    dag_id = '2019_Accidents_UK_etl_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['2019_Accidents_UK-pipeline'],
)as dag:
    extract_clean_task= PythonOperator(
        task_id = 'milestone_1',
        python_callable = milestone1,
        op_kwargs={
            "df": '/opt/airflow/data/2019_Accidents_UK.csv'
        },
    )
    encoding_load_task= PythonOperator(
        task_id = 'milestone_2',
        python_callable = milestone2,
        op_kwargs={
            "df": '/opt/airflow/data/2019_Accidents_UK.csv',
            "dfdf": "/opt/airflow/data/Accidents_UK_2019_new_columns.csv"
        },
    )
    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "filename": "/opt/airflow/data/output/Accidents_UK_2019_new_data_added.csv",
            "filename1": "/opt/airflow/data/output/Accidents_UK_2019_key.csv"
        },
    )
    create_dashboard_task= PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = create_dashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/output/Accidents_UK_2019_new_data_added.csv"
        },
    )
    


    extract_clean_task >> encoding_load_task >> load_to_postgres_task >> create_dashboard_task
