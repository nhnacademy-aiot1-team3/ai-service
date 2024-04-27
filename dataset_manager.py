from influxdb_client import InfluxDBClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 받아오기 및 데이터 전처리 담당하는 클래스
class DatasetManager:
    
    def __init__(self, db_url, token, org, bucket):
        self.client = InfluxDBClient(url=db_url, token=token, org=org, timeout=30_000)
        self.bucket = bucket
    
    # 온도 데이터 가져오기
    def query_temperature(self, branch):
        query_api = self.client.query_api()
        
        query = f'from(bucket: "{self.bucket}")\
            |> range(start: -2d)\
            |> filter(fn: (r) => r.branch == "{branch}")\
            |> filter(fn: (r) => r.endpoint == "temperature")\
            |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")\
            |> group(columns: ["place", "endpoint", "branch"])'

        print("Query: ",query)

        result_df = query_api.query_data_frame(query=query)
        
        return result_df

    # 도어 데이터 가져오기
    def query_di(self, branch):
        query_api = self.client.query_api()

        query = f'from(bucket: "{self.bucket}")\
            |> range(start: -4d)\
            |> filter(fn: (r) => r.branch == "{branch}")\
            |> filter(fn: (r) => r.endpoint == "di")\
            |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")\
            |> group(columns: ["place", "endpoint", "branch"])'

        result_df = query_api.query_data_frame(query=query)
        print(result_df)
        return result_df
    
    # DB connection 닫기
    def close_connection(self):
        self.client.close()

    # 데이터 전처리
    @staticmethod
    def data_preprocessing(raw_dataset):

        # dataframe 중요한 column 가져오기
        df = raw_dataset[['_time', 'branch', 'device', 'endpoint', 'place', 'site', 'sensor_value']].copy()

        # 기존 column 이름 변경하기
        df.rename(columns={'_time':'date'}, inplace=True)
        df.rename(columns={'branch':'organization'}, inplace=True)
        df.rename(columns={'device':'device_id'}, inplace=True)
        df.rename(columns={'endpoint':'sensor_type'}, inplace=True)

        # 시간별로 column 만들기
        df['year'] = df['date'].apply(lambda x : x.year)
        df['month'] = df['date'].apply(lambda x : x.month)
        df['day'] = df['date'].apply(lambda x : x.day)
        df['hour'] = df['date'].apply(lambda x : x.hour)

        # 시별로 aggregate
        agg_df = df.groupby(['year', 'month', 'day', 'hour', 'place', 'site']).agg({ \
            'date': 'first', \
            'organization': 'first', \
            'sensor_type': 'first', \
            'sensor_value': 'mean' \
        }).reset_index()

        agg_df['date'] = agg_df['date'].dt.strftime('%Y-%m-%d %H:00:00')

        # 결측지를 전체의 평균값으로 채우기

        DatasetManager.print_df_info(df)
        DatasetManager.print_df_info(agg_df)
        DatasetManager.draw_table(df)
        
        return df
    
    # dataset의 정보 출력
    @staticmethod
    def print_df_info(dataset):
        print('='*30)
        print(dataset.info())
        print('='*30)
        print("Column Names: ",dataset.columns)
        print('='*30)
        print("Data types:\n",dataset.dtypes)
        print('='*30)
        print("Summary statistics:\n",dataset.describe())
        print('='*30)
        print(dataset.to_string())

    # dataset을 그래프로 표현
    @staticmethod
    def draw_table(dataset):
        plt.figure(figsize=(30,7))
        sns.lineplot(data=dataset, x='date', y='sensor_value', hue='place', palette='Set1')
        plt.xlabel('Time')
        plt.ylabel('Sensor Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        