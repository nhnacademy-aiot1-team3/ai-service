from influxdb_client import InfluxDBClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 받아오기 및 데이터 전처리 담당하는 클래스
class DatasetManager:
    
    def __init__(self, db_url, token, org, bucket, sensor_type):
        self.client = InfluxDBClient(url=db_url, token=token, org=org, timeout=30_000)
        self.bucket = bucket
        self.sensor_type = sensor_type
    
    # 온도 데이터 가져오기
    def query_sensor_data(self, branch):
        query_api = self.client.query_api()
        
        query = f'from(bucket: "{self.bucket}")\
            |> range(start: -4d)\
            |> filter(fn: (r) => r.branch == "{branch}")\
            |> filter(fn: (r) => r["endpoint"] == "{self.sensor_type}")\
            |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)\
            |> yield(name: "sensor_value")'
        
        print("Query: ",query)
        result_df = query_api.query_data_frame(query=query)
        return result_df
    
    # 전량 데이터 가져오기
    def query_energy(self, branch):
        query_api = self.client.query_api()

        query = f'from(bucket: "{self.bucket}")\
                |> range(start: -4d)\
                |> filter(fn: (r) => r["branch"] == "{branch}")\
                |> filter(fn: (r) => r["endpoint"] == "{self.sensor_type}")\
                |> filter(fn: (r) => r["phase"] == "total")\
                |> filter(fn: (r) => r["description"] == "w")\
                |> group(columns: ["site"])\
                |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)\
                |> yield(name: "sensor_value")'
        
        print("Query: ", query)
        result_df = query_api.query_data_frame(query=query)
        return result_df

    # DB connection 닫기
    def close_connection(self):
        self.client.close()

    # 데이터 전처리
    def data_preprocessing(self, raw_dataset):
        # dataframe 중요한 column 가져오기
        if 'place' in raw_dataset.columns:
            df = raw_dataset[['_time', 'place', '_value']].copy()
            
            # 냉장고 지우기
            if '냉장고' in df['place'].values:
                df = df[df['place'] != '냉장고']
                df = df.drop(columns=['place'])
        else:
            df = raw_dataset[['_time', '_value']].copy()
        
        # 기존 column 이름 변경하기
        df.rename(columns={'_time':'date'}, inplace=True)
        df.rename(columns={'_value': self.sensor_type}, inplace=True)

        # date asia 시간대로 바꾸기
        df['date'] = pd.to_datetime(df['date']).dt.tz_convert('Asia/Seoul').dt.tz_localize(None)

        if self.sensor_type=='electrical_energy':
            df = self.second2minute(df)
            df['date'] = pd.to_datetime(df['date'])

        # date를 index로 set하기
        df = df.set_index('date')

        # 결측지를 전체의 평균값으로 채우기
        df[self.sensor_type] = df[self.sensor_type].interpolate()

        self.print_df_info(df)
        self.draw_table(df)
        
        return df
    
    # dataset의 정보 출력
    def print_df_info(self, dataset):
        pd.set_option('display.max_columns', None)  # Display all columns
        pd.set_option('display.max_rows', None)     # Display all rows
        pd.set_option('display.width', None)        # Set the display width to None

        print('='*30)
        print(dataset.info())
        print('='*30)
        print("Column Names: ",dataset.columns)
        print('='*30)
        print("Data types:\n",dataset.dtypes)
        print('='*30)
        print("Summary statistics:\n",dataset.describe())
        print('='*30)
        print(dataset.isnull().sum())
        print('='*30)
        print(dataset.head(30))
        print('='*30)
        print('Index of Dataframe: ',dataset.index.name)
        print('='*30)
        print(dataset.corr()[self.sensor_type].sort_values(ascending=False))

    # dataset을 그래프로 표현
    def draw_table(self, dataset):
        plt.figure(figsize=(16,8))
        sns.lineplot(data=dataset, palette='Set1')
        plt.xlabel('Time')
        plt.ylabel(self.sensor_type)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 초별 -> 분별 변환
    def second2minute(self, df):
        # 시간별로 column 만들기
        df['year'] = df['date'].apply(lambda x : x.year)
        df['month'] = df['date'].apply(lambda x : x.month)
        df['day'] = df['date'].apply(lambda x : x.day)
        df['hour'] = df['date'].apply(lambda x : x.hour)
        df['minute'] = df['date'].apply(lambda x : x.minute)
        df['second'] = df['date'].apply(lambda x : x.second)

        # 시별로 aggregate
        agg_df = df.groupby(['year', 'month', 'day', 'hour', 'minute']).agg({ \
            'date': 'first', \
            self.sensor_type : 'mean' \
        }).reset_index()

        agg_df['date'] = agg_df['date'].dt.strftime('%Y-%m-%d %H:%M:00')
        return agg_df
    
    # 분별 -> 시별 변환
    def minute2hour(self, df):
        # 시간별로 column 만들기
        df['year'] = df['date'].apply(lambda x : x.year)
        df['month'] = df['date'].apply(lambda x : x.month)
        df['day'] = df['date'].apply(lambda x : x.day)
        df['hour'] = df['date'].apply(lambda x : x.hour)

        # 시별로 aggregate
        agg_df = df.groupby(['year', 'month', 'day', 'hour']).agg({ \
            'date': 'first', \
            self.sensor_type : 'mean' \
        }).reset_index()

        agg_df['date'] = agg_df['date'].dt.strftime('%Y-%m-%d %H:00:00')
        return agg_df