from influxdb_client import InfluxDBClient
import pandas as pd

# 데이터셋 받아오기 및 데이터 전처리 담당하는 클래스
class DatasetManager:
    
    def __init__(self, db_url, token, org, bucket):
        self.client = InfluxDBClient(url=db_url, token=token, org=org, timeout=30_000)
        self.bucket = bucket
    
    # _time, _value, branch, endpoint(hmmm), place, site
    def query_temperature(self, branch):
        query_api = self.client.query_api()
        
        query = f'from(bucket: "{self.bucket}")\
            |> range(start: -1d)\
            |> filter(fn: (r) => r.branch == "{branch}")\
            |> filter(fn: (r) => r.endpoint == "temperature")\
            |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")\
            |> group(columns: ["place", "endpoint", "branch"])'

        # query = f'SELECT "_time", "_value", "branch", "endpoint", "place", "site" FROM "{self.bucket}" WHERE "branch" = "{branch}" AND "endpoint" = "temperature"'
        print(query)

        result_df = query_api.query_data_frame(query=query)
        # result_df = self.client.query(query)
        print('======================')
        print(result_df)
        print('======================')
        print("Result Type: ",type(result_df))
        print('======================')
        print("Column Names: ",result_df.columns)
        print('======================')
        print("Data types:\n",result_df.dtypes)
        print('======================')
        print("Summary statistics:\n",result_df.describe())
        print(result_df.to_string())
        return result_df

    def query_di(self, branch):
        query_api = self.client.query_api()

        query = f'from(bucket: "{self.bucket}") |> range(start: -1d) |> filter(fn: (r) => r.endpoint == "di")'

        result_df = query_api.query_data_frame(query=query)
        print(result_df)
        return result_df
    
    def close_connection(self):
        self.client.close()

    def data_preprocessing(self, raw_dataset):

        # dataframe 중요한 column 가져오기
        df = raw_dataset[['_time', 'branch', 'device', 'endpoint', 'place', 'site', 'sensor_value']].copy()

        # column 이름 변경하기
        df.rename(columns={'_time':'date'}, inplace=True)
        df.rename(columns={'branch':'organization'}, inplace=True)
        df.rename(columns={'device':'device_id'}, inplace=True)
        df.rename(columns={'endpoint':'sensor_type'}, inplace=True)
        df.rename(columns={'_time':'date'}, inplace=True)

        return df 