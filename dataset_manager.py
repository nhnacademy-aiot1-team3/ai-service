from influxdb_client import InfluxDBClient # type: ignore
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 데이터셋 받아오기 및 데이터 전처리 담당하는 클래스
class DatasetManager:
    """
    데이터셋을 관리하고 전처리하는 클래스입니다.
    
    Attributes:
        db_url (str): InfluxDB의 URL.
        token (str): InfluxDB에 대한 액세스 토큰.
        org (str): InfluxDB 조직 이름.
        bucket (str): 데이터가 저장된 InfluxDB 버킷.
        sensor_type (str): 센서 타입.
    """

    def __init__(self, db_url, token, org, bucket, sensor_type):
        """
        DatasetManager 클래스의 생성자입니다.
        
        Args:
            db_url (str): InfluxDB의 URL.
            token (str): InfluxDB에 대한 액세스 토큰.
            org (str): InfluxDB 조직 이름.
            bucket (str): 데이터가 저장된 InfluxDB 버킷.
            sensor_type (str): 센서 타입.
        """

        self.client = InfluxDBClient(url=db_url, token=token, org=org, timeout=30_000)
        self.bucket = bucket
        self.sensor_type = sensor_type
    
    # 데이터 쿼리를 위한 시작 및 종료 시간을 반환
    def start_end_time(self):
        """
        데이터 쿼리를 위한 시작 및 종료 시간을 반환합니다.
        
        Returns:
            list: 시작 및 종료 시간을 나타내는 ISO 형식 문자열의 리스트.
        """

        now = datetime.now()
        start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        start_iso = start_time.isoformat() + 'Z'
        end_iso = now.isoformat() + 'Z'

        return [start_iso, end_iso]
    
    # 온도 데이터 가져오기
    def query_sensor_data(self, branch):
        """
        환경 센서 데이터를 쿼리하여 DataFrame으로 반환합니다.
        
        Args:
            branch (str): 지점 이름.
        
        Returns:
            DataFrame: 쿼리 결과로 생성된 데이터 프레임.
        """

        query_api = self.client.query_api()
        
        query = f'from(bucket: "{self.bucket}")\
            |> range(start: -7d)\
            |> filter(fn: (r) => r.branch == "{branch}")\
            |> filter(fn: (r) => r["endpoint"] == "{self.sensor_type}")\
            |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)\
            |> yield(name: "sensor_value")'
        
        print("Query: ",query)
        result_df = query_api.query_data_frame(query=query)
        return result_df
    
    # 전량 데이터 가져오기
    def query_energy(self, branch):
        """
        에너지 데이터를 쿼리하여 DataFrame으로 반환합니다.
        
        Args:
            branch (str): 지점 이름.
        
        Returns:
            DataFrame: 쿼리 결과로 생성된 데이터 프레임.
        """

        query_api = self.client.query_api()
        datetimes = self.start_end_time()
        query = f'from(bucket: "{self.bucket}")\
                |> range(start: {datetimes[0]}, stop: {datetimes[1]})\
                |> filter(fn: (r) => r["branch"] == "{branch}")\
                |> filter(fn: (r) => r["endpoint"] == "{self.sensor_type}")\
                |> filter(fn: (r) => r["phase"] == "total")\
                |> filter(fn: (r) => r["description"] == "w")\
                |> group(columns: ["site"])\
                |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)\
                |> yield(name: "sensor_value")'
        
        print("Query: ", query)
        result_df = query_api.query_data_frame(query=query)
        return result_df

    # DB connection 닫기
    def close_connection(self):
        """InfluxDB 클라이언트 연결을 닫습니다."""
        
        self.client.close()

    # 데이터 전처리
    def data_preprocessing(self, raw_dataset):
        """
        데이터 프레임을 전처리하여 반환합니다.
        
        Args:
            raw_dataset (DataFrame): 전처리할 원본 데이터 프레임.
        
        Returns:
            DataFrame: 전처리된 데이터 프레임.
        """

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

        # date를 index로 set하기
        df = df.set_index('date')

        # 결측지를 전체의 평균값으로 채우기
        df = self.outlier_processing(df)        

        # 가공된 dataframe의 정보 출력
        self.print_df_info(df)
        
        return df
    
    # dataset의 정보 출력
    def print_df_info(self, dataset):
        """
        데이터 프레임에 대한 정보를 출력합니다.
        
        Args:
            dataset (DataFrame): 출력할 데이터 프레임.
        """

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
        """
        데이터 프레임을 그래프로 표현합니다.
        
        Args:
            dataset (DataFrame): 그래프로 표현할 데이터 프레임.
        """

        plt.figure(figsize=(16,8))
        sns.lineplot(data=dataset, palette='Set1')
        plt.xlabel('Time')
        plt.ylabel(self.sensor_type)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # 초별 -> 분별 변환
    def second2minute(self, df):
        """
        초 단위 데이터를 분 단위로 변환합니다.
        
        Args:
            df (DataFrame): 초 단위 데이터 프레임.
        
        Returns:
            DataFrame: 분 단위로 변환된 데이터 프레임.
        """

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
        """
        분 단위 데이터를 시간 단위로 변환합니다.
        
        Args:
            df (DataFrame): 분 단위 데이터 프레임.
        
        Returns:
            DataFrame: 시간 단위로 변환된 데이터 프레임.
        """

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
    
    # 이상치 제거
    def outlier_processing(self, raw_dataset):
        """
        데이터 프레임에서 이상치를 제거하고 보간합니다.
        
        Args:
            raw_dataset (DataFrame): 이상치를 처리할 원본 데이터 프레임.
        
        Returns:
            DataFrame: 이상치를 제거하고 보간한 데이터 프레임.
        """
        raw_dataset[self.sensor_type] = raw_dataset[self.sensor_type].interpolate()
        
        print('-'*30)
        print(np.percentile(raw_dataset,25))
        print(np.percentile(raw_dataset,50))
        print(np.median(raw_dataset))
        print(np.percentile(raw_dataset,75))

        iqr_value = np.percentile(raw_dataset,75) - np.percentile(raw_dataset,25)
        print('IQR_value : {}'.format(iqr_value))

        upper_bound = iqr_value * 1.5 + np.percentile(raw_dataset, 75)
        print('upper_bound : {}'.format(upper_bound)) # 보다 큰 값은 이상치

        lower_bound = np.percentile(raw_dataset,25) - iqr_value * 1.5
        print('lower_bound : {}'.format(lower_bound)) # 보다 작은 값은 이상치

        # 우리 데이터에서 이상치를 출력
        print(raw_dataset[(raw_dataset > upper_bound) | (raw_dataset < lower_bound)])

        result_data = raw_dataset[(raw_dataset<=upper_bound) & (raw_dataset >= lower_bound)]
        print('이상치 제거 후 데이터 : {}'.format(result_data))
        print('-'*30)

        return result_data