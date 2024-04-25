from influxdb_client import InfluxDBClient
import pandas as pd

class DatasetManager:
    
    def __init__(self, db_url, token, org, bucket):
        self.client = InfluxDBClient(url=db_url, token=token, org=org, timeout=30_000)
        self.bucket = bucket
    
    def query_influx(self, measurement):
        query_api = self.client.query_api()

        query = f'from(bucket: "{self.bucket}") |> range(start: -1d) |> filter(fn: (r) => r.endpoint == "temperature")'

        result_df = query_api.query_data_frame(query=query)
        print(result_df)
        return result_df
    
    def close_connection(self):
        self.client.close()