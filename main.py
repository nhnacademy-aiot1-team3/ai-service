from env_config import load_env_var
from dataset_manager import DatasetManager
from ai_model_manager import ModelManager
import pandas as pd

# Main
def main():
    env_var_list = load_env_var()
    dataset_manager = DatasetManager(env_var_list['db_url'], env_var_list['token'], env_var_list['org'], env_var_list['bucket'], 'temperature')
    result_df = dataset_manager.query_sensor_data('gyeongnam')
    dataset_manager.close_connection()

    result_df = dataset_manager.data_preprocessing(result_df)
    
    # result_df = pd.read_csv('Occupancy.csv')
    # result_df = result_df.drop(columns=['Humidity', 'Light', 'CO2', 'HumidityRatio'])
    # result_df.rename(columns={'Temperature':'temp'}, inplace=True)
    # result_df.rename(columns={'Occupancy':'occ'}, inplace=True)
    # result_df = result_df.set_index('date')
    # DatasetManager.print_df_info(result_df)

    model_manager = ModelManager(result_df, 'temperature')
    model_manager.split_datasets()
    # model_manager.train_rf_model()
    # model_manager.train_auto_arima_model()
    # model_manager.train_lstm_model()
    
    model_manager.train_lr_model()
    accuracies = model_manager.accuracies

    # best_model = max(accuracies, key=lambda x: accuracies[x]['Explained Variance'])
    # best_metrics = accuracies[best_model]

    for model, metrics in accuracies.items():
        print(f"Model: {model}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        print('-'*30)

    # result_df['date'] = pd.to_datetime(result_df['date'])
    # result_df['year'] = result_df['date'].apply(lambda x : x.year)
    # result_df['month'] = result_df['date'].apply(lambda x : x.month)
    # result_df['day'] = result_df['date'].apply(lambda x : x.day)
    # result_df['hour'] = result_df['date'].apply(lambda x : x.hour)

    # agg_df = result_df.groupby(['year', 'month', 'day', 'hour']).agg({ \
    #         'date': 'first', \
    #         'Temperature': 'mean', \
    #         'Humidity': 'mean', \
    #         'Light': 'mean', \
    #         'CO2' : 'mean', \
    #         'HumidityRatio' : 'mean', \
    #         'Occupancy' : 'first'
    #     }).reset_index()
    
    # agg_df['date'] = agg_df['date'].dt.strftime('%Y-%m-%d %H:00:00')
    # DatasetManager.print_df_info(result_df)


if __name__ == '__main__':
    main()