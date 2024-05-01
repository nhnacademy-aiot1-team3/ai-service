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

    model_manager = ModelManager(result_df, 'temperature')
    model_manager.split_datasets()
    model_manager.train_rf_model()
    model_manager.train_lr_model()
    # model_manager.train_auto_arima_model()
    accuracies = model_manager.evaluate_models()

    # 모든 모델 출력
    for model, metrics in accuracies.items():
        print(f"Model: {model}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        print('-'*30)


if __name__ == '__main__':
    main()