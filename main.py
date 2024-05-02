from env_config import load_env_var
from dataset_manager import DatasetManager
from ai_model_manager import ModelManager
import pandas as pd

# Main
def main():
    accuracies = {}
    best_models = {}
    env_var_list = load_env_var()

    dataset_manager_temp = DatasetManager(env_var_list['db_url'], env_var_list['token'], env_var_list['org'], env_var_list['bucket'], 'temperature')
    temp_df = dataset_manager_temp.query_sensor_data('gyeongnam')
    dataset_manager_temp.close_connection()
    temp_df = dataset_manager_temp.data_preprocessing(temp_df)

    dataset_manager_energy = DatasetManager(env_var_list['db_url'], env_var_list['token'], env_var_list['org'], env_var_list['bucket'], 'electrical_energy')
    energy_df = dataset_manager_energy.query_energy('gyeongnam')
    dataset_manager_energy.close_connection()
    energy_df = dataset_manager_energy.data_preprocessing(energy_df)

    model_manager_temp = ModelManager(temp_df, 'temperature')
    model_manager_temp.split_datasets()
    model_manager_temp.train_rf_model()
    model_manager_temp.train_lr_model()
    accuracies['Temperature'] = model_manager_temp.evaluate_models()
    best_models['Temperature'] = model_manager_temp.models['LinearRegression']

    model_manager_energy = ModelManager(energy_df, 'electrical_energy')
    model_manager_energy.split_datasets()
    model_manager_energy.train_rf_model()
    model_manager_energy.train_lr_model()
    accuracies['Energy'] = model_manager_energy.evaluate_models()
    best_models['Energy'] = model_manager_energy.models['LinearRegression']

    # 모든 모델 출력
    print('EVALUATION SCORES: ')
    for sensor, models in accuracies.items():
        print(f"Sensor: {sensor}")
        for model, scores in models.items():
            print(f"{model}: {scores}")
        print('-'*30)


if __name__ == '__main__':
    main()