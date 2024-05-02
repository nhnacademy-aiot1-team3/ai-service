from env_config import load_env_var
from dataset_manager import DatasetManager
from ai_model_manager import ModelManager
from flask import Flask, request, jsonify
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)
temp_model = joblib.load('temperature_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('dates')
    print(data)

    # Parse each date string into a datetime object
    data = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in data]
    data = np.array(data)
    data = data.reshape(-1,1)
    print(data)
    
    prediction = temp_model.predict(data)

    response = {'prediction': prediction}
    return jsonify(response)

# Main
def main():
    accuracies = {}
    best_models = {}
    env_var_list = load_env_var()

    dataset_manager_temp = DatasetManager(env_var_list['db_url'], env_var_list['token'], env_var_list['org'], env_var_list['bucket'], 'temperature')
    temp_df = dataset_manager_temp.query_sensor_data('gyeongnam')
    dataset_manager_temp.close_connection()
    temp_df = dataset_manager_temp.data_preprocessing(temp_df)

    # dataset_manager_energy = DatasetManager(env_var_list['db_url'], env_var_list['token'], env_var_list['org'], env_var_list['bucket'], 'electrical_energy')
    # energy_df = dataset_manager_energy.query_energy('gyeongnam')
    # dataset_manager_energy.close_connection()
    # energy_df = dataset_manager_energy.data_preprocessing(energy_df)

    model_manager_temp = ModelManager(temp_df, 'temperature')
    model_manager_temp.split_datasets()
    model_manager_temp.train_rf_model()
    model_manager_temp.train_lr_model()
    accuracies['Temperature'] = model_manager_temp.evaluate_models()
    best_models['Temperature'] = model_manager_temp.models['LinearRegression']
    model_manager_temp.save_model()

    # model_manager_energy = ModelManager(energy_df, 'electrical_energy')
    # model_manager_energy.split_datasets()
    # model_manager_energy.train_rf_model()
    # model_manager_energy.train_lr_model()
    # accuracies['Energy'] = model_manager_energy.evaluate_models()
    # best_models['Energy'] = model_manager_energy.models['LinearRegression']

    # 모든 모델 출력
    print('EVALUATION SCORES: ')
    for sensor, models in accuracies.items():
        print(f"Sensor: {sensor}")
        for model, scores in models.items():
            print(f"{model}: {scores}")
        print('-'*30)
    
    app.run()

if __name__ == '__main__':
    main()