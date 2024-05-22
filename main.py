import threading
from env_config import load_env_var
from dataset_manager import DatasetManager
from ai_model_manager import ModelManager
from flask import Flask, request, jsonify # type: ignore
from pathlib import Path
import joblib # type: ignore
import pandas as pd
import schedule # type: ignore
import time

app = Flask(__name__)

@app.route('/predict/temp', methods=['POST'])
def temp_predict():
    """
    온도 예측을 수행하는 엔드포인트입니다.
    """

    temp_model = joblib.load('temperature_model.joblib')
    data = request.json
    print(data)

    df = pd.DataFrame(data)
    df.columns=['date', 'temperature.L1']
    df['date'] = pd.to_datetime(df['date'])
    df['temperature.L1'] = df['temperature.L1'].astype(float)
    df.set_index('date', inplace=True)
    print(df)

    features = df['temperature.L1'].values.reshape(-1, 1)
    
    prediction = temp_model.predict(features).tolist()

    response = {'prediction': prediction}
    return jsonify(response)

@app.route('/predict/elect', methods=['POST'])
def elec_predict():
    """
    전기 에너지 예측을 수행하는 엔드포인트입니다.
    """

    elec_model = joblib.load('electrical_energy_model.joblib')
    data = request.json
    print(data)

    df = pd.DataFrame(data)
    df.columns=['date', 'electrical_energy.L1']
    df['date'] = pd.to_datetime(df['date'])
    df['electrical_energy.L1'] = df['electrical_energy.L1'].astype(float)
    df.set_index('date', inplace=True)
    print(df)

    features = df['electrical_energy.L1'].values.reshape(-1, 1)

    prediction = elec_model.predict(features).tolist()

    response = {'prediction' :  prediction}
    return jsonify(response)

def make_and_upload_model(sensor_type):
    """
    모델을 생성하고 업로드하는 함수입니다.

    Args:
        sensor_type (str): 센서 타입.
    """

    accuracies = {}
    best_models = {}
    env_var_list = load_env_var()

    dataset_manager = DatasetManager(env_var_list['db_url'], env_var_list['token'], env_var_list['org'], env_var_list['bucket'], sensor_type)

    # 온도 센서 데이터 불러오기
    if sensor_type=='temperature':
        df = dataset_manager.query_sensor_data('nhnacademy')
    # 전기 에너지 센서 데이터 불러오기
    else:
        df = dataset_manager.query_energy('nhnacademy')

    dataset_manager.close_connection()

    # 데이터 전처리
    df = dataset_manager.data_preprocessing(df)

    # 모델 학습, 평가 및 저장
    model_manager = ModelManager(df, sensor_type)

    # 데이터셋 분할
    model_manager.split_datasets()

    # 선형 회귀 모델 학습
    model_manager.train_lr_model()
    accuracies[sensor_type] = model_manager.evaluate_models()
    best_models[sensor_type] = model_manager.models['LinearRegression']

    # 모델 저장
    model_manager.save_model()

    # 모든 모델 출력
    print('EVALUATION SCORES: ')
    for sensor, models in accuracies.items():
        print(f"Sensor: {sensor}")
        for model, scores in models.items():
            print(f"{model}: {scores}")
        print('-'*30)

# Main
def main():
    """
    메인 함수입니다.
    """

    file_path_elect = Path("electrical_energy_model.joblib")
    file_path_temp = Path("temperature_model.joblib")

    # 모델 파일이 없는 경우 모델 생성 및 업로드
    # if file_path_elect.exists()==False:
    make_and_upload_model('electrical_energy')
    
    # if file_path_temp.exists()==False:
    make_and_upload_model('temperature')

    # 주기적으로 모델 생성 및 업로드
    # schedule.every(1).minutes.do(make_and_upload_model, 'electrical_energy')    
    # schedule.every(1).minutes.do(make_and_upload_model,'temperature')
    
    # flask_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000})
    # flask_thread.start()

    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)

if __name__ == '__main__':
    main()