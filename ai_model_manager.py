import numpy as np
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import joblib # type: ignore

# 모델 생성, 학습 및 테스트 담당하는 클래스
class ModelManager:
    """
    모델을 생성하고 학습하며 평가하는 클래스입니다.

    Attributes:
        df (DataFrame): 데이터셋.
        sensor_type (str): 센서 타입.
        models (dict): 학습된 모델들을 저장하는 딕셔너리.
        predictions (dict): 모델로부터 얻은 예측값을 저장하는 딕셔너리.
        accuracies (dict): 모델의 성능 지표를 저장하는 딕셔너리.
        X_tests (dict): 테스트 데이터 X를 저장하는 딕셔너리.
        y_tests (dict): 테스트 데이터 y를 저장하는 딕셔너리.
    """

    def __init__(self, df, sensor_type):
        """
        ModelManager 클래스의 생성자입니다.

        Args:
            df (DataFrame): 데이터셋.
            sensor_type (str): 센서 타입.
        """

        self.df = df
        self.sensor_type = sensor_type
        self.models = {}
        self.predictions = {}
        self.accuracies = {}
        self.X_tests = {}
        self.y_tests = {}

    # 가공된 데이터셋을 train & test dataset으로 나누기
    def split_datasets(self):
        """
        데이터셋을 학습용과 테스트용으로 나눕니다.
        """

        # train & test dataset X,y별로 나누기
        X = self.df.index.values.reshape(-1,1)
        y = self.df[self.sensor_type]

        print('-'*30)
        print('X shape: ', X.shape)
        print('X : ', X)
        print('y shape: ', y.shape)
        print('y : ', y.head(10))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=False)
        
        print('-'*30)
        print('X_train : ', self.X_train[:10])
        print('-'*30)
        print('y_train : ', self.y_train[:10])
        print('-'*30)
        print('X_test : ', self.X_test[:10])
        print('X_test type :', type(self.X_test))
        print('X_test value type :', type(self.X_test[0,0]))
        print('-'*30)
        print('y_test : ', self.y_test[:10])

    # Random Forest 모델 학습
    def train_rf_model(self):
        """
        Random Forest 모델을 학습합니다.
        """

        # 모델 생성 및 학습
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = rf_model
        self.predictions['RandomForest'] = rf_model.predict(self.X_test)
        self.X_tests['RandomForest'] = self.X_test
        self.y_tests['RandomForest'] = self.y_test

    # 선형 회귀 모델 학습
    def train_lr_model(self):
        """
        선형 회귀 모델을 학습합니다.
        """

        # linear regression model용 dataset 만들기
        split_ratio = int(len(self.df) * 0.9)
        lr_df = self.df
        lr_df[self.sensor_type+'.L1'] = lr_df[self.sensor_type].shift(1)
        lr_df.dropna(inplace=True)
        lr_y = lr_df[self.sensor_type]
        lr_X = lr_df.drop(columns=self.sensor_type)
        X_train, y_train = lr_X.iloc[:split_ratio], lr_y.iloc[:split_ratio]
        X_test, y_test = lr_X.iloc[split_ratio:], lr_y.iloc[split_ratio:]

        y_pred_baseline = [y_train.mean()] * len(y_train)
        mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
        print("Mean Close Prices:", round(y_train.mean(), 2))
        print("Baseline MAE:", round(mae_baseline, 2))
        print('-'*30)
        print('LR X_test : ', X_test[:10])
        print('LR X_test type :', type(X_test))

        # 모델 생성 및 학습
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        self.models['LinearRegression'] = lr_model
        self.predictions['LinearRegression'] = pd.Series(lr_model.predict(X_test), index=y_test.index)
        self.X_tests['LinearRegression'] = X_test
        self.y_tests['LinearRegression'] = y_test

    # 모델 평가
    def evaluate_models(self):
        """
        학습된 모델들을 평가합니다.

        Returns:
            dict: 모델의 평가 지표들을 담은 딕셔너리.
        """

        accuracies = {}
        for name, prediction in self.predictions.items():
            evs = explained_variance_score(self.y_tests[name], prediction)
            mae = mean_absolute_error(self.y_tests[name], prediction)
            mse = mean_squared_error(self.y_tests[name], prediction)
            rmse = mean_squared_error(self.y_tests[name], prediction, squared=False)
            r2 = r2_score(self.y_tests[name], prediction)

            accuracies[name] = {'Explained Variance': evs,
                                'Mean Absolute Error': mae,
                                'Mean Squared Error': mse,
                                'Root Mean Squared Error': rmse,
                                'R² Score': r2}

        return accuracies
    
    # 예측 그래프으로 그리기
    def draw_prediction(self, model_name, X_test, y_test, y_pred):
        """
        예측 결과를 그래프로 시각화합니다.

        Args:
            model_name (str): 모델 이름.
            X_test (DataFrame): 테스트 데이터 X.
            y_test (DataFrame): 테스트 데이터 y.
            y_pred (array): 모델의 예측값.
        """

        X_test_sorted = y_test.index
        y_test_sorted = y_test
        y_pred_sorted = y_pred

        if model_name=='RandomForest':
            sorted_indices = np.argsort(X_test[:, 0])
            X_test_sorted = X_test[sorted_indices]
            y_test_sorted = y_test.iloc[sorted_indices]
            y_pred_sorted = y_pred[sorted_indices]

        plt.figure()
        plt.plot(X_test_sorted, y_test_sorted, label='Actual Values')
        plt.plot(X_test_sorted, y_pred_sorted, label='Predicted Values')
        plt.title(f'{model_name} Predictions vs Actual')
        plt.xlabel('Date')
        plt.ylabel(self.sensor_type)
        plt.legend()
        plt.grid()
        plt.show()

    # 모델 저장
    def save_model(self):
        """
        학습된 모델을 저장합니다.
        """
        
        joblib.dump(self.models['LinearRegression'], self.sensor_type+'_model.joblib')