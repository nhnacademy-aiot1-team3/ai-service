import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import pmdarima as pm
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd

class ModelManager:
    def __init__(self, df, sensor_type):
        self.df = df
        self.sensor_type = sensor_type
        self.models = {}
        self.predictions = {}
        self.accuracies = {}
        self.y_tests = {}

    def split_datasets(self):
        # train & test dataset으로 나누기
        split_ratio = int(len(self.df) * 0.9)
        self.train_datasets = self.df[:split_ratio]
        self.test_datasets = self.df[split_ratio:]
        
        # train & test dataset X,y별로 나누기
        X = self.df.index.values.reshape(-1,1)
        y = self.df[self.sensor_type]

        print('-'*30)
        print('X shape: ', X.shape)
        print('y shape: ', y.shape)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=False)
        
        print('-'*30)
        print('X_train : ', self.X_train[:10])
        print('-'*30)
        print('y_train : ', self.y_train[:10])
        print('-'*30)
        print('X_test : ', self.X_test[:10])
        print('-'*30)
        print('y_test : ', self.y_test[:10])

    def train_rf_model(self):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = rf_model
        self.predictions['RandomForest'] = rf_model.predict(self.X_test)
        self.y_tests['RandomForest'] = self.y_test

    def train_lstm_model(self):

        look_back = 52
        lstm_model = Sequential()

        # x_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1]))
        # print(x_train.shape)
        # x_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1]))

        lstm_model.add(LSTM(50, activation='relu',
                       return_sequences=True, input_shape=(1, 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(self.X_train, self.y_train, batch_size=1,
                       epochs=5, validation_data=(self.X_test, self.y_test))
        self.models['LSTM'] = lstm_model
        self.predictions['LSTM'] = lstm_model.predict(self.X_test)
        self.y_tests['LSTM'] = self.y_test

    def train_auto_arima_model(self):
        kpss_diffs = pm.arima.ndiffs(self.train_datasets, alpha=0.05, test='kpss', max_d=5)
        adf_diffs = pm.arima.ndiffs(self.train_datasets, alpha=0.05, test='adf', max_d=5)
        n_diffs = max(kpss_diffs, adf_diffs)

        print(f"Optimized 'd' = {n_diffs}")

        aa_model = pm.auto_arima(self.train_datasets, d=n_diffs, seasonal=False, trace=True)
        self.models['AutoARIMA'] = aa_model
        self.predictions['AutoARIMA'] = aa_model.predict(n_periods=len(self.test_datasets)).to_list()
        self.y_tests['AutoARIMA'] = self.test_datasets
    
    def train_lr_model(self):
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

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        self.models['LinearRegression'] = lr_model
        self.predictions['LinearRegression'] = lr_model.predict(X_test)
        self.y_tests['LinearRegression'] = pd.Series(y_test, index=y_test.index)

    def evaluate_models(self):
        for name, prediction in self.predictions.items():
            evs = explained_variance_score(self.y_tests[name], prediction)
            mae = mean_absolute_error(self.y_tests[name], prediction)
            mse = mean_squared_error(self.y_tests[name], prediction)
            rmse = mean_squared_error(self.y_tests[name], prediction, squared=False)
            r2 = r2_score(self.y_tests[name], prediction)

            self.accuracies[name] = {'Explained Variance': evs,
                                'Mean Absolute Error': mae,
                                'Mean Squared Error': mse,
                                'Root Mean Squared Error': rmse,
                                'R² Score': r2}
            
            self.draw_prediction(name, y_test, y_pred)

        return self.accuracies
    
    def draw_prediction(self, model_name, y_test, y_pred):
        plt.plot(self.df.index, self.df.values)
        plt.plot(y_test.index, y_test.values, label='Actual Values')
        plt.plot(y_test.index, y_pred.values, label='Predicted Values')
        plt.title(f'{model_name} Predictions vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.legend()
        plt.grid()
        plt.show()