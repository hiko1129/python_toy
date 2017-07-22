import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
import numpy as np
from skopt.plots import plot_convergence
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import seaborn
import math

def set_params(params, reg):
        """パラメータの設定."""
        n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf = params

        reg.set_params(n_estimators=n_estimators,
                       max_depth=max_depth,
                       max_features=max_features,
                       min_samples_split=min_samples_split,
                       min_samples_leaf=min_samples_leaf)
        return reg

def cross_validation(reg, X_train, y_train):
    return -np.mean(cross_val_score(reg, X_train, y_train, cv=10, n_jobs=-1,
                                    scoring='neg_mean_squared_error'))

class PreparingData:
    # 非線形回帰にR2は使用できないため平均二乗誤差のみ使用
    def get_weather_df(self, input_files, output_file, skiprows_dict):
        if os.path.exists(output_file):
            return pd.read_csv(output_file, encoding='sjis')
        temperture_df = pd.read_csv(input_files['temperture'], encoding='sjis', skiprows=skiprows_dict['temperture'])
        amount_of_precipitation_df = pd.read_csv(input_files['amount_of_precipitation'], encoding='sjis', skiprows=skiprows_dict['amount_of_precipitation'])
        hours_of_sunlight_df = pd.read_csv(input_files['hours_of_sunlight'], encoding='sjis', skiprows=skiprows_dict['hours_of_sunlight'])
        wind_speed_df = pd.read_csv(input_files['wind_speed'], encoding='sjis', skiprows=skiprows_dict['wind_speed'])
        weather_df = pd.DataFrame(data=temperture_df.iloc[:, 0].values, columns=['年月日時'])
        weather_df['気温'] = temperture_df.iloc[:, 1].values
        weather_df['降水量'] = amount_of_precipitation_df.iloc[:, 1].values
        weather_df['日照時間'] = hours_of_sunlight_df.iloc[:, 1].values
        weather_df['風速'] = wind_speed_df.iloc[1:, 1].values
        weather_df.to_csv(output_file, index=None, encoding='sjis')
        return weather_df


    def get_demand_date_list(self, filename, header):
        """データの準備."""
        demand_df = pd.read_csv(filename, encoding='sjis', header=header)
        date_list = demand_df['DATE'].apply(lambda x: x.split('/')).values
        time_list = demand_df['TIME'].apply(lambda x: x.split(':')[0]).values
        month_list = []
        day_list = []
        for year, month, day in date_list:
            month_list.append(month)
            day_list.append(day)
        return (demand_df.iloc[:, 2], month_list, day_list, time_list)


    def prepare_data(self, input_filenames, output_filename, objective_filename, skiprows_list=[4, 4, 4, 4], demand_header=1):
        input_files = {
            'temperture': input_filenames[0],
            'amount_of_precipitation': input_filenames[1],
            'hours_of_sunlight': input_filenames[2],
            'wind_speed': input_filenames[3]
        }
        skiprows_dict = {
            'temperture': skiprows_list[0],
            'amount_of_precipitation': skiprows_list[1],
            'hours_of_sunlight': skiprows_list[2],
            'wind_speed': skiprows_list[3]
        }
        weather_df = self.get_weather_df(input_files, output_filename, skiprows_dict)
        demand_list, month_list, day_list, time_list = self.get_demand_date_list(objective_filename, demand_header)
        weather_df['月'] = np.array(month_list, dtype='int')
        weather_df['日'] = np.array(day_list, dtype='int')
        weather_df['時'] = np.array(time_list, dtype='int')
        X_y = weather_df
        X_y['電力需要'] = demand_list
        # nanを平均で補完
        X_y = X_y.fillna(X_y.mean())
        self.X_y = X_y
        return (X_y.drop(['年月日時', '電力需要'], axis=1).astype('float').values, X_y['電力需要'].astype('float').values.reshape(-1, 1))

    def get_created_df(self):
        return self.X_y


if __name__ == '__main__':
    preparing_data = PreparingData()
    input_filenames = ['temperture_2008.csv',
                       'amount_of_precipitation_2008.csv',
                       'hours_of_sunlight_2008.csv',
                       'wind_speed_2008.csv']
    X_train, y_train = preparing_data.prepare_data(input_filenames, 'weather_2008.csv', 'demand_2008.csv')

    input_filenames = ['temperture_2010.csv',
                       'amount_of_precipitation_2010.csv',
                       'hours_of_sunlight_2010.csv',
                       'wind_speed_2010.csv']
    temp_X, temp_y = preparing_data.prepare_data(input_filenames, 'weather_2010.csv', 'demand_2010.csv')
    X_train = np.vstack((X_train, temp_X))
    y_train = np.vstack((y_train, temp_y))

    input_filenames = ['temperture_2011.csv',
                       'amount_of_precipitation_2011.csv',
                       'hours_of_sunlight_2011.csv',
                       'wind_speed_2011.csv']
    temp_X, temp_y = preparing_data.prepare_data(input_filenames, 'weather_2011.csv', 'demand_2011.csv')
    X_train = np.vstack((X_train, temp_X))
    y_train = np.vstack((y_train, temp_y))

    input_filenames = ['temperture_2012.csv',
                       'amount_of_precipitation_2012.csv',
                       'hours_of_sunlight_2012.csv',
                       'wind_speed_2012.csv']
    temp_X, temp_y = preparing_data.prepare_data(input_filenames, 'weather_2012.csv', 'demand_2012.csv')
    X_train = np.vstack((X_train, temp_X))
    y_train = np.vstack((y_train, temp_y))

    input_filenames = ['temperture_2014.csv',
                       'amount_of_precipitation_2014.csv',
                       'hours_of_sunlight_2014.csv',
                       'wind_speed_2014.csv']
    temp_X, temp_y = preparing_data.prepare_data(input_filenames, 'weather_2014.csv', 'demand_2014.csv')
    X_train = np.vstack((X_train, temp_X))
    y_train = np.vstack((y_train, temp_y))

    input_filenames = ['temperture_2016.csv',
                       'amount_of_precipitation_2016.csv',
                       'hours_of_sunlight_2016.csv',
                       'wind_speed_2016.csv']
    X_test, y_test = preparing_data.prepare_data(input_filenames, 'weather_2016.csv', 'demand_2016.csv', demand_header=2)
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    reg = RandomForestRegressor(random_state=0)

    def objective(params):
        global reg
        reg = set_params(params, reg)
        return cross_validation(reg, X_train, y_train)
    space = [(1, 1e5),  # n_estimators
             (1, 1e5),  # max_depth
             (1, X_train.shape[1]),  # max_features
             (2, 1e5),  # min_samples_split
             (1, 1e5)]  # min_samples_leaf
    res_gp = gp_minimize(objective, space, n_calls=10, random_state=0, verbose=0, n_jobs=-1)
    math.sqrt(res_gp.fun)
    reg = set_params(res_gp.x, reg)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    math.sqrt(mean_squared_error(y_pred, y_test))
    plot_convergence(res_gp)
    plt.figure(figsize=(20, 10))
    plt.plot(range(0, X_test.shape[0]), y_test)
    plt.plot(range(0, X_test.shape[0]), y_pred)
    plt.show()
