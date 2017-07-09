# coding: utf-8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
import math

# 作物統計.
# http://www.e-stat.go.jp/SG1/estat/List.do?bid=000001024932&cycode=0

WEATHER_URL = 'http://www.data.jma.go.jp/obd/stats/etrn/view/annually_s.php?prec_no=45&block_no=47682&year=&month=&day=&view='

AMOUNT_OF_CROP = '収穫量'
PLANTED_AREA = '作付面積'


def clean(x):
    if isinstance(x, str):
        return x.replace(']', '')


def get_weather_df(start_year, end_year):
    weather_dfs = pd.read_html(WEATHER_URL, skiprows=3, index_col=0)
    weather_df = weather_dfs[0]
    weather_data_list = []
    for i in weather_df.columns.values:
        weather_data_list.append(
            weather_df[i].apply(clean).loc[start_year:end_year].values
        )
    X = pd.DataFrame({
        '降雨量': weather_data_list[2],
        '平均気温': weather_data_list[6],
        '平均湿度': weather_data_list[11],
        '平均風速': weather_data_list[13],
        '日照時間': weather_data_list[18]
    }, index=range(start_year, end_year+1), dtype='float')
    return X


def calculate_rmse_all_data(X, y):
    """収穫量及び、気象情報を用いた生産量の予測."""
    reg = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    space = [(1, len(X.columns)),  # max_features
             (1, 1e3),  # n_estimators
             (1, 1e3)]  # min_samples_leaf

    def objective(params):
        max_features, n_estimators, min_samples_leaf = params

        reg.set_params(max_features=max_features,
                       n_estimators=n_estimators,
                       min_samples_leaf=min_samples_leaf)

        return -np.mean(cross_val_score(reg, X_train, y_train, cv=5, n_jobs=-1,
                                        scoring="neg_mean_squared_error"))

    res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)
    # memo: 290.268
    # 332.78
    return math.sqrt(res_gp.fun)


def calculate_rmse_only_planted_area(X, y):
    """収穫量のみを用いた生産量の予測."""
    reg = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(df[PLANTED_AREA],
                                                        df[AMOUNT_OF_CROP],
                                                        test_size=0.2)
    space = [(1, 1e3),  # n_estimators
             (1, 1e3)]  # min_samples_leaf

    def objective(params):
        n_estimators, min_samples_leaf = params

        reg.set_params(n_estimators=n_estimators,
                       min_samples_leaf=min_samples_leaf)

        return -np.mean(cross_val_score(reg, X_train.values[:, None], y_train,
                                        cv=5, n_jobs=-1,
                                        scoring="neg_mean_squared_error"))

    res_gp = gp_minimize(objective, space, n_calls=10, random_state=0)
    # memo 466.93
    # 453.32
    return math.sqrt(res_gp.fun)


def calculate_rmse(filepath, header, weather_period, vegetable_period):
    vegetable_df = pd.read_excel(filepath, header=header)

    vegetable_df = vegetable_df[[PLANTED_AREA, AMOUNT_OF_CROP]]
    start_year, end_year = weather_period
    df = get_weather_df(start_year, end_year)
    start_year, end_year = vegetable_period
    vegetable_df = vegetable_df.loc[start_year:end_year]
    df[PLANTED_AREA] = vegetable_df[PLANTED_AREA].astype('float').values
    df[AMOUNT_OF_CROP] = vegetable_df[AMOUNT_OF_CROP].astype('float').values
    rmse_all_data = calculate_rmse_all_data(df.drop(AMOUNT_OF_CROP, axis=1),
                                            df[AMOUNT_OF_CROP])
    rmse_only_planted_area = calculate_rmse_only_planted_area(
        df[PLANTED_AREA],
        df[AMOUNT_OF_CROP]
    )
    return {'rmse_all_data': rmse_all_data,
            'rmse_only_planted_area': rmse_only_planted_area}


if __name__ == '__main__':
    calculate_rmse(filepath='soy.xls',
                   header=[4, 5],
                   weather_period=[1966, 2012],
                   vegetable_period=['昭.41(1966)', '平.24(2012)'])
    calculate_rmse(filepath='corn.xls',
                   header=[4, 5],
                   weather_period=[1966, 2012],
                   vegetable_period=['昭.41(1966)', '平.24(2012)'])
    calculate_rmse(filepath='wheat.xls',
                   header=[4, 5],
                   weather_period=[1966, 2012],
                   vegetable_period=['昭.41(1966)', '平.24(2012)'])
