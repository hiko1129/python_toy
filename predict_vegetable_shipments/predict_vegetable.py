# coding: utf-8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
import math

AMOUNT_OF_CROP = '収穫量'
PLANTED_AREA = '作付面積'
weather_dfs = pd.read_html('http://www.data.jma.go.jp/obd/stats/etrn/view/annually_s.php?prec_no=45&block_no=47682&year=&month=&day=&view=', skiprows=3, index_col=0)

chiba_df = pd.read_excel('chiba.xls', header=[4, 5, 6], skip_footer=3)

weather_df = weather_dfs[0]

def clean(x):
    if isinstance(x, str):
        return x.replace(']', '')


def get_weather_df(start_year, end_year):
    weather_data_list = []
    for i in weather_df.columns.values:
        weather_data_list.append(weather_df[i].apply(clean).loc[start_year:end_year].values)
    X = pd.DataFrame({
        '降雨量': weather_data_list[2],
        '平均気温': weather_data_list[6],
        '平均湿度': weather_data_list[11],
        '平均風速': weather_data_list[13],
        '日照時間': weather_data_list[18]
    }, index=range(start_year, end_year+1), dtype='float')
    return X


X = get_weather_df(1973, 2012)
# 「…」と「_」はnanに置き換え
chiba_df = chiba_df.apply(lambda x: x.replace('…', np.nan))
chiba_df = chiba_df.apply(lambda x: x.replace('-', np.nan))

# データ数が少ないため欠損値があるものは除外.
chiba_df = chiba_df.dropna(axis=1)

vegetable_name_list = chiba_df.columns.levels[0].values

# 収穫量用.
amount_of_crops = {}
# 作付面積用.
planted_areas = {}
for vegetable_name in vegetable_name_list:
    vegetable_df = chiba_df[vegetable_name]
    if 'ha' not in vegetable_df[PLANTED_AREA].columns.values\
            or 't' not in vegetable_df[AMOUNT_OF_CROP].columns.values:
        continue
    amount_of_crops[vegetable_name] = vegetable_df[AMOUNT_OF_CROP].t.values
    planted_areas[vegetable_name] = vegetable_df[PLANTED_AREA].ha.values


X_cp = X.copy()
# 平均二乗誤差用.
scores = []
# 予測値用.
predictions = []
# テストのインデックス用.
test_index = []
# y_test保持用.
y_tests = []
n_features = len(X.columns)
space = [(1, n_features),  # max_features
         (1, 1e2),  # n_estimators
         (1, 1e2)]  # min_samples_leaf
for vegetable_name in amount_of_crops.keys():
    df = X_cp
    df[PLANTED_AREA] = planted_areas[vegetable_name]
    df[AMOUNT_OF_CROP] = amount_of_crops[vegetable_name]
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([AMOUNT_OF_CROP], axis=1),
        df[AMOUNT_OF_CROP],
        test_size=0.3
    )
    y_tests.append(y_test)
    test_index.append(X_test.index)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    reg = RandomForestRegressor()

    def objective(params):
        max_features, n_estimators, min_samples_leaf = params

        reg.set_params(max_features=max_features,
                       n_estimators=n_estimators,
                       min_samples_leaf=min_samples_leaf)

        return -np.mean(cross_val_score(reg, X_train, y_train, cv=5, n_jobs=-1,
                                        scoring="neg_mean_squared_error"))

    res_gp = gp_minimize(objective, space, n_calls=15, random_state=0)
    scores.append(res_gp.fun)
    reg.set_params(max_features=res_gp.x[0],
                   n_estimators=res_gp.x[1],
                   min_samples_leaf=res_gp.x[2])
    reg.fit(X_train, y_train)
    predictions.append(reg.predict(X_test))

for i, j, k, l, m in zip(amount_of_crops.keys(), test_index, scores, predictions, y_tests):
    print(i, j, k, l, m)

reg = RandomForestRegressor()
soy_df = pd.read_excel('soy.xls', header=[4, 5])
soy_df = soy_df[[PLANTED_AREA, AMOUNT_OF_CROP]]
df = get_weather_df(1966, 2012)
soy_df = soy_df.loc['昭.41(1966)':]
df[PLANTED_AREA] = soy_df[PLANTED_AREA].astype('float').values
df[AMOUNT_OF_CROP] = soy_df[AMOUNT_OF_CROP].astype('float').values
X_train, X_test, y_train, y_test = train_test_split(df.drop(AMOUNT_OF_CROP, axis=1), df[AMOUNT_OF_CROP], test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

space = [(1, len(df.drop(AMOUNT_OF_CROP, axis=1).columns)),  # max_features
         (1, 1e3),  # n_estimators
         (1, 1e3)]  # min_samples_leaf


def objective(params):
    max_features, n_estimators, min_samples_leaf = params

    reg.set_params(max_features=max_features,
                   n_estimators=n_estimators,
                   min_samples_leaf=min_samples_leaf)

    return -np.mean(cross_val_score(reg, X_train, y_train, cv=5, n_jobs=-1,
                                    scoring="neg_mean_squared_error"))


res_gp = gp_minimize(objective, space, n_calls=200, random_state=0)
# memo: 290.268
math.sqrt(res_gp.fun)

X_train, X_test, y_train, y_test = train_test_split(df[PLANTED_AREA], df[AMOUNT_OF_CROP], test_size=0.2)
space = [(1, 1e3),  # n_estimators
         (1, 1e3)]  # min_samples_leaf


reg = RandomForestRegressor()
reg.fit(X_train.values[:, None], y_train)
def objective(params):
    n_estimators, min_samples_leaf = params

    reg.set_params(n_estimators=n_estimators,
                   min_samples_leaf=min_samples_leaf)

    return -np.mean(cross_val_score(reg, X_train.values[:, None], y_train, cv=5, n_jobs=-1,
                                    scoring="neg_mean_squared_error"))

res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)
# memo 466.93
math.sqrt(res_gp.fun)
