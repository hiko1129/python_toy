# coding: utf-8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize

weather_dfs = pd.read_html('http://www.data.jma.go.jp/obd/stats/etrn/view/annually_s.php?prec_no=45&block_no=47682&year=&month=&day=&view=', skiprows=3, index_col=0)

chiba_df = pd.read_excel('chiba.xls', header=[4, 5, 6], skip_footer=3)

weather_df = weather_dfs[0]


def clean(x):
    if isinstance(x, str):
        return x.replace(']', '')


weather_data_list = []
for i in weather_df.columns.values:
    weather_data_list.append(weather_df[i].apply(clean).loc[1973:2012].values)
# X = []
# X.append(weather_data_list[2])  # 降雨量.
# X.append(weather_data_list[6])  # 平均の気温.
# X.append(weather_data_list[11])  # 平均湿度.
# X.append(weather_data_list[13])  # 平均風速.
# X.append(weather_data_list[18])  # 日照時間.
X = pd.DataFrame({
    '降雨量': weather_data_list[2],
    '平均気温': weather_data_list[6],
    '平均湿度': weather_data_list[11],
    '平均風速': weather_data_list[13],
    '日照時間': weather_data_list[18]
}, index=range(1973, 2013), dtype='float')

# 「…」と「_」はnanに置き換え
chiba_df = chiba_df.apply(lambda x: x.replace('…', np.nan))
chiba_df = chiba_df.apply(lambda x: x.replace('-', np.nan))

# データ数が少ないため欠損値があるものは除外.
chiba_df = chiba_df.dropna(axis=1)

# naを0で埋める
# chiba_df = chiba_df.fillna(value=0)
vegetable_name_list = chiba_df.columns.levels[0].values

amount_of_crops = {}
planted_areas = {}
for vegetable_name in vegetable_name_list:
    vegetable_df = chiba_df[vegetable_name]
    if 'ha' not in vegetable_df['作付面積'].columns.values or 't' not in vegetable_df['収穫量'].columns.values:
        continue
    amount_of_crops[vegetable_name] = vegetable_df['収穫量'].t.values
    planted_areas[vegetable_name] = vegetable_df['作付面積'].ha.values


X_cp = X.copy()
scores = []
predictions = []
test_index = []
n_features = len(X.columns)
space = [(1, n_features),  # max_features
         (1, 1e2),  # n_estimators
         (1, 1e2)]  # min_samples_leaf
for vegetable_name in amount_of_crops.keys():
    df = X_cp
    df['作付面積'] = planted_areas[vegetable_name]
    df['収穫量'] = amount_of_crops[vegetable_name]
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['収穫量'], axis=1), df['収穫量'], test_size=0.3)
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
                                        scoring="neg_mean_absolute_error"))
    res_gp = gp_minimize(objective, space, n_calls=15, random_state=0)
    scores.append(res_gp.fun)
    reg.set_params(max_features=res_gp.x[0],
                   n_estimators=res_gp.x[1],
                   min_samples_leaf=res_gp.x[2])
    reg.fit(X_train, y_train)
    predictions.append(reg.predict(X_test))

for i, j, k, l in zip(amount_of_crops.keys(), test_index, scores, predictions):
    print(i, j, k, l)
