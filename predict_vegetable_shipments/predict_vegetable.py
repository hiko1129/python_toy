# coding: utf-8
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
import math
from scipy import stats

# 作物統計.
# http://www.e-stat.go.jp/SG1/estat/List.do?bid=000001024932&cycode=0

# オープンデータを組み合わせた作物の収穫量予測.
# 気象情報と作物の作付面積から収穫量を予測するモデルを構築.
# 農林水産省の提供している作物統計のみを用いて予測した場合と気象庁が公開している気象情報を組み合わせた場合を比較.
# 公的機関によって公開されている情報のみでも組み合わせることでモデルの構築やモデルの予測精度を向上させることが可能であることを示す.
# 公的機関であるため突然公開されなくなるなどの恐れが少ない.民間企業の場合ビジネスと絡んでいるデータは非公開にされる恐れがある.
# 以前利用できたAPIが利用できなくなった事例は多い.
# 需要が予測可能であるならば、ある時点での作付面積のデータ（統計を取る割合を増やす)と気象情報の長期予報を利用して生産量を増やしてもらうように施策を打つなどして価格の変動
# を抑えることができ、物価安定などの施策が早い段階で打てる.
# バター不足のようなことを防げる可能性がある（野菜においては）.


class Statistics:
    WEATHER_URL = 'http://www.data.jma.go.jp/obd/stats/etrn/view/annually_s.php?prec_no=45&block_no=47682&year=&month=&day=&view='
    AMOUNT_OF_CROP = '収穫量'
    PLANTED_AREA = '作付面積'

    def clean(self, x):
        if isinstance(x, str):
            return x.replace(']', '')

    def get_weather_df(self, start_year, end_year):
        weather_dfs = pd.read_html(Statistics.WEATHER_URL,
                                   skiprows=3,
                                   index_col=0)
        weather_df = weather_dfs[0]
        weather_data_list = []
        for i in weather_df.columns.values:
            weather_data_list.append(
                weather_df[i].apply(self.clean).loc[start_year:end_year].values
            )
        X = pd.DataFrame({
            '降水量': weather_data_list[2],
            '平均気温': weather_data_list[6],
            '平均湿度': weather_data_list[11],
            '平均風速': weather_data_list[13],
            '日照時間': weather_data_list[18]
        }, index=range(start_year, end_year+1), dtype='float')
        return X

    def optimize_using_all_data(self, X, y):
        """収穫量及び、気象情報を用いた生産量の予測."""
        reg = RandomForestRegressor()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)
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

            return -np.mean(cross_val_score(reg, X_train, y_train,
                                            cv=3, n_jobs=-1,
                                            scoring="neg_mean_squared_error"))

        res_gp = gp_minimize(objective, space, n_calls=50, random_state=0, n_jobs=-1)
        return res_gp

    def optimize_only_using_planted_area(self, X, y):
        """収穫量のみを用いた生産量の予測."""
        reg = RandomForestRegressor()
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2)
        space = [(1, 1e3),  # n_estimators
                 (1, 1e3)]  # min_samples_leaf

        def objective(params):
            n_estimators, min_samples_leaf = params

            reg.set_params(n_estimators=n_estimators,
                           min_samples_leaf=min_samples_leaf)

            return -np.mean(cross_val_score(reg,
                                            X_train.values[:, None], y_train,
                                            cv=3, n_jobs=-1,
                                            scoring="neg_mean_squared_error"))

        res_gp = gp_minimize(objective, space, n_calls=50, random_state=0, n_jobs=-1)
        return res_gp

    def create_df(self, filepath, header, weather_period, vegetable_period):
        vegetable_df = pd.read_excel(filepath, header=header)

        vegetable_df = vegetable_df[[Statistics.PLANTED_AREA,
                                     Statistics.AMOUNT_OF_CROP]]
        start_year, end_year = weather_period
        df = self.get_weather_df(start_year, end_year)
        start_year, end_year = vegetable_period
        vegetable_df = vegetable_df.loc[start_year:end_year]
        df[Statistics.PLANTED_AREA] = vegetable_df[Statistics.PLANTED_AREA].astype('float').values
        df[Statistics.AMOUNT_OF_CROP] = vegetable_df[Statistics.AMOUNT_OF_CROP].astype('float').values
        dfs = {'all_data': {'X': df.drop(Statistics.AMOUNT_OF_CROP, axis=1),
                            'y': df[Statistics.AMOUNT_OF_CROP]},
               'only_planted_area': {'X': df[Statistics.PLANTED_AREA],
                                     'y': df[Statistics.AMOUNT_OF_CROP]}}
        return dfs

    def optimize(self, dfs):
        # それぞれのデータを入れるように変更する.
        res_gp_all_data = self.optimize_using_all_data(dfs['all_data']['X'], dfs['all_data']['y'])

        res_gp_only_planted_area = self.optimize_only_using_planted_area(dfs['only_planted_area']['X'], dfs['only_planted_area']['y'])
        return {'all_data': res_gp_all_data,
                'only_planted_area': res_gp_only_planted_area}

    def t_test(self, dfs, all_data_params, only_planted_area_params):
        # 最適化後のパラメーターを用いている→汎化性能の高いモデルを用いてcross_val_scoreを計算している
        # 特定のデータに対してのみ良い値を取るようなことがない.
        max_features, n_estimators, min_samples_leaf = all_data_params
        reg = RandomForestRegressor(max_features=max_features,
                                    n_estimators=n_estimators,
                                    min_samples_leaf=min_samples_leaf)
        train_num = math.floor(dfs['all_data']['X'].shape[0] / 2)
        X, y = dfs['all_data']['X'], dfs['all_data']['y']
        reg.fit(X[:train_num], y[:train_num])
        temp = reg.predict(X[train_num:]) - y[train_num:]

        n_estimators, min_samples_leaf = only_planted_area_params
        reg = RandomForestRegressor(n_estimators=n_estimators,
                                    min_samples_leaf=min_samples_leaf)
        X, y = dfs['only_planted_area']['X'][:, None], dfs['only_planted_area']['y']
        reg.fit(X[:train_num], y[:train_num])
        t, p = stats.ttest_rel(
            temp,
            reg.predict(X[train_num:]) - y[train_num:]
        )
        return (t, p)


if __name__ == '__main__':
    statistics = Statistics()
    dfs = statistics.create_df(filepath='soy.xls',
                               header=[4, 5],
                               weather_period=[1966, 2012],
                               vegetable_period=['昭.41(1966)', '平.24(2012)'])
    res_gp = statistics.optimize(dfs)

    dfs = statistics.create_df(filepath='corn.xls',
                               header=[4, 5],
                               weather_period=[1966, 2012],
                               vegetable_period=['昭.41(1966)', '平.24(2012)'])
    res_gp = statistics.optimize(dfs)

    dfs = statistics.create_df(filepath='wheat.xls',
                               header=[4, 5],
                               weather_period=[1966, 2012],
                               vegetable_period=['昭.41(1966)', '平.24(2012)'])
    res_gp = statistics.optimize(dfs)
