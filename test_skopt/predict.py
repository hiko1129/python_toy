import numpy as np
from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
# skoptのuserwarningへの対応
warnings.warn('ignore')

TEST_SIZE = 0.3


def calculate_score(estimator, X_train, X_test, y_train, y_test):
    estimator.fit(X_train, y_train)
    return estimator.score(X_test, y_test)


def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return (X_train, X_test)


def iris():
    X_train, X_test, y_train, y_test, default_score = response_common_data(
        load_iris(), SVC()
    )

    def objective(params):
        gamma, C = params
        estimator = SVC()
        estimator.set_params(gamma=gamma, C=C)

        return -np.mean(cross_val_score(estimator, X_train, y_train, cv=5,
                                        n_jobs=-1))
    space = [(1e-2, 1e2),
             (1e-2, 1e2)]
    res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)
    return (default_score, -res_gp.fun)


def boston():
    X_train, X_test, y_train, y_test, default_score = response_common_data(
        load_boston(),
        SVR()
    )

    def objective(params):
        gamma, C = params
        estimator = SVR()
        estimator.set_params(gamma=gamma, C=C)

        return -np.mean(cross_val_score(estimator, X_train, y_train, cv=5,
                                        n_jobs=-1, scoring='r2'))
    space = [(1e-2, 1e2),
             (1e-2, 1e2)]
    res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)
    return (default_score, -res_gp.fun)


def diabetes():
    X_train, X_test, y_train, y_test, default_score = response_common_data(
        load_diabetes(),
        SVR()
    )

    def objective(params):
        gamma, C = params
        estimator = SVR()
        estimator.set_params(gamma=gamma, C=C)

        return -np.mean(cross_val_score(estimator, X_train, y_train, cv=5,
                                        n_jobs=-1, scoring='r2'))
    space = [(1e-2, 1e2),
             (1e-2, 1e2)]
    res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)
    return (default_score, -res_gp.fun)


def digits():
    X_train, X_test, y_train, y_test, default_score = response_common_data(
        load_digits(), SVC()
    )

    def objective(params):
        gamma, C = params
        estimator = SVC()
        estimator.set_params(gamma=gamma, C=C)

        return -np.mean(cross_val_score(estimator, X_train, y_train, cv=5,
                                        n_jobs=-1))
    space = [(1e-2, 1e2),
             (1e-2, 1e2)]
    res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)
    return (default_score, -res_gp.fun)


def response_common_data(dataset, estimator):
    X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                        dataset.target,
                                                        test_size=TEST_SIZE)
    default_score = calculate_score(estimator, X_train, X_test, y_train,
                                    y_test)
    return (X_train, X_test, y_train, y_test, default_score)


if __name__ == '__main__':
    # hydrogen使用
    iris_result = iris()
    iris_result
    boston_result = boston()
    boston_result
    diabetes_result = diabetes()
    diabetes_result
    digits_result = digits()
    digits_result
