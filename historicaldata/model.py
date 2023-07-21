import pandas as pd
import numpy as np
from finta import TA
import streamlit as st

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics

INDICATORS = [
    "RSI",
    "MACD",
    "STOCH",
    "ADL",
    "ATR",
    "MOM",
    "MFI",
    "ROC",
    "OBV",
    "CCI",
    "EMV",
    "VORTEX",
]


def predictions(data):
    data.rename(
        columns={
            "Close": "close",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Open": "open",
        },
        inplace=True,
    )
    data = _exponential_smooth(data, 0.65)
    data = _get_indicator_data(data)
    data = _produce_prediction(data, window=15)


def _exponential_smooth(data, alpha):
    """
    Function that exponentially smooths dataset so values are less 'rigid'
    :param alpha: weight factor to weight recent values more
    """

    return data.ewm(alpha=alpha).mean()


def _get_indicator_data(data):
    """
    Function that uses the finta API to calculate technical indicators used as the features
    :return:
    """

    for indicator in INDICATORS:
        ind_data = eval("TA." + indicator + "(data)")
        if not isinstance(ind_data, pd.DataFrame):
            ind_data = ind_data.to_frame()
        data = data.merge(ind_data, left_index=True, right_index=True)
    data.rename(columns={"14 period EMV.": "14 period EMV"}, inplace=True)

    # Also calculate moving averages for features
    data["ema50"] = data["close"] / data["close"].ewm(50).mean()
    data["ema21"] = data["close"] / data["close"].ewm(21).mean()
    data["ema15"] = data["close"] / data["close"].ewm(14).mean()
    data["ema5"] = data["close"] / data["close"].ewm(5).mean()

    # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
    data["normVol"] = data["volume"] / data["volume"].ewm(5).mean()

    # Remove columns that won't be used as features
    del data["open"]
    del data["high"]
    del data["low"]
    del data["volume"]
    del data["Adj Close"]

    return data


def _produce_prediction(data, window):
    """
    Function that produces the 'truth' values
    At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
    :param window: number of days, or rows to look ahead to see what the price did
    """

    prediction = data.shift(-window)["close"] >= data["close"]
    prediction = prediction.iloc[:-window]
    data["pred"] = prediction.astype(int)
    cross_Validation
    return data


def _train_random_forest(X_train, y_train, X_test, y_test):
    """
    Function that uses random forest classifier to train the model
    :return:
    """

    # Create a new random forest classifier
    rf = RandomForestClassifier()

    # Dictionary of all values we want to test for n_estimators
    params_rf = {"n_estimators": [110, 130, 140, 150, 160, 180, 200]}

    # Use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)

    # Fit model to training data
    rf_gs.fit(X_train, y_train)

    # Save best model
    rf_best = rf_gs.best_estimator_

    return rf_best


def _train_KNN(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {"n_neighbors": np.arange(1, 25)}

    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)

    # Fit model to training data
    knn_gs.fit(X_train, y_train)

    # Save best model
    knn_best = knn_gs.best_estimator_

    return knn_best


def _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test):
    # Create a dictionary of our models
    estimators = [("knn", knn_model), ("rf", rf_model), ("gbt", gbt_model)]

    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting="hard")

    # fit model to training data
    ensemble.fit(X_train, y_train)

    # test our model on the test data
    print(ensemble.score(X_test, y_test))

    return ensemble


def cross_Validation(data):
    # Split data into equal partitions of size len_train

    num_train = 10  # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 40  # Length of each train-test set

    # Lists to store the results from each model
    rf_RESULTS = []
    knn_RESULTS = []
    ensemble_RESULTS = []

    i = 0
    while True:
        # Partition the data into chunks of size len_train every num_train days
        df = data.iloc[i * num_train : (i * num_train) + len_train]
        i += 1

        if len(df) < 40:
            break

        y = df["pred"]
        features = [x for x in df.columns if x not in ["pred"]]
        X = df[features]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=7 * len(X) // 10, shuffle=False
        )

        rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
        knn_model = _train_KNN(X_train, y_train, X_test, y_test)
        ensemble_model = _ensemble_model(
            rf_model, knn_model, X_train, y_train, X_test, y_test
        )

        rf_prediction = rf_model.predict(X_test)
        knn_prediction = knn_model.predict(X_test)
        ensemble_prediction = ensemble_model.predict(X_test)

        st.write("rf prediction is ", rf_prediction)
        st.write("knn prediction is ", knn_prediction)
        st.write("ensemble prediction is ", ensemble_prediction)
        st.write("truth values are ", y_test.values)
