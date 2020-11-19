import requests
import pandas as pd
import scipy
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import PolynomialFeatures
from sklearn import preprocessing
TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"




def predict_price(area):
    area = area.reshape(-1 , 1)
    train_df = pd.read_csv("./linreg_train.csv" , header=None)
    train_df = train_df.T[1:]
    train_df.columns = ["area" , "price"]
    test_df = pd.read_csv("./linreg_test.csv" , header=None)
    test_df = test_df.T[1:]
    test_df.columns = ["area" , "price"]
    x = np.array(train_df["area"]).reshape(-1 , 1)
    y = np.array(train_df["price"]).reshape(-1 , 1)
    x_test = np.array(test_df["area"]).reshape(-1 , 1)
    y_test = np.array(test_df["price"]).reshape(-1 , 1)
    scaler = preprocessing.StandardScaler().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)
    area = scaler.transform(area)
    reg = LinearRegression()
    reg.fit(x , y)
    preds = reg.predict(area)
    return preds


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
