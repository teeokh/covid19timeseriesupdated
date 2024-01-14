import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):  # Tests the stationarity of time series using ADF test
    rolmean = timeseries.rolling(window=5).mean()
    rolstd = timeseries.rolling(window=5).std()

    orig = plt.plot(timeseries, label="Original")
    mean = plt.plot(timeseries, label="Rolling Mean")
    std = plt.plot(timeseries, label="Rolling std")

    plt.legend(loc="best")
    plt.title("Timeseries data with rolling mean and std")
    plt.show()

    dftest = adfuller(timeseries)
    dfoutput = pd.Series(dftest[0:4],
                         index=["The test statistic", "Mackinnon's approximate p-value", "#usedLags", "NOBS"])

    print(dfoutput)


dtf = '/Users/terrellokhiria/PycharmProjects/tensorEnv/covid19timeseriesupdated/covid_19_data.csv'

covid_19_dtf = pd.read_csv(dtf)

covid_19_dtf['ObservationDate']=pd.to_datetime(covid_19_dtf['ObservationDate'], infer_datetime_format=True)
idx_covid_19_dtf = covid_19_dtf.set_index(['ObservationDate'])
print(idx_covid_19_dtf.head())

