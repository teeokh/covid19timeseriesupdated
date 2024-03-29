import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def test_stationarity(timeseries):  # Tests the stationarity of time series using ADF test
    rolmean = timeseries.rolling(window=5).mean()
    rolstd = timeseries.rolling(window=5).std()

    orig = plt.plot(timeseries, label="Original")
    mean = plt.plot(rolmean, label="Rolling Mean")
    std = plt.plot(rolstd, label="Rolling std")

    plt.legend(loc="best")
    plt.title("Timeseries data with rolling mean and std")
    plt.show()

    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4],
                         index=["Test Statistic", "p-value", "Number of Lags Used", "Number of Observations Used"])

    print(dfoutput)


dtf = '/Users/terrellokhiria/PycharmProjects/tensorEnv/covid19timeseriesupdated/covid_19_data.csv'

covid_19_dtf = pd.read_csv(dtf)

covid_19_dtf['ObservationDate']=pd.to_datetime(covid_19_dtf['ObservationDate'], infer_datetime_format=True)
idx_covid_19_dtf = covid_19_dtf.set_index(['ObservationDate'])

# print(idx_covid_19_dtf['Country/Region'].unique())
covid_data_UK = idx_covid_19_dtf['Country/Region'] == 'UK'
covid_UK_dataset = idx_covid_19_dtf[covid_data_UK]

covid_UK_confirmed_case_data = covid_UK_dataset['Confirmed']
covid_UK_confirmed_agg_data = covid_UK_confirmed_case_data.groupby(['ObservationDate']).sum()

#test_stationarity(covid_UK_confirmed_agg_data)

covid_UK_confirmed_agg_data_log = np.log(covid_UK_confirmed_agg_data)
#test_stationarity(covid_UK_confirmed_agg_data_log)

ma = covid_UK_confirmed_agg_data_log.rolling(window=5).mean()
covid_UK_confirmed_agg_data_log_minus_ma = covid_UK_confirmed_agg_data_log - ma
covid_UK_confirmed_agg_data_log_minus_ma.dropna(inplace=True)
#test_stationarity(covid_UK_confirmed_agg_data_log_minus_ma)


lag_acf = acf(covid_UK_confirmed_agg_data_log_minus_ma, nlags=32)
lag_pacf = pacf(covid_UK_confirmed_agg_data_log_minus_ma, nlags=16)

fig, ax = plt.subplots(1,2, figsize=(20,5))

plot_acf(lag_acf, ax=ax[0])
plot_pacf(lag_pacf, lags=7, ax=ax[1])

plt.show()