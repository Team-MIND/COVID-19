# -*- coding: utf-8 -*-
"""Copy of Copy of Welcome To Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1evJdl55DkHAgA89tzzramoYxZsNjhrA-
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    # Getting our data into a usable form
    df = pd.read_csv('/content/sample_data/us_daily.csv')
    positives = np.flipud(df.iloc[:, 2].values.reshape(-1, 1))
    for i in range(0, positives.size-7):
      positives = np.delete(positives, 0)
    recovered = np.flipud(df.iloc[:, 11].values.reshape(-1, 1))
    for i in range(0, recovered.size-7):
      recovered = np.delete(recovered, 0)
    deaths = np.flipud(df.iloc[:, 14].values.reshape(-1, 1))
    for i in range(0, deaths.size-7):
      deaths = np.delete(deaths, 0)

    # Creating date windows for the past 7 days
    x = np.arange(0,positives.size,1).reshape(-1,1)
    dates_for_positives = pd.DataFrame.from_records(x)
    x = np.arange(0,recovered.size,1).reshape(-1,1)
    dates_for_recovered = pd.DataFrame.from_records(x)
    x = np.arange(0,deaths.size,1).reshape(-1,1)
    dates_for_deaths = pd.DataFrame.from_records(x)

    # Creating linear regression models for each of the 3 paramaters
    positive_lr = LinearRegression()
    positive_lr.fit(dates_for_positives, positives)
    positive_pred = positive_lr.predict(dates_for_positives)
    recovered_lr = LinearRegression()
    recovered_lr.fit(dates_for_recovered, recovered)
    recovered_pred = recovered_lr.predict(dates_for_recovered)
    death_lr = LinearRegression()
    death_lr.fit(dates_for_deaths, deaths)
    deaths_pred = death_lr.predict(dates_for_deaths)

    # Creating the three scatter plots
    plt.scatter(dates_for_positives, positives)
    plt.title("Confirmed Cases Over the Past 7 Days - USA")
    plt.xlabel("Past 7 Days (6 = today)")
    plt.ylabel("Number of Confirmed Cases")
    plt.plot(dates_for_positives, positive_pred, color='red')
    plt.show()
    plt.scatter(dates_for_recovered, recovered)
    plt.title("Number Recovered Over the Past 7 Days - USA")
    plt.xlabel("Past 7 Days (6 = today)")
    plt.ylabel("Number of Recovered Cases")
    plt.plot(dates_for_recovered, recovered_pred, color='red')
    plt.show()
    plt.scatter(dates_for_deaths, deaths)
    plt.title("Number of Deaths Over the Past 7 Days - USA")
    plt.xlabel("Past 7 Days (6 = today)")
    plt.ylabel("Number of Deaths")
    plt.plot(dates_for_deaths, deaths_pred, color='red')
    plt.show()

    # Creating predictions for tomorrow and printing them out
    pred_positive = positive_lr.predict(np.arange(7,8,1).reshape(-1,1))
    pred_recovered = recovered_lr.predict(np.arange(7,8,1).reshape(-1,1))
    pred_deaths = death_lr.predict(np.arange(7,8,1).reshape(-1,1))

    print("There is a predicted %d positive tests in the US on April 15th" % (pred_positive))
    print("There is a predicted %d recovered in the US on April 15th" % (pred_recovered))
    print("There is a predicted %d deaths in the US on April 15th" % (pred_deaths))

if __name__ == "__main__": 
    main()

"""# New Section"""