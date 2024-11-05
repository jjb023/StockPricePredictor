import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        #Skip first 3 rows
        for i in range(3):
            next(csvFileReader)
        for row in csvFileReader:
            date = datetime.strptime(row[0].split()[0], '%Y-%m-%d')
            dates.append(date.toordinal())  # Convert to a numerical format
            prices.append(float(row[1]))  # Make sure this corresponds to your intended column
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel= 'linear', C= 1e3)
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color= 'black', label= 'Data')
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model')
    plt.plot(dates, svr_lin.predict(dates), color= 'green', label= 'Linear model')
    plt.plot(dates, svr_poly.predict(dates), color= 'blue', label= 'Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('AAPL.csv')
print("Dates:", dates[:5])  # Print first 5 dates
print("Prices:", prices[:5])  # Print first 5 prices


x = np.array([[datetime(2024, 10, 5).toordinal()]])  # Example date
predicted_price = predict_prices(dates, prices, x)

print(predicted_price)


