import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import yfinance as yf


dates = []
prices = []

data = yf.download('AAPL', start='2024-01-01', end='2024-11-01')
data.to_csv('AAPL.csv')

with open('AAPL.csv', 'r') as csvfile:
