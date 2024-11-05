import yfinance as yf

data = yf.download('AAPL', start='2024-10-01', end='2024-10-31')

# Save data to csv file
data.to_csv('AAPL.csv')

