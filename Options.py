import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scy
from scipy.stats import norm
import yfinance as yf
import mplfinance as mpl
from fredapi import Fred
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import requests

def black_scholes(S, E, r, T, vol):
    """
    S = Current stock price
    E = Strike price
    r = Current risk free rate
    T = time to expiry (y)
    vol = Current calculated historical volatility
    """
    d1 = (np.log(S/E) + (r + (vol**2)/2)*T) / (vol*np.sqrt(T)) 
    d2 = d1 - vol*np.sqrt(T)

    call_price = S*norm.cdf(d1) - (E * np.exp(-r*T) * norm.cdf(d2))
    put_price = (E * np.exp(-r*T) * norm.cdf(-d2)) - S*norm.cdf(-d1)

    return call_price, put_price

def vega(S, E, r, T, vol):
    """
    Function to calculate vega, the derivative of an option price with respecxt to implied volatility
    S = Current stock price
    E = Strike price
    r = Current risk free rate
    T = time to expiry (y)
    vol = Current calculated implied volatility
    """
    d1 = (np.log(S/E) + (r + (vol**2)/2)*T) / (vol*np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) 

    return vega

def implied_volatility(S, E, r, T, market_price):
    """
    Reverse Black Schole to calculate implied volatility via newton raphson, function is blackscholes - option price. For inital guess could use 0.5*(option price / stock price) or historical volatility if available
    S = Current stock price
    E = Strike price
    r = Current risk free rate
    T = time to expiry (y)
    vol = Implied volatility
    market_price = live market price of option
    """
    vol = 0.3

    for i in range(1000): #No. of times Newton-Raphson is iterated 
        price, _ = black_scholes(S, E, T, r, vol )
        diff = price - market_price 
        
        if abs(diff) < 1e-4: #Tolerance
            return vol 
        
        vol -= diff / vega(S, E, T, r, vol)  # Newton-Raphson step
    
    raise ValueError("Couldn't converge")


def historical_volatility(ticker):
    """
    Reads in historical stock data and calculates annualised historical volatility as standar deviation of the log daily returns 
    ticker = Ticker of stock 
    """
    ts = TimeSeries(key="I928OTG1KGG7WNC9", output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')
    
    close = data['4. close']
    returns = close / close.shift(1)
    log_returns = np.log(returns).dropna()
    daily_vol = np.std(log_returns)
    
    return daily_vol * np.sqrt(252)

def historical_return(ticker):
    """
    Function to retun historical return
    """
    ts = TimeSeries(key="I928OTG1KGG7WNC9", output_format='pandas')
    data, meta_data = ts.get_daily(symbol=ticker, outputsize='compact')
    
    close = data['4. close']
    returns = close / close.shift(1)
    log_returns = np.log(returns).dropna()
    avg = log_returns.mean()
    
    return avg * 252

def get_risk_free_rate():
    """
Function to read in 1 year US bond yield via FRED api
    """
    fred = Fred(api_key='7fa74711b91f4f4e0e2aa6c6b3345a3e')
    r_value = fred.get_series('GS1').iloc[-1]
    
    return r_value / 100

def get_lastest_stock_price(ticker):
    """
    """
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={"I928OTG1KGG7WNC9"}'

    response = requests.get(url)
    data = response.json()
    latest_price = data['Global Quote']['05. price']
    
    return float(latest_price)

def stock_model(S, mu, sigma, T, dt, n_paths):
    """
    Geometric Brownian Motion stock model that outputs price at each step of simulation
    S = Initial stock price
    mu = Expected annual return
    sigma = volatility (implied)
    T = Overall time frame
    dt = Time step
    n_paths = Number of simulated paths
    """
    dt = 1/365
    steps = int(T/dt)
    t = np.linspace(0, T, steps) #start, finish, number of steps

    paths = np.zeros((steps, n_paths))
    paths[0] = S

    for i in range(1, steps):
        z = np.random.standard_normal(n_paths)
        paths[i] = paths[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    return t, paths

def trading_simulation(S, E, T, mu, sigma, market_price):
    """
    Simple simulation that trades a strategy based off of option price, due to the stochastic component of the stock model, could do multiple simulations and average out the outcome to outweigh the brownian motion
    out put graph should be accumulated equity, i.e. summed pnl from exercising option at expirey. PNL will be calculated from final price in paths array outptued by stock model equation.
    When running, use the volatility you belive the stock has, when generating own numbers, use the bs price volatility, not the volatiltiy predicted by the market 
    """
    strat = input("Which strategy would You like to use? (Long/Short):")
    dt = 1/365
    n = 1000 # number of trade iterations
    equity = np.zeros(n)
    equity[0] = 0

    if strat == "Long":
        for i in range(1, n):
            _ , paths = stock_model(S, mu, sigma, T, dt, 1)
            pnl = max((paths[-1] - E), 0) - market_price
            equity[i] = equity[i-1] + pnl
        
    else:
        for i in range(1, n):
            _ , paths = stock_model(S, mu, sigma, T, dt, 1)
            pnl = market_price - max((paths[-1] - E), 0)
            equity[i] = equity[i-1] + pnl
        
    plt.plot(equity)
    plt.title("Accumulated Equity Over 1000 Trades")
    plt.xlabel("Trade Iteration")
    plt.ylabel("Cumulative Equity")
    plt.grid(True)
    plt.show()

    return equity
    
def main():
    """
    Main execution block
    """
    # User inputs
    ticker =input("Enter stock ticker: ")
    E = float(input("Enter option strike price: "))
    market_price = float(input("Enter current option market price: "))
    T = eval(input("Time till option expires (in years): "))
    # Fetching and calculating values
    r = get_risk_free_rate()
    S = get_lastest_stock_price(ticker)
    historical_vol = historical_volatility(ticker)
    mu = historical_return(ticker)
    implied_vol = implied_volatility(S, E, r, T, market_price)
    call_price, _ = black_scholes(S, E, r, T, historical_vol)
    # Printing calculated values
    print(f"Theoretical option price is {call_price:.2f}")
    print(f"Historical volatility is {historical_vol:.2f}")
    print(f"Implied volatility is {implied_vol:.2f}")
    # Outputing acculated equity as value and graph
    equity = trading_simulation(S, E, T, mu, implied_vol, market_price)
    print(f"Accululated equity is {equity[-1]:.2f}")


if __name__ == "__main__":
    main()





# Plot stock model results
#plt.figure(figsize=(12,6))
#for i in range(1): # number of paths is range
 #   plt.plot(t, paths[:, i], lw=1.5)
#plt.title('Simulated Geometric Brownian Motion (Stock Prices)')
#plt.xlabel('Time (Years)')
#plt.ylabel('Stock Price')
#plt.grid(True)
#plt.show()

