#pip install yfinance
#pip install backtrader 
#pip install lxml
import yfinance as yf
import pandas as pd
import numpy as np
import backtrader as bt

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].tolist()
spx = yf.Ticker("^GSPC")
period = "2y"
spx_data = spx.history(period=period)['Close']
data = {}
for ticker in tickers:
    try:
        stock_data = yf.Ticker(ticker).history(period="1y")['Close']
        data[ticker] = stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
df = pd.DataFrame(data)
df = df.dropna(axis=1, how='any')
ratios = df.div(spx_data, axis=0)
ratio_stds = ratios.std()#相对指数涨跌幅标准差
top_20_stocks = ratio_stds.nsmallest(20).index
top_20_ratios = ratios[top_20_stocks]
top_20_ratios.to_csv("top_20_ratios.csv")
print("Top 20 stocks with most stable ratios saved to top_20_ratios.csv")


ratios = pd.read_csv("top_20_ratios.csv", index_col=0, parse_dates=True).mean()
top_20_stocks = ratios.index.to_list()
class NeutralStrategy(bt.Strategy):
    def __init__(self):
        self.spx = self.datas[0]
        self.stocks = self.datas[1:]
        self.ratios = ratios.to_dict()
        self.total_value = pd.Series(name='total_value',dtype=float)
    def next(self):
        self.total_value[self.data.datetime.date(-1)] = self.broker.getvalue()
        spx_close = self.spx.close[0]
        for i, stock in enumerate(self.stocks):
            ticker = stock._name
            stock_close = stock.close[0]
            current_ratio = stock_close / spx_close#相对指数涨跌幅
            threshold = self.ratios[ticker]
            if current_ratio < threshold:
                self.buy(data=stock)
            elif current_ratio > threshold:
                self.sell(data=stock)
    def stop(self):
        self.total_value.plot()
        self.total_value.to_csv('total_value.csv')
cerebro = bt.Cerebro()
data = yf.download("^GSPC", period = period,multi_level_index=False,progress=False)
data = bt.feeds.PandasData(dataname=data)
cerebro.adddata(data)
for ticker in top_20_stocks:
    data = yf.download(ticker, period = period,multi_level_index=False,progress=False)
    data = bt.feeds.PandasData(dataname=data,name = ticker)
    cerebro.adddata(data)
cerebro.addstrategy(NeutralStrategy)
cerebro.broker.setcash(100000)
cerebro.addobserver(bt.observers.Cash)
cerebro.run()
print(f'Final value: {cerebro.broker.getvalue():.2f}')