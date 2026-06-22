import numpy as np
import pandas as pd
import backtrader as bt
from sklearn.svm import SVR
from tqdm import tqdm
import warnings
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# numpy 和 pandas：用于数据处理和数值计算。
# backtrader：用于构建回测框架。
# SVR：支持向量回归模型，用于预测股票市值。
# tqdm：用于显示进度条。
# minimize：用于优化投资组合权重。
# matplotlib：用于绘制图表

warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
plt.switch_backend('agg')

import os
# 获取当前文件的完整路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
print("current_dir",current_dir)


# signlog：对数据进行符号对数变换，用于处理数据的非线性关系。
def signlog(X):
    return np.sign(X) * np.log(1.0 + abs(X))
# reverse_signlog：对符号对数变换后的数据进行逆变换。
def reverse_signlog(X):
    return np.sign(X) * (np.exp(X)-1)

# MarketFundamentalData：继承自 bt.feeds.PandasData，用于加载股票的基本面数据（如市值、净利润、负债等）。
# lines：定义了额外的数据字段（如 shares、assets 等）。
# params：将数据字段映射到 Pandas DataFrame 的列。
class MarketFundamentalData(bt.feeds.PandasData):
    lines = ('shares',
            'assets',
            'liabilities',
            'equity',
            'net_income',
            'total_revenue',
            "liabilitiesequity",
            'cash_and_cash_equivalents','debt','eps',
            'market_capital','enterprise_value','de_ratio','pb_ratio','pe_ratio','bvps')
    
    params = (
        ('datetime',None),
        ('open','open'),
        ('low','low'),
        ('high','high'),
        ('close','close'),
        ('volume','vol'),
        ('market_capital','market_capital'),
        ('net_income','net_income'),
        ('equity','equity'),
        ('liabilitiesequity','liabilitiesequity'),
        ('shares','shares'),
        ('assets','assets'),
        ('liabilities','liabilities'),
        ('total_revenue','total_revenue'),
        ('cash_and_cash_equivalents','cash_and_cash_equivalents'),
        ('debt','debt'),
        ('eps','eps'),
        ('de_ratio','de_ratio'),
        ('pb_ratio','pb_ratio'),
        ('pe_ratio','pe_ratio'),
        ('enterprise_value','enterprise_value'),
        ('bvps','bvps'),
        # ('industry_code',-1),
    )

# 【之前回测收益不好是加上了市盈率、市净率、市销率等干扰因子，去掉之后好很多】

# 策略类
class FactorRotationStrategy(bt.Strategy):
    def __init__(self):
        # FactorRotationStrategy：继承自 bt.Strategy，定义了因子轮动策略。
        # current_factors：当前使用的因子列表。
        # top_n：每次选股的数量。
        # look_back：回看期（用于计算收益率）。
        # current_holdings_index：当前持有的股票索引。
        # total_value：记录总资产价值。
        # high_position：记录权重最高的股票。
        self.current_factors = ['net_income',
                                'total_revenue',
                                'equity',
                                # 'liabilities',
                                "liabilitiesequity",#这个因子无法构建
                                ]#这里使用利润、营收、资产、负债，实际上使用负债率更好
        self.top_n = 10
        self.look_back = 1 #【最稳定】但是收益不如10天高一些
        # self.look_back = 10 #回看时间从1天变成10天收益确实增加了，从100%变成了200%
        # self.look_back = 100 #回看时间从10天变成100天收益明显减少了
        self.current_holdings_index = []
        self.total_value = pd.Series(name='total_value',dtype=float)
        self.high_position = pd.DataFrame(columns=['1','2','3','4','5','6','7','8','9','10'])
    #获取基本面数据
    def get_fund_data(self):
        fields = ['market_capital'] + self.current_factors
        df = pd.DataFrame(columns=fields)
        for data in self.datas:
            row = pd.DataFrame()
            for field in fields:
                row[field] = pd.Series(eval(f'data.{field}[0]'))
            df = pd.concat([df,row],ignore_index=True)
        df = signlog(df)
        return df
    #选股
    def select_stocks(self,df):
        # df=df[df[]]
        y = df['market_capital']
        X = df.drop(['market_capital'],axis=1)
        model = SVR(kernel='rbf').fit(X,y)
        pred = model.predict(X)
        df['residual'] = y - pd.Series(pred.flatten(),y.index)
        selected = df[df['residual']<0].nsmallest(self.top_n,'residual').index#【实际值低于理论值】
        # selected = df[df['residual']>0].nlargest(self.top_n,'residual').index#【实际值高于理论值】
        return selected
    #计算收益率
    def get_return_data(self,selected_index):
        df = pd.DataFrame()
        for i in selected_index:
            df[i] = pd.Series([self.datas[i].close[-j] for j in range(self.look_back)[::-1]])#.pct_change().drop(0)
        df = np.log((df / df.shift(1)).drop(0)) * 252
        return df
    
    def statistics(self,df,weights):
        ret = np.sum(df.mean() * weights)
        var = np.dot(weights.T,np.dot(df.cov(),weights))
        std = np.sqrt(var)
        sharpe = ret / std
        return ret,var,std,sharpe

    def get_weights(self,df,selected_index):
        n =  len(selected_index)
        if len(self.data) < self.look_back:
            return np.ones(n)/n
       
        cons = ({'type':'eq','fun':lambda x: np.sum(x)-1})
        bnds = tuple((0,1) for x in range(n))#都是0.1
        opt = minimize(lambda w: -self.statistics(df,w)[3],np.ones(n)/n,method='SLSQP',bounds=bnds,constraints=cons)
        weights = opt.x
        return weights

    def get_total_position_ratio(self,df,weights):
        rets =  np.dot(df,weights)
        gain = rets[rets>0]
        loss = rets[rets<0]
        a = loss.mean()
        b = gain.mean()
        p = len(gain) / len(rets)
        q = 1 - p
        return - p/a - q/b

    def rebalance_portfolio(self,selected_index,weights):
        for i in self.current_holdings_index:
            # if i not in selected_index:
                self.close(self.datas[i])
        
        # diff = set(selected_index).difference(set(self.current_holdings_index))
        # if len(diff) == 0:
        #     return
        # value = self.broker.getcash() * 0.9 / len(diff)
        # for i in diff:
        #     self.order_target_value(self.datas[i],value)
        #     print(f'Buy {self.datas[i]._name},price: {self.datas[i].open[0]},value: {value}')

        cash = self.broker.getcash()
        for i,w in zip(selected_index,weights):
            # self.order_target_percent(self.datas[i],target=w)
            self.order_target_value(self.datas[i],cash * w)
            # print(cash * w)

        self.current_holdings_index = selected_index

    def record_high_position(self,current_date,selected_index,weights):
        idx = np.argsort(weights)[-10:][::-1]
        row = pd.DataFrame([[f'{self.datas[i]._name}: {j}' for i,j in zip(selected_index.values[idx],weights[idx])]],
                           index=[current_date],columns=['1','2','3','4','5','6','7','8','9','10'])
        self.high_position = pd.concat((self.high_position,row))
            
    def next(self):
        # if current_date.day == 2:
        self.total_value[self.data.datetime.date(-1)] = self.broker.getvalue()
        print(f'Total value: {self.broker.getvalue()}')

        current_date = self.data.datetime.date(0)
        # if current_date.day == 2:
        print(f"Current date: {current_date}")

        if len(self.data) < self.look_back:
            return
        
        train_data = self.get_fund_data()
        selected_index = self.select_stocks(train_data)

        df = self.get_return_data(selected_index)
        weights = self.get_weights(df,selected_index)
        self.record_high_position(current_date,selected_index,weights)
        print("打印权重",weights)

        # position_ratio = self.get_total_position_ratio(df,weights)
        # if position_ratio > 1: position_ratio = 1
        # elif position_ratio < 0: position_ratio = 0
        position_ratio = 0.8
        # print(f'Position ratio: {position_ratio}')

        self.rebalance_portfolio(selected_index,weights * position_ratio)

    def notify_order(self,order):
        if order.status in [order.Submitted,order.Accepted]:
            return
        # if order.status in [order.Completed]:
        #     print('Order Completed')
        # elif order.status in [order.Canceled,order.Margin,order.Rejected]:
        #     print('Order Canceled/Margin/Rejected')
        if order.status in [order.Margin]:
            print('Order Margin')
            print(self.broker.getcash())
        if order.status in [order.Rejected]:
            print('Order Rejected')
        if order.status in [order.Canceled]:
            print('Order Canceled')
        self.order = None

    def stop(self):
        self.total_value.plot()
        self.total_value.to_csv('total_value.csv')
        self.high_position.to_csv('high_position.csv')

def load_sp500_data():
    df = pd.read_csv(rf'{current_dir}/spx500_拼接数据.csv',index_col=0)#需要有这个拼接好的数据，否则无法执行，也就是数据拼接里面生成的【组合后.csv文件】
    # market_capital市值
    # net_income
    # total_revenue
    # equity
    # # liabilities【失效是因为使用负债代替的资产负债率】
    df["liabilitiesequity"]=df["liabilities"]/df["equity"]#需要删掉本地的backtest_data.csv文件
    df = df.groupby('time').apply(lambda x: x[list(df['market_capital'][:516]>0)])
    df.to_csv('backtest_data.csv')
    df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%d')
    datas = []
    for name in tqdm(df['asset'].unique(),desc="Loading data",total=len(df['asset'].unique())):
        data = df[df['asset'] == name].drop(columns=['asset']).set_index('time').interpolate(method='time').fillna(0).replace([np.inf,-np.inf],0)
        datas.append((name,data))
    return datas

def run_backtest():
    cerebro = bt.Cerebro()
    
    sp500_stocks = load_sp500_data()
    for name,data in tqdm(sp500_stocks,desc="Adding data",total=len(sp500_stocks)):
        data = MarketFundamentalData(dataname=data,name=name)
        cerebro.adddata(data)

    cerebro.broker.setcash(1000000.0)
    cerebro.addstrategy(FactorRotationStrategy)
    cerebro.addwriter(bt.WriterFile,csv=True,out='backtest_results.csv')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,_name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown,_name='drawdown')

    # cerebro.broker.setcommission(commission=0.001)
    # cerebro.broker.set_slippage_perc(0.003)

    results = cerebro.run()

    print(f'Final value: {cerebro.broker.getvalue():.2f}')

if __name__ == '__main__':
    run_backtest()