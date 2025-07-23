from click import group
import pandas as pd
import numpy as np
from statsmodels.api import OLS,add_constant
import matplotlib.pyplot as plt

def align_factor(factors):
	'''
	将factors,按照aim_factor完成对齐,factors=[aim_factor,factor1,factor2]
	以第一个df为标准对齐
	需要考虑第一个factor有的日期,后面的factor可能会没有,所以,第一步应该是确保factor[1:]中的日期要有,没有的话向前填充
	'''
	index = factors[0].index
	columns = factors[0].columns
	result = [factors[0]]
	for i in factors[1:]:
		to_append = i.reindex(index=index,method='pad')
		to_append = to_append.reindex(columns=columns)
		result.append(to_append)
	return result


def FactorIC(factor1,factor2,min_valid_num=0):
	'''
	计算factor1与factor2的横截面相关系数(Pearson,Spearman)
	:param factor1(pd.DataFrame):因子1
	:param factor2(pd.DataFrame):因子2
	:param min_valid_num(float):横截面上计算一期相关系数最小的样本个数要求,默认最小是1
	:return pearson_corr,spearman_corr
	'''
	factor1 = factor1.fillna(np.nan)
	factor1 = factor1.astype(float)
	factor2 = factor2.fillna(np.nan)
	factor2 = factor2.astype(float)	
	
	factor1_sum = factor1.notnull().sum(axis=1)
	factor1.loc[factor1_sum<min_valid_num,:]=np.nan
	factor2_sum = factor2.notnull().sum(axis=1)
	factor2.loc[factor2_sum<min_valid_num,:]=np.nan
	
	pearson_corr=factor1.corrwith(factor2,axis=1)	
	spearman_corr=factor1.rank(axis=1).corrwith(factor2.rank(axis=1),axis=1)
	
	return pearson_corr,spearman_corr


def FactorGroup(factor,split_method='average',split_num=5,industry_factor=None,limit_df=None):
	'''
	将因子进行分类,按照行分类。
	:param factor:要分类的因子,或者打分
	:param split_method:默认为等比例分组,'average',还可以有'largest','smallest','largest_ratio','smallest_ratio'
	:param split_num:若split_method=='average',则等分split_num组,若为'largest',则最大n个,若'smallest',则最小n个,若largest_ratio,则最大百分比,若smallest_ratio,则最小百分比
	:param industry_factor(pd.DataFrame or None):行业因子
	:param limit_df(pd.DataFrame or None):None或者TrueFalse构成的df,来自于FactorTool_GetLimitDf的结果
	:return:factor_split_result, df
	'''
	if limit_df is not None:
		[factor,limit_df] = align_factor([factor,limit_df])
		limit_df = limit_df.fillna(value=True).astype('bool')
		factor = factor[limit_df]
		
	if industry_factor is None:
		industry_factor = pd.DataFrame(index=factor.index,columns=factor.columns,data='Market')
		industry_factor = industry_factor[factor.notnull()].astype('object')
	else:
		[factor,industry_factor] = align_factor([factor,industry_factor])
		industry_factor = industry_factor.astype('object')
		industry_factor = industry_factor.fillna(value='others')
		industry_factor = industry_factor[factor.notnull()]
		
	data = pd.DataFrame(index=pd.MultiIndex.from_product([factor.index,factor.columns],names=['date','asset']))
	data['group'] = industry_factor.stack()
	data['factor'] = factor.stack()
	data = data.dropna(subset=['group'])
	
	data_factor_array = data['factor'].values
	data_final_split = np.full((len(data_factor_array),),np.nan)
	
	grouper = [data.index.get_level_values('date'),'group']
	data_groupby = data.groupby(grouper)
	data_groupby_indices = data_groupby.indices
	data_groupby_indices = list(data_groupby_indices.values())
	
	def auxilary_get_split_array(data_factor_array,data_final_split,data_groupby_indices,split_method,split_num):
		def quantile_split(_this_split_result,_this_array,_split_percentile):
			split_value = np.nanpercentile(_this_array,_split_percentile)
			split_value[0] -= 1
			split_value[-1] += 1
			for i in range(len(split_value)-1):
				_this_split_result[(_this_array<=split_value[i+1])&(_this_array>split_value[i])] = i
			return _this_split_result
			
		if split_method=='average':
			split_percentile = np.linspace(0,100,split_num+1)
		elif split_method=='largest_ratio':
			split_percentile = np.array([0,100-split_num*100,100])
		elif split_method=='smallest_ratio':
			split_percentile = np.array([0,split_num*100,100])
		
		for this_group_place in range(len(data_groupby_indices)):
			this_indice_place = data_groupby_indices[this_group_place]
			this_factor_array = data_factor_array[this_indice_place]
			this_split_result = data_final_split[this_indice_place]
			# if split_method in ['average','largest','smallest']:
			if split_method =='average':
				this_data_final_split = quantile_split(this_split_result,this_factor_array,split_percentile)
				data_final_split[this_indice_place] = this_data_final_split
			elif split_method=='smallest':
				this_factor_array_sort = np.sort(this_factor_array[~np.isnan(this_factor_array)])
				split_value = this_factor_array_sort[min(len(this_factor_array_sort)-1,split_num-1)]			
				if len(split_value)>0:
					this_split_result[this_factor_array<=split_value]=0
					this_split_result[this_factor_array>split_value]=1
					data_final_split[this_indice_place] = this_split_result
			elif split_method=='largest':
				this_factor_array_sort = np.sort(this_factor_array[~np.isnan(this_factor_array)])[::-1]
				split_value = this_factor_array_sort[min(len(this_factor_array_sort)-1,split_num-1)]
				if len(split_value)>0:
					this_split_result[this_factor_array<split_value] = 0
					this_split_result[this_factor_array>=split_value] = 1
					data_final_split[this_indice_place] = this_split_result
		return data_final_split
	
	data_final_split = auxilary_get_split_array(data_factor_array,data_final_split,data_groupby_indices,split_method,split_num)	
	data.loc[:,'factor'] = data_final_split	
	final_data = data['factor'].unstack().reindex(index=factor.index,columns=factor.columns)
	
	return final_data


def FactorNeutralize(factor, elifactor):
    # elifactor = [pd.DataFrame(),pd.DataFrame()...]
    '''
    对因子进行中性化处理。
    :param factor: 需要被中性化的因子, pd.DataFrame, 
    :param elifactor: 数值型因子列表, 每个元素为pd.DataFrame,
    :return: pd.DataFrame, 中性化后的因子
    '''
    new_elifactor = []
    for f in elifactor:
        new_elifactor.append(f.reindex(index = factor.index, columns = factor.columns))
    factor_res = pd.DataFrame()
    for idx in factor.dropna(how = 'all', axis=0).index:
        this_factor = factor.loc[idx,:]
        for f in new_elifactor:
            this_elifactor = f.loc[idx,:]
            this_factor = pd.concat([this_factor, this_elifactor], axis=1)
        this_factor = this_factor.astype(float).dropna()
        model = OLS(this_factor.iloc[:,0], add_constant(this_factor.iloc[:,1:]))
        res = model.fit()
        factor_res = pd.concat([factor_res, res.resid.rename(idx)], axis=1)
    return factor_res.T


def IndustryNeutralize(factor, industry_data):
	'''
    对因子进行行业中性化处理。
	:param factor: 需要被中性化的因子, pd.DataFrame,
	:param industry_data: 行业数据, pd.DataFrame, 包含'Code'和'Industry'列
	:return: pd.DataFrame, 中性化后的因子
	'''
	industry_dummies = pd.get_dummies(industry_data.set_index('Code')['Industry'], prefix='Industry')
	industry_dummies = industry_dummies.reindex(columns=factor.columns).fillna(0)
	neutralized_factor = pd.DataFrame(index=factor.index, columns=factor.columns)

	for date in factor.index:
		target = factor.loc[date, :]
		industry = industry_dummies.loc[target.index, :]
		
		data = pd.concat([target, industry], axis=1)
		data = data.dropna()  
		if data.empty or data.shape[1] < 2:  
			continue

		y = data.iloc[:, 0]  
		X = data.iloc[:, 1:]  
		X = add_constant(X)  
		model = OLS(y, X).fit()

		residuals = model.resid
		neutralized_factor.loc[date, residuals.index] = residuals
    
	return neutralized_factor


def simple_factor_test(factor, ret_close):
    '''
    因子回测数据计算,计算因子的IC和RankIC,以及分组收益率
    :param factor: pd.DataFrame, 因子数据
	:param ret_close: pd.DataFrame, 收盘收益率数据
    :return: ic, rankic, group_ret
	'''
    this_ret_data = ret_close.shift(-1)
    ic,rankic = FactorIC(factor,this_ret_data) 
    factor_group = FactorGroup(factor)
    condata = pd.concat([factor_group.unstack(),this_ret_data.unstack()],axis=1).dropna().reset_index()
    condata.columns =['stockcode','date','group_id','ret']
    group_ret = condata.groupby(['date','group_id'])['ret'].mean().unstack()

    factor_statistics(ic, rankic, group_ret)

    return ic,rankic,group_ret


def factor_statistics(ic, rankic, group_ret):
	'''
	计算因子统计数据,包括IC均值和标准差, RankIC均值和标准差, IC t-stat, IC > 0的胜率, IR等
	:param ic: pd.Series, 因子IC
	:param rankic: pd.Series, 因子RankIC
	:param group_ret: pd.DataFrame, 分组收益率
	:return: None
	'''
	ic_mean = ic.mean()
	ic_std = ic.std()
	rankic_mean = rankic.mean()
	rankic_std = rankic.std()

	ir = ic_mean / ic_std

	ic_win_rate = (ic > 0).mean()   

	ic_t_stat = ic_mean / (ic_std / np.sqrt(len(ic.dropna())))
 
	print('--'*30)
	print("因子统计数据")
	print(f"IC Mean: {ic_mean:.2%},\n IC Std: {ic_std:.2%}")
	print(f"Rank IC Mean: {rankic_mean:.2%},\n Rank IC Std: {rankic_std:.2%}")
	print(f"IC t-stat: {ic_t_stat:.2f}")
	print(f"IC > 0 Win Rate: {ic_win_rate:.2%}")
	print(f"IR: {ir:.2f}")
	print('--'*30)
	print(group_ret)


def plot_factor_performance(factor, ret_close, ic, rankic, group_ret):
	'''
    绘制因子表现图, 包括因子值与收益率的散点图、IC和Rank IC的年度统计、累积收益率图、多空组合收益率图等
	:param factor: pd.DataFrame, 因子数据
	:param ret_close: pd.DataFrame, 收盘收益率数据
	:param ic: pd.Series, 因子IC
	:param rankic: pd.Series, 因子RankIC
	:param group_ret: pd.DataFrame, 分组收益率
	:return: None
	'''
	# 绘制因子值和收益率的散点图
	plt.figure(figsize=(10, 6))
	plt.scatter(factor.values.flatten(), ret_close.values.flatten(), alpha=0.5, color='blue')
	plt.xlabel(f'Factor Value ({factor.name})')
	plt.ylabel('Monthly Returns')
	plt.title('Scatter Plot of Factor Values vs Monthly Returns')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()

	# 分别绘制IC和Rank IC的时序图
	plt.figure(figsize=(14, 7))

	# 绘制IC时序图
	plt.subplot(2, 1, 1)
	plt.plot(ic.index, ic, label='IC', color='blue', alpha=0.7)
	ic_ma = ic.rolling(window=21).mean()  # 添加月度移动平均线
	plt.plot(ic_ma.index, ic_ma, label='IC (21-day MA)', color='purple', linestyle='dashed')
	plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
	plt.title('IC Time Series with Moving Average')
	plt.xlabel('Date')
	plt.ylabel('IC Value')
	plt.legend()
	plt.grid(True, alpha=0.3)

	# 绘制Rank IC时序图
	plt.subplot(2, 1, 2)
	plt.plot(rankic.index, rankic, label='Rank IC', color='green', alpha=0.7)
	rankic_ma = rankic.rolling(window=21).mean()  # 添加月度移动平均线
	plt.plot(rankic_ma.index, rankic_ma, label='Rank IC (21-day MA)', color='purple', linestyle='dashed')
	plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
	plt.title('Rank IC Time Series with Moving Average')
	plt.xlabel('Date')
	plt.ylabel('Rank IC Value')
	plt.legend()
	plt.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()

	# IC和Rank IC的年度统计
	ic_by_year = ic.groupby(ic.index.year).agg(['mean', 'std'])
	rankic_by_year = rankic.groupby(rankic.index.year).agg(['mean', 'std'])

	plt.figure(figsize=(16, 6))
	plt.subplot(1, 2, 1)
	plt.bar(ic_by_year.index, ic_by_year['mean'], yerr=ic_by_year['std'], alpha=0.7, color='blue', edgecolor='black', capsize=5)
	plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
	plt.xlabel('Year')
	plt.ylabel('IC Value')
	plt.title('Annual IC Statistics (Mean ± Std)')
	plt.grid(True, alpha=0.3)

	plt.subplot(1, 2, 2)
	plt.bar(rankic_by_year.index, rankic_by_year['mean'], yerr=rankic_by_year['std'], alpha=0.7, color='green', edgecolor='black', capsize=5)
	plt.axhline(0, color='red', linestyle='dashed', linewidth=1)
	plt.xlabel('Year')
	plt.ylabel('Rank IC Value')
	plt.title('Annual Rank IC Statistics (Mean ± Std)')
	plt.grid(True, alpha=0.3)

	plt.tight_layout()
	plt.show()


	# 转换收益率为累积收益率
	cumulative_returns = (1 + group_ret).cumprod() - 1

	# 绘制累积收益率
	plt.figure(figsize=(14, 7))
	for column in cumulative_returns.columns:
		plt.plot(cumulative_returns.index, cumulative_returns[column], label=f'Group {column}')

	plt.title('Cumulative Group Returns Over Time')
	plt.xlabel('Date')
	plt.ylabel('Cumulative Returns')
	plt.legend()
	plt.grid(True)
	plt.show()

	# 计算多空组合收益率 
	group_returns_mean = group_ret.mean()
	long_group = group_returns_mean.idxmax()
	long_position = group_ret[long_group]
	print(f"Long position: Group {long_group}")

	short_group = group_returns_mean.idxmin()
	short_position = group_ret[short_group]
	print(f"Short position: Group {short_group}")

	long_short_returns = long_position - short_position


	# 累计收益率
	cumulative_long_short = (1 + long_short_returns.dropna()).cumprod() - 1

	# 累计收益率图
	plt.figure(figsize=(14, 7))
	plt.plot(cumulative_long_short.index, cumulative_long_short, label= f'Long Group {long_position.name} - Short Group {short_position.name}', color='purple', linewidth=2)
	plt.title('Cumulative Long-Short Returns')
	plt.xlabel('Date')
	plt.ylabel('Cumulative Returns')
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.show()

	# 多空组合月度表现
	monthly_returns = long_short_returns.dropna().resample('ME').sum()
	monthly_returns.index = pd.to_datetime(monthly_returns.index).strftime('%y-%m-%d')

	plt.figure(figsize=(16, 6))
	monthly_returns.plot(kind='bar', color=np.where(monthly_returns > 0, 'red', 'green'))
	plt.title('Monthly Long-Short Returns')
	plt.xlabel('Month')
	plt.ylabel('Returns')
	plt.xticks(rotation=45)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.show()


	# 多空组合年化收益率、波动率、夏普比率和最大回撤
	annual_return = long_short_returns.mean() * 252
	annual_vol = long_short_returns.std() * np.sqrt(252)
	sharpe = annual_return / annual_vol
	max_drawdown = (cumulative_long_short.cummax() - cumulative_long_short).max()

	print('--'*30)
	print("多空组合表现数据")
	print(f"Annual Return: {annual_return:.2%}")
	print(f"Annual Volatility: {annual_vol:.2%}")
	print(f"Sharpe Ratio: {sharpe:.2f}")
	print(f"Max Drawdown: {max_drawdown:.2%}")
	print('--'*30)
