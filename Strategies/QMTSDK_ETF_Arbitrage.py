# pip install loguru pywencai xcsc-tushare

# #测试里面买不了深证的是因为没开相关记录,上证的正常买入没有限制
import os
# 获取当前文件的完整路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
print("current_dir",current_dir)
# 配置日志
basepath=current_dir
# pip install loguru # 这个框架可以解决中文不显示的问题
from loguru import logger
logger.add(
    # sink=f"{basepath}/log.log",#sink: 创建日志文件的路径。
    sink=f"log.log",#sink: 创建日志文件的路径。
    level="INFO",#level: 记录日志的等级,低于这个等级的日志不会被记录。等级顺序为 debug < info < warning < error。设置 INFO 会让 logger.debug 的输出信息不被写入磁盘。
    rotation="00:00",#rotation: 轮换策略,此处代表每天凌晨创建新的日志文件进行日志 IO；也可以通过设置 "2 MB" 来指定 日志文件达到 2 MB 时进行轮换。   
    retention="7 days",#retention: 只保留 7 天。 
    # compression="zip",#compression: 日志文件较大时会采用 zip 进行压缩。
    encoding="utf-8",#encoding: 编码方式
    enqueue=True,#enqueue: 队列 IO 模式,此模式下日志 IO 不会影响 python 主进程,建议开启。
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"#format: 定义日志字符串的样式,这个应该都能看懂。
)

#通过问财获取ETF近期的成交额数据
import datetime
import time
# import math
import pandas as pd#conda install pandas
import numpy as np#pip install numpy
import requests
import json
import math

def symbol_convert(stock):#股票代码加后缀
    #北交所的股票8字开头，包括82、83、87、88，其中82开头的股票表示优先股；83和87开头的股票表示普通股票、88开头的股票表示公开发行的。
    if (stock.startswith("60"))or(#上交所主板
        stock.startswith("68"))or(#上交所科创板
        stock.startswith("11"))or(#上交所可转债
        (stock.startswith("5"))):#上交所ETF：51、52、56、58都是
        return str(str(stock)+".SH")
        # return str(str(stock)+".SS")
    elif (stock.startswith("00"))or(#深交所主板
        stock.startswith("30"))or(#深交所创业板
        stock.startswith("12"))or(#深交所可转债
        (stock.startswith("159"))):#深交所ETF：暂时只有159的是深交所ETF
        return str(str(stock)+".SZ")
    else:
        print("不在后缀转换名录",str(stock))
        return str(str(stock))
 
def getiopv(targetway):#有可能错误的参数会导致报错      获取IOPV相关数据
    if targetway=="东方财富" or targetway=="东财":
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Connection": "keep-alive",
            "Origin": "https://emrnweb.eastmoney.com",
            "Referer": "https://emrnweb.eastmoney.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36 Edg/130.0.0.0",
            "sec-ch-ua": "\"Chromium\";v=\"130\", \"Microsoft Edge\";v=\"130\", \"Not?A_Brand\";v=\"99\"",
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": "\"Android\""
        }
        url = "https://datacenter.eastmoney.com/stock/etfselector/api/data/get"
        params = {
            "type": "RPTA_APP_ETFSELECT",
            "sty": "ETF_TYPE_CODE,DEAL_AMOUNT,SECUCODE,SECURITY_CODE,CHANGE_RATE_1W,CHANGE_RATE_1M,CHANGE_RATE_3M,CHANGE_RATE_12M,ETF_SCALE,YTD_CHANGE_RATE,DEC_TOTALSHARE,DEC_NAV,SECURITY_NAME_ABBR,DERIVE_INDEX_CODE,INDEX_CODE,INDEXNAME,NW_PRICE,CHANGE_RATE,CHANGE,VOLUME,PREMIUM_DISCOUNT_RATIO,QUANTITY_RELATIVE_RATIO,HIGH_PRICE,LOW_PRICE,STOCK_ID,PRE_CLOSE_PRICE,PREMIUM_RATIO,HIGH,LOW,SPEED_UP,STOCK_ID",
            "source": "SECURITIES",
            "client": "APP",
            "filter": "",
            "p": "1",
            # "ps": "30",
            "ps": "2000",#最大值
            "st": "CHANGE_RATE,CHANGE,SECURITY_CODE",
            "sr": "-1,-1,1",
            "isIndexFilter": "0"
        }
        response = requests.get(url, headers=headers, params=params)
        # print(json.loads(response.text)["result"])
        # print(response)
        df=pd.DataFrame(json.loads(response.text)["result"]["data"])
        print(df)


        df=df.rename(columns={
                            "SECURITY_CODE":"code",#不含.sh后缀
                            "SECUCODE":"代码",#含.sh后缀
                            "CHANGE_RATE_1W":"近1周涨幅",#百分数需要处理
                            "CHANGE_RATE_1M":"近1月涨幅",#百分数需要处理
                            "CHANGE_RATE_3M":"近3月涨幅",#百分数需要处理
                            "CHANGE_RATE_12M":"近12月涨幅",#百分数需要处理
                            "last_est_time":"数据时间",
                            "DEAL_AMOUNT":"成交额",
                            "VOLUME":"成交量",
                            "ETF_SCALE":"ETF规模",
                            "DEC_TOTALSHARE":"最新份额",
                            "SECURITY_NAME_ABBR":"名称",
                            "DERIVE_INDEX_CODE":"指数代码",
                            "INDEXNAME":"指数名称",
                            "NW_PRICE":"现价",
                            "CHANGE_RATE":"涨跌幅",#百分数需要处理
                            "CHANGE":"涨跌额",
                            "QUANTITY_RELATIVE_RATIO":"量比",
                            "QUANTITY_RELATIVE_RATIO":"量比",
                            "PRE_CLOSE_PRICE":"前收",
                            "PREMIUM_RATIO":"折价率",#百分数需要处理
                            "HIGH":"最高",
                            "LOW":"最低",
                            "SPEED_UP":"涨速",
                            })
        df=df[df["折价率"]!="-"]#去掉不含溢价率数据的标的【空值干扰】
        df["折价率"]=df["折价率"].apply(lambda x:float(x)/100)
        df["溢价率"]=-df["折价率"]
        # df["涨跌幅"]=df["涨跌幅"].apply(lambda x:float(x)/100)#这种格式就能修改带负号的浮点数了
        df["代码"]=df["code"].apply(lambda x:symbol_convert(str(x)))#生成六位字符串
    return df



def getlist(trade_code,group):
    url = "https://basic.10jqka.com.cn/fundf10/etf/v1/base"
    headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Connection": "keep-alive",
            "Origin": "https://emh5.eastmoney.com",
            "Referer": "https://emh5.eastmoney.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36 Edg/131.0.0.0",
            "sec-ch-ua": "\"Microsoft Edge\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": "\"Android\""
        }
    params = {
        "trade_code": f"{trade_code}",
        "group": group
    }
    response = requests.get(url,headers=headers, params=params)
    response = json.loads(response.text)["data"]
    # print(response,type(response))
    try:
        df=pd.DataFrame(response)
    except:#finFundShareInfo、baseInfo数据本身就只有一层【只有一层的数据直接转dataframe会报错】
        print("直接转换报错，字典可能只有一层")
        response=[response]
        df=pd.DataFrame(response)
    return df



# pip install xcsc-tushare
import xcsc_tushare as ts
def gettradedays(lastday,startday):
    # ts.set_token('7ed61c98882a320cadce6481aef04ebf7853807179d45ee7f72089d7')
    # ts.pro_api(server='http://116.128.206.39:7172')   #指定tocken对应的环境变量，此处以生产为例
    pro = ts.pro_api('7ed61c98882a320cadce6481aef04ebf7853807179d45ee7f72089d7',server='http://116.128.206.39:7172')
    tradedaydf = pro.query('trade_cal',
                    exchange='SSE',#交易所 SZN-深股通,SHN-沪股通,SSE-上海证券交易所,SZSE-深圳证券交易所
                    start_date=lastday,
                    end_date=startday)
    tradedaydf['trade_date'] = pd.to_datetime(tradedaydf['trade_date'], format='%Y%m%d')
    tradedaydf['trade_date'] = tradedaydf['trade_date'].dt.strftime('%Y-%m-%d')
    logger.info(tradedaydf)
    tradedays=tradedaydf["trade_date"].tolist()
    return tradedays

# #需要提前安装node.js抓网页数据，命令行查看node版本node -v，如果没有安装node.js会报错
# node -v
# pip install pywencai -U#python3.7版本调用这个函数就报错：module 'pywencai' has no attribute 'get'
def getetfkline(startday):
    import pywencai#同花顺的最新净值数据不是iopv不能用来计算溢价率
    try:
        df=pd.read_csv(f"{startday}ETFkline_同花顺.csv")
        df=df.iloc[:, 1:]
        df["代码"]=df["code"].apply(lambda x:symbol_convert(str(x)))#生成六位字符串
        logger.info(f"{startday}ETFkline_同花顺.csv","存在")
    except Exception as e:
        logger.info(e)
        logger.info(f"{startday}ETFkline_同花顺.csv","不存在")
        # ETF净值数据【当天】
        # 【申购、赎回费率】跟官网对比了一下问财的最高申购赎回费率是准确的【具体说法是：申购赎回代理券商可按照不超过 0.4%的标准收取佣金】
        # 【价格及净值数据】，不是实时的，
        # 【最近十日成交额】和【最近十日逐日成交额】有差异，但是都是准确的现在用的【最近十日逐日成交额】
        # 【成分股详情】只包含重仓股
        word=f'{startday}所有ETF，最高申购费率、最高赎回费率，最近十日逐日成交额'
        df=pywencai.get(question=word,#query参数
                        loop=True,
                        query_type="fund",
                        # pro=True, #付费版才使用
                        # cookie='xxxx',
                        )
        logger.info(df)
        df["代码"]=df["code"].apply(lambda x:symbol_convert(str(x)))#生成六位字符串
        df.to_csv(f"{startday}ETFkline_同花顺.csv")
    return df



#获取iopv数据，跟上面的成交额数据和费率数据合并【这个清单里面有一些拿出来是空值】
# 特殊字段【"unit_incr":"规模变化",#疑似跟当日已经消耗的申赎额度数量有关系】也未必对主要是有那种申购了之后又赎回的只看规模变化的净值不太准确
# import getiopv#网页里面请求的网络资源当中{:}符号标注的一般是后端的数据库
def getetfinfoandlist(df,startday):
    try:
        etfinfodf=pd.read_csv(f"{startday}ETF申赎额度.csv")
        etflistdf=pd.read_csv(f"{startday}ETF申赎清单.csv")
        logger.info(f"{startday}ETF申赎清单.csv","存在",f"{startday}ETF申赎额度.csv","存在",)
    except Exception as e:
        logger.info(e)
        logger.info(f"{startday}ETF申赎清单.csv","不存在",f"{startday}ETF申赎额度.csv","不存在",)
        # now=datetime.datetime.now()
        # thistime=now.strftime("%Y-%m-%d")#上交所
        # thattime=now.strftime("%Y%m%d")#深交所
        etfinfodf=pd.DataFrame({})
        etflistdf=pd.DataFrame({})
        for symbol in df["代码"].tolist():
            try:
                logger.info("获取申赎清单和申赎详情",symbol)
                logger.info(symbol)
                index=symbol.replace(".SZ","").replace(".SH","")
                logger.info(index,type(index))
                #【申赎详情】
                etfinfo=getlist(trade_code=index,group="finFundShareInfo")#申赎详情
                etfinfo=etfinfo.rename(columns={
                    "subRedListPreDate":"昨日净值份额更新时间",#最小申购、赎回单位的现金差额，最小申购、赎回单位净值，基金份额净值
                    "subRedListMinUnitNetAssetValue":"最小申购、赎回单位净值",
                    "subRedListCashDifference":"最小申购、赎回单位的现金差额",
                    "subRedListFundUnitNetValue":"基金份额净值",
                    "subRedListShouldPubIopv":"是否需要公布IOPV",#1代表是
                    "subRedListMinSubRedUnit":"最小申购、赎回单位",
                    "subRedListCashSubLimit":"现金替代比例上限",#需要除以100
                    "subRedListAllowedSub":"是否允许申购",
                    "subRedListAllowedRed":"是否允许赎回",
                    "subRedListDailySubLimit":"当日累计申购上限",#无或者具体数额
                    "subRedListDailyRedLimit":"当日累计赎回上限",
                    "subRedListEstimatedCash":"预估现金部分",
                    "subRedListDate":"当日申赎详情更新时间",
                })
                #【申赎清单】
                etflist=getlist(trade_code=index,group="subRedListStock")#成分股列表
                etflist=etflist.rename(columns={
                    "market":"交易所",#33深证，17上证
                    "cashSubDisPct":"赎回折价比例",#需要除以100
                    "thscodeHq":"同花顺行情代码",
                    "stockRate":"市值占比",#需要除以100
                    "stockName":"成分股简称",
                    "cashSubPremiumPct":"现金替代溢价比例",
                    "endDate":"更新时间",
                    "fixedSubAmt":"固定替代金额",#上交所标的该值为空
                    "cashSubFlag":"现金替代标志",#上证为允许、深京为退补
                    "stockNum":"股票数量",
                    "seccode":"股票代码",
                    # "secType":"",#不清楚什么意思【上证深证均为001001】
                })
                etfinfo["代码"]=symbol
                print(etfinfo)
                etflist["代码"]=symbol
                print(etflist)#新上市的159363会返回空值导致后续报错
                #【拼接前先验证是否有非A股标的】
                etflist['全A股标的'] = etflist['股票代码'].apply(lambda x: '是' if (isinstance(x, str) and len(x) == 6) else '否')
                if (etflist['全A股标的'] == '否').any():
                    etflist['全A股标的'] = '否'#有一个不是A股标的,则A股标的列整列的值重置为否
                #【成分股列表的股票代码需要单独生成六位字符串】
                etflist["股票代码"]=etflist["股票代码"].apply(lambda x:symbol_convert(str(x)))
                #【数据拼接】
                etfinfodf=pd.concat([etfinfodf,etfinfo])
                etflistdf=pd.concat([etflistdf,etflist])
                # #【数据输出】
                # etfinfodf.to_csv(f"{startday}ETF申赎额度.csv")
                # etflistdf.to_csv(f"{startday}ETF申赎清单.csv")
            except Exception as e:
                print("标的刚刚上市数据为空值",e)
        etfinfodf.to_csv(f"{startday}ETF申赎额度.csv")
        etflistdf.to_csv(f"{startday}ETF申赎清单.csv")
    
    #【目前{现金替代标志}有{退补}、{允许}两种】
    etflistdf=etflistdf[etflistdf["全A股标的"]=="是"]#只保留成分股全都是A股标的的ETF
    droplistdf=etflistdf.copy()#过滤存在申赎风险【现金替代标志为必须的情况或者说其实就是涨停了】
    droplist=droplistdf[(droplistdf['现金替代标志']=='必须')]["代码"].unique().tolist()#【有一些要求必须现金替代的不在考虑范围内】
    print("需删减的ETF",len(droplist))#存在申赎风险敞口的情况，但是这样未必就不能交易，只要买入申购没涨停，赎回卖出没跌停就行
    etflistdf.loc[etflistdf['代码'].isin(droplist),"是否存在现金替代风险敞口"]="是"

    #【计算剩余额度】不设上限、无、xxxx份
    etfinfodf.loc[etfinfodf["当日累计申购上限"]=="不设上限","当日累计申购上限"]="0"
    etfinfodf.loc[etfinfodf["当日累计赎回上限"]=="不设上限","当日累计赎回上限"]="0"
    etfinfodf.loc[etfinfodf["当日累计申购上限"]=="无","当日累计申购上限"]="0"
    etfinfodf.loc[etfinfodf["当日累计赎回上限"]=="无","当日累计赎回上限"]="0"
    #.replace("份","")无法解决异常数据1700000-------的问题，这里改成去掉所有非数字的字符串
    etfinfodf["当日累计申购上限"]=etfinfodf["当日累计申购上限"].str.extract(r'(\d+)',expand=False).astype(float)
    etfinfodf["当日累计赎回上限"]=etfinfodf["当日累计赎回上限"].str.extract(r'(\d+)',expand=False).astype(float)
    #【注意在这里又份额转换为金额了】
    etfinfodf["当日累计申购金额上限"]=etfinfodf["当日累计申购上限"]*etfinfodf["基金份额净值"].astype(float)#最小申购赎回单位（份），当日累计申购上限（份）
    etfinfodf["当日累计赎回金额上限"]=etfinfodf["当日累计赎回上限"]*etfinfodf["基金份额净值"].astype(float)#最小申购赎回单位（份），当日累计赎回上限（份）
    etfinfodf.loc[etfinfodf["当日累计申购金额上限"]==0,"当日累计申购上限"]="不设上限"
    etfinfodf.loc[etfinfodf["当日累计赎回金额上限"]==0,"当日累计赎回上限"]="不设上限"
    etfinfodf=etfinfodf[etfinfodf["代码"].isin(etflistdf["代码"].unique().tolist())]#这里也只要成分股都是A股的部分
    return etfinfodf,etflistdf

#_________________________________________________________________________________________________________________________________
#设置交易参数并且获取买卖计划
bidrate=0.005#设置盘口价差为0.004
timecancellwait=60#设置撤单函数筛选订单的确认时间
# ETF_timecancellwait=600#设置在ETF进行申购赎回的时候撤单函数筛选订单的确认时间
timetickwait=600#设置每次下单时确认是否是最新tick的确认时间【3秒一根，但是模拟盘的tick滞后五分钟左右】
timeseconds=60#设置获取tick的函数的时间长度【避免没有数据】
targetmoney=20000#设置下单时对手盘需要达到的厚度（即单笔目标下单金额,因为手数需要向下取整,所以实际金额比这个值低）
traderate=2#设置单次挂单金额是targetmoney的traderate倍
# cancellorder=False#取消一分钟不成交或者已成交金额达到目标值自动撤单并回补撤单金额的任务
cancellorder=True#设置一分钟不成交或者已成交金额达到目标值自动撤单并回补撤单金额的任务


while True:
    thistime=datetime.datetime.now()
    print(thistime,datetime.time(14,30))
    if thistime.time()>=datetime.time(9,30):
        now=datetime.datetime.now()
        startday=now.strftime("%Y%m%d")
        # lastday=(now-datetime.timedelta(days=365)).strftime("%Y%m%d")
        lastday=(now-datetime.timedelta(days=60)).strftime("%Y%m%d")
        #获取交易日期列表【湘财证券tushare】
        tradedays=gettradedays(lastday,startday)#获取指定范围内的交易日list

        #获取ETF标的的近期交易量详情df【同花顺问财】
        df=getetfkline(startday)
        #【总耗时{非交易日}大概32分钟】etfinfodf申赎清单,etflistdf申赎额度【爬虫爬取上交所、深交所】申赎清单和额度信息从10:55执行到了11:23,大概耗时30分钟
        etfinfodf,etflistdf=getetfinfoandlist(df,startday)#过滤条件是成分股都是A股、没有必须要求现金替代的成分股，所以数量会比较少
        print("etfinfodf,etflistdf",len(etfinfodf),len(etflistdf))
        etfinfodf["基金代码无后缀"]=etfinfodf["代码"].replace(".SH","").replace(".SZ","")
        etflistdf["基金代码无后缀"]=etflistdf["代码"].replace(".SH","").replace(".SZ","")
        etflistdf["股票代码无后缀"]=etflistdf["股票代码"].replace(".SH","").replace(".SZ","")
        #过滤最近十天的成交额【主要是对同花顺数据进行处理】
        for day in tradedays[-10:]:
            logger.info(day.replace('-',''),startday)
            if day.replace('-','')!=startday:#跳过当前这一天的数据过滤
                logger.info(day)
                df[f"{day.replace('-','')}日成交额"]=df[f"基金@成交额[{day.replace('-','')}]"].astype(float)
                df=df.sort_values(by=f"{day.replace('-','')}日成交额",ascending=False)#成交额降序排列
                df=df[df[f"{day.replace('-','')}日成交额"]>10000000]#卡在最近十天每天成交额大于一个亿从999只变成了418只
        # logger.info(df)
        df=df[["基金代码","基金简称","code","基金@最高申购费率","基金@最高赎回费率"]]
        df["code"]=df["code"].astype(str)
        df["代码"]=df["code"].str.zfill(6).apply(lambda x:symbol_convert(x)).astype(str)#需要指定类型为字符串
        logger.info(df)

        # #【启动交易引擎】勾选独立交易之后，行情、交易、交易+行情选项一个都不要选择，才能启动miniqmt成功，否则无法执行订单
        # xtdata提供和MiniQmt的交互接口,本质是和MiniQmt建立连接,由MiniQmt处理行情数据请求,再把结果回传返回到python层。使用的行情服务器以及能获取到的行情数据和MiniQmt是一致的,要检查数据或者切换连接时直接操作MiniQmt即可。
        # 对于数据获取接口,使用时需要先确保MiniQmt已有所需要的数据,如果不足可以通过补充数据接口补充,再调用数据获取接口获取。
        # 对于订阅接口,直接设置数据回调,数据到来时会由回调返回。订阅接收到的数据一般会保存下来,同种数据不需要再单独补充。
        from xtquant import xtdata
        #交易模块
        import random
        from xtquant.xttype import StockAccount
        from xtquant.xttrader import XtQuantTrader
        from xtquant import xtconstant
        ##from cancel_retrade import cancel_retrade
        
        # QMT账号
        # mini_qmt_path = r"D:\迅投极速交易终端 睿智融科版\userdata_mini"# miniQMT安装路径
        # account_id = "2011506"# QMT账号
        # account_id = "2011908"
        # mini_qmt_path = r"D:\国金QMT交易端模拟\userdata_mini"# miniQMT安装路径
        mini_qmt_path = r"D:\Program Files\国金QMT交易端模拟\bin.x64\XtItClient.exe"# miniQMT安装路径
        account_id = "55013189"
        if (account_id=='55013189')or(account_id=='2011506')or(account_id=="2011908"):#密码:wth000
            # choosename="可转债"
            choosename="微盘股"
            tradeway="taker"#设置主动吃单
            # tradeway="maker"#设置被动吃单
        else:
            choosename="微盘股"
            tradeway="taker"#设置主动吃单
        #【启动QMT】
        
        session_id = int(random.randint(100000,999999))# 创建session_id
        trade_api = XtQuantTrader(mini_qmt_path,session_id)# 创建交易对象
        trade_api.start()# 启动交易对象

        while True:
            connect_result = trade_api.connect()# 连接客户端
            logger.info("连接结果",connect_result)
            if connect_result==0:
                logger.info("连接成功")
                break
            else:
                logger.info("重新链接")
                time.sleep(1)
        acc = StockAccount(account_id)# 创建账号对象
        trade_api.subscribe(acc)# 订阅账号


        #查询资产
        portfolio=trade_api.query_stock_asset(account=acc)
        logger.info(f"查询资产,portfolio")#收盘之后估计会返回空值
        available_cash=portfolio.cash#available_cash可用资金
        market_value=portfolio.market_value#market_value证券市值
        frozen_cash=portfolio.frozen_cash#frozen_cash冻结资金
        total_value=portfolio.total_asset#total_asset总资产
        logger.info(f"******"+"可用资金"+str(available_cash)+"证券市值"+str(market_value)+"冻结资金"+str(frozen_cash)+"总资产"+str(total_value))

        # 【其他注意：现金替代需要t+2结算，存在涨停股补券的风险，遇到有涨停的或者需要现金替代的不要买就好，等不涨停了自己买】
        # 【特殊情况：格力电器停牌之后，存在买入深F120ETF兑换出来格力电器的机会（深F120在4月1日~4月8日期间，格力电器替代状态均未变化。），同期如159901、159961ETF，在4月2日开始格力电器调整为必须现金替代。】
        
        #策略主体
        while True:
            #每一轮交易都需要获取一次
            iopvdf=getiopv(targetway="东方财富")#获取实时iopv
            logger.info(iopvdf)
            iopvdf["code"]=iopvdf["code"].astype(str)
            #【拼接iopv和问财的量价的数据】
            df=df.merge(iopvdf,on=["代码","code"],how="inner")
            df=df.sort_values(by='溢价率',ascending=False)
            df["基金@最高申购费率"]=df["基金@最高申购费率"].astype(float)
            df["基金@最高赎回费率"]=df["基金@最高赎回费率"].astype(float)
            # df.loc[df["溢价率"]>0,"申购费后利润率"]=abs(df["溢价率"])-df["基金@最高申购费率"]/100#原则上申购费率在国金证券可以降低到0
            # df.loc[df["溢价率"]<0,"赎回费后利润率"]=abs(df["溢价率"])-df["基金@最高赎回费率"]/100#原则上申购费率在国金证券可以降低到0
            #TODO:在程序运行过程当中出现过几次“溢价率”kryerror的情况，但重复运行程序之后问题又消失了
            df.loc[df["溢价率"]>0,"申购费后利润率"]=abs(df["溢价率"])-0.002#原则上申购费率在国金证券可以降低到0【但是滑点按0.002】
            df.loc[df["溢价率"]<=0,"申购费后利润率"]=-1#默认方向错误的记做亏完
            df.loc[df["溢价率"]<0,"赎回费后利润率"]=abs(df["溢价率"])-0.002#原则上申购费率在国金证券可以降低到0【但是滑点按0.002】
            df.loc[df["溢价率"]>=0,"赎回费后利润率"]=-1#默认方向错误的记做亏完

            #iopv数据尽量使用东方财富（集思录不准确）
            etfinfodf=etfinfodf[etfinfodf["最小申购、赎回单位净值"]<100*10**4]#只要最小赎回单位净值小于100w的
            df=df.merge(etfinfodf,on=["代码"],how="inner")#跟ETF申赎详情，只拼接两边都有的数据

            # df.to_csv("申赎数据.csv")#【ETF总池子过少会导致这里变成空值（没有符合要求的ETF）】
            #【{研究}确认是否还有申购余额{通过总额度-实时申赎额度=当日剩余额度，判断是否可以执行交易}】ETF申赎只做不限制额度的方向或者挨个测试【其实应该根据同花顺的申购赎回详情页是否会从允许改成不允许判断是否还有余额】
            #TODO:下面溢价率这个“>”有的时候会报错，应该是数据类型不太规范导致的
            df.loc[((df["当日累计赎回金额上限"]*0.99>df["成交额"])|(df["当日累计赎回上限"]=="不设上限")),"是否达到赎回上限"]="否"
            df.loc[((df["当日累计申购金额上限"]*0.99>df["成交额"])|(df["当日累计申购上限"]=="不设上限")),"是否达到申购上限"]="否"
            df.loc[~(df["是否达到申购上限"]=="否"),'申购费后利润率']=None
            df.loc[~(df["是否达到赎回上限"]=="否"),'赎回费后利润率']=None
            


            #【主要用于处理单笔下单金额】这里使用沪深交易所申赎清单里面的数据【之前使用集思录数据的时候直接用就行】
            df["最小申购、赎回单位"]=df["最小申购、赎回单位"].astype(float)#将最小申购赎回单位转浮点数
            # df["申赎利润率"]=np.maximum(df['申购费后利润率'], df['赎回费后利润率'])#判断申赎利润率和申购赎回方向
            df["申赎利润率"]=df[['申购费后利润率','赎回费后利润率']].max(axis=1)#判断申赎利润率和申购赎回方向，判断做哪一边更划算
            df.loc[df['申购费后利润率']>=df['赎回费后利润率'],"申赎方向"]="申购"#申购
            df.loc[df['赎回费后利润率']>=df['申购费后利润率'],"申赎方向"]="赎回"#赎回
            df=df.sort_values(by="申赎利润率",ascending=False)#申赎利润率由大到小排序
            
            #【折溢价率达到targetrate才执行交易】
            targetrate=0.001#目标利润率
            df=df[df["申赎利润率"]>targetrate]#【前面扣了千分之二的费用了】只要申赎利润率足够的标的
            df.to_csv("申赎利润率.csv")#后面才能判断成分券是否涨停跌停，是否可以执行套利
            if len(df)>0:#【】
                #【下单是使用无后缀代码进行下单的】
                #【这里需要对涨跌停金额进行验证再决定是否执行交易】判断代替金额
                #同花顺内打出来的数据（字符串数据）【申赎清单当中上证和深证的替代金额表示方式不同】
                for index,thisdf in df.iterrows():#遍历所有ETF
                    #判断申购赎回方向【鼠绘时】
                    thisincome=thisdf["申赎方向"]
                    thisrate=thisdf["申赎利润率"]#浮点数
                    #TODO:但从运行结果来看好像symbolsymbol还是有后缀，但好在下单的时候格式要求带后缀，所以也不用动了
                    symbolsymbol=thisdf["基金代码无后缀"]#其实事实上还是有后缀
                    print(symbolsymbol)
                    #TODO:从上交所深交所的ETF申赎记录来看，ETF的最小申赎份数差别比较大，但是最小申赎单位的IOPV基本都是100w左右
                    #TODO:如果按100M资金，每笔交易100w的情况考虑，基本只能够得上热门ETF的最小申赎单位，所以只用下一单就行
                    mix_unit = int(thisdf["最小申购、赎回单位"])#最少申赎单位
                    thislistdf=etflistdf[etflistdf["基金代码无后缀"]==symbolsymbol]#包含在指定ETF内的所有成分股组成的list

                    #赎回验证涨跌停【成分券】
                    goodnum=0#统计涨停标的
                    badnum=0#统计跌停标的
                    #TODO:在运行过程当中发现159900.SZ总是出现，而且根本查不到对应的个股或者ETF，所以设置了except去捕捉这个keyerror错误
                    try:#收集到错误股票代码159900.SZ
                        for index,thisdf in thislistdf.iterrows():#遍历同一ETF内的所有成分股【将同一ETF内的所有个股记录在案】这个循环单纯是记录用的
                            thisstock=thisdf["股票代码无后缀"]#成分股代码
                            thisvolume=thisdf["股票数量"]#成分股对应的数量
                            tick=xtdata.get_full_tick([thisstock])
                            # print(tick)
                            tick=tick[thisstock]#临近收盘没数据了
                            ask_price_1=tick["askPrice"][0]
                            ask_volume_1=tick["askVol"][0]
                            ask_price_2=tick["askPrice"][1]
                            ask_volume_2=tick["askVol"][1]
                            bid_price_1=tick["bidPrice"][0]
                            bid_volume_1=tick["bidVol"][0]
                            bid_price_2=tick["bidPrice"][1]
                            bid_volume_2=tick["bidVol"][1]
                            logger.info(f"{thisstock},tick{tick},{ask_price_1},{ask_volume_1},{ask_price_2},{ask_volume_2},{bid_price_1},{bid_volume_1},{bid_price_2},{bid_volume_2}")
                            if (ask_price_2==0)and(ask_price_1==0):
                                logger.info(f"{thisstock},成分券涨停")#无法买入
                                goodnum+=1
                            if (bid_price_2==0)and(bid_price_1==0):
                                logger.info(f"{thisstock},成分券跌停")#无法卖出
                                badnum+=1

                        symbol=thisdf["代码"]#symbol是ETF带后缀的名字
                        logger.info("symbol",type(thisrate),symbol)
                    #TODO:设置了except去捕捉这个keyerror错误
                    except KeyError:
                        logger.info(f"发生了 keyerror错误，股票代码{thisstock}错误")
                        continue#收集到错误股票代码自然无法继续，看下一个ETF


                    #开始套利
                    if thisincome=="申购":
                        if thisrate>targetrate:#只做高溢价率的标的
                            logger.info(f"{symbolsymbol}申购费后利润率较大，适合申购套利")
                            print(thislistdf)
                            if goodnum==0:#没有涨停的才执行赎回交易
                                
                                for index,thisdf in thislistdf.iterrows():#遍历所有属于同一个ETF内的成分股 【买入同一个ETF下的所有成分股】
                                    thisstock=thisdf["股票代码无后缀"]#第一个成分股
                                    thisvolume=thisdf["股票数量"]#一份ETF所含的数量
                                    #正股下单
                                    #返回五档数据
                                    tick=xtdata.get_full_tick([thisstock])
                                    tick=tick[thisstock]
                                    ask_price_1=tick["askPrice"][0]
                                    ask_volume_1=tick["askVol"][0]
                                    ask_price_2=tick["askPrice"][1]
                                    ask_volume_2=tick["askVol"][1]
                                    bid_price_1=tick["bidPrice"][0]
                                    bid_volume_1=tick["bidVol"][0]
                                    bid_price_2=tick["bidPrice"][1]
                                    bid_volume_2=tick["bidVol"][1]
                                    logger.info(f"tick{tick},{ask_price_1},{ask_volume_1},{ask_price_2},{ask_volume_2},{bid_price_1},{bid_volume_1},{bid_price_2},{bid_volume_2}")
                                    #TODO:对这个下单函数的参数进行了修改，因为我认为这个函数是用来遍历然后购买所有ETF包含的成分股的，所以code写成了thisstock,
                                    #TODO:因为我看基本ETF最小申购单位包含的个股股数净值不大，所以没必要分批下单，就算是一笔下单也不会对市场价格有巨大冲击
                                    buyorder=trade_api.order_stock(acc, stock_code=thisstock,
                                                        order_type=xtconstant.STOCK_BUY,
                                                        order_volume=thisvolume,
                                                        price_type=xtconstant.FIX_PRICE,#限价
                                                        strategy_name=choosename,#策略名称
                                                        price=ask_price_1)#按照卖一价格买入
                                    logger.info(f"下单成功{buyorder}")
                                    buyorder0 = buyorder
                                    #TODO:我设置了每个个股在购买之后进行验证，验证是否购买成功，价格是否变动过大需要重新下单（大体复制的是原函数赎回部分的验证部分）
                                    # #【这里要验证一下是否都买到了，没买到的话需要撤单重新下】
                                    #______________________________________________________________________
                                    while True:  # 【撤单处理这里有问题不能这样处理】
                                        holdvolume = 0
                                        positions = trade_api.query_stock_positions(account=acc)  # 获取持仓列表
                                        for position in positions:
                                            possymbol = position.stock_code  # 获取对应股票代码
                                            logger.info("stock", thisstock, "possymbol", possymbol)
                                            if thisstock == possymbol:
                                                logger.info(thisstock, position.volume)
                                                holdvolume = position.volume
                                                logger.info("当前持仓数量", holdvolume)
                                        if holdvolume == thisvolume:
                                            logger.info("当前个股持仓数量达标ETF要求数量买入计划结束")
                                            
                                        else:
                                            orderalls = trade_api.query_stock_orders(account=acc, cancelable_only=False)  # 仅查询可撤委托
                                            for orderall in orderalls:
                                                # #模拟盘下午无法识别到撤单（orderall.status_msg无数据）把这块拿出来单独研究
                                                # logger.info(f"{orderall},{type(orderall.offset_flag)},{orderall.direction},{orderall.price_type},{orderall.order_id}")
                                                # 账号状态(account_status)
                                                # xtconstant.ORDER_UNREPORTED	48	未报
                                                # xtconstant.ORDER_WAIT_REPORTING	49	待报
                                                # xtconstant.ORDER_REPORTED	50	已报
                                                # xtconstant.ORDER_REPORTED_CANCEL	51	已报待撤
                                                # xtconstant.ORDER_PARTSUCC_CANCEL	52	部成待撤
                                                # xtconstant.ORDER_PART_CANCEL	53	部撤
                                                # xtconstant.ORDER_CANCELED	54	已撤
                                                # xtconstant.ORDER_PART_SUCC	55	部成
                                                # xtconstant.ORDER_SUCCEEDED	56	已成
                                                # xtconstant.ORDER_JUNK	57	废单【这个也得算金额】
                                                # xtconstant.ORDER_UNKNOWN	255	未知
                                                # 拼接orderall的数据【不对已成（56）、待报（49）、未报（48）订单进行处理】大部分是54已撤、55部成、56已成、57废单
                                                #TODO:为了避免出现在核对某个个股委托状态的时候查到其他个股，加了一步核对订单编号是否与下单的订单编号一致
                                                if buyorder0['order_id'] == orderall.order_id:
                                                    if (orderall.order_status != int(56)) and (orderall.order_status != int(49)) and (orderall.order_status != int(48)):
                                                        dforderall = pd.DataFrame({
                                                            "order_status": [orderall.order_status],
                                                            "order_id": [orderall.order_id],
                                                            "status_msg": [orderall.status_msg],
                                                            "symbol": [orderall.stock_code],
                                                            "amount": [orderall.order_volume],
                                                            "trade_amount": [orderall.traded_volume],
                                                            "trade_price": [orderall.traded_price],
                                                            "order_type": [orderall.order_type],  # int,24卖出,23买入
                                                            "direction": [orderall.direction],  # int,多空方向,股票不需要；参见数据字典
                                                            "offset_flag": [orderall.offset_flag],  # int,交易操作,用此字段区分股票买卖,期货开、平仓,期权买卖等；参见数据字典
                                                            "price": [orderall.price],
                                                            "price_type": [orderall.price_type],
                                                            "datetime": [datetime.datetime.fromtimestamp(orderall.order_time).strftime("%Y%m%d %H:%M:%S")],
                                                            "secondary_order_id": [orderall.order_id]
                                                        })
                                                        dforderalls = pd.concat([dforderalls, dforderall], ignore_index=True)
                                                        if (orderall.order_status == int(55)) or (orderall.order_status == int(50)):
                                                            logger.info(f"******,不是已成交订单,{orderall.order_id}")
                                                            # 60秒内不成交就撤单【这个是要小于当前时间,否则就一直无法执行】
                                                            if (datetime.datetime.fromtimestamp(orderall.order_time) + datetime.timedelta(seconds=timecancellwait)) < datetime.datetime.now():  # 成交额还得超过targetmoney才可以最终撤单
                                                                if (orderall.traded_volume * orderall.price > targetmoney):
                                                                    try:
                                                                        cancel_result = trade_api.cancel_order_stock(account=acc, order_id=orderall.order_id)
                                                                        # .cancel_order(orderall.order_id)
                                                                        logger.info(f"******,已成交金额达标执行撤单,{orderall.order_id, cancel_result}")
                                                                    except:
                                                                        logger.info(f"******", "已完成或取消中的条件单不允许取消")
                                                                elif orderall.traded_volume == 0:  # 未成交撤单
                                                                    try:  # 如果该委托已成交或者已撤单则会报错
                                                                        cancel_result = trade_api.cancel_order_stock(account=acc, order_id=orderall.order_id)
                                                                        # .cancel_order(orderall.order_id)
                                                                        logger.info(f"******,执行撤单,{orderall.order_id}, cancel_result,{cancel_result}")
                                                                    except:
                                                                        logger.info(f"******,已完成或取消中的条件单不允许取消")
                                                        else:  # 撤单或者废单之后的金额回补
                                                            # 交易操作(offset_flag)
                                                            # 枚举变量名	值	含义
                                                            # xtconstant.OFFSET_FLAG_OPEN	48	买入,开仓
                                                            # xtconstant.OFFSET_FLAG_CLOSE	49	卖出,平仓
                                                            # xtconstant.OFFSET_FLAG_FORCECLOSE	50	强平
                                                            # xtconstant.OFFSET_FLAG_CLOSETODAY	51	平今
                                                            # xtconstant.OFFSET_FLAG_ClOSEYESTERDAY	52	平昨
                                                            # xtconstant.OFFSET_FLAG_FORCEOFF	53	强减
                                                            # xtconstant.OFFSET_FLAG_LOCALFORCECLOSE	54	本地强平
                                                            if (orderall.order_type == int(23)):  # 这里只计算BUY方向的订单,24是卖23是买
                                                                # logger.info("该订单是买入")
                                                                # time.sleep(10)
                                                                if (orderall.order_status == int(54)):
                                                                    thiscancel_amount = orderall.order_volume - orderall.traded_volume
                                                                    logger.info(f"{orderall}")
                                                                    logger.info(f"******,撤单成功,{orderall},{orderall.order_status},{thiscancel_amount}")
                                                                    if dfordercancelled.empty:  # dfordercancelled一开始是个空值,这里主要是确认一下之前有没有数据,有数据才需要检验之前是否撤销过
                                                                        dfordercancelled = pd.concat([dfordercancelled, dforderall], ignore_index=True)
                                                                        cancel_money = thiscancel_amount  # 然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                        retradenum += cancel_money
                                                                    else:
                                                                        if orderall.order_id not in dfordercancelled["order_id"].tolist():
                                                                            dfordercancelled = pd.concat([dfordercancelled, dforderall], ignore_index=True)
                                                                            cancel_money = thiscancel_amount  # 然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                            retradenum += cancel_money
                                                                elif (orderall.order_status == int(57)):
                                                                    logger.info(f"******,废单处理,{orderall},{orderall.order_status},{orderall.order_volume}")
                                                                    if dfordercancelled.empty:  # dfordercancelled一开始是个空值,这里主要是确认一下之前有没有数据,有数据才需要检验之前是否撤销过
                                                                        dfordercancelled = pd.concat([dfordercancelled, dforderall], ignore_index=True)
                                                                        cancel_money = orderall.order_volume  # 然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                        retradenum += cancel_money
                                                                    if orderall.order_id not in dfordercancelled["order_id"].tolist():
                                                                        dfordercancelled = pd.concat([dfordercancelled, dforderall], ignore_index=True)
                                                                        cancel_money = orderall.order_volume  # 然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                        retradenum += cancel_money
                                        tradestatus = False  # 初始化交易状态【判断是否可以执行申赎交易】为False
                                        for num in range(0, 100):
                                            if retradenum <= 0:  # 如果已经下单结束就不再执行下单任务
                                                logger.info("下单完成结束下单进程")
                                                logger.info("剩余应下单数量", retradenum)
                                                tradestatus = True
                                                break
                                            
                                            else:
                                                volume = retradenum
                                                retradenum -= volume
                                                logger.info("撤单重下数量", volume)
                                            
                                            #这个位置不能按10w一笔下单，某些个股没有到10w

                                            # 正股下单【需要根据单笔最大金额确定需要下单的次数】
                                            tick = xtdata.get_full_tick([thisstock])
                                            tick = tick[thisstock]
                                            ask_price_1 = tick["askPrice"][0]
                                            ask_volume_1 = tick["askVol"][0]
                                            ask_price_2 = tick["askPrice"][1]
                                            ask_volume_2 = tick["askVol"][1]
                                            bid_price_1 = tick["bidPrice"][0]
                                            bid_volume_1 = tick["bidVol"][0]
                                            bid_price_2 = tick["bidPrice"][1]
                                            bid_volume_2 = tick["bidVol"][1]
                                            logger.info(f"tick{tick},{ask_price_1},{ask_volume_1},{ask_price_2},{ask_volume_2},{bid_price_1},{bid_volume_1},{bid_price_2},{bid_volume_2}")

                                            if (ask_price_2 == 0) and (ask_price_1 == 0):
                                                logger.info(f"{symbol},涨停不参与交易")  # 无法买入
                                                retradenum = 0
                                            else:
                                                buyorder0 = trade_api.order_stock(acc, stock_code=thisstock,
                                                                                order_type=xtconstant.STOCK_BUY,
                                                                                order_volume=volume,
                                                                                price_type=xtconstant.FIX_PRICE,  # 限价
                                                                                strategy_name=choosename,  # 策略名称
                                                                                price=ask_price_1)
                                            logger.info(f"下单成功{buyorder0}")

                                        time.sleep(1)  # 休息【避免空转】
                                        if tradestatus == True:
                                            logger.info("完成撤单再交易")
                                            break
                                #for循环(遍历买入所有成分股的程序的位置)______________________________________________________________________(cancel_retrade完成)
                                    
                                #【成分股购买完成，开始申购ETF份额】
                                #返回五档数据
                                #TODO:添加了ETF申购部分，购买玩所有包含个股之后，可以提交申购申请
                                tick=xtdata.get_full_tick([symbolsymbol])
                                tick=tick[symbolsymbol]
                                ask_price_1=tick["askPrice"][0]
                                ask_volume_1=tick["askVol"][0]
                                ask_price_2=tick["askPrice"][1]
                                ask_volume_2=tick["askVol"][1]
                                bid_price_1=tick["bidPrice"][0]
                                bid_volume_1=tick["bidVol"][0]
                                bid_price_2=tick["bidPrice"][1]
                                bid_volume_2=tick["bidVol"][1]
                                logger.info(f"tick{tick},{ask_price_1},{ask_volume_1},{ask_price_2},{ask_volume_2},{bid_price_1},{bid_volume_1},{bid_price_2},{bid_volume_2}")
                                #【ETF申购】实盘ETF申赎API需要开通ETF申赎权限（认证专业投资者）【申赎下单的时候报价为空值，与传入的值无关，后面会在新的订单上自动生成一个成交价格作为订单查询时回报的成交价格】
                                purchaseorder=trade_api.order_stock(acc, stock_code=symbolsymbol,
                                                    order_type=xtconstant.ETF_PURCHASE,#ETF申购
                                                    order_volume=mix_unit,#按照申购单位的最小值进行申购（其实基本最小单位净值都在百万级了）
                                                    price_type=xtconstant.FIX_PRICE,#限价
                                                    strategy_name=choosename,#策略名称
                                                    price=ask_price_1)
                                logger.info(f"申购下单成功{purchaseorder}")
                                purchaseorder0 = purchaseorder

                                #【检查一下是否申购成功】——————————————————————————————————————————————————————————————————————
                                #TODO:在申购函数下单完成后，通过查看持仓状态验证是否成功
                                while True:
                                    # 验证持仓当中的该ETF是否已经申购成功
                                    positions = trade_api.query_stock_positions(account=acc)
                                    goodnum = 0
                                    for position in positions:
                                        if symbolsymbol == position.stock_code:
                                            print(position.stock_code, "position.volume", position.volume)
                                            if position.volume == mix_unit:
                                                print("全部申购成功停止等待", symbol)
                                                goodnum += 1
                                    if goodnum != 0:  #
                                        break
                                    time.sleep(1)


                                #————————————————————————————————————————————————————————————————————————————————————————————
                                #TODO:验证完ETF申购成功之后，通过卖出ETF份额完成套利
                                #【卖出申购的ETF份额】完成套利
                                tick=xtdata.get_full_tick([symbolsymbol])
                                tick=tick[symbolsymbol]
                                ask_price_1=tick["askPrice"][0]
                                ask_volume_1=tick["askVol"][0]
                                ask_price_2=tick["askPrice"][1]
                                ask_volume_2=tick["askVol"][1]
                                bid_price_1=tick["bidPrice"][0]
                                bid_volume_1=tick["bidVol"][0]
                                bid_price_2=tick["bidPrice"][1]
                                bid_volume_2=tick["bidVol"][1]
                                logger.info(f"tick{tick},{ask_price_1},{ask_volume_1},{ask_price_2},{ask_volume_2},{bid_price_1},{bid_volume_1},{bid_price_2},{bid_volume_2}")
                                sellorder=trade_api.order_stock(acc, stock_code=symbolsymbol,
                                                    order_type=xtconstant.STOCK_SELL,
                                                    order_volume=mix_unit,
                                                    price_type=xtconstant.FIX_PRICE,#限价
                                                    strategy_name=choosename,#策略名称
                                                    price=bid_price_1)#按照买一价格卖出
                                logger.info(f"下单成功{sellorder}")
                                sellorder0=sellorder
                                # #【这里要验证一下是否都卖掉了，没买到的话需要撤单重新下】
                                #______________________________________________________________________
                                #TODO:验证ETF是否卖出成功，这里也是通过订单编号限制检查范围，通过委托状态进行等待或撤单并重新下单
                                while True:  
                                    holdvolume = 0
                                    positions = trade_api.query_stock_positions(account=acc)  # 获取持仓列表
                                    for position in positions:
                                        possymbol = position.stock_code  # 获取对应股票代码
                                        logger.info("symbolsymbol", symbolsymbol, "possymbol", possymbol)
                                        if symbolsymbol == possymbol:
                                            logger.info(symbolsymbol, position.volume)
                                            holdvolume = position.volume
                                            logger.info("当前持仓数量", holdvolume)
                                    if holdvolume == 0:
                                        logger.info("当前ETF卖出计划结束")
                                        
                                    else:
                                        orderalls = trade_api.query_stock_orders(account=acc, cancelable_only=False)  # 仅查询可撤委托
                                        for orderall in orderalls:
                                            # #模拟盘下午无法识别到撤单（orderall.status_msg无数据）把这块拿出来单独研究
                                            # logger.info(f"{orderall},{type(orderall.offset_flag)},{orderall.direction},{orderall.price_type},{orderall.order_id}")
                                            # 账号状态(account_status)
                                            # xtconstant.ORDER_UNREPORTED	48	未报
                                            # xtconstant.ORDER_WAIT_REPORTING	49	待报
                                            # xtconstant.ORDER_REPORTED	50	已报
                                            # xtconstant.ORDER_REPORTED_CANCEL	51	已报待撤
                                            # xtconstant.ORDER_PARTSUCC_CANCEL	52	部成待撤
                                            # xtconstant.ORDER_PART_CANCEL	53	部撤
                                            # xtconstant.ORDER_CANCELED	54	已撤
                                            # xtconstant.ORDER_PART_SUCC	55	部成
                                            # xtconstant.ORDER_SUCCEEDED	56	已成
                                            # xtconstant.ORDER_JUNK	57	废单【这个也得算金额】
                                            # xtconstant.ORDER_UNKNOWN	255	未知
                                            # 拼接orderall的数据【不对已成（56）、待报（49）、未报（48）订单进行处理】大部分是54已撤、55部成、56已成、57废单
                                            if sellorder0['order_id'] == orderall.order_id:
                                                if (orderall.order_status != int(56)) and (orderall.order_status != int(49)) and (orderall.order_status != int(48)):
                                                    dforderall = pd.DataFrame({
                                                        "order_status": [orderall.order_status],
                                                        "order_id": [orderall.order_id],
                                                        "status_msg": [orderall.status_msg],
                                                        "symbol": [orderall.stock_code],
                                                        "amount": [orderall.order_volume],
                                                        "trade_amount": [orderall.traded_volume],
                                                        "trade_price": [orderall.traded_price],
                                                        "order_type": [orderall.order_type],  # int,24卖出,23买入
                                                        "direction": [orderall.direction],  # int,多空方向,股票不需要；参见数据字典
                                                        "offset_flag": [orderall.offset_flag],  # int,交易操作,用此字段区分股票买卖,期货开、平仓,期权买卖等；参见数据字典
                                                        "price": [orderall.price],
                                                        "price_type": [orderall.price_type],
                                                        "datetime": [datetime.datetime.fromtimestamp(orderall.order_time).strftime("%Y%m%d %H:%M:%S")],
                                                        "secondary_order_id": [orderall.order_id]
                                                    })
                                                    dforderalls = pd.concat([dforderalls, dforderall], ignore_index=True)
                                                    if (orderall.order_status == int(55)) or (orderall.order_status == int(50)):
                                                        logger.info(f"******,不是已成交订单,{orderall.order_id}")
                                                        # 60秒内不成交就撤单【这个是要小于当前时间,否则就一直无法执行】
                                                        if (datetime.datetime.fromtimestamp(orderall.order_time) + datetime.timedelta(seconds=timecancellwait)) < datetime.datetime.now():  # 成交额还得超过targetmoney才可以最终撤单
                                                            if (orderall.traded_volume * orderall.price > targetmoney):
                                                                try:
                                                                    cancel_result = trade_api.cancel_order_stock(account=acc, order_id=orderall.order_id)
                                                                    # .cancel_order(orderall.order_id)
                                                                    logger.info(f"******,已成交金额达标执行撤单,{orderall.order_id, cancel_result}")
                                                                except:
                                                                    logger.info(f"******", "已完成或取消中的条件单不允许取消")
                                                            elif orderall.traded_volume == 0:  # 未成交撤单
                                                                try:  # 如果该委托已成交或者已撤单则会报错
                                                                    cancel_result = trade_api.cancel_order_stock(account=acc, order_id=orderall.order_id)
                                                                    # .cancel_order(orderall.order_id)
                                                                    logger.info(f"******,执行撤单,{orderall.order_id}, cancel_result,{cancel_result}")
                                                                except:
                                                                    logger.info(f"******,已完成或取消中的条件单不允许取消")
                                                    else:  # 撤单或者废单之后的金额回补
                                                        # 交易操作(offset_flag)
                                                        # 枚举变量名	值	含义
                                                        # xtconstant.OFFSET_FLAG_OPEN	48	买入,开仓
                                                        # xtconstant.OFFSET_FLAG_CLOSE	49	卖出,平仓
                                                        # xtconstant.OFFSET_FLAG_FORCECLOSE	50	强平
                                                        # xtconstant.OFFSET_FLAG_CLOSETODAY	51	平今
                                                        # xtconstant.OFFSET_FLAG_ClOSEYESTERDAY	52	平昨
                                                        # xtconstant.OFFSET_FLAG_FORCEOFF	53	强减
                                                        # xtconstant.OFFSET_FLAG_LOCALFORCECLOSE	54	本地强平
                                                        if (orderall.order_type == int(24)):  # 这里只计算SELL方向的订单,24是卖23是买
                                                            # logger.info("该订单是买入")
                                                            # time.sleep(10)
                                                            if (orderall.order_status == int(54)):
                                                                thiscancel_amount = orderall.order_volume - orderall.traded_volume
                                                                logger.info(f"{orderall}")
                                                                logger.info(f"******,撤单成功,{orderall},{orderall.order_status},{thiscancel_amount}")
                                                                if dfordercancelled.empty:  # dfordercancelled一开始是个空值,这里主要是确认一下之前有没有数据,有数据才需要检验之前是否撤销过
                                                                    dfordercancelled = pd.concat([dfordercancelled, dforderall], ignore_index=True)
                                                                    cancel_money = thiscancel_amount  # 然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                    retradenum += cancel_money
                                                                else:
                                                                    if orderall.order_id not in dfordercancelled["order_id"].tolist():
                                                                        dfordercancelled = pd.concat([dfordercancelled, dforderall], ignore_index=True)
                                                                        cancel_money = thiscancel_amount  # 然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                        retradenum += cancel_money
                                                            elif (orderall.order_status == int(57)):
                                                                logger.info(f"******,废单处理,{orderall},{orderall.order_status},{orderall.order_volume}")
                                                                if dfordercancelled.empty:  # dfordercancelled一开始是个空值,这里主要是确认一下之前有没有数据,有数据才需要检验之前是否撤销过
                                                                    dfordercancelled = pd.concat([dfordercancelled, dforderall], ignore_index=True)
                                                                    cancel_money = orderall.order_volume  # 然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                    retradenum += cancel_money
                                                                if orderall.order_id not in dfordercancelled["order_id"].tolist():
                                                                    dfordercancelled = pd.concat([dfordercancelled, dforderall], ignore_index=True)
                                                                    cancel_money = orderall.order_volume  # 然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                    retradenum += cancel_money
                                    tradestatus = False  # 初始化交易状态【判断是否可以执行申赎交易】为False
                                    for num in range(0, 100):
                                        if retradenum <= 0:  # 如果已经下单结束就不再执行下单任务
                                            logger.info("下单完成结束下单进程")
                                            logger.info("剩余应下单数量", retradenum)
                                            tradestatus = True
                                            break
                                        
                                        else:
                                            volume = retradenum
                                            retradenum -= volume
                                            logger.info("撤单重下数量", volume)
                                        #TODO:分批下单避免对市场价格造成冲击
                                        if retradenum > 100000:#卖出ETF的过程避免对盘面造成冲击
                                            volume = 100000  # 每次1000手10w股
                                            retradenum -= 100000
                                            logger.info("剩余应下单数量", retradenum)

                                        # 正股下单【需要根据单笔最大金额确定需要下单的次数】
                                        tick = xtdata.get_full_tick([symbolsymbol])
                                        tick = tick[symbolsymbol]
                                        ask_price_1 = tick["askPrice"][0]
                                        ask_volume_1 = tick["askVol"][0]
                                        ask_price_2 = tick["askPrice"][1]
                                        ask_volume_2 = tick["askVol"][1]
                                        bid_price_1 = tick["bidPrice"][0]
                                        bid_volume_1 = tick["bidVol"][0]
                                        bid_price_2 = tick["bidPrice"][1]
                                        bid_volume_2 = tick["bidVol"][1]
                                        logger.info(f"tick{tick},{ask_price_1},{ask_volume_1},{ask_price_2},{ask_volume_2},{bid_price_1},{bid_volume_1},{bid_price_2},{bid_volume_2}")

                                        if (bid_price_2 == 0) and (bid_price_1 == 0):
                                            logger.info(f"{symbolsymbol},跌停不参与交易")  # 无法买入
                                            retradenum = 0
                                        else:
                                            sellorder0 = trade_api.order_stock(acc, stock_code=symbolsymbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=volume,#每次按照数量卖出
                                                                            price_type=xtconstant.FIX_PRICE,  # 限价
                                                                            strategy_name=choosename,  # 策略名称
                                                                            price=bid_price_1)#按照买一价格卖出
                                        logger.info(f"下单成功{sellorder0}")

                                    time.sleep(1)  # 休息【避免空转】
                                    if tradestatus == True:
                                        logger.info("完成撤单再交易")
                                        break
                                #______________________________________________________________________
                            else:
                                logger.info("成分券存在涨停板不宜执行交易")

            #----------------------------------------------------------------------------------------------------------------
                    elif thisincome=="赎回":
                        if thisrate>targetrate:#只做高溢价率的标的
                            logger.info("赎回费后利润率较大，适合赎回套利")
                            print(thislistdf)
                            if badnum==0:#没有跌停的才执行赎回交易
                            # 后面才能判断成分券是否涨停跌停，是否可以执行套利
                                try:
                                    #返回五档数据
                                    tick=xtdata.get_full_tick([symbolsymbol])
                                    tick=tick[symbolsymbol]
                                    ask_price_1=tick["askPrice"][0]
                                    ask_volume_1=tick["askVol"][0]
                                    ask_price_2=tick["askPrice"][1]
                                    ask_volume_2=tick["askVol"][1]
                                    bid_price_1=tick["bidPrice"][0]
                                    bid_volume_1=tick["bidVol"][0]
                                    bid_price_2=tick["bidPrice"][1]
                                    bid_volume_2=tick["bidVol"][1]
                                    logger.info(f"tick{tick},{ask_price_1},{ask_volume_1},{ask_price_2},{ask_volume_2},{bid_price_1},{bid_volume_1},{bid_price_2},{bid_volume_2}")
                                    if (ask_price_1==0)and(ask_price_2==0):
                                        logger.info(f"{symbolsymbol},ETF涨停不参与交易")#无法买入
                                    else:
                                        #股票下单
                                        allvolume=mix_unit#下单的单位是股数不是手数【软件里最大是100万股】#最小下单单位mixunit
                                        #减去当前持仓当中该ETF的数量
                                        holdvolume=0
                                        positions=trade_api.query_stock_positions(account=acc)
                                        for position in positions:
                                            possymbol=position.stock_code
                                            logger.info("symbolsymbol",symbolsymbol,"possymbol",possymbol)
                                            if symbolsymbol==possymbol:
                                                logger.info(symbol,position.volume)
                                                holdvolume=position.volume
                                                logger.info("当前持仓数量",holdvolume)#获得持仓数
                                        thisvolume=allvolume-holdvolume#减去当前持仓

                                        #金额处理
                                        thismoney=ask_price_1*thisvolume#买1*目标量=目标成交额
                                        cashmoney=trade_api.query_stock_asset(account=acc).cash#可用资金余额
                                        logger.info(thismoney,type(thismoney),cashmoney,type(cashmoney))
                                        if cashmoney>thismoney*1.1:#余额是最小下单金额的1.1倍
                                            logger.info("余额充足适合下单")
                                            for num in range(0,1000):
                                                if thisvolume<=0:#如果已经下单结束就不再执行下单任务
                                                    logger.info("下单完成结束下单进程")
                                                    logger.info("剩余应下单数量",thisvolume)
                                                    break
                                                if thisvolume>100000:
                                                    volume=100000#每次1000手10w股
                                                    thisvolume-=100000
                                                    logger.info("剩余应下单数量",thisvolume)
                                                else:
                                                    volume=thisvolume
                                                    thisvolume-=volume
                                                    logger.info("剩余应下单数量",thisvolume)
                                                #ETF分批下单
                                                buyorder=trade_api.order_stock(acc, stock_code=symbolsymbol,
                                                                    order_type=xtconstant.STOCK_BUY,
                                                                    order_volume=volume,
                                                                    price_type=xtconstant.FIX_PRICE,#限价
                                                                    strategy_name=choosename,#策略名称
                                                                    price=ask_price_1)
                                                logger.info(f"下单成功{buyorder}")
                                                buyorder0 = buyorder

                                            #【未成交的撤单重新下】
                                            retradenum=0#重置应下单数量
                                            timecancellwait=10#timecancellwait秒不成交就撤单
                                            targetmoney=20000#对手盘需要达到的厚度
                                            
                                            #_________________________________________________________________
                                            #TODO:这里验证的时候也是按照订单编号限制的检查范围
                                            while True:
                                                holdvolume=0
                                                positions=trade_api.query_stock_positions(account=acc)#获取持仓列表
                                                for position in positions:
                                                    possymbol=position.stock_code#获取对应股票
                                                    logger.info("stock",symbolsymbol,"possymbol",possymbol)
                                                    if symbolsymbol==possymbol:
                                                        logger.info(symbolsymbol,position.volume)
                                                        holdvolume=position.volume
                                                        logger.info("当前ETF持仓数量",holdvolume)
                                                if holdvolume==allvolume:#最后thisvolume总归还是0,还得是用allvolume
                                                    logger.info("当前持仓数量达标ETF买入计划结束")
                                                else:
                                                    orderalls = trade_api.query_stock_orders(account=acc,cancelable_only=False)#仅查询可撤委托
                                                    for orderall in orderalls:
                                                        if buyorder0['order_id'] == orderall.order_id:
                                                        # #模拟盘下午无法识别到撤单（orderall.status_msg无数据）把这块拿出来单独研究
                                                        # logger.info(f"{orderall},{type(orderall.offset_flag)},{orderall.direction},{orderall.price_type},{orderall.order_id}")
                                                        # 账号状态(account_status)
                                                        # xtconstant.ORDER_UNREPORTED	48	未报
                                                        # xtconstant.ORDER_WAIT_REPORTING	49	待报
                                                        # xtconstant.ORDER_REPORTED	50	已报
                                                        # xtconstant.ORDER_REPORTED_CANCEL	51	已报待撤
                                                        # xtconstant.ORDER_PARTSUCC_CANCEL	52	部成待撤
                                                        # xtconstant.ORDER_PART_CANCEL	53	部撤
                                                        # xtconstant.ORDER_CANCELED	54	已撤
                                                        # xtconstant.ORDER_PART_SUCC	55	部成
                                                        # xtconstant.ORDER_SUCCEEDED	56	已成
                                                        # xtconstant.ORDER_JUNK	57	废单【这个也得算金额】
                                                        # xtconstant.ORDER_UNKNOWN	255	未知
                                                        #拼接orderall的数据【不对已成（56）、待报（49）、未报（48）订单进行处理】大部分是54已撤、55部成、56已成、57废单
                                                            if ((orderall.order_status!=int(56))and(orderall.order_status!=int(49))and(orderall.order_status!=int(48))):
                                                                dforderall=pd.DataFrame({
                                                                    "order_status":[orderall.order_status],
                                                                    "order_id":[orderall.order_id],
                                                                    "status_msg":[orderall.status_msg],
                                                                    "symbol":[orderall.stock_code],
                                                                    "amount":[orderall.order_volume],
                                                                    "trade_amount":[orderall.traded_volume],
                                                                    "trade_price":[orderall.traded_price],
                                                                    "order_type":[orderall.order_type],#int,24卖出,23买入
                                                                    "direction":[orderall.direction],#int,多空方向,股票不需要；参见数据字典
                                                                    "offset_flag":[orderall.offset_flag],#int,交易操作,用此字段区分股票买卖,期货开、平仓,期权买卖等；参见数据字典
                                                                    "price":[orderall.price],
                                                                    "price_type":[orderall.price_type],
                                                                    "datetime":[datetime.datetime.fromtimestamp(orderall.order_time).strftime("%Y%m%d %H:%M:%S")],
                                                                    "secondary_order_id":[orderall.order_id]})
                                                                dforderalls=pd.concat([dforderalls,dforderall],ignore_index=True)
                                                                if ((orderall.order_status==int(55))or(orderall.order_status==int(50))):
                                                                    logger.info(f"******,不是已成交订单,{orderall.order_id}")
                                                                    #60秒内不成交就撤单【这个是要小于当前时间,否则就一直无法执行】
                                                                    if (datetime.datetime.fromtimestamp(orderall.order_time)+datetime.timedelta(seconds=timecancellwait))<datetime.datetime.now():#成交额还得超过targetmoney才可以最终撤单
                                                                        if (orderall.traded_volume*orderall.price>targetmoney):
                                                                            try:
                                                                                cancel_result = trade_api.cancel_order_stock(account=acc,order_id=orderall.order_id)
                                                                                # .cancel_order(orderall.order_id)
                                                                                logger.info(f"******,已成交金额达标执行撤单,{orderall.order_id,cancel_result}")
                                                                            except:
                                                                                logger.info(f"******","已完成或取消中的条件单不允许取消")
                                                                        elif orderall.traded_volume==0:#未成交撤单
                                                                            try:#如果该委托已成交或者已撤单则会报错
                                                                                cancel_result = trade_api.cancel_order_stock(account=acc,order_id=orderall.order_id)
                                                                                # .cancel_order(orderall.order_id)
                                                                                logger.info(f"******,执行撤单,{orderall.order_id},cancel_result,{cancel_result}")
                                                                            except:
                                                                                logger.info(f"******,已完成或取消中的条件单不允许取消")
                                                                else:#撤单或者废单之后的金额回补
                                                                    # 交易操作(offset_flag)
                                                                    # 枚举变量名	值	含义
                                                                    # xtconstant.OFFSET_FLAG_OPEN	48	买入,开仓
                                                                    # xtconstant.OFFSET_FLAG_CLOSE	49	卖出,平仓
                                                                    # xtconstant.OFFSET_FLAG_FORCECLOSE	50	强平
                                                                    # xtconstant.OFFSET_FLAG_CLOSETODAY	51	平今
                                                                    # xtconstant.OFFSET_FLAG_ClOSEYESTERDAY	52	平昨
                                                                    # xtconstant.OFFSET_FLAG_FORCEOFF	53	强减
                                                                    # xtconstant.OFFSET_FLAG_LOCALFORCECLOSE	54	本地强平
                                                                    if (orderall.order_type==int(23)):#这里只计算BUY方向的订单,24是卖23是买
                                                                        # logger.info("该订单是买入")
                                                                        # time.sleep(10)
                                                                        if (orderall.order_status==int(54)):
                                                                            thiscancel_amount=orderall.order_volume-orderall.traded_volume
                                                                            logger.info(f"{orderall}")
                                                                            logger.info(f"******,撤单成功,{orderall},{orderall.order_status},{thiscancel_amount}")
                                                                            if dfordercancelled.empty:#dfordercancelled一开始是个空值,这里主要是确认一下之前有没有数据,有数据才需要检验之前是否撤销过
                                                                                dfordercancelled=pd.concat([dfordercancelled,dforderall],ignore_index=True)
                                                                                cancel_money=thiscancel_amount#然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                                retradenum+=cancel_money
                                                                            else:
                                                                                if orderall.order_id not in dfordercancelled["order_id"].tolist():
                                                                                    dfordercancelled=pd.concat([dfordercancelled,dforderall],ignore_index=True)
                                                                                    cancel_money=thiscancel_amount#然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                                    retradenum+=cancel_money
                                                                        elif (orderall.order_status==int(57)):
                                                                            logger.info(f"******,废单处理,{orderall},{orderall.order_status},{orderall.order_volume}")
                                                                            if dfordercancelled.empty:#dfordercancelled一开始是个空值,这里主要是确认一下之前有没有数据,有数据才需要检验之前是否撤销过
                                                                                dfordercancelled=pd.concat([dfordercancelled,dforderall],ignore_index=True)
                                                                                cancel_money=orderall.order_volume#然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                                retradenum+=cancel_money
                                                                            if orderall.order_id not in dfordercancelled["order_id"].tolist():
                                                                                dfordercancelled=pd.concat([dfordercancelled,dforderall],ignore_index=True)
                                                                                cancel_money=orderall.order_volume#然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                                retradenum+=cancel_money
                                                tradestatus=False#初始化交易状态【判断是否可以执行申赎交易】为False
                                                for num in range(0,100):
                                                    if retradenum<=0:#如果已经下单结束就不再执行下单任务
                                                        logger.info("下单完成结束下单进程")
                                                        logger.info("剩余应下单数量",retradenum)
                                                        tradestatus=True
                                                        break
                                                    if retradenum>100000:
                                                        volume=100000#每次1000手10w股
                                                        retradenum-=100000
                                                        logger.info("剩余应下单数量",retradenum)
                                                    else:
                                                        volume=retradenum
                                                        retradenum-=volume
                                                        logger.info("剩余应下单数量",retradenum)
                                                    #正股下单【需要根据单笔最大金额确定需要下单的次数】
                                                    buyorder0=trade_api.order_stock(acc, stock_code=symbolsymbol,
                                                                        order_type=xtconstant.STOCK_BUY,
                                                                        order_volume=volume,
                                                                        price_type=xtconstant.FIX_PRICE,#限价
                                                                        strategy_name=choosename,#策略名称
                                                                        price=ask_price_1)
                                                    logger.info(f"下单成功{buyorder}")
                                                time.sleep(1)#休息一秒,避免空转
                                                if tradestatus==True:
                                                    logger.info("买入ETF任务结束执行赎回任务")
                                                    break#如果可以执行赎回交易则打断前面的循环

                                            #_________________________________________________________    
                                            #【ETF赎回】实盘ETF申赎API需要开通ETF申赎权限（认证专业投资者）【申赎下单的时候报价为空值，与传入的值无关，后面会在新的订单上自动生成一个成交价格作为订单查询时回报的成交价格】
                                            buyorder=trade_api.order_stock(acc, stock_code=symbolsymbol,
                                                            order_type=xtconstant.ETF_REDEMPTION,#ETF赎回
                                                            order_volume=allvolume,
                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                            strategy_name=choosename,#策略名称
                                                            price=ask_price_1)
                                            logger.info(f"赎回下单成功{buyorder}")
                                            # #【ETF申购】实盘ETF申赎API需要开通ETF申赎权限（认证专业投资者）【申赎下单的时候报价为空值，与传入的值无关，后面会在新的订单上自动生成一个成交价格作为订单查询时回报的成交价格】
                                            # buyorder=trade_api.order_stock(acc, stock_code=symbol,
                                            #                     order_type=xtconstant.ETF_PURCHASE,#ETF申购
                                            #                     order_volume=thisvolume,
                                            #                     price_type=xtconstant.FIX_PRICE,#限价
                                            #                     strategy_name=choosename,#策略名称
                                            #                     price=ask_price_1)
                                            # logger.info(f"申购下单成功{buyorder}")
                                except Exception as e:#报索引越界一般是tick数据没出来
                                    logger.info(f"******,发生bug:,{symbol},{e}")
                            else:
                                print("成分券存在跌停板不宜执行交易")

                        #————————————————————————————————————————————————————————————————
                        while True:
                            #验证持仓当中的该ETF是否已经赎回成功
                            positions=trade_api.query_stock_positions(account=acc)
                            badnum=0
                            for position in positions:
                                if symbolsymbol==position.stock_code:
                                    print(position.stock_code,"position.volume",position.volume)
                                    if position.volume==0:
                                        print("全部赎回成功停止等待",symbol)
                                        badnum+=1
                            if badnum!=0:#
                                break
                            time.sleep(1)
                        #______________________________________________________________________
                        #获取这个ETF的申赎清单【将处于申赎清单当中的标的卖掉完成套利】
                        for index,thisdf in thislistdf.iterrows():
                            thisstock=thisdf["股票代码无后缀"]
                            thisvolume=thisdf["股票数量"]
                            #正股下单
                            #返回五档数据
                            tick=xtdata.get_full_tick([thisstock])
                            tick=tick[thisstock]
                            ask_price_1=tick["askPrice"][0]
                            ask_volume_1=tick["askVol"][0]
                            ask_price_2=tick["askPrice"][1]
                            ask_volume_2=tick["askVol"][1]
                            bid_price_1=tick["bidPrice"][0]
                            bid_volume_1=tick["bidVol"][0]
                            bid_price_2=tick["bidPrice"][1]
                            bid_volume_2=tick["bidVol"][1]
                            logger.info(f"tick{tick},{ask_price_1},{ask_volume_1},{ask_price_2},{ask_volume_2},{bid_price_1},{bid_volume_1},{bid_price_2},{bid_volume_2}")
                            sellorder0=trade_api.order_stock(acc, stock_code=thisstock,
                                                order_type=xtconstant.STOCK_SELL,
                                                order_volume=thisvolume,
                                                price_type=xtconstant.FIX_PRICE,#限价
                                                strategy_name=choosename,#策略名称
                                                price=bid_price_1)
                            logger.info(f"下单成功{sellorder0}")
                            # break#也可以不主动停止而是等待任务结束的时候再停止，这样每一轮都验证一下自己估算的净值
                            
                            #【未成交的撤单重新下】
                            retradenum=0#重置应下单数量
                            timecancellwait=10#timecancellwait秒不成交就撤单
                            targetmoney=2000#对手盘需要达到的厚度
                            
                            #_________________________________________________________________
                            #TODO:验证赎回的个股是否卖出成功
                            while True:
                                holdvolume=0
                                positions=trade_api.query_stock_positions(account=acc)#获取持仓列表
                                for position in positions:
                                    possymbol=position.stock_code#获取对应股票
                                    logger.info("thisstock",thisstock,"possymbol",possymbol)
                                    if thisstock==possymbol:
                                        logger.info(thisstock,position.volume)
                                        holdvolume=position.volume
                                        logger.info(f"当前{thisstock}持仓数量",holdvolume)
                                if holdvolume==0:
                                    logger.info("当前赎回后个股卖出计划结束")
                                else:
                                    orderalls = trade_api.query_stock_orders(account=acc,cancelable_only=False)#仅查询可撤委托
                                    for orderall in orderalls:
                                        if sellorder0['order_id'] == orderall.order_id:
                                        # #模拟盘下午无法识别到撤单（orderall.status_msg无数据）把这块拿出来单独研究
                                        # logger.info(f"{orderall},{type(orderall.offset_flag)},{orderall.direction},{orderall.price_type},{orderall.order_id}")
                                        # 账号状态(account_status)
                                        # xtconstant.ORDER_UNREPORTED	48	未报
                                        # xtconstant.ORDER_WAIT_REPORTING	49	待报
                                        # xtconstant.ORDER_REPORTED	50	已报
                                        # xtconstant.ORDER_REPORTED_CANCEL	51	已报待撤
                                        # xtconstant.ORDER_PARTSUCC_CANCEL	52	部成待撤
                                        # xtconstant.ORDER_PART_CANCEL	53	部撤
                                        # xtconstant.ORDER_CANCELED	54	已撤
                                        # xtconstant.ORDER_PART_SUCC	55	部成
                                        # xtconstant.ORDER_SUCCEEDED	56	已成
                                        # xtconstant.ORDER_JUNK	57	废单【这个也得算金额】
                                        # xtconstant.ORDER_UNKNOWN	255	未知
                                        #拼接orderall的数据【不对已成（56）、待报（49）、未报（48）订单进行处理】大部分是54已撤、55部成、56已成、57废单
                                            if ((orderall.order_status!=int(56))and(orderall.order_status!=int(49))and(orderall.order_status!=int(48))):
                                                dforderall=pd.DataFrame({
                                                    "order_status":[orderall.order_status],
                                                    "order_id":[orderall.order_id],
                                                    "status_msg":[orderall.status_msg],
                                                    "symbol":[orderall.stock_code],
                                                    "amount":[orderall.order_volume],
                                                    "trade_amount":[orderall.traded_volume],
                                                    "trade_price":[orderall.traded_price],
                                                    "order_type":[orderall.order_type],#int,24卖出,23买入
                                                    "direction":[orderall.direction],#int,多空方向,股票不需要；参见数据字典
                                                    "offset_flag":[orderall.offset_flag],#int,交易操作,用此字段区分股票买卖,期货开、平仓,期权买卖等；参见数据字典
                                                    "price":[orderall.price],
                                                    "price_type":[orderall.price_type],
                                                    "datetime":[datetime.datetime.fromtimestamp(orderall.order_time).strftime("%Y%m%d %H:%M:%S")],
                                                    "secondary_order_id":[orderall.order_id]})
                                                dforderalls=pd.concat([dforderalls,dforderall],ignore_index=True)
                                                if ((orderall.order_status==int(55))or(orderall.order_status==int(50))):
                                                    logger.info(f"******,不是已成交订单,{orderall.order_id}")
                                                    #60秒内不成交就撤单【这个是要小于当前时间,否则就一直无法执行】
                                                    if (datetime.datetime.fromtimestamp(orderall.order_time)+datetime.timedelta(seconds=timecancellwait))<datetime.datetime.now():#成交额还得超过targetmoney才可以最终撤单
                                                        if (orderall.traded_volume*orderall.price>targetmoney):
                                                            try:
                                                                cancel_result = trade_api.cancel_order_stock(account=acc,order_id=orderall.order_id)
                                                                # .cancel_order(orderall.order_id)
                                                                logger.info(f"******,已成交金额达标执行撤单,{orderall.order_id,cancel_result}")
                                                            except:
                                                                logger.info(f"******","已完成或取消中的条件单不允许取消")
                                                        elif orderall.traded_volume==0:#未成交撤单
                                                            try:#如果该委托已成交或者已撤单则会报错
                                                                cancel_result = trade_api.cancel_order_stock(account=acc,order_id=orderall.order_id)
                                                                # .cancel_order(orderall.order_id)
                                                                logger.info(f"******,执行撤单,{orderall.order_id},cancel_result,{cancel_result}")
                                                            except:
                                                                logger.info(f"******,已完成或取消中的条件单不允许取消")
                                                else:#撤单或者废单之后的金额回补
                                                    # 交易操作(offset_flag)
                                                    # 枚举变量名	值	含义
                                                    # xtconstant.OFFSET_FLAG_OPEN	48	买入,开仓
                                                    # xtconstant.OFFSET_FLAG_CLOSE	49	卖出,平仓
                                                    # xtconstant.OFFSET_FLAG_FORCECLOSE	50	强平
                                                    # xtconstant.OFFSET_FLAG_CLOSETODAY	51	平今
                                                    # xtconstant.OFFSET_FLAG_ClOSEYESTERDAY	52	平昨
                                                    # xtconstant.OFFSET_FLAG_FORCEOFF	53	强减
                                                    # xtconstant.OFFSET_FLAG_LOCALFORCECLOSE	54	本地强平
                                                    if (orderall.order_type==int(24)):#这里只计算BUY方向的订单,24是卖23是买
                                                        logger.info("该订单是卖出")
                                                        # time.sleep(10)
                                                        if (orderall.order_status==int(54)):
                                                            thiscancel_amount=orderall.order_volume-orderall.traded_volume
                                                            logger.info(f"{orderall}")
                                                            logger.info(f"******,撤单成功,{orderall},{orderall.order_status},{thiscancel_amount}")
                                                            if dfordercancelled.empty:#dfordercancelled一开始是个空值,这里主要是确认一下之前有没有数据,有数据才需要检验之前是否撤销过
                                                                dfordercancelled=pd.concat([dfordercancelled,dforderall],ignore_index=True)
                                                                cancel_money=thiscancel_amount#然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                retradenum+=cancel_money
                                                            else:
                                                                if orderall.order_id not in dfordercancelled["order_id"].tolist():
                                                                    dfordercancelled=pd.concat([dfordercancelled,dforderall],ignore_index=True)
                                                                    cancel_money=thiscancel_amount#然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                    retradenum+=cancel_money
                                                        elif (orderall.order_status==int(57)):
                                                            logger.info(f"******,废单处理,{orderall},{orderall.order_status},{orderall.order_volume}")
                                                            if dfordercancelled.empty:#dfordercancelled一开始是个空值,这里主要是确认一下之前有没有数据,有数据才需要检验之前是否撤销过
                                                                dfordercancelled=pd.concat([dfordercancelled,dforderall],ignore_index=True)
                                                                cancel_money=orderall.order_volume#然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                retradenum+=cancel_money
                                                            if orderall.order_id not in dfordercancelled["order_id"].tolist():
                                                                dfordercancelled=pd.concat([dfordercancelled,dforderall],ignore_index=True)
                                                                cancel_money=orderall.order_volume#然后就是计算撤销了的订单的未完成金额,加给下单金额当中
                                                                retradenum+=cancel_money
                                tradestatus=False#初始化交易状态【判断是否可以执行申赎交易】为False
                                for num in range(0,100):
                                    if retradenum<=0:#如果已经下单结束就不再执行下单任务
                                        logger.info("下单完成结束下单进程")
                                        logger.info("剩余应下单数量",retradenum)
                                        tradestatus=True
                                        break
                                    if retradenum>100000:
                                        volume=100000#每次1000手10w股
                                        retradenum-=100000
                                        logger.info("剩余应下单数量",retradenum)
                                    else:
                                        volume=retradenum
                                        retradenum-=volume
                                        logger.info("剩余应下单数量",retradenum)
                                    #正股下单【需要根据单笔最大金额确定需要下单的次数】
                                    sellorder0=trade_api.order_stock(acc, stock_code=thisstock,
                                                        order_type=xtconstant.STOCK_SELL,
                                                        order_volume=volume,
                                                        price_type=xtconstant.FIX_PRICE,#限价
                                                        strategy_name=choosename,#策略名称
                                                        price=bid_price_1)
                                    logger.info(f"下单成功{sellorder0}")
                                time.sleep(1)#休息一秒,避免空转
                                if tradestatus==True:
                                    logger.info("买入ETF任务结束执行赎回任务")
                                    break#如果可以执行赎回交易则打断前面的循环
                        

                    else:
                        print("达到交易上限无法发起申购或者赎回")
            logger.info("结束当前轮次的套利交易")
            break
    logger.info("休息10秒")
    time.sleep(10)


    #后续整体调整————————————————————————————————————————————————————————————————
    logger.info("任务结束清仓卖出")#避免之前持仓太多的影响
    #重置并获取持仓信息【这里的目的是重新获取最新持仓以避免执行卖出成功后数据没有更新导致的持仓数量不对的情况】
    dfposition=pd.DataFrame([])
    positions=trade_api.query_stock_positions(account=acc)
    for position in positions:
        symbol=position.stock_code  #symbol为持仓股代码
        logger.info(symbol,position.volume)
        if position.volume>0:#获取持仓股票信息
            dfposition=pd.concat([dfposition,pd.DataFrame({"symbol":[symbol],
                                                            "volume":[position.volume],
                                                            "can_use_volume":[position.can_use_volume],
                                                            "frozen_volume":[position.frozen_volume],
                                                            "market_value":[position.market_value],
                                                            })],ignore_index=True)
    logger.info(f"******,本轮持仓,{dfposition}")
    dfposition.to_csv(str(basepath)+"_dfposition.csv")
    logger.info(f"******,卖出")
    if not dfposition.empty:#有持仓则卖出
        for symbol in dfposition["symbol"].tolist():
                thisposition=dfposition[dfposition["symbol"]==symbol]
                logger.info(f"{thisposition},{thisposition.can_use_volume.values[0]}")
                if (thisposition.can_use_volume.values[0]>0):#个股持有余额及可用余额都要大于0才执行卖出动作
                    logger.info(f"******,{symbol},持仓数量,{thisposition}")
                    try:
                        #返回五档数据
                        tick=xtdata.get_full_tick([symbol])
                        tick=tick[symbol]
                        logger.info(f"{tick}")
                        ask_price_1=tick["askPrice"][0]
                        ask_volume_1=tick["askVol"][0]
                        bid_price_1=tick["bidPrice"][0]
                        bid_volume_1=tick["bidVol"][0]
                        ask_price_2=tick["askPrice"][1]
                        ask_volume_2=tick["askVol"][1]
                        bid_price_2=tick["bidPrice"][1]
                        bid_volume_2=tick["bidVol"][1]
                        lastPrice=tick["lastPrice"]
                        timetag= datetime.datetime.strptime(tick["timetag"],"%Y%m%d %H:%M:%S")
                        logger.info(f"{lastPrice},{type(lastPrice)},timetag:{timetag},{type(timetag)}")
                        if (timetag+datetime.timedelta(seconds=timetickwait)>datetime.datetime.now()):
                            logger.info(f"******,确认是最新tick,执行交易")
                            if (ask_price_1-bid_price_1)<=(((ask_price_1+bid_price_1)/2)*bidrate):#盘口价差
                                logger.info(f"******,盘口价差适宜,适合执行交易")
                                if ((symbol.startswith("12")) or (symbol.startswith("11"))):#针对11开头或者12开头的转债单独处理
                                    ask_volume_1*=10
                                    bid_volume_1*=10
                                    if tradeway=="maker":#maker下单【不需要考虑深度问题】#分单卖出
                                        if (thisposition.can_use_volume.values[0]*ask_price_1)<(traderate*targetmoney):
                                            logger.info("******","剩余全部卖出")
                                            sellvolume =(math.floor(thisposition.can_use_volume.values[0]/10)*10)
                                            sellorder=trade_api.order_stock(acc, stock_code=symbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=sellvolume,
                                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                                            strategy_name=choosename,#策略名称
                                                                            price=ask_price_1)
                                            logger.info(f"下单成功{sellorder},{ask_price_1},{sellvolume}")
                                        else:#限价卖出最小下单金额
                                            logger.info(f"******,卖出目标金额")
                                            sellvolume=(math.floor((targetmoney/ask_price_1)/10)*10)
                                            if (thisposition.can_use_volume.values[0]*bid_price_1)>500000:#针对余额高于500000的标的单独扩大下单数量
                                                sellvolume*=10
                                            logger.info(f"sellvolume{sellvolume},{sellvolume*ask_price_1}")
                                            sellorder=trade_api.order_stock(acc, stock_code=symbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=sellvolume,
                                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                                            strategy_name=choosename,#策略名称
                                                                            price=ask_price_1)
                                            logger.info(f"下单成功{sellorder},{ask_price_1},{sellvolume}")
                                        time.sleep(1)
                                    if tradeway=="taker":#maker下单【需要考虑深度问题】
                                        if (bid_price_1*bid_volume_1)>targetmoney:#盘口深度【己方一档买入】（转债价格较高,一档深度相对小一些）                                 
                                            if (thisposition.can_use_volume.values[0]*bid_price_1)<(traderate*targetmoney):
                                                logger.info(f"******,剩余全部卖出")
                                                sellvolume =(math.floor(thisposition.can_use_volume.values[0]/10)*10)
                                                sellorder=trade_api.order_stock(acc, stock_code=symbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=sellvolume,
                                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                                            strategy_name=choosename,#策略名称
                                                                            price=bid_price_1)
                                                logger.info(f"下单成功{sellorder},{bid_price_1},{sellvolume}")
                                            else:#限价卖出最小下单金额
                                                logger.info(f"******,卖出目标金额")
                                                sellvolume=(math.floor((targetmoney/bid_price_1)/10)*10)
                                                if (thisposition.can_use_volume.values[0]*bid_price_1)>500000:#针对余额高于500000的标的单独扩大下单数量
                                                    sellvolume*=10
                                                logger.info(f"sellvolume{sellvolume},{sellvolume*bid_price_1}")
                                                sellorder=trade_api.order_stock(acc, stock_code=symbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=sellvolume,
                                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                                            strategy_name=choosename,#策略名称
                                                                            price=bid_price_1)
                                                logger.info(f"下单成功{sellorder},{bid_price_1},{sellvolume}")
                                else:#非可转债交易方式
                                    ask_volume_1*=100
                                    bid_volume_1*=100
                                    if tradeway=="maker":#maker下单【不需要考虑深度问题】
                                        if (thisposition.can_use_volume.values[0]*ask_price_1)<(traderate*targetmoney):
                                            logger.info(f"******,剩余全部卖出")
                                            sellvolume =(math.floor(thisposition.can_use_volume.values[0]/100)*100)
                                            sellorder=trade_api.order_stock(acc, stock_code=symbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=sellvolume,
                                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                                            strategy_name=choosename,#策略名称
                                                                            price=ask_price_1)
                                            logger.info(f"下单成功{sellorder},{ask_price_1},{sellvolume}")
                                        else:#限价卖出最小下单金额
                                            logger.info("******","卖出目标金额")
                                            sellvolume=(math.floor((targetmoney/ask_price_1)/100)*100)
                                            if (thisposition.can_use_volume.values[0]*ask_price_1)>500000:#针对余额高于500000的标的单独扩大下单数量
                                                sellvolume*=10
                                            logger.info(f"sellvolume{sellvolume},{sellvolume*ask_price_1}")
                                            sellorder=trade_api.order_stock(acc, stock_code=symbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=sellvolume,
                                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                                            strategy_name=choosename,#策略名称
                                                                            price=ask_price_1)
                                            logger.info(f"下单成功{sellorder},{ask_price_1},{sellvolume}")
                                    if tradeway=="taker":#maker下单【需要考虑深度问题】
                                        if (bid_price_1*bid_volume_1)>targetmoney:#盘口深度【对手盘一档买入】                                            
                                            if (thisposition.can_use_volume.values[0]*bid_price_1)<(traderate*targetmoney):
                                                logger.info(f"******,剩余全部卖出")
                                                sellvolume =(math.floor(thisposition.can_use_volume.values[0]/100)*100)
                                                sellorder=trade_api.order_stock(acc, stock_code=symbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=sellvolume,
                                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                                            strategy_name=choosename,#策略名称
                                                                            price=bid_price_1)
                                                logger.info(f"下单成功{sellorder},{bid_price_1},{sellvolume}")
                                            else:#限价卖出最小下单金额
                                                logger.info(f"******,卖出目标金额")
                                                sellvolume=(math.floor((targetmoney/bid_price_1)/100)*100)
                                                if (thisposition.can_use_volume.values[0]*bid_price_1)>500000:#针对余额高于500000的标的单独扩大下单数量
                                                    sellvolume*=10
                                                logger.info(f"sellvolume{sellvolume},{sellvolume*bid_price_1}")
                                                sellorder=trade_api.order_stock(acc, stock_code=symbol,
                                                                            order_type=xtconstant.STOCK_SELL,
                                                                            order_volume=sellvolume,
                                                                            price_type=xtconstant.FIX_PRICE,#限价
                                                                            strategy_name=choosename,#策略名称
                                                                            price=bid_price_1)
                                                logger.info(f"下单成功{sellorder},{bid_price_1},{sellvolume}")
                    except Exception as e:#报索引越界一般是tick数据没出来
                        logger.info("******","发生bug:",symbol,e)