import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style

style.use('ggplot')
yf.pdr_override()

#Creating datestamps
start =dt.datetime(2000,1,1) 
end = dt.datetime(2016,12,31)

#Getting TSLA datas from yahoo finance
df = web.get_data_yahoo('TSLA', start, end)
df.head()

#Transform the dataframe in csv file
df.to_csv('tsla.csv')

#Read the csv using parse_dates and index_col to get the dates as the index column
df=pd.read_csv('tsla.csv',parse_dates=True,index_col=0)
print(df.head())

#Print only the open and high values
print(df[['Open','High']].head())

#Plot only the adjusted close graph in function of time
df['Adj Close'].plot()
plt.show()

#We use min_periods to take the mean for the datas  for which we can't do a 100ma
df['100ma'] = df['Adj Close'].rolling(window=100,min_periods=0).mean() 

#We could have dropped the first 99 with which we can't do a mobile average, if we didn't use min_periods=0
df.dropna(inplace=True) 

print(df.head())

#We plot the graph of the datas we are intersted in
ax1=plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2=plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)

ax1.plot(df.index,df['Adj Close'])
ax1.plot(df.index,df['100ma'])
ax2.bar(df.index,df['Volume'])

plt.show()

from mplfinance.original_flavor import candlestick_ohlc

#We resample without a rolling, with 'mean' it takes the mean every ten days so it divides by 10 the number of observations, with'ohlc' we have open high low close
df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()
df_ohlc.reset_index(inplace=True)
df_ohlc['Date']=df_ohlc['Date'].map(mdates.date2num)
print(df_ohlc.head())

#We plot the datas using candlestick graph
ax1=plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2=plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
ax1.xaxis_date()
candlestick_ohlc(ax1,df_ohlc.values,width=2,colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)

plt.show()

import bs4 as bs
import pickle
import requests

#We save sp_500 tickers to have the current list
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text,'lxml')
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker=row.findAll('td')[0].text
        tickers.append(ticker)
        new_tickers  = [x[:-1] for x in tickers]
    print(new_tickers)
    with open('sp500tickers.pickle','wb') as f:
        pickle.dump(new_tickers,f)

    return new_tickers

save_sp500_tickers()

import bs4 as bs
import pickle
import os
import requests
import numpy as np

#We get the datas for all the stocks in the sp_500 using yahoo finance and then we stock them in csv files
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open ('sp500tickers.pickle','rb') as f :
            tickers = pickle.load(f)
    start = dt.datetime(2010,1,1)
    end=dt.datetime(2023,12,31)
    for ticker in tickers:
        print(ticker)
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df= web.get_data_yahoo(ticker, start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Aready have {}'.format(ticker))

get_data_from_yahoo()

#We then creta an unique dataframe with all the datas from the sp_500 companies and we transform it in csv file
def compile_data():
    with open('sp500tickers.pickle','rb') as f:
        tickers=pickle.load(f)
        main_df=pd.DataFrame()

    for count,ticker in enumerate(tickers):
        df=pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)
        df.rename(columns={'Adj Close':ticker},inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],axis=1,inplace=True)

        if main_df.empty:
            main_df=df
        else:
            main_df=main_df.join(df,how='outer')
        #We will follow to see 
        if count % 10 ==0 :
            print(count)

    print (main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

compile_data()

#we do a heat map of the correlation of the stowks of sp_500
def visualize_data():
    df=pd.read_csv('sp500_joined_closes.csv')
    #df['AAPL'].plot()
    #plt.show()
    df_corr=df.corr(numeric_only=True)
    #print(df_corr.head())
    data =df_corr.values
    fig = plt.figure()
    ax=fig.add_subplot(1,1,1)

    heatmap=ax.pcolor(data,cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0])+0.5,minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5,minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()

visualize_data()




