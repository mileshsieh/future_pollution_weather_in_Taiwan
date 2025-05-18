import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('xtick',labelsize=25)
matplotlib.rc('ytick',labelsize=25)

def group_by_winter_month(dd):
    mon=dd.month
    if mon in [1,2,3,4]:
        winter='%04d-%04dwinter'%(dd.year-1,dd.year)
    else:
        winter='%04d-%04dwinter'%(dd.year,dd.year+1)
    return winter,mon

if __name__=='__main__':
  #define weather regimes
  regimes={'TC':['TYW','TC100','TC200','TC300','TC500','TC1000'],
           'FT':['FT',],
           'CS':['CS',],
           'NE':['NE','SNE'],
           'SW':['SWF','SSWF'],
          }

  #load weather events table
  a=pd.read_csv('../data/weather_event_1980to2020.csv',header=0,parse_dates=[0])
  print(a.shape)
  m='TaiESM1'
  yr_start=2015
  yr_end=2020
  weather=a[['yyyymmdd','CS','TYW','TC100','TC200','TC300','TC500','TC1000','FT','NE','SNE','SWF','SSWF']].copy()
  print(weather.columns)
  weather.fillna(0.0,inplace=True)
  weather['total']=weather.apply(lambda ser: ser.iloc[1:13].sum(),axis=1)
  weather['month']=weather.apply(lambda ser: ser.yyyymmdd.month,axis=1)
  weather['doy']=weather.apply(lambda ser: ser.yyyymmdd.dayofyear,axis=1)
  weather['year']=weather.apply(lambda ser: ser.yyyymmdd.year,axis=1)
  #get the JFMA+OND from yr_start to yr_end
  stime=pd.to_datetime('%d-01-01'%yr_start)
  etime=pd.to_datetime('%d-12-31'%yr_end)
  wx=weather[(weather.yyyymmdd>=stime)&(weather.yyyymmdd<=etime)]
  wx_cold_season=wx[wx.month.isin([1,2,3,4,10,11,12])].copy()
  for c in wx_cold_season.columns:
    print(c,wx_cold_season[c].unique())

  def getWeatherClass(ser):
    wr='LCD'
    for r in regimes.keys():
      cnt=np.sum([ser[c] for c in regimes[r]])
      if (cnt>0)&(wr=='LCD'):
        wr=r
    return wr
  wx_cold_season['regime']=wx_cold_season.apply(getWeatherClass,axis=1)

  #open reanalysis indices and merge
  #columns: date,yyyymmdd,meanU,meanV,wdCor,meanWS,maxWS,ish_u,ish_v,meanWD,ish_wd,ish_ws,year,month,prRatio
  df_idx=pd.read_csv('../data/reanalysis.%s_grid.%d-%d.indices.csv'%(m,yr_start,yr_end),header=0,parse_dates=[0])
  wx_cold_season.rename(columns={'yyyymmdd':'date'},inplace=True)
  final=wx_cold_season[['date','doy','regime']].merge(df_idx,on='date')
  final.to_csv('../data/cold_season_weather_regimes_with_idx.%s.%d-%d.csv'%(m,yr_start,yr_end),index=False)

    
