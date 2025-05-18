#!/home/mileshsieh/anaconda3/bin/python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('xtick',labelsize=10)
matplotlib.rc('ytick',labelsize=10)

pltCfg={'prRatio':['Rainy Area Ratio'],
        'meanWS':['Mean Wind Speed (m/s)'],
        'maxWS':['Maximum Wind Speed (m/s)'],
        'std_uv':['Std. of Wind Component Speed (m/s)'],
        'meanWD':['Mean Wind Direction ($^{\circ}$)'],
        'wdCor':['Wind Direction Coherence ($^{\circ}$)'],}

wdList=[60,90,120,150,180]
def getUFR(ser,wdList=wdList):
  idx,r=divmod(ser.meanWD+15,30)
  return wdList[int(idx)-2]

if __name__=='__main__':
  m='TaiESM1'
  sce='ssp585'
  ana=pd.read_csv('../data/cold_season_weather_regimes_with_idx.csv',header=0,parse_dates=[0])
  mdl=pd.read_csv('../data/cold_season_%s_historical.indices.csv'%m,header=0,parse_dates=[0])
  
  ana=ana[(ana.prRatio<=0.3)&(ana.meanWS>=4.0)&(ana.meanWS<=12.0)&(ana.meanWD>=45)&(ana.meanWD<=195)]
  mdl=mdl[(mdl.prRatio<=0.3)&(mdl.meanWS>=4.0)&(mdl.meanWS<=12.0)&(mdl.meanWD>=45)&(mdl.meanWD<=195)]
  
  for lbl,df in zip(['ana','mdl'],[ana,mdl]):
    df['UFR']=df.apply(getUFR,axis=1)  
    print(lbl,[df[(df.UFR==150)&(df.month==mon)].date.size for mon in [10,11,12,1,2,3,4]])

  cntDict={}
  yrList=[]
  df_yr={}
  #for ii,yr in enumerate(range(2015,2086,10)): 
  for ii,yr in enumerate(range(2025,2086,10)): 
    df=pd.read_csv('../data/cold_season_%s_%s_%04d0101-%04d1231.indices.csv'%(m,sce,yr,yr+9),header=0,parse_dates=[0])
    df=df[(df.prRatio<=0.3)&(df.meanWS>=4.0)&(df.meanWS<=12.0)&(df.meanWD>=45)&(df.meanWD<=195)]
    df['UFR']=df.apply(getUFR,axis=1)
    df_yr[yr]=df
    
    #monthly trend
    yrList.append(yr)
    cnt=[df[(df.UFR==150)&(df.month==mon)].date.size for mon in [10,11,12,1,2,3,4]]
    cnt_OND=np.sum([df[(df.UFR==150)&(df.month==mon)].date.size for mon in [10,11,12]])
    cnt_JFMA=[df[(df.UFR==150)&(df.month==mon)].date.size for mon in [1,2,3,4]]
    cnt_JFMA.insert(0,cnt_OND)
    cntDict[yr]=cnt_JFMA
    #cntDict[yr]=[df[(df.UFR==150)&(df.month==mon)].date.size for mon in [1,2,3,4]]
    print(yr,cnt,cntDict[yr])

  monList=['OCT-DEC','JAN','FEB','MAR','APR']
  x=np.arange(len(monList))  # the label locations
  width=0.1  # the width of the bars
  mul=-4
  offset=0.5
  matplotlib.rc('xtick',labelsize=15)
  matplotlib.rc('ytick',labelsize=15)
  plt.close()
  fig,ax=plt.subplots(figsize=(16,8))
  for yr in yrList:
    total=np.sum(cntDict[yr])
    rects=ax.bar(x+width*(mul+offset),cntDict[yr],width,align='edge',label='%04d-%04d(%d days)'%(yr,yr+9,total))
    ax.bar_label(rects,fmt='%d',padding=3,fontsize=12)
    mul+=1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Occurence',fontsize=20)
  ax.set_title('Trend of WD150 Day Occurence in Month [TaiESM1-SSP585]',fontsize=20)
  ax.set_xticks(x, monList)
  ax.set_xticklabels(monList)
  ax.legend(loc='upper left',fontsize=15)
  ax.set_ylim(0, )

  plt.savefig('../figures/fig7_LVD_count_by_month.png',dpi=200)
  
