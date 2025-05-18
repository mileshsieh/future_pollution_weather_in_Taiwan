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
  sce='SSP-585'

  df_yr={}
  for yr in range(2021,2100,10): 
    df=pd.read_csv('../data/LVD.%s_%s_%d0101-%d1231.csv'%(m,sce,yr,yr+9),header=0,parse_dates=[0])  
    df['UFR']=df.apply(getUFR,axis=1)
    df['consecutive'] = df.UFR.groupby((df.UFR != df.UFR.shift()).cumsum()).transform('size')
    df_yr[yr]=df
    #find the max consecutive days in WD150
    maxCon=sorted(df[df.UFR==150].consecutive.unique())[-1]
     
    if maxCon<3:
      #print('Most consecutive WD150 day is %d in %d'%(maxCon,yr))
      print(yr,df[df.UFR==150].date.size,0.0)
    else:
      #print('Most consecutive WD150 day is %d in %d'%(maxCon,yr))
      n_event=0
      for nday in range(3,maxCon+1):
        print('consectutive %d days:'%nday)
        print(df[(df.UFR==150)&(df.consecutive==nday)][['date','meanWS','meanWD','consecutive']])
        n_event=n_event+df[(df.UFR==150)&(df.consecutive==nday)].date.size/nday
      print(yr,df[df.UFR==150].date.size,n_event)
    ##count the episodes of the consecutive WD150 >= 3 days
    #cons_150=sorted(df[df.UFR==150].consecutive.unique())
    #cnt_over3=0
    #for cons in cons_150:
    #  if cons<3:
    #    continue
    #  cnt_over3=cnt_over3+df[(df.UFR==150)&(df.consecutive==cons)].date.size/cons
    #print(yr,df[df.UFR==150].date.size/df.date.size*100,cnt_over3)
    
    #for wd in wdList:
    #  print(yr,wd,df_yr[yr][df.UFR==wd].consecutive.unique().max())

