#!/home/mileshsieh/anaconda3/bin/python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('xtick',labelsize=10)
matplotlib.rc('ytick',labelsize=10)

wdList=[60,90,120,150,180]
def getUFR(ser,wdList=wdList):
  idx,r=divmod(ser.meanWD+15,30)
  return wdList[int(idx)-2]

if __name__=='__main__':
  m='TaiESM1'
  sce='ssp585'
  #yr=2085
  #mdl=pd.read_csv('../data/cold_season_%s_historical.indices.csv'%m,header=0,parse_dates=[0])
  #df_yr=pd.read_csv('../data/cold_season_%s_%s_%04d0101-%04d1231.indices.csv'%(m,sce,yr,yr+9),header=0,parse_dates=[0])

  dfDict={}
  #for lbl,df in zip(['SSP585[','SSP585[%d-%d]'%(yr,yr+9)],[mdl,df_yr]):
  for yr in [2025,2085]:
    df=pd.read_csv('../data/cold_season_%s_%s_%04d0101-%04d1231.indices.csv'%(m,sce,yr,yr+9),header=0,parse_dates=[0])
    df=df[(df.prRatio<=0.3)&(df.meanWS>=4.0)&(df.meanWS<=12.0)&(df.meanWD>=45)&(df.meanWD<=195)]
    df['UFR']=df.apply(getUFR,axis=1)  
    dfDict['SSP585[%d-%d]'%(yr,yr+9)]=df[(df.UFR==150)&(df.month.isin([3,4]))]

  #plot LTS PDF
  plt.close()  
  fig=plt.figure(figsize=(8,6))
  ax1=plt.subplot()
  for yr,c in zip([2025,2085],['blue','salmon']):
    dfDict['SSP585[%d-%d]'%(yr,yr+9)].ish_LTS.plot(kind='hist',density=False,bins=np.linspace(10,25,16)\
                                                   ,label='%d-%d(%d days)'%(yr,yr+9,dfDict['SSP585[%d-%d]'%(yr,yr+9)].date.size)\
                                                   ,color=c,alpha=0.5)
  ax1.grid()
  ax1.set_ylim(0,)
  ax1.set_ylabel('Occurence',fontsize=20)
  #ax1.set_xlim(5,35)
  #ax1.set_xticks(np.linspace(0.0,1.0,6))
  ax1.set_xlabel('LTS (K)',fontsize=20)
  ax1.legend(fontsize=14)
  '''
  #add kde plot but remove the yaxis
  ax2=ax1.twinx()
  for yr,c in zip([2025,2085],['b','r']):
    dfDict['SSP585[%d-%d]'%(yr,yr+9)].ish_LTS.plot(kind='kde',color=c,label='SSP585[%d-%d]'%(yr,yr+9))
  ax2.legend(loc=1,fontsize=14)
  ax2.set_yticks([])
  ax2.set_ylim(0,0.3)
  ax2.set_ylabel('')
  '''
  plt.title('LTS Drift in %s Future Projection'%m,fontsize=20)
  plt.savefig('../figures/fig9_LTS_comparison.png',dpi=200)

