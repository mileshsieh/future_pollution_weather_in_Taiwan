#!/home/mileshsieh/anaconda3/envs/dask/bin/python
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

if __name__=='__main__':
  m='TaiESM1'
  yr_start=2005
  yr_end=2014
  ana=pd.read_csv('../data/cold_season_weather_regimes_with_idx.%s.%d-%d.csv'%(m,yr_start,yr_end),header=0,parse_dates=[0])
  #plot count with prRatio criteria
  rList=['LCD','NE','CS','FT','TC']
  cnt={'all':[],'RAR':[],'RAR+UFR':[]}
  for label,thd in zip(['all','RAR',],[1.0,0.3]):
    for r in rList:
      cnt[label].append(ana[(ana.prRatio<=thd)&(ana.regime==r)].date.size)
      if label=='RAR':
        cnt['RAR+UFR'].append(ana[(ana.prRatio<=thd)&(ana.regime==r)\
                          &(ana.meanWS>=4.0)&(ana.meanWS<=12.0)&(ana.meanWD>=45)&(ana.meanWD<=195)].date.size)  
   

  x=np.arange(len(rList))  # the label locations
  width=0.3  # the width of the bars
  mul=-1.5
  matplotlib.rc('xtick',labelsize=15)
  matplotlib.rc('ytick',labelsize=15)
  plt.close()
  fig,ax=plt.subplots(figsize=(12,8))
  for g,cntList in cnt.items():
    rects=ax.bar(x+width*mul,cntList,width,align='edge',label=g)
    ax.bar_label(rects,fmt='%d',padding=3,fontsize=15)
    mul+=1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Occurence',fontsize=20)
  ax.set_title('Cold-Season Weather Event Occurence under Criteria(%d-%d)'%(yr_start,yr_end),fontsize=20)
  rList[0]='WS'
  ax.set_xticks(x, rList)
  ax.set_xticklabels(rList)
  ax.legend(loc='upper right',fontsize=15)
  ax.set_ylim(0, )

  plt.savefig('../figures/fig2_event_count.png',dpi=200)
  
