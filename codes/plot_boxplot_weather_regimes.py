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

if __name__=='__main__':

  df=pd.read_csv('../data/weather_regimes_with_idx.csv',header=0,parse_dates=[0])
  rList=['LCD', 'NE', 'FT', 'TC', 'CS']

  plt.close()
  fig,axes=plt.subplots(2,2,figsize=(12,8))
  for var,ax in zip(['prRatio','std_uv','meanWD','wdCor'],axes.ravel()):
    df_var=df[['regime',var]]
    data={}
    for r,grp in df_var.groupby('regime'):
      data[r]=grp[var].values
    
    #boxplot
    #ax.boxplot([data[a] for a in rList],labels=rList
    #            ,showmeans=True,boxprops= dict(linewidth=2.0, color='black')
    #            ,whiskerprops=dict(linewidth=2.0, color='black')
    #            ,capprops=dict(linewidth=2.0, color='black')
    #            ,medianprops=dict(linewidth=2.0, color='red')
    #            ,meanprops=dict(markersize=15, color='blue'))

    #violinplot
    ax.violinplot([data[a] for a in rList],vert=True,showmeans=True,showmedians=True)
    ax.set_xticks(np.linspace(1,len(rList),len(rList)))
    lblList=['%s(%d)'%(r,data[r].size) for r in rList]
    ax.set_xticklabels(lblList)
    ax.set_ylim(0,)
    ax.grid(True)
    ax.set_ylabel(pltCfg[var][0])
    ax.set_title(var)
  plt.suptitle('Synoptic Factors of Cold Season Weather Regimes (2001-2010)',fontsize=20)
  plt.savefig('../figures/factors.png')
