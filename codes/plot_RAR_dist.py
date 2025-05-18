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
  m='TaiESM1'
  ana=pd.read_csv('../data/cold_season_weather_regimes_with_idx.csv',header=0,parse_dates=[0])
  mdl=pd.read_csv('../data/cold_season_%s_historical.indices.csv'%m,header=0,parse_dates=[0])

  #plot prRatio PDF
  plt.close()  
  fig=plt.figure(figsize=(8,6))
  ax1=plt.subplot()
  ana.prRatio.plot(kind='hist',density=False,bins=10,color='blue',alpha=0.5)
  mdl.prRatio.plot(kind='hist',density=False,bins=10,color='salmon',alpha=0.5)
  ax1.grid()
  ax1.set_ylim(0,850)
  ax1.set_ylabel('Occurence',fontsize=20)
  ax1.set_xlim(-0.2,1.2)
  ax1.set_xticks(np.linspace(0.0,1.0,6))
  ax1.set_xlabel(pltCfg['prRatio'][0],fontsize=20)
  #add kde plot but remove the yaxis
  ax2=ax1.twinx()
  ana.prRatio.plot(kind='kde',color='b',label='IMERG')
  mdl.prRatio.plot(kind='kde',color='r',label=m)
  ax2.legend(fontsize=14)
  ax2.set_yticks([])
  ax2.set_ylabel('')
  plt.title('Cold Season Daily Rainy Areal Ratio\n%s (2005-2014)'%m,fontsize=20)
  plt.savefig('../figures/comp.RAR.%s.png'%m,dpi=200)
  #sns.distplot(ana.prRatio.values,hist=False,kde_kws={'color':'red','linestyle':'-'},norm_hist=True,label='IMERG')
  #sns.distplot(ana[ana.regime=='LCD'].prRatio.values,hist=False,kde_kws={'color':'red','linestyle':'-'},norm_hist=True,label='IMERG')
  #sns.distplot(ana[ana.regime.isin(['FT','TC'])].prRatio.values,hist=False,kde_kws={'color':'red','linestyle':'-'},norm_hist=True,label='IMERG')
  #sns.distplot(mdl.prRatio.values,hist=False,kde_kws={'color':'blue','linestyle':'-'},norm_hist=True,label='TaiESM1')
