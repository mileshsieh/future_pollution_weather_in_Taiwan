#!/home/mileshsieh/anaconda3/bin/python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('xtick',labelsize=12)
matplotlib.rc('ytick',labelsize=12)

wdList=[60,90,120,150,180]
def getUFR(ser,wdList=wdList):
  idx,r=divmod(ser.meanWD+15,30)
  return wdList[int(idx)-2]

if __name__=='__main__':
  m='TaiESM1'
  sce='ssp585'
  yr_start=2015
  yr_end=2020

  runDict={'ERA5':['ERA5',],
            m:[m,],
           # 'diff':['%s - ERA5\nFrequency Difference'%m],
          }
  runList=list(runDict.keys())
  nRun=len(runList)
  #create ws bins    
  spd_bins = np.array([0.0,2.0,4.0,6.0,8.0,10.0,12.0])
  spd_labels = ['%.1f'%k for k in spd_bins]
  #spd_labels[-1]='Inf'
  dir_bins = np.linspace(45,195,6)
  dir_labels =0.5*(dir_bins[1:]+dir_bins[:-1])
    
  col='meanWD'
  ws_col='meanWS'

  ana=pd.read_csv('../data/cold_season_weather_regimes_with_idx.%s.%d-%d.csv'%(m,yr_start,yr_end),header=0,parse_dates=[0])
  mdl=pd.read_csv('../data/cold_season_%s_%s_%d0101-%d1231.indices.csv'%(m,sce,yr_start,yr_start+9),header=0,parse_dates=[0])
  mdl=mdl.query("date >= '%d-01-01' and date <= '%d-12-31'"%(yr_start,yr_end))
  
  ana=ana[(ana.prRatio<=0.3)&(ana.meanWS>=4.0)&(ana.meanWS<=12.0)&(ana.meanWD>=45)&(ana.meanWD<=195)]
  mdl=mdl[(mdl.prRatio<=0.3)&(mdl.meanWS>=4.0)&(mdl.meanWS<=12.0)&(mdl.meanWD>=45)&(mdl.meanWD<=195)]
 
  fqDict={} 
  for lbl,df in zip(['ERA5',m],[ana,mdl]):
    df['UFR']=df.apply(getUFR,axis=1)  
    total_count=df.shape[0]
    df['dir_bin']=pd.cut(df[col],dir_bins)
    df['spd_bin']=pd.cut(df[ws_col],spd_bins)
    size_2D=df.groupby(by=['spd_bin', 'dir_bin']).count().yyyymmdd.fillna(0)
    x=size_2D.unstack().values
    cnt_all=x.sum()
    cnt_str=np.sum(x,axis=0).reshape(1,-1)
    print(cnt_all,total_count,cnt_str)
    freq=100*x/cnt_all
    fqDict[lbl]=freq
    runDict[lbl].append(cnt_all)

  fqDict['diff']=fqDict[m]-fqDict['ERA5']
    
  plt.close()
  #plt.figure(figsize=(32,16))
  fig,axes=plt.subplots(1,nRun,figsize=(6.5*nRun, 8),subplot_kw={'projection': 'polar'})
        
  for ax,run in zip(axes,runList):
    #plot
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')
    ax.set_thetamin(45)
    ax.set_thetamax(195)
    if run!='diff':
      cs=ax.pcolormesh(dir_bins/180*np.pi,spd_bins,fqDict[run],cmap='viridis',shading='flat',vmin=0.0,vmax=14.0)
    else:
      cs2=ax.pcolormesh(dir_bins/180*np.pi,spd_bins,fqDict[run],cmap='bwr',shading='flat',vmin=-5.0,vmax=5.0)
    #label the numbers on plot
    for (j,i),days in np.ndenumerate(fqDict[run]):
      ax.text(0.5*(dir_bins[i]+dir_bins[i+1])/180*np.pi,0.5*(spd_bins[j]+spd_bins[j+1]),\
              '%.1f'%days,color='white',ha='center',va='center',fontsize=11)

    #ax.grid()

    ax.set_xticks(np.array([60,90,120,150,180])*np.pi/180)
    # tick locations
    ax.xaxis.set_tick_params(pad=10)
    ax.set_yticks(spd_bins[::2])
    ax.set_yticklabels(spd_labels[::2])
    ax.text(0.1*np.pi,7,'Wind Speed (m/s)',size=12,rotation=45,ha='center',va='center',)

    ax.set_ylim(4,12)
    ax.set_rorigin(-1)
    ax.text(0.7*np.pi,16,'Wind Direction ($^\circ$)',size=12,rotation=35,ha='center',va='center',)

    if run!='diff':
      ax.set_title('%s Frequency\n(%d days)\n'%(runDict[run][0],runDict[run][1]),fontsize=22,loc='left')
    else:
      ax.set_title('%s\n'%runDict[run][0],fontsize=22,loc='left')
  fig.subplots_adjust(bottom=0.1)
  if nRun==3:
    ax_cb = fig.add_axes([0.22, 0.13, 0.18, 0.03])
    cb=plt.colorbar(cs,cax=ax_cb,orientation='horizontal',extend='max')
    cb.set_label('Frequency (%)',fontsize=15)
    ax_cb2 = fig.add_axes([0.72, 0.13, 0.18, 0.03])
    cb2=plt.colorbar(cs2,cax=ax_cb2,orientation='horizontal',extend='both')
    cb2.set_label('Frequency (%)',fontsize=15)
    #plt.tight_layout()
    plt.suptitle('%s Bias in Frequency of Lee Vortex Days (%d-%d)'%(m,yr_start,yr_end),fontsize=30)
    plt.annotate('(a)', xy=(0.05, 0.94), xytext=(0.055, 0.86),xycoords='figure fraction',fontsize=22)
    plt.annotate('(b)', xy=(0.37, 0.95), xytext=(0.36, 0.86),xycoords='figure fraction',fontsize=22)
    plt.annotate('(c)', xy=(0.68, 0.95), xytext=(0.665, 0.86),xycoords='figure fraction',fontsize=22)
  else:
    ax_cb = fig.add_axes([0.35, 0.13, 0.3, 0.03])
    cb=plt.colorbar(cs,cax=ax_cb,orientation='horizontal',extend='max')
    cb.set_label('Frequency (%)',fontsize=15)

    plt.suptitle('%s Bias in Frequency of Lee Vortex Days (%d-%d)'%(m,yr_start,yr_end),fontsize=30)
    plt.annotate('(a)', xy=(0.05, 0.94), xytext=(0.06, 0.81),xycoords='figure fraction',fontsize=22)
    plt.annotate('(b)', xy=(0.67, 0.95), xytext=(0.48, 0.81),xycoords='figure fraction',fontsize=22)
  plt.savefig('../figures/fig6_%s_UFR_bias.png'%m,bbox_inches='tight',dpi=300)

