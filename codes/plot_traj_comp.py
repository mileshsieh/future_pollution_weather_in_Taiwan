#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mc
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.ticker as ticker

def plotTW(ax,lonTW,latTW,topo):
    xx,yy=np.meshgrid(lonTW,latTW)
    cf=ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    ax.contour(xx,yy,topo,levels=[0.05,],colors=['k'])
    ax.set_xlim(118,lonTW[-1])
    ax.set_ylim(latTW[0],26)
    ax.set_ylabel('Latitude ($^\circ N$)')
    ax.set_xlabel('Longitude ($^\circ E$)')
    cb=plt.colorbar(cf,extend='max')
    cb.set_label('Height (km)')
    return

if __name__=='__main__':
  #domain and step of the original TaiwanVVM 2km simulation
  (xstart,xend,ystart,yend)=(20,321,150,451)
  topo=np.load('../data/topodata.npy')[ystart:yend,xstart:xend]
  lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend]
  latTW=np.load('../data/tw_s_lat.npy')[ystart:yend]

  #get emission xy index
  lonlat={'TPP':[120.4848,24.20943],'SNC':[120.2058,23.80023],\
          'TY':[121.314328,24.991278],'HsinChu':[120.983333,24.816667],\
          'NewTaipei':[121.445833,25.01111],'Taipei':[121.5625,25.0375],\
          'TC':[120.666667,24.25]}
  #load historical/ssp-585 upstream flow regimes and local flows
  m='TaiESM1'
  sce='ssp585'
  site='TPP'
  site='TC'
  runDict={'historical':['Historical',2001,2010],}
  for yr in range(2025,2095,10):
    runDict['%04d'%yr]=['SSP-585',yr,yr+9]

  trajDict={}
  dfDict={}
  for run in ['historical','2025','2085']:
    dfDict[run]=pd.read_csv('../data/LVD.%s_%s_%04d0101-%04d1231.csv'%\
               (m,runDict[run][0],runDict[run][1],runDict[run][2]),header=0)
    
    #trajDict[run]=np.load('../data/traj.%s.%s_%s_%04d0101-%04d1231.npy'%\
    #               (site,m,runDict[run][0],runDict[run][1],runDict[run][2]))
    trajDict[run]=np.load('../data/back_traj.%s.%s_%s_%04d0101-%04d1231.npy'%\
                   (site,m,runDict[run][0],runDict[run][1],runDict[run][2]))
    print(run,dfDict[run].date.size,trajDict[run].shape)
    #nday,_,nt=traj.shape

  #for specific synoptic wind direction
  for wd_cat in [60,90,120,150,180]:
  #for wd_cat in [60,]:

    for hr in range(0,25):
    #for hr in [24,]:
      plt.close()
      ax=plt.subplot(111)
      plotTW(ax,lonTW,latTW,topo)
      ax.plot(lonlat[site][0],lonlat[site][1],'rs',zorder=5) 
  
      for clr,run in zip(['b','r'],['2025','2085']):
        synoptic_wind=dfDict[run].meanWD.values
        idxList=np.where(np.logical_and(synoptic_wind>=(wd_cat-15),synoptic_wind<(wd_cat+15)))[0]
        traj_cat=trajDict[run][idxList]
        ncase,_,nt=traj_cat.shape
        #get synoptic informaton for plotting
        df_cat=dfDict[run].query(f'meanWD >= {wd_cat-15} and meanWD < {wd_cat+15}')[['date','meanWD','meanWS']]
        print(run,idxList.shape[0],df_cat.date.size)
        plt.scatter(traj_cat[:,0,hr*2],traj_cat[:,1,hr*2],color=clr,s=2)
        plt.title(f'Upstream Wind Direction={wd_cat}$\pm15^\circ$',fontsize=12)
        plt.title(f'{m}\n{runDict[run][0]}',fontsize=8,loc='left')
        plt.title(f'{hr:02d} hour',fontsize=8,loc='right')
        #plt.title(f'TaiESM1\nSSP-585',fontsize=8)
        #plt.title(f'{hr:02d} hour',loc='right',fontsize=8)
        plt.plot(traj_cat[0,0,:hr*2+1],traj_cat[0,1,:hr*2+1],lw=0.4,color=clr,\
                 label=f'{runDict[run][1]}-{runDict[run][2]}({df_cat.date.size} days)')
        for k in range(1,ncase):
          plt.plot(traj_cat[k,0,:hr*2+1],traj_cat[k,1,:hr*2+1],lw=0.4,color=clr,label='_')
      plt.legend(loc=2,fontsize=10)  
      #plt.savefig(f'../figures/traj/traj.{wd_cat:03d}{site}.hr{hr:02d}.png')
      plt.savefig(f'../figures/traj/back_traj.{wd_cat:03d}{site}.hr{hr:02d}.png')
  


