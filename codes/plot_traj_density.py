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
    ax.contour(xx,yy,topo,levels=[0.05,],colors=['k'],zorder=7)
    ax.set_xlim(118,lonTW[-1])
    ax.set_ylim(latTW[0],26)
    #cb=plt.colorbar(cf,extend='max')
    #cb.set_label('Height (km)')
    return cf

def calcDensity(xy,xbins,ybins):
    #xtk=ytk=0.5*(bins[1:]+bins[:-1])
    xbin=np.digitize(xy[:,0],xbins)
    ybin=np.digitize(xy[:,1],ybins)
    cnt=np.zeros((ybins.shape[0]-1,xbins.shape[0]-1))

    for i in range(xbins.shape[0]-1):
        for j in range(ybins.shape[0]-1):
            #print(i,j,xbin[(xbin==i)&(ybin==j)].shape[0])
            cnt[j,i]=xbin[(xbin==i+1)&(ybin==j+1)].shape[0]
    return cnt

if __name__=='__main__':
  #domain and step of the original TaiwanVVM 2km simulation
  (xstart,xend,ystart,yend)=(20,321,150,451)
  topo=np.load('../data/topodata.npy')[ystart:yend,xstart:xend]
  lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend]
  latTW=np.load('../data/tw_s_lat.npy')[ystart:yend]

  #get emission xy index
  lonlat={'TPP':[120.4848,24.20943],'SNC':[120.2058,23.80023]}

  #load historical/ssp-585 upstream flow regimes and local flows
  m='TaiESM1'
  sce='ssp585'
  site='TPP'
  runDict={'historical':['Historical',2001,2010],}
  for yr in range(2025,2095,10):
    runDict['%04d'%yr]=['SSP-585',yr,yr+9]

  trajDict={}
  dfDict={}
  for run in ['historical','2025','2085']:
    dfDict[run]=pd.read_csv('../data/sites.%s_%s_%04d0101-%04d1231.csv'%\
               (m,runDict[run][0],runDict[run][1],runDict[run][2]),header=0)
    
    trajDict[run]=np.load('../data/traj.%s.%s_%s_%04d0101-%04d1231.npy'%\
                   (site,m,runDict[run][0],runDict[run][1],runDict[run][2]))
    print(run,dfDict[run].date.size,trajDict[run].shape)
    #nday,_,nt=traj.shape

  xbins=np.arange(119.0,122.0+0.1,0.25)
  ybins=np.arange(22.0,25.5+0.1,0.25)

  #for specific synoptic wind direction
  #for wd_cat in [60,90,120,150,180]:
  for wd_cat in [60,90,120,150]:
      hr=24
      plt.close()
      fig,axes=plt.subplots(1,2,figsize=(18,8))
      #for clr,run in zip(['b','r'],['2025','2085']):
      for ax,run in zip(axes.ravel(),['2025','2085']):
        cf=plotTW(ax,lonTW,latTW,topo)
        ax.plot(lonlat[site][0],lonlat[site][1],'bs',zorder=7) 
        synoptic_wind=dfDict[run].meanWD.values
        idxList=np.where(np.logical_and(synoptic_wind>=(wd_cat-15),synoptic_wind<(wd_cat+15)))[0]
        traj_cat=trajDict[run][idxList]
        ncase,_,nt=traj_cat.shape
        print(wd_cat,run,ncase,idxList.shape[0])
        xy=np.swapaxes(traj_cat[:,:,1:],1,2).reshape((ncase*(nt-1),2))
        cnt=calcDensity(xy,xbins,ybins)
        for (j,i),days in np.ndenumerate(cnt):
          if days>0:
            ax.text(0.5*(xbins[i]+xbins[i+1]),0.5*(ybins[j]+ybins[j+1]),'%d'%days,color='skyblue',ha='center',va='center',fontsize=8,zorder=4)
        cnt[cnt==0]=np.nan
        cf2=ax.pcolormesh(xbins,ybins,cnt,cmap='YlOrRd',vmin=0.0,vmax=300.0,zorder=3)
        ax.scatter(traj_cat[:,0,hr*2],traj_cat[:,1,hr*2],color='k',s=2,zorder=6)
        for k in range(ncase):
          ax.plot(traj_cat[k,0,:hr*2+1],traj_cat[k,1,:hr*2+1],color='grey',linewidth=0.3,alpha=0.3,zorder=6)
        ax.set_title(f'{m} {runDict[run][0]}\n{runDict[run][1]}-{runDict[run][2]}')
      fig.subplots_adjust(right=0.8)
      ax_cb = fig.add_axes([0.82, 0.12, 0.03, 0.75])
      cbar=plt.colorbar(cf,cax=ax_cb)
      cbar.set_label('Height (km)',fontsize=14)
      ax_cb_ws = fig.add_axes([0.9, 0.12, 0.03, 0.75])
      cbar_ws=plt.colorbar(cf2,cax=ax_cb_ws,extend='max')
      cbar_ws.set_label('Hours',fontsize=14)

      plt.suptitle(f'Pollutant Transport Pathway Density (WD={wd_cat})',fontsize=20)  
      plt.savefig(f'../figures/traj.{wd_cat:03d}{site}.hr{hr:02d}.png')
  


