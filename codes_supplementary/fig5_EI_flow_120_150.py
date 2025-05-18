#!/home/mileshsieh/anaconda3/envs/torch/bin/python

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from VAE import variationalAutoEncoder,Encoder,Decoder,Reshape
#from scipy.optimize import fsolve
from scipy.optimize import fsolve
import matplotlib.colors as mc
import matplotlib,glob
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter,FixedLocator,FixedFormatter
matplotlib.rc('xtick',labelsize=20)
matplotlib.rc('ytick',labelsize=20)

#[min_lat,max_lat,min_lon,max_lon]=[21.7,25.5,119.8,122.2]
[min_lat,max_lat,min_lon,max_lon]=[21.2,26,119.3,122.7]
EPA_threshold=54.4 #mug/m^3

pltCfg={'PM25':[20,80,'jet'],'ratio':[0.0,3.0,'hot_r']}

pm25_cmap = colors.ListedColormap(['green', 'yellow', 'orange', 'red'])
pm25_cmap.set_over('purple')
#cmap.set_under('blue')
pm25_bounds = [0.0, 15.5, 35.5, 54.5, 120.0]
pm25_norm = colors.BoundaryNorm(pm25_bounds, pm25_cmap.N)

colors_under1=plt.cm.PuBu_r(np.linspace(0.0,1.0,8))
#colors_over1=plt.cm.YlOrRd(np.linspace(0.3,1.0,8))
colors_over1=plt.cm.hot_r(np.linspace(0.3,1.0,8))
#上面兩個色階的數量(8)要平衡才能做出中間的分隔色
all_colors=np.vstack((colors_under1,colors_over1))
r_cmap=colors.LinearSegmentedColormap.from_list('raito_map',all_colors)
r_norm=colors.TwoSlopeNorm(vmin=0.0,vcenter=1.0,vmax=3.0)

def plotTW(var,ax,lon,lat,data,title,norm=None,cmap=None):
    idx=np.argsort(data)
    #print(idx.shape,idx[:5])
    ax.contour(lonTW,latTW,topo,levels=[0.01,],colors='k',linewidths=2)
    ax.contourf(lonTW,latTW,topo_m,cmap='Greys')

    cs=ax.scatter(lon[idx],lat[idx],c=data[idx],s=50,norm=norm,cmap=cmap)
    ax.set_title(title,fontsize=15)
    plt.xticks([])
    plt.yticks([])
    return cs


[a,b,x0,y0]=[0.08303117,0.95491014,8.25855574,0.11692898]
[a0,a1,wd0]=[2.93885019,-1.29016647,103.75438081]
def ws_h(x, y):
    return a * (x - x0)**2 + b * (y - y0)**2

def wd_h(x, y):
    return a1*np.arctan(y/(x-a0))/np.pi*180+wd0

def wind2xy(wd,ws):
    def equations(vars):
        x, y = vars
        alpha=np.tan((wd-wd0)*np.pi/180/a1)
        eq1 = x-a0-y/alpha
        eq2 = a*(x-x0)**2+b*(y-y0)**2-ws
        return [eq1, eq2]

    x, y =  fsolve(equations, (2.1, 0))
    return x,y

def plotStreamLine(ax,topo,vardata,title,rtitle,ltitle):
    ny,nx=topo.shape
    xx,yy=np.meshgrid(np.arange(nx),np.arange(ny))
    ax.contour(topo,levels=[0.05,],colors=['k'])
    #ax.contour(topo,levels=[0.05,0.2],colors=['k'])
    ws=np.sqrt(vardata[0,:,:]**2+vardata[1,:,:]**2)
    #print(ws.min(),ws.max())
    strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
            color=ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),density=0.6,linewidth=2.2,arrowsize=2,zorder=3)
    #ax.plot([xstart,xend,xend,xstart,xstart],[ystart,ystart,yend,yend,ystart],lw=2,ls='--')
    ax.contourf(topo,levels=[0.2,5.0],colors=['darkgreen'],zorder=5)
    ax.set_xlim(0,nx)
    ax.set_ylim(0,ny)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=25,loc='center')
    return strm

cities={'Taipei':[121.5598,25.09108],'Kaohsiung':[120.311922,22.620856]}

if __name__=='__main__':
  num_input_channels=2
  latent_dim=2
  beta=0.01
  dataset='ctrl'
  thd=11
  seed=3
  ts=6
  te=54
  #for write out
  sf='vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,ts,te,seed,thd)
  device = torch.device("cpu")
  # Load model
  model_ae = torch.load('../data/leevortex.%s.pth'%sf,map_location=torch.device('cpu'))
  model_ae.eval()

  #load topo and lat/lon of generated circulation
  ys=150
  ye=451
  xs=20
  xe=321
  lonTW=np.load('../data/tw_s_lon.npy')[xs:xe]
  latTW=np.load('../data/tw_s_lat.npy')[ys:ye]
  topo=np.load('../data/topodata.npy')[ys:ye,xs:xe]
  topo_m=np.copy(topo)
  topo_m[topo_m==0]=np.nan

  xx,yy=np.meshgrid(lonTW[::5],latTW[::5])

  #load pm25 data
  pm25_data=pd.read_csv('../data/PM25daily_74sta_20082019ColdSeason.csv').values
  ratio_data=pd.read_csv('../data/PM25ratio_74sta_20082019ColdSeason.csv').values
  lon=pm25_data[0,1:].astype(float)
  lat=pm25_data[1,1:].astype(float)
  idx_west=~((lon>121)&(lat<24.8))
  lon=lon[idx_west]
  lat=lat[idx_west]
  pm25=pm25_data[2:,1:].astype(np.float32)[:,idx_west]
  pm25r=ratio_data[2:,1:].astype(float)[:,idx_west]
  dateList=ratio_data[2:,0].astype(int)
  nOBS_all=lon.shape[0]

  #load flow regime
  yr_start=2008
  yr_end=2010
  era5=pd.read_csv('../data/cold_season_weather_regimes_with_idx.csv',header=0,parse_dates=[0])
  era5=era5[(era5.yyyymmdd>=(yr_start*10000+101))&(era5.yyyymmdd<=(yr_end*10000+1231))&(era5.prRatio<=0.3)&(era5.meanWS>=4.0)&(era5.meanWS<=12.0)&(era5.meanWD>=45)&(era5.meanWD<=195)]
  def getDateList(df,wd):
    df_within=df[(df.meanWD>=wd-15)&(df.meanWD<=wd+15)]
    dlist=df_within.yyyymmdd.values.astype(int)
    dtlist=df_within.date.values
    mean_ws=df_within.meanWS.mean()
    mean_wd=df_within.meanWD.mean()
    return dlist,dtlist,mean_wd,mean_ws

  #plot
  plt.close()
  fig=plt.figure(figsize=(12,8))
  for i,wd in enumerate([120,150]):
    dlist,dtlist,meanWD,meanWS=getDateList(era5,wd)
    print(wd,dlist.shape[0],meanWD,meanWS)

    x_temp,y_temp=wind2xy(meanWD,meanWS)
    local_flow=model_ae.decode(torch.Tensor([x_temp,y_temp]).to(device)).detach().cpu().numpy()[0]*thd

    #calculate enhancement index
    idxList=[np.where(dateList==idate)[0][0] for idate in dlist]
    #calculate enhancement index
    pm25r_days=pm25r[idxList,:]
    conc=[]
    enh_idx=[]
    lon_enh=[]
    lat_enh=[]
    for k in range(pm25r_days.shape[1]):
      stn_pm25r=pm25r_days[:,k]
      stn_pm25r=stn_pm25r[~np.isnan(stn_pm25r)]
      if stn_pm25r.shape[0]==0:
        continue
      enh=(np.where(stn_pm25r>1.0,1,0).sum())/stn_pm25r.shape[0]
      #print(k,enh,np.where(stn_pm25r>1.0,1,0).sum(),stn_pm25r.shape[0])
      enh_idx.append(enh)
      lon_enh.append(lon[k])
      lat_enh.append(lat[k])
      conc.append(np.nanmean(pm25[idxList,k]))
    enh_idx=np.array(enh_idx)
    #conc=np.array(conc)
    lon_enh=np.array(lon_enh)
    lat_enh=np.array(lat_enh)
    #rOrange=[]
    #rRed=[]
    #plot pm25 of Taiwan
    ax=plt.subplot(1,2,i+1)
    lbl='$WD=%d\pm15^\circ(%d days)$\n$[WD=%.2f^\circ, WS=%.2fms^{-1}]$'%(wd,pm25r_days.shape[0],meanWD,meanWS)
    r_thd=0.7
    r_percentage_TW=(enh_idx>=r_thd).sum()/enh_idx.shape[0]*100
    cs_ratio=plotTW('ratio',ax,lon_enh,lat_enh,enh_idx,lbl,norm=colors.Normalize(0.0,1.0),cmap='jet')

    generated_ws=np.sqrt(local_flow[0,:,:]**2+local_flow[1,:,:]**2)
    strm=plt.streamplot(xx,yy,local_flow[0,:,:],local_flow[1,:,:],color=generated_ws,cmap='YlGnBu',
                        norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=2,zorder=3,arrowsize=2.5)
    ax.set_xlim(119.8,122.2)
    ax.set_ylim(21.7,25.5)

  fig.subplots_adjust(left=0.02,right=0.75,bottom=0.05,top=0.85)
  ax_cb_ratio = fig.add_axes([0.79, 0.05, 0.03, 0.8])
  cbar_ratio=plt.colorbar(cs_ratio,cax=ax_cb_ratio, ticks=np.linspace(0.0,1.0,6))
  cbar_ratio.set_label('Enhancement Index',fontsize=15)
  ax_cb_ws = fig.add_axes([0.9, 0.05, 0.03, 0.8])
  cbar_ws=plt.colorbar(strm.lines,cax=ax_cb_ws,extend='max')
  cbar_ws.set_label('Wind Speed (m/s)',fontsize=15)
  plt.annotate('(a)', xy=(0.04, 0.7), xytext=(0.01, 0.88),xycoords='figure fraction',fontsize=20)
  plt.annotate('(b)', xy=(0.68, 0.7), xytext=(0.41, 0.88),xycoords='figure fraction',fontsize=20)

  plt.suptitle('Enhancement Index and Generated Local Circulation(%d-%d)'%(yr_start,yr_end),fontsize=20)
  plt.savefig('../figures/fig5_EI_flow.png')
  
  

