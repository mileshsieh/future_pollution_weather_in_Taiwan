#!/home/mileshsieh/anaconda3/envs/dask/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 08:52:46 2023

@author: miles
"""

import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from VAE import variationalAutoEncoder,Encoder,Decoder,Reshape
from scipy.optimize import fsolve
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib.colors as mc
import matplotlib,glob
from matplotlib import dates as mdates
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

#for local circulation generation
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

def uv2wd(u, v):
  return np.where(u==0,np.where(v>0,180.0,0.0),180.0 + np.arctan2(u, v) * 180.0 / np.pi)

def plotTW(var,ax,lon,lat,data,title,norm=None,cmap=None):
    idx=np.argsort(data)
    #print(idx.shape,idx[:5])
    ax.contour(lonTW,latTW,topo,levels=[0.01,],colors='k',linewidths=2)
    ax.contourf(lonTW,latTW,topo_m,cmap='Greys')

    cs=ax.scatter(lon[idx],lat[idx],c=data[idx],s=50,norm=norm,cmap=cmap)
    #for N TW
    #ax.set_xlim(119.9,122.1)
    #ax.set_ylim(24.2,25.4)
    #TW
    ax.set_xlim(min_lon,max_lon)
    ax.set_ylim(min_lat,max_lat)

    ax.set_title(title,fontsize=15)
    plt.xticks([])
    plt.yticks([])
    return cs

if __name__=='__main__':
    cities={'Taipei':[121.5598,25.09108],'Kaohsiung':[120.311922,22.620856]}
    #map plotting setup
    ys=150
    ye=451
    xs=20
    xe=321
    lonTW=np.load('../data/tw_s_lon.npy')[xs:xe]
    latTW=np.load('../data/tw_s_lat.npy')[ys:ye]
    topo=np.load('../data/topodata.npy')[ys:ye,xs:xe]
    topo_m=np.copy(topo)
    topo_m[topo_m==0]=np.nan
    lon_flow=lonTW[::5]
    lat_flow=latTW[::5]
    xx_TW,yy_TW=np.meshgrid(lonTW, latTW)
    xx_flow,yy_flow=np.meshgrid(lon_flow, lat_flow)


    #AI-TaiwanVVM setup
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

    ish=pd.read_csv('../data/noWx.obs.47918_wind.2008to2019.csv',header=0,parse_dates=[0,])
    ish['ymd']=ish.apply(lambda ser: pd.to_datetime(ser.yyyymmdd).strftime('%Y%m%d'),axis=1)
    ish=ish[(ish.yyyymmdd>='2008-01-01')&(ish.yyyymmdd<='2010-12-31')&(ish.ws925>=4.0)&(ish.ws925<=12.0)&(ish.wd925>=45)&(ish.wd925<=195)]
    
    def getDateList(df,wd_bin):
        df_within=df[(df.wd925>=wd_bin-15)&(df.wd925<=wd_bin+15)]
        dlist=df_within.ymd.values.astype(int)
        dtlist=df_within.yyyymmdd.values
        ishWsList=df_within.ws925.values
        ishWdList=df_within.wd925.values
        return dlist,dtlist,ishWsList,ishWdList
    wdList=[60,90,120,150,180]
    nRegime=len(wdList)
    plt.close()
    fig=plt.figure(figsize=(20,5))
    eilbl=['a','b','c','d','e']
    for i,wd_bin in enumerate(wdList):
        dlist,dtlist,ishWsList,ishWdList=getDateList(ish,wd_bin)
        print(wd_bin,dlist.shape[0],np.mean(ishWdList),np.mean(ishWsList))
        #generate local circulation
        temp_xy=np.array([wind2xy(ishWdList[k],ishWsList[k]) for k in range(len(ishWdList))])
        local_flow=model_ae.decode(torch.Tensor(temp_xy).to(device)).detach().cpu().numpy()*thd
        #interpolate
        u=np.array([griddata((xx_flow.flatten(),yy_flow.flatten()),local_flow[k,0,:,:].flatten(),\
                   (xx_TW,yy_TW),method='cubic') for k in range(len(local_flow))])
        v=np.array([griddata((xx_flow.flatten(),yy_flow.flatten()),local_flow[k,1,:,:].flatten(),\
                   (xx_TW,yy_TW),method='cubic') for k in range(len(local_flow))])
        local_flow_2km=np.mean(np.swapaxes(np.array([u,v]),0,1),axis=0)
 
        #calculate enhancement index
        idxList=[np.where(dateList==idate)[0][0] for idate in dlist]
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
        conc=np.array(conc)
        lon_enh=np.array(lon_enh)
        lat_enh=np.array(lat_enh)
        #mean concentration
        ax=plt.subplot(1,nRegime,i+1)
        
        r_thd=0.7
        r_percentage_TW=(enh_idx>=r_thd).sum()/enh_idx.shape[0]*100
        #strm=ax.streamplot(xx_flow,yy_flow,local_flow[0,:,:],local_flow[1,:,:],color='blue')
        spd=np.sqrt(local_flow_2km[0,:,:]*local_flow_2km[0,:,:]+local_flow_2km[1,:,:]*local_flow_2km[1,:,:])
        strm=ax.streamplot(xx_TW,yy_TW,local_flow_2km[0,:,:],local_flow_2km[1,:,:],\
                           color=spd,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),\
                           linewidth=2,density=0.8,zorder=3,arrowsize=1.5)
        lbl='(%s) $WD=%d\pm15^\circ(%d days)$\n[%.1f%% station EI over %.1f]'%(eilbl[i],wd_bin,pm25r_days.shape[0],r_percentage_TW,r_thd)         
        cs_ratio=plotTW('ratio',ax,lon_enh,lat_enh,enh_idx,lbl,norm=colors.Normalize(0.0,1.0),cmap='jet')
        ax.set_xlim(119,122)
        ax.set_ylim(21.5,26)

    fig.subplots_adjust(left=0.02,right=0.82,bottom=0.02,top=0.78)
    ax_cb_ratio = fig.add_axes([0.84, 0.08, 0.02, 0.7])
    cbar_ratio=plt.colorbar(cs_ratio,cax=ax_cb_ratio, ticks=[0.0,0.5,1.0,2.0,3.0])
    cbar_ratio.set_label('Enhancement Index',fontsize=15)
    ax_cb_ws = fig.add_axes([0.92, 0.08, 0.02, 0.7])
    cbar_ws=plt.colorbar(strm.lines,cax=ax_cb_ws,extend='max')
    cbar_ws.set_label('WS (m/s)',fontsize=15)

    plt.suptitle('Observed Pollution Deterioration Corresponding to Various Local Circulation Patterns (%d-%d)'%(yr_start,yr_end),fontsize=20)
    plt.savefig('../figures/fig2_validation.png',dpi=300)
