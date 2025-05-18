#!/home/mileshsieh/anaconda3/envs/dask/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 08:52:46 2023

@author: miles
"""

import numpy as np
import pandas as pd
import xarray as xr
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
    
    #load ERA5 fro plot composite wind 
    ERA5_SFC_DIR='/data/dadm1/reanalysis/ERA5/SFC/day'
    ERA5_PRS_DIR='/data/dadm1/reanalysis/ERA5/PRS/day'
    #ERA5 wind field at 1000mb
    tmp=[]
    for var in ['u','v']:
      tmp.append(np.concatenate([np.array(sorted(glob.glob('%s/%s/%04d/*'%(ERA5_PRS_DIR,var,yr)))) for yr in range(yr_start,yr_end+1)]))
    fList=np.concatenate(tmp)
    #preprocess
    def _preprocess_wind(ds):
      return ds.reindex(latitude=list(reversed(ds.latitude))).sel(latitude=slice(min_lat-1,max_lat+1),longitude=slice(min_lon-1,max_lon+1),\
                                                               level=1000.0)
    ds_ERA5_wind=xr.open_mfdataset(fList,preprocess=_preprocess_wind)
    ds_ERA5_wind.coords['time']=ds_ERA5_wind.time.dt.floor('1D')

    lonERA5=ds_ERA5_wind.longitude.values
    latERA5=ds_ERA5_wind.latitude.values
    xx,yy=np.meshgrid(lonERA5,latERA5)

    wdList=[60,90,120,150,180]
    nRegime=len(wdList)
    plt.close()
    fig=plt.figure(figsize=(20,8))
    pmlbl=['a','b','c','d','e']
    eilbl=['f','g','h','i','j']
    for i,wd in enumerate(wdList):
        dlist,dtlist,meanWD,meanWS=getDateList(era5,wd)
        print(wd,dlist.shape[0],meanWD,meanWS)

        #get ERA5 composite
        uERA5=ds_ERA5_wind.sel(time=dtlist).mean(dim='time').u.values
        vERA5=ds_ERA5_wind.sel(time=dtlist).mean(dim='time').v.values

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
        conc=np.array(conc)
        lon_enh=np.array(lon_enh)
        lat_enh=np.array(lat_enh)
        #mean concentration
        #conc=np.nanmean(pm25[idxList,:],axis=0)
        #plot pm25 of Taiwan
        ax=plt.subplot(2,nRegime,i+1)
        pPolluted_TW=(conc>=54.5).sum()/conc.shape[0]*100
        lbl='(%s) $WD=%d\pm15^\circ(%d days)$\n[%.1f%% stations polluted]'%(pmlbl[i],wd,pm25r_days.shape[0],pPolluted_TW)
        strm=ax.streamplot(xx,yy,uERA5,vERA5,color='blue')         
        cs_pm25=plotTW('PM25',ax,lon_enh,lat_enh,conc,lbl,norm=pm25_norm,cmap=pm25_cmap)
        #q1=ax.quiver(xx,yy,uERA5,vERA5,scale=40,color='blue',width=0.02,units='xy')
        
        ax=plt.subplot(2,nRegime,nRegime+i+1)
        
        r_thd=0.7
        r_percentage_TW=(enh_idx>=r_thd).sum()/enh_idx.shape[0]*100
        strm=ax.streamplot(xx,yy,uERA5,vERA5,color='blue')         
        cs_ratio=plotTW('ratio',ax,lon_enh,lat_enh,enh_idx,'(%s) [%.1f%% station EI over %.1f]'%(eilbl[i],r_percentage_TW,r_thd),norm=colors.Normalize(0.0,1.0),cmap='jet')
        #q1=ax.quiver(xx,yy,uERA5,vERA5,scale=40,color='blue',width=0.02,units='xy')

    fig.subplots_adjust(left=0.02,right=0.85,bottom=0.02,top=0.85)
    ax_cb_pm25 = fig.add_axes([0.87, 0.47, 0.02, 0.38])
    cbar_pm25=plt.colorbar(cs_pm25,cax=ax_cb_pm25, ticks=pm25_bounds,spacing='proportional')
    cbar_pm25.set_label('$PM_{2.5} Concentration ({\mu}g/m^3)$',fontsize=15)
    ax_cb_ratio = fig.add_axes([0.87, 0.02, 0.02, 0.38])
    cbar_ratio=plt.colorbar(cs_ratio,cax=ax_cb_ratio, ticks=[0.0,0.5,1.0,2.0,3.0])
    cbar_ratio.set_label('Enhancement Index',fontsize=15)
    #veclenght = 10
    #maxstr = '%3.1f m/s' % veclenght
    #plt.quiverkey(q1,0.88,0.87,10,'10.0 m/s',labelpos='E', coordinates='figure').set_zorder(11)

    plt.suptitle('$PM_{2.5}$ Concentration and Enhancement Index under Various Flow Regime(%d-%d)'%(yr_start,yr_end),fontsize=30)
    plt.savefig('../figures/fig3_bifurcation_enh_tw.png')
