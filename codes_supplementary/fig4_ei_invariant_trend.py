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

EPA_threshold=54.4 #mug/m^3

if __name__=='__main__':
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
    era5=pd.read_csv('../data/cold_season_weather_regimes_with_idx.csv',header=0,parse_dates=[0])

    def getDateList(df,wd):
        df_within=df[(df.meanWD>=wd-15)&(df.meanWD<=wd+15)]
        dlist=df_within.yyyymmdd.values.astype(int)
        dtlist=df_within.date.values
        mean_ws=df_within.meanWS.mean()
        mean_wd=df_within.meanWD.mean()
        return dlist,dtlist,mean_wd,mean_ws
    

    wdList=[60,90,120,150,180]
    nRegime=len(wdList)
    ei={2008:[],2011:[]}
    pm={2008:[],2011:[]}
    for yr_start in [2008,2011]:
        yr_end=yr_start+2
        era5_3yr=era5[(era5.yyyymmdd>=(yr_start*10000+101))&(era5.yyyymmdd<=(yr_end*10000+1231))&(era5.prRatio<=0.3)&(era5.meanWS>=4.0)&(era5.meanWS<=12.0)&(era5.meanWD>=45)&(era5.meanWD<=195)]
        print(yr_start,yr_end,era5_3yr.date.size)
        for i,wd in enumerate(wdList):
            dlist,dtlist,meanWD,meanWS=getDateList(era5_3yr,wd)
            #print(wd,dlist.shape[0],meanWD,meanWS)

            #calculate enhancement index
            idxList=[np.where(dateList==idate)[0][0] for idate in dlist]
            #calculate enhancement index
            pm25r_days=pm25r[idxList,:]
            conc=[]
            enh_idx=[]
            for k in range(pm25r_days.shape[1]):
                stn_pm25r=pm25r_days[:,k]
                stn_pm25r=stn_pm25r[~np.isnan(stn_pm25r)]
                if stn_pm25r.shape[0]==0:
                    continue
                enh=(np.where(stn_pm25r>1.0,1,0).sum())/stn_pm25r.shape[0]
                #print(k,enh,np.where(stn_pm25r>1.0,1,0).sum(),stn_pm25r.shape[0])
                enh_idx.append(enh)
                conc.append(np.nanmean(pm25[idxList,k]))
            enh_idx=np.array(enh_idx)
            conc=np.array(conc)
            #mean concentration
            #conc=np.nanmean(pm25[idxList,:],axis=0)
            #plot pm25 of Taiwan
            pPolluted_TW=(conc>=54.5).sum()/conc.shape[0]*100
            pm[yr_start].append(pPolluted_TW)
            r_thd=0.7
            r_percentage_TW=(enh_idx>=r_thd).sum()/enh_idx.shape[0]*100
            ei[yr_start].append(r_percentage_TW) 
            print(wd,pm25r_days.shape[0],pPolluted_TW,r_percentage_TW)
    lblList=['$WD=%d\pm15^\circ$'%k for k in wdList]
    x=np.arange(len(wdList))  # the label locations
    matplotlib.rc('xtick',labelsize=12)
    matplotlib.rc('ytick',labelsize=15)
    plt.close()
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(24,8))
    for i,(ax,dataDict) in enumerate(zip(axes,[pm,ei])):
        width=0.3  # the width of the bars
        mul=-1
        for yr,cntList in dataDict.items():
            rects=ax.bar(x+width*mul,cntList,width,align='edge',label='%d-%d'%(yr,yr+2))
            ax.bar_label(rects,fmt='%.1f',padding=3,fontsize=13)
            mul+=1

        if i==0:
            ax.set_title('Polluted: $PM_{2.5}$ Concentration Exceeded 54.5 ${\mu}g/m^3$',fontsize=20)
        else:
            ax.set_title('Polluted: Enhancement Index Exceeded 0.7',fontsize=20)

        ax.set_ylabel('Ratio of Polluted Station (%)',fontsize=20)
        ax.set_xticks(x, lblList)
        ax.set_xticklabels(lblList)
        ax.legend(loc='upper left',fontsize=15)
        ax.set_ylim(0,100)
    plt.suptitle('Ratio of Polluted Stations in Lee Vortex Days\nunder Various Upstream Flow Regimes',fontsize=30,y=1.04)
    plt.annotate('(a)', xy=(0.22, 0.94), xytext=(0.063, 0.83),xycoords='figure fraction',fontsize=22)
    plt.annotate('(b)', xy=(0.42, 0.95), xytext=(0.49, 0.83),xycoords='figure fraction',fontsize=22)
    plt.savefig('../figures/fig4_polluted_event_count.png',bbox_inches='tight',dpi=300)

