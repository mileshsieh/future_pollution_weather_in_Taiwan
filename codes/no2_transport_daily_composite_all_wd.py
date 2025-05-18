#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mc
import geopandas as gpd

def plotTW(ax,lonTW,latTW,topo):
    xx,yy=np.meshgrid(lonTW,latTW)
    #cf=ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    ax.contour(xx,yy,topo,levels=[0.05,],colors=['k'])
    ax.contour(xx,yy,topo,levels=[0.5,],colors=['k'],linewidths=[0.5])
    #cb=plt.colorbar(cf,extend='max')
    #cb.set_label('Height (km)')
    return

def plotStreamLine(ax,lonTW,latTW,topo,vardata,title,rtitle,ltitle):
#def plotStreamLine(ax,lonTW,latTW,topo,step,vardata,slon,slat,title,rtitle,ltitle):
    #xx_flow,yy_flow=np.meshgrid(lonTW[::step],latTW[::step])
    xx,yy=np.meshgrid(lonTW,latTW)
    plotTW(ax,lonTW,latTW,topo)
    #ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    #ax.contour(xx,yy,topo,levels=[0.05,],colors=['k'])
    #ax.contour(topo,levels=[0.05,0.2],colors=['k'])
    ws=np.sqrt(vardata[0,:,:]**2+vardata[1,:,:]**2)
    #print(ws.min(),ws.max())
    strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],\
            color='gray',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=1.2,density=1.0,zorder=3,arrowsize=1)
    #strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
    #        color=ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=0.8,zorder=3,arrowsize=1.5)
    ax.set_xlim(118,lonTW[-1])
    ax.set_ylim(21.7,25.7)
    plt.grid()
    #ax.xaxis.set_major_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=15,loc='center')
    #cb=plt.colorbar(strm.lines,extend='max')
    #cb.set_label('WS (m/s)')
    return strm

# no2 colormap
colors_below=plt.cm.Blues(np.linspace(0.0,0.5,5))
colors_middle=plt.cm.Greens(np.linspace(0.2,0.7,5))
colors_over=plt.cm.hot_r(np.linspace(0.3,0.8,5))
all_colors=np.vstack((colors_below,colors_middle,colors_over))
#all_colors=np.vstack((colors_middle,colors_over))
r_cmap=mc.LinearSegmentedColormap.from_list('no2_map',all_colors)

def plot_no2(ax,data,lonTW,latTW,topo,levels,cmap,ext,title,ltitle='',rtitle='',only_conc=False):
    plotTW(ax,lonTW,latTW,topo)
    if only_conc:
        no2=data
    else:
        no2=data[0]
        uv=data[1:]
        strm=plotStreamLine(ax,lonTW,latTW,topo,uv,'','','')
    print(no2.max())
    cf_no2=ax.contourf(lonTW,latTW,no2,\
                 levels=levels,extend=ext,cmap=cmap)
    ax.set_title(title,fontsize=8)
    ax.set_title(ltitle,loc='left')
    ax.set_title(rtitle,loc='right')
    ax.set_xlim(119,122)
    ax.set_ylim(21.7,25.7)
    ax.grid()
    if only_conc:
        return cf_no2
    else:
        return strm,cf_no2

def plot_no2_noWind(ax,no2,lonTW,latTW,topo,levels,cmap,ext,title,ltitle='',rtitle=''):
    plotTW(ax,lonTW,latTW,topo)
    print(no2.max())
    cf_no2=ax.contourf(lonTW,latTW,composite,\
                 levels=levels,extend=ext,cmap=cmap)
    ax.set_title(title)
    ax.set_title(ltitle,loc='left')
    ax.set_title(rtitle,loc='right')
    ax.set_xlim(119,122)
    ax.set_ylim(21.7,26)
    ax.grid()
    return cf_no2

if __name__=='__main__':
    (xstart,xend,ystart,yend)=(20,321,150,451)
    step=1
    lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend:step]
    latTW=np.load('../data/tw_s_lat.npy')[ystart:yend:step]
    topo=np.load('../data/topodata.npy')[ystart:yend:step,xstart:xend:step]
    xx_TW,yy_TW=np.meshgrid(lonTW, latTW)
    county = gpd.read_file('/data/huanghuai/DATA/MAP/twcounty/twcounty.shp')
    great_taipei=gpd.GeoSeries(county[county.COUNTYNAME.isin(['臺北市','新北市'])].geometry.union_all())

    lonlat={'TPP':[120.4848,24.20943],'SNC':[120.2058,23.80023]}

    #load historical/ssp-585 upstream flow regimes and local flows
    m='TaiESM1'
    sce='SSP-585'

    dfDict={}
    no2Dict={}
    for yr_start in [2021,2091]:
        #flow=np.load('../data/local_flow.%s_%s_%d0101-%d1231.npy'%\
        #             (m,sce,yr_start,yr_start+9))
        df=pd.read_csv('../data/LVD.%s_%s_%04d0101-%04d1231.csv'%\
               (m,sce,yr_start,yr_start+9),header=0)
        no2_24hr=np.load('../data/no2_transport_24hr.%s_%s_%s0101-%s1231.npy'%\
                     (m,sce,yr_start,yr_start+9))
        dfDict[yr_start]=df.copy()
        no2Dict[yr_start]=no2_24hr.copy()
        #flowDict[run]=flow.copy()
        print(yr_start,df.date.size,no2_24hr.shape)
 
    #composite of different prevailing wind direction
    no2_wd={}
    wdList=[60,90,120,150,180]
    ndays_wd={}
    for wd_cat in wdList:
        conc=[]
        ndays=[]
        for yr_start in [2021,2091]:
            synoptic_wind=dfDict[yr_start].meanWD.values
            idxList=np.where(np.logical_and(synoptic_wind>=(wd_cat-15),synoptic_wind<(wd_cat+15)))[0]
            composite=no2Dict[yr_start][idxList,:,:].mean(axis=0)*1.88
            conc.append(composite)
            ndays.append(idxList.shape[0])
        conc.append(conc[1]-conc[0])
        ndays.append(ndays[1]-ndays[0])
        no2_wd[wd_cat]=conc
        ndays_wd[wd_cat]=ndays
    lbl=['a','b','c','d','e','f','g','h','i','j']
    ilbl=0
    plt.close()
    fig,axes=plt.subplots(2,5,figsize=(12,6))
    for ax,wd_cat in zip(axes[0,:],wdList):
        print(ax)
        print(wd_cat,)
        #plot 2091-2100 concentration
        #cf_no2=ax.contourf(lonTW,latTW,no2_wd[wd_cat][1],\
        #         levels=np.linspace(0,300,16),extend='max',cmap=r_cmap)

        cf_no2=plot_no2(ax,no2_wd[wd_cat][1],lonTW,latTW,topo,\
              np.linspace(0,600,16),r_cmap,'max',f'({lbl[ilbl]}) $WD={wd_cat}\pm15^\circ$\n $NO_2$ Conc. ({ndays_wd[wd_cat][1]} days)',\
              only_conc=True)
        great_taipei.boundary.plot(ax=ax,lw=1.2,color='navy',zorder=5)
        ilbl=ilbl+1
    for ax,wd_cat in zip(axes[1,:],wdList):
        #plot difference between 2020s and 2090s
        cf_diff=plot_no2(ax,no2_wd[wd_cat][2],lonTW,latTW,topo,\
              np.linspace(-90,90,19),'bwr','both',f'({lbl[ilbl]}) [2091-2100] - [2021-2030]\n({ndays_wd[wd_cat][2]:+d} days)',only_conc=True)
        great_taipei.boundary.plot(ax=ax,lw=1.2,color='navy',zorder=5)
        ilbl=ilbl+1
    for icol in range(len(wdList)):   
        axes[0,icol].set_xticklabels([])
        axes[1,icol].set_xlabel('Longitude ($^\circ E$)')
        if icol>0:
            axes[0,icol].set_yticklabels([])
            axes[1,icol].set_yticklabels([])
        else:
            axes[0,icol].set_ylabel('Latitude ($^\circ E$)')
            axes[1,icol].set_ylabel('Latitude ($^\circ E$)')
        
    fig.subplots_adjust(left=0.05,right=0.83)
    ax_cb_no2=fig.add_axes([0.87, 0.52, 0.03, 0.35])
    cbar_no2=plt.colorbar(cf_no2,cax=ax_cb_no2,extend='max')
    cbar_no2.set_label('$NO_2$ Conc. (${\mu}g/m^3$)',fontsize=10)
    ax_cb_diff=fig.add_axes([0.87, 0.11, 0.03, 0.35])
    cbar_diff=plt.colorbar(cf_diff,cax=ax_cb_diff,extend='both')
    cbar_diff.set_label('$NO_2$ Conc.Difference (${\mu}g/m^3$)',fontsize=10)
    
    plt.suptitle(f'Composite of $NO_2$ Distribution on Lee Vortex Days and Climate Shifts (2091-2100)',fontsize=16)
    plt.savefig(f'../figures/no2_2020s_2090s_WD.png')
             
