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

def plot_no2(ax,data,lonTW,latTW,topo,levels,cmap,title,ltitle='',rtitle=''):
    #cf_topo=plotTW(ax,lonTW,latTW,topo)
    no2=data[0]
    uv=data[1:]
    strm=plotStreamLine(ax,lonTW,latTW,topo,uv,'','','')
    print(no2.max())
    cf_no2=ax.contourf(lonTW,latTW,no2,\
                 levels=levels,extend='max',cmap=cmap)
    ax.set_title(title,fontsize=8)
    ax.set_title(ltitle,loc='left')
    ax.set_title(rtitle,loc='right')
    ax.set_xlim(119,122)
    ax.set_ylim(21.7,25.7)
    ax.grid()
    return strm,cf_no2

def plot_no2_noWind(ax,no2,lonTW,latTW,topo,levels,cmap,title,ltitle='',rtitle=''):
    plotTW(ax,lonTW,latTW,topo)
    print(no2.max())
    cf_no2=ax.contourf(lonTW,latTW,composite,\
                 levels=levels,extend='max',cmap=cmap)
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

    lonlat={'TPP':[120.4848,24.20943],'SNC':[120.2058,23.80023]}
    county = gpd.read_file('/data/huanghuai/DATA/MAP/twcounty/twcounty.shp')
    great_taipei=gpd.GeoSeries(county[county.COUNTYNAME.isin(['臺北市','新北市'])].geometry.union_all())

    #load historical/ssp-585 upstream flow regimes and local flows
    m='TaiESM1'
    sce='ssp585'
    runDict={'historical':['Historical',2001,2010],}
    for yr in range(2021,2100,10):
        runDict['%04d'%yr]=['SSP-585',yr,yr+9]

    dfDict={}
    no2Dict={}
    flowDict={}
    for run in ['2021','2091']:
        flow=np.load('../data/local_flow.%s_%s_%s0101-%s1231.npy'%\
                     (m,runDict[run][0],runDict[run][1],runDict[run][2]))
        df=pd.read_csv('../data/LVD.%s_%s_%04d0101-%04d1231.csv'%\
               (m,runDict[run][0],runDict[run][1],runDict[run][2]),header=0)
        no2_24hr=np.load('../data/no2_transport_24hr.%s_%s_%s0101-%s1231.npy'%\
                     (m,runDict[run][0],runDict[run][1],runDict[run][2]))
        dfDict[run]=df.copy()
        no2Dict[run]=no2_24hr.copy()
        flowDict[run]=flow.copy()
        print(run,df.date.size,no2_24hr.shape)

    lbl=['a','b']
    ilbl=0 
    #plot of entire composite
    plt.close()
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,run in zip(axes.ravel(),['2021','2091']):
        composite=no2Dict[run].mean(axis=0)*1.88
        cf_no2=plot_no2_noWind(ax,composite,lonTW,latTW,topo,\
              np.linspace(0,600,16),r_cmap,f'({lbl[ilbl]}) {runDict[run][1]}-{runDict[run][2]}\n({no2Dict[run].shape[0]} days)')
        great_taipei.boundary.plot(ax=ax,lw=1.2,color='navy',zorder=5)
        if ilbl==0:
            ax.set_ylabel('Latitude ($^\circ N$)')
        ax.set_xlabel('Longitude ($^\circ E$)')
        ilbl=ilbl+1
    fig.subplots_adjust(right=0.85)
    #ax_cb = fig.add_axes([0.82, 0.12, 0.03, 0.75])
    #cbar=plt.colorbar(cf_topo,cax=ax_cb)
    #cbar.set_label('Height (km)',fontsize=14)
    ax_cb_no2 = fig.add_axes([0.87, 0.12, 0.03, 0.75])
    cbar_no2=plt.colorbar(cf_no2,cax=ax_cb_no2,extend='max')
    cbar_no2.set_label('$NO_x$ (${\mu}g/m^3$)',fontsize=14)

    plt.suptitle(f'Comparison of $NO_x$ Distribution Composite in {m} {runDict[run][0]}',fontsize=15)
    plt.savefig(f'../figures/no2_comp_all.png')
    
