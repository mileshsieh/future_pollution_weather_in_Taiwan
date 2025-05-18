#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mc

def plotTW(ax,lonTW,latTW,topo):
    xx,yy=np.meshgrid(lonTW,latTW)
    #cf=ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    ax.contour(xx,yy,topo,levels=[0.05,],colors=['k'])
    ax.set_ylabel('Latitude ($^\circ N$)')
    ax.set_xlabel('Longitude ($^\circ E$)')
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
            color='gray',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=1.5,density=1.0,zorder=3,arrowsize=1.5)
    #strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
    #        color=ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=0.8,zorder=3,arrowsize=1.5)
    ax.set_xlim(118,lonTW[-1])
    ax.set_ylim(latTW[0],26)
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
    ax.set_title(title)
    ax.set_title(ltitle,loc='left')
    ax.set_title(rtitle,loc='right')
    ax.set_xlim(119,122)
    ax.set_ylim(21.5,26)
    ax.grid()
    return strm,cf_no2

if __name__=='__main__':
    (xstart,xend,ystart,yend)=(20,321,150,451)
    step=1
    lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend:step]
    latTW=np.load('../data/tw_s_lat.npy')[ystart:yend:step]
    topo=np.load('../data/topodata.npy')[ystart:yend:step,xstart:xend:step]
    xx_TW,yy_TW=np.meshgrid(lonTW, latTW)

    lonlat={'TPP':[120.4848,24.20943],'SNC':[120.2058,23.80023]}

    #load historical/ssp-585 upstream flow regimes and local flows
    m='TaiESM1'
    sce='ssp585'
    runDict={'historical':['Historical',2001,2010],}
    for yr in range(2025,2095,10):
        runDict['%04d'%yr]=['SSP-585',yr,yr+9]
   
    #composite of different prevailing wind direction
    for wd_cat in [60,90,120,150,180]:
        plt.close()
        fig,axes=plt.subplots(1,2,figsize=(12,6))
        for ax,run in zip(axes.ravel(),['2025','2085']):
            composite=np.load('../data/local_NO2_composite.WD%03d.%s_%s.%s-%s.npy'%\
                (wd_cat,m,runDict[run][0],runDict[run][1],runDict[run][2]))
            df=pd.read_csv('../data/sites.%s_%s_%04d0101-%04d1231.csv'%\
                   (m,runDict[run][0],runDict[run][1],runDict[run][2]),header=0)
            synoptic_wind=df.meanWD.values
            idxList=np.where(np.logical_and(synoptic_wind>=(wd_cat-15),synoptic_wind<(wd_cat+15)))[0]
            print(wd_cat,run,idxList.shape[0])
            strm,cf_no2=plot_no2(ax,composite,lonTW,latTW,topo,\
              np.linspace(0,300,16),r_cmap,f'WD={wd_cat}\n{runDict[run][1]}-{runDict[run][2]}\n({idxList.shape[0]} days)')
        fig.subplots_adjust(right=0.8)
        ax_cb = fig.add_axes([0.82, 0.12, 0.03, 0.75])
        cbar=plt.colorbar(strm.lines,cax=ax_cb)
        cbar.set_label('WS (m/s)',fontsize=14)
        ax_cb_no2 = fig.add_axes([0.9, 0.12, 0.03, 0.75])
        cbar_no2=plt.colorbar(cf_no2,cax=ax_cb_no2,extend='max')
        cbar_no2.set_label('NO2 (ppb)',fontsize=14)
    
        plt.suptitle(f'NO2 Distribution Composite (WD={wd_cat})',fontsize=20)
        plt.savefig(f'../figures/no2_comp.wd{wd_cat:03}.png')
    
             
