#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import pandas as pd
from glob import glob

def plotTW(ax,lonTW,latTW,topo):
    xx,yy=np.meshgrid(lonTW,latTW)
    #cf=ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    ax.contour(xx,yy,topo,levels=[0.05,],colors=['k'])
    #ax.set_ylabel('Latitude ($^\circ N$)')
    #ax.set_xlabel('Longitude ($^\circ E$)')
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
    sce='SSP-585'
    eventDict={'WD150':[2085,536],
               'WD60':[2085,0],
               'WD120':[2075,511],}
    event='WD150'
    (yr,start_idx)=eventDict[event]
    nday=5

    df=pd.read_csv(f'../data/LVD.{m}_{sce}_{yr}0101-{yr+9}1231.csv',header=0)
    print(df.date[start_idx:start_idx+nday])
    flow=np.load(f'../data/local_flow.{m}_{sce}_{yr}0101-{yr+9}1231.npy')[start_idx:start_idx+nday,:,:,:]
    no2_hourly=np.array([np.load('/data/mileshsieh/future_daily_NO2/local_NO2.%s_%s.%s.npy'%\
                (m,sce,dd)) for dd in df.date[start_idx:start_idx+nday]])
    print(flow.shape,no2_hourly.shape)

    for hr in range(25):
    #for hr in [24,]:
        plt.close()
        fig,axes=plt.subplots(1,nday,figsize=(4.5*nday,6))
        for ax,iday in zip(axes.ravel(),np.arange(nday)):
            dd=df.date[start_idx+iday]
            wd=df.meanWD[start_idx+iday]
            ws=df.meanWS[start_idx+iday]
            data=np.stack((no2_hourly[iday,hr,:,:],flow[iday,0,:,:],flow[iday,1,:,:]),axis=0)
            print(hr,dd,iday,no2_hourly[iday,hr,:,:].max())
            strm,cf_no2=plot_no2(ax,data,lonTW,latTW,topo,\
               np.linspace(0,300,16),r_cmap,'',f'{m} {sce}\n{dd}',\
               #np.linspace(0,300,16),'hot_r','',f'{m} {sce}\n{dd}',\
               f'WD={wd:.1f}$^\circ$\nWS={ws:.1f}m$s^{{-1}}$')
            if iday==0:
                ax.set_ylabel('Latitude ($^\circ E$)')
            ax.set_xlabel('Longitude ($^\circ E$)')
        fig.subplots_adjust(right=0.85,top=0.85)
        ax_cb_no2 = fig.add_axes([0.87, 0.12, 0.03, 0.75])
        cbar_no2=plt.colorbar(cf_no2,cax=ax_cb_no2,extend='max')
        cbar_no2.set_label('NO2 (ppb)',fontsize=14)

        plt.suptitle(f'NO2 Dispersion within {hr} hours',fontsize=16)
        plt.savefig(f'../figures/no2_consecutive_{event}.{hr:02}.png')

