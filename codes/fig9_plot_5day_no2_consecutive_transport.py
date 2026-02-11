#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.dates as mdates
import geopandas as gpd
from shapely.geometry import Point
from matplotlib.gridspec import GridSpec

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
    ax.set_title(title,fontsize=10)
    ax.set_title(ltitle,loc='left')
    ax.set_title(rtitle,loc='right')
    ax.set_xlim(119,122)
    ax.set_ylim(21.5,26)
    ax.grid()
    return strm,cf_no2

def upwind_adv(q0,q,u,v,dt,dx,dy):
    """
    Performs a single time step using forward difference in time and
    upwind difference in space for the 2D advection equation.
    """

    # Update the field, preserving concentration
    dqdx_p=(q0-np.roll(q0,1,axis=1))/dx
    dqdx_n=(np.roll(q0,-1,axis=1)-q0)/dx
    dqdy_p=(q0-np.roll(q0,1,axis=0))/dy
    dqdy_n=(np.roll(q0,-1,axis=0)-q0)/dy
    q=q0-u*dt*np.where(u>0,dqdx_p,np.where(u<0,dqdx_n,0))\
        -v*dt*np.where(v>0,dqdy_p,np.where(v<0,dqdy_n,0))

    # remove negtive values and preserve the total quantity
    q[q<0]=0.0
    q=q*q0.sum()/q.sum()
    return q

def simple_adv(q0,q,u,v,dt,dx,dy):
    """
    Performs a single time step using forward difference in time and 
    central difference in space for the 2D advection equation.
    """
    # Calculate gradients using numpy.gradient
    dQy,dQx=np.gradient(q0,dx,dy)

    # Update the field
    q[1:-1,1:-1]=q0[1:-1,1:-1]-dt*(u[1:-1,1:-1]*dQx[1:-1,1:-1]+v[1:-1,1:-1]*dQy[1:-1,1:-1])
    # remove negtive values and preserve the total quantity
    q[q<0]=0.0
    q=q*q0.sum()/q.sum()

    return q

if __name__=='__main__':
    (xstart,xend,ystart,yend)=(20,321,150,451)
    lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend]
    latTW=np.load('../data/tw_s_lat.npy')[ystart:yend]
    topo=np.load('../data/topodata.npy')[ystart:yend,xstart:xend]
    xx_TW,yy_TW=np.meshgrid(lonTW, latTW)
    grid = np.zeros((len(latTW), len(lonTW)))
    county = gpd.read_file('/data/huanghuai/DATA/MAP/twcounty/twcounty.shp')

    great_taipei=gpd.GeoSeries(county[county.COUNTYNAME.isin(['臺北市','新北市'])].geometry.union_all())
    #taipei = county[county['COUNTYNAME']=='臺北市'].geometry
    #new_taipei = county[county['COUNTYNAME']=='新北市'].geometry
    for i in range(len(lonTW)):
        for j in range(len(latTW)):
            grid[j,i] = great_taipei.contains(Point(lonTW[i],latTW[j])).item()

    #load NO2 emission data and regrid to TaiwanVVM 2km grid
    emi=xr.open_dataset('../data/2021_tropomi_winter_3month_mean_emiss_all_tw_update.nc')
    no2=emi.no2.values/100.0*24.45/46. #change unit to ppb/s
    #no2=emi.no2.values #unit: mug m-2 s-1
    elon=emi.longitude.values
    elat=emi.latitude.values
    xx,yy=np.meshgrid(elon,elat)
    no2_intp=griddata((xx.flatten(),yy.flatten()),no2.flatten(),(xx_TW,yy_TW),method='linear')
    no2_intp[np.isnan(no2_intp)]=0.0
    #intergration for 24 hrs
    dt=1*60
    dx=dy=2000
    nt=86400//dt

    #load historical/ssp-585 upstream flow regimes and local flows
    m='TaiESM1'
    sce='SSP-585'
    nday=5
    eventDict={'WD150':[2091,105], #2092-04-13
              }
    for wd in eventDict.keys():
        [yr,idx]=eventDict[wd]
        flow=np.load('../data/local_flow.%s_%s_%s0101-%s1231.npy'%\
                     (m,sce,yr,yr+9))
        df=pd.read_csv('../data/LVD.%s_%s_%04d0101-%04d1231.csv'%\
               (m,sce,yr,yr+9),header=0)
        print(wd,df.date.size,flow.shape)

        #empty array for saving daily NO2 distribution
        plt.close()
        fig=plt.figure(figsize=(12,7))
        gs=GridSpec(3,5,figure=fig)
        #fig,axes=plt.subplots(1,nday,figsize=(12,8))
        
        no2_24hr=[]
        no2_hourly=[]
        lbl=['a','b','c','d','e']
        ilbl=0
        #for ax,iday in zip(axes,np.arange(nday)):
        for iday in np.arange(nday):
            ax=fig.add_subplot(gs[:2,iday])
            dd=df.date.values[idx+iday]
            uv=flow[idx+iday,:,:,:]    
            results=[]
            if iday==0:
                results.append(no2_intp.copy()*dt)
            else:
                results.append(no2_24hr[iday-1])
            for n in range(nt):
                q0=results[n].copy()
                #q=simple_adv(q0,np.zeros_like(q0),uv[0,:,:],uv[1,:,:],dt,dx,dy)
                q=upwind_adv(q0,np.zeros_like(q0),uv[0,:,:],uv[1,:,:],dt,dx,dy)
                results.append(q+dt*no2_intp)  
            evo=np.array(results)
            print(evo.shape)
            print(iday,dd,evo[::3600//dt,:,:][-1].max())
            no2_24hr.append(evo[::3600//dt,:,:][-1])
            if iday==(nday-1):
                no2_hourly.append(evo[::3600//dt,:,:])
            else:
                no2_hourly.append(evo[::3600//dt,:,:][:-1])
            data=np.stack((evo[::3600//dt,:,:][-1]*1.88,flow[idx+iday,0,:,:],flow[idx+iday,1,:,:]),axis=0)
            strm,cf_no2=plot_no2(ax,data,lonTW,latTW,topo,\
               np.linspace(0,600,16),r_cmap,f'({lbl[ilbl]}) {dd}\nWD={df.meanWD.values[idx+iday]:.1f}$^\circ$ WS={df.meanWS.values[idx+iday]:.1f}m$s^{{-1}}$')
            great_taipei.boundary.plot(ax=ax,lw=1.2,color='navy',zorder=5)
            if iday==0:
                ax.set_ylabel('Latitude ($^\circ N$)')
            else:
                ax.set_yticklabels([])
            ax.set_xlabel('Longitude ($^\circ E$)')
            ax.grid(True)
            ilbl=ilbl+1

        no2_hourly=np.concatenate(no2_hourly,axis=0)
        no2_tpe=[no2_hourly[k][grid>0.5].mean().item() for k in range(len(no2_hourly))]
        tmList=[pd.to_datetime(df.date.values[idx])+np.timedelta64(tt,'h') for tt in range(len(no2_tpe))]         
        ax=fig.add_subplot(gs[2,:])
        ax.plot(tmList,np.array(no2_tpe)*1.88) #change unit to mug/m3
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H\n%m/%d'))
        ax.set_xlim(tmList[0],tmList[-1])
        ax.set_ylim(0,)
        ax.set_xlabel('Local Time')
        ax.grid(True)
        ax.set_ylabel(r'$NO_x$ Conc. (${\mu}g/m^3$)')
        plt.axhline(25,color='g')
        plt.axhline(120,color='r')
        plt.text(tmList[-1],25+15,'WHO AQG level',color='g',horizontalalignment='right',verticalalignment='center')
        plt.text(tmList[-1],120+15,'WHO AQG interim target 1',color='r',horizontalalignment='right',verticalalignment='center')

        ax.set_title('(f) Average $NO_x$ Concentration of Greater Taipei Area in 5-Day Pollution Episode')

        #fig.subplots_adjust(left=0.07,top=0.9,bottom=0.1,right=0.89)
        fig.subplots_adjust(left=0.07,right=0.89)
        ax_cb_no2 = fig.add_axes([0.91, 0.48, 0.03, 0.32])
        cbar_no2=plt.colorbar(cf_no2,cax=ax_cb_no2,extend='max')
        cbar_no2.set_label('$NO_x$ Conc. (${\mu}g/m^3$)',fontsize=14)
        plt.suptitle(f'An Episode of Consecutive 5-Day WD150 Pollution Scenario',fontsize=16)
        plt.savefig(f'../figures/fig9_consecutive_scenario.png',dpi=300)
