#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mc

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
    #print(no2.max())
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
        no2_24hr=[]
        for iday in range(nday):
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
            #print(evo.shape)
            #print(iday,dd,evo[::3600//dt,:,:][-1].max())
            no2_24hr.append(evo[::3600//dt,:,:][-1])

            for hr in range(24):
              plt.close()
              fig=plt.figure(figsize=(6,8))
              ax=plt.subplot()
              print(dd,hr,evo[::3600//dt,:,:][hr].max())
              data=np.stack((evo[::3600//dt,:,:][hr],flow[idx+iday,0,:,:],flow[idx+iday,1,:,:]),axis=0)
              strm,cf_no2=plot_no2(ax,data,lonTW,latTW,topo,\
                np.linspace(0,300,16),r_cmap,f'{dd} {hr:02}:00')
              ax.set_ylabel('Latitude ($^\circ N$)')
              ax.set_xlabel('Longitude ($^\circ E$)')
              ax.grid(True)
              fig.subplots_adjust(right=0.85)
              ax_cb_no2 = fig.add_axes([0.89, 0.12, 0.03, 0.75])
              cbar_no2=plt.colorbar(cf_no2,cax=ax_cb_no2,extend='max')
              cbar_no2.set_label('NO2 (ppb)',fontsize=14)
              plt.savefig(f'../figures/episode/no2.{iday*24+hr:03}.png')

