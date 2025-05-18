#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mc
from scipy.interpolate import griddata
import xarray as xr

def plotTW(ax,lonTW,latTW,topo):
    xx,yy=np.meshgrid(lonTW,latTW)
    #cf=ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    ax.contour(xx,yy,topo,levels=[0.05,1.5],colors=['k'])
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
    #strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
    #        color='lightblue',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=1.5,density=1.0,zorder=3,arrowsize=1.5)
    strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
            color=ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=0.8,zorder=3,arrowsize=1.5)
    #strm_site=ax.streamplot(xx_flow,yy_flow,vardata[0,:,:],vardata[1,:,:], color='r', linewidth=2,
    #strm_site=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:], color='r', linewidth=3,
    #                 start_points=np.array([[slon,],[slat,]]).T,integration_direction='forward',maxlength=1,zorder=5)
    #ax.plot([xstart,xend,xend,xstart,xstart],[ystart,ystart,yend,yend,ystart],lw=2,ls='--')
    #ax.plot(slon,slat,'rs',zorder=5)
    #ax.contourf(xx,yy,topo,levels=[0.2,5.0],colors=['darkgreen'],zorder=5)
    ax.set_xlim(118,lonTW[-1])
    ax.set_ylim(latTW[0],26)
    plt.grid()
    #ax.xaxis.set_major_locator(ticker.NullLocator())
    #ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.set_title(rtitle,fontsize=10,loc='right')
    ax.set_title(ltitle,fontsize=10,loc='left')
    ax.set_title(title,fontsize=15,loc='center')
    cb=plt.colorbar(strm.lines,extend='max')
    cb.set_label('WS (m/s)')
    return strm

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

# no2 colormap
colors_below=plt.cm.Blues(np.linspace(0.0,0.5,5))
colors_middle=plt.cm.Greens(np.linspace(0.2,0.7,5))
colors_over=plt.cm.hot_r(np.linspace(0.3,0.8,5))
all_colors=np.vstack((colors_below,colors_middle,colors_over))
#all_colors=np.vstack((colors_middle,colors_over))
r_cmap=mc.LinearSegmentedColormap.from_list('no2_map',all_colors)

if __name__=='__main__':
    (xstart,xend,ystart,yend)=(20,321,150,451)
    step=1
    lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend:step]
    latTW=np.load('../data/tw_s_lat.npy')[ystart:yend:step]
    topo=np.load('../data/topodata.npy')[ystart:yend:step,xstart:xend:step]
    xx_TW,yy_TW=np.meshgrid(lonTW, latTW)

    emi=xr.open_dataset('../data/2021_tropomi_winter_3month_mean_emiss_all_tw_update.nc')
    no2=emi.no2.values/100.0*24.45/46. #change unit to ppb/s
    #no2=emi.no2.values #unit: mug m-2 s-1
    elon=emi.longitude.values
    elat=emi.latitude.values
    xx,yy=np.meshgrid(elon,elat)
    no2_intp=griddata((xx.flatten(),yy.flatten()),no2.flatten(),(xx_TW,yy_TW),method='linear')
    no2_intp[np.isnan(no2_intp)]=0.0

    selection={'NE':[2025,560],'SE':[2091,105]}
    #flow=np.load('../data/local_flow.TaiESM1_SSP-585_20250101-20341231.npy')
    #uv=flow[560,:,::step,::step]
    flow=np.load('../data/local_flow.TaiESM1_SSP-585_20850101-20941231.npy')
    uv=flow[523,:,::step,::step]
    uv_uniform=np.zeros_like(uv)
    uv_uniform[1,:,:]=3.0
    
    lonlat={'TPP':[120.4848,24.20943],'SNC':[120.2058,23.80023]}
    #domain and step of the original TaiwanVVM 2km simulation
    (xstart,xend,ystart,yend)=(20,321,150,451)
    '''
    plt.close()  
    ax=plt.subplot(111)
    strm=plotStreamLine(ax,lonTW,latTW,topo,uv,'','','')
                        #f'{df.date.values[idx]}',\
                        #f'{m}\n{runDict[run][0]}\n{site}',\
                        #f'UFR:\nWS={df.meanWS.values[idx]:.2f}$m s^{{-1}}$\nWD={df.meanWD.values[idx]:.0f}$^\circ$')
    '''
    dt=1*60
    dx=dy=2000
    nt=86400//dt

    results=[]
    #results.append(np.zeros(no2_intp.shape))
    results.append(no2_intp.copy()*dt)
    #tmp=np.zeros_like(no2_intp)
    #tmp[10:50,200:250]=100.0
    #results.append(tmp)
    for n in range(nt):
        q0=results[n].copy()
        #q=simple_adv(q0,np.zeros_like(q0),uv[0,:,:],uv[1,:,:],dt,dx,dy)
        #q=simple_adv(q0,np.zeros_like(q0),uv_uniform[0,:,:],uv_uniform[1,:,:],dt,dx,dy)
        #q=upwind_adv(q0,np.zeros_like(q0),uv_uniform[0,:,:],uv_uniform[1,:,:],dt,dx,dy)
        q=upwind_adv(q0,np.zeros_like(q0),uv[0,:,:],uv[1,:,:],dt,dx,dy)
        print(f'step {n:0>4d}: {q0.sum():.3f} {q.sum():.3f}')
        results.append(q+dt*no2_intp)  
        #results.append(q)  
    evo=np.array(results)
    print(evo.shape)
    for k in range(25):
        print(k,np.percentile(evo[k*3600//dt,:,:],95))
        plt.close()
        ax=plt.subplot(111)
        strm=plotStreamLine(ax,lonTW,latTW,topo,uv,'','','')
        #strm=plotStreamLine(ax,lonTW,latTW,topo,uv_uniform,'','','')
        
        plt.contourf(lonTW,latTW,np.where(evo[k*3600//dt,:,:]>0,evo[k*3600//dt,:,:],np.nan),\
                     levels=np.linspace(0,300,16),extend='max',cmap=r_cmap)
        #plt.contourf(evo[k*3600//dt,:,:])
        cb=plt.colorbar()
        cb.set_label('$NO_2$ Conc. (ppb)')
        
        plt.savefig(f'test.adv.{k:03d}.png')
