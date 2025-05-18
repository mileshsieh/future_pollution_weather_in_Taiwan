#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import xarray as xr

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
    sce='ssp585'
    runDict={'historical':['Historical',2001,2010],}
    for yr in range(2025,2095,10):
        runDict['%04d'%yr]=['SSP-585',yr,yr+9]

    compositeDict={}
    for run in ['2025','2085']:
        flow=np.load('../data/local_flow.%s_%s_%s0101-%s1231.npy'%\
                     (m,runDict[run][0],runDict[run][1],runDict[run][2]))
        df=pd.read_csv('../data/sites.%s_%s_%04d0101-%04d1231.csv'%\
               (m,runDict[run][0],runDict[run][1],runDict[run][2]),header=0)
        print(run,df.date.size,flow.shape)
        nday=len(flow)
        synoptic_wind=df.meanWD.values
        for wd_cat in [60,90,120,150,180]:
            idxList=np.where(np.logical_and(synoptic_wind>=(wd_cat-15),synoptic_wind<(wd_cat+15)))[0]
            composite=flow[idxList,:,:,:].mean(axis=0)
            compositeDict[run]=flow[idxList]
            #empty array for saving hourly NO2 distribution
            results=[]
            print(wd_cat,run,idxList.shape[0])
            results.append(no2_intp.copy()*dt)
            for n in range(nt):
                q0=results[n].copy()
                #q=simple_adv(q0,np.zeros_like(q0),uv[0,:,:],uv[1,:,:],dt,dx,dy)
                q=upwind_adv(q0,np.zeros_like(q0),composite[0,:,:],composite[1,:,:],dt,dx,dy)
                results.append(q+dt*no2_intp)  
            evo=np.array(results)
            print(evo.shape,evo[::3600//dt,:,:][-1].max())
            np.save('../data/local_NO2_composite.WD%03d.%s_%s.%s-%s.npy'%\
                (wd_cat,m,runDict[run][0],runDict[run][1],runDict[run][2]),\
                np.array([evo[::3600//dt,:,:][-1],composite[0,:,:],composite[1,:,:]]))

