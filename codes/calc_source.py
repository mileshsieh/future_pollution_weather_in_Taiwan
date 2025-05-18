#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from glob import glob
from scipy.interpolate import griddata

def uv2deg(u,v):
    if u==0:
        return 180.0 if v>0 else 0.0
    else:
        tmpdir=270-np.arctan(v/u)/np.pi*180
        return tmpdir if u>0 else tmpdir-180

def getUV(site_lon,site_lat,lon_flow,lat_flow,uv_data):
    if (site_lon<lon_flow[0])or(site_lat<lat_flow[0]):
        return [np.nan,np.nan]
    try:
        e_idx=np.where(lon_flow>=site_lon)[0][0]
    except:
        return [np.nan,np.nan]
    w_idx=e_idx-1
    try:
        n_idx=np.where(lat_flow>=site_lat)[0][0]
    except:
        return [np.nan,np.nan]
    s_idx=n_idx-1
    n_site_flow=uv_data[:,n_idx,w_idx]*((site_lon-lon_flow[w_idx])/\
                (lon_flow[e_idx]-lon_flow[w_idx]))-\
                uv_data[:,n_idx,e_idx]*((site_lon-lon_flow[e_idx])/\
                (lon_flow[e_idx]-lon_flow[w_idx]))
    s_site_flow=uv_data[:,s_idx,w_idx]*((site_lon-lon_flow[w_idx])/\
                (lon_flow[e_idx]-lon_flow[w_idx]))-\
                uv_data[:,s_idx,e_idx]*((site_lon-lon_flow[e_idx])/\
                (lon_flow[e_idx]-lon_flow[w_idx]))
    site_flow=n_site_flow*((site_lat-lat_flow[s_idx])/(lat_flow[n_idx]-lat_flow[s_idx]))-\
              s_site_flow*((site_lat-lat_flow[n_idx])/(lat_flow[n_idx]-lat_flow[s_idx]))           
    return site_flow

def calculate_back_trajectory(site_lon,site_lat,lon_flow,lat_flow,uv_data,total_time_in_hr,step_in_hr):
    # Calculate the number of steps based on time intervals
    num_steps=int(total_time_in_hr/step_in_hr) + 1  # +1 to include the starting point
    trajectory=np.zeros((2,num_steps))
    trajectory[:,0]=[site_lon,site_lat]
    for t in range(1,num_steps):
        # Calculate the current position based on speed and time
        #print(t,trajectory[0,t-1],trajectory[1,t-1])
        site_flow=getUV(trajectory[0,t-1],trajectory[1,t-1],lon_flow,lat_flow,uv_data)
        trajectory[:,t]=[trajectory[0,t-1]-site_flow[0]*step_in_hr*3600/101695.,\
                         trajectory[1,t-1]-site_flow[1]*step_in_hr*3600/101695.]
        #stop trajectory calculation at boundaries
        if (trajectory[0,t]<lon_flow[0])or(trajectory[0,t]>lon_flow[-1])or\
           (trajectory[1,t]<lat_flow[0])or(trajectory[1,t]>lat_flow[-1]):
            trajectory[0,t:]=trajectory[0,t-1]
            trajectory[1,t:]=trajectory[1,t-1]
            break
    return trajectory

if __name__=='__main__':
  #domain and step of the original TaiwanVVM 2km simulation
  (xstart,xend,ystart,yend)=(20,321,150,451)
  #step=5
  topo=np.load('../data/topodata.npy')[ystart:yend,xstart:xend]
  lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend]
  latTW=np.load('../data/tw_s_lat.npy')[ystart:yend]

  #get emission xy index
  lonlat={'TPP':[120.4848,24.20943],'SNC':[120.2058,23.80023],\
          'TY':[121.314328,24.991278],'HsinChu':[120.983333,24.816667],\
          'NewTaipei':[121.445833,25.01111],'Taipei':[121.5625,25.0375],\
          'TC':[120.666667,24.25]}
  #lon_flow=lonTW[::step]
  #lat_flow=latTW[::step]

  #xx_TW,yy_TW=np.meshgrid(lonTW, latTW)
  #xx_flow,yy_flow=np.meshgrid(lon_flow, lat_flow)
  

  #load historical/ssp-585 upstream flow regimes and local flows
  m='TaiESM1'
  sce='ssp585'
  runDict={'historical':['Historical',2001,2010],}
  for yr in range(2025,2095,10):
    runDict['%04d'%yr]=['SSP-585',yr,yr+9]

  for run in runDict.keys():
  #for run in ['historical',]:
    print(run)
    df=pd.read_csv('../data/LVD.%s_%s_%04d0101-%04d1231.csv'%\
               (m,runDict[run][0],runDict[run][1],runDict[run][2]),header=0)
    local_flow=np.load('../data/local_flow.%s_%s_%04d0101-%04d1231.npy'%\
               (m,runDict[run][0],runDict[run][1],runDict[run][2]))
    #u=np.array([griddata((xx_flow.flatten(),yy_flow.flatten()),local_flow[k,0,:,:].flatten(),\
    #            (xx_TW,yy_TW),method='cubic') for k in range(len(local_flow))])
    #v=np.array([griddata((xx_flow.flatten(),yy_flow.flatten()),local_flow[k,1,:,:].flatten(),\
    #            (xx_TW,yy_TW),method='cubic') for k in range(len(local_flow))])
    #local_flow=np.swapaxes(np.array([u,v]),0,1)
    print(local_flow.shape)
    for site in lonlat.keys():
      site_flow=np.array([getUV(lonlat[site][0],lonlat[site][1],\
                                lonTW,latTW,local_flow[k]) for k in range(len(local_flow))])
    print(run,df.date.size)
    
    #for site in lonlat.keys():
    for site in['TC',]:

      dest_lon=[]
      dest_lat=[] 
      traj_list=[]     
      for idx in range(df.date.size):
      #for idx in [29,]:
        #calculate trajectory for 24 hr
        traj=calculate_back_trajectory(lonlat[site][0],lonlat[site][1],\
                                  lonTW,latTW,local_flow[idx],24,0.5)
        datestr=df.date.values[idx][:4]+df.date.values[idx][5:7]+df.date.values[idx][-2:]
        print(site,idx)
        dest_lon.append(traj[0,-1])
        dest_lat.append(traj[1,-1])
        traj_list.append(traj)

      fname=f'../data/back_traj.{site}.{m}_{runDict[run][0]}_{runDict[run][1]:04d}0101-{runDict[run][2]:04d}1231.npy'
      np.save(fname,np.array(traj_list))




