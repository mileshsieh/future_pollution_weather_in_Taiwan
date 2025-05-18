#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import torch
import torch.nn as nn
from VAE import variationalAutoEncoder,Encoder,Decoder,Reshape
from scipy.optimize import fsolve
import matplotlib
matplotlib.rc('xtick',labelsize=20)
matplotlib.rc('ytick',labelsize=20)

[a,b,x0,y0]=[0.08303117,0.95491014,8.25855574,0.11692898]
[a0,a1,wd0]=[2.93885019,-1.29016647,103.75438081]
def ws_h(x, y):
    return a * (x - x0)**2 + b * (y - y0)**2

def wd_h(x, y):
    return a1*np.arctan(y/(x-a0))/np.pi*180+wd0

def wind2xy(wd,ws):
    def equations(vars):
        x, y = vars
        alpha=np.tan((wd-wd0)*np.pi/180/a1)
        eq1 = x-a0-y/alpha
        eq2 = a*(x-x0)**2+b*(y-y0)**2-ws
        return [eq1, eq2]

    x, y =  fsolve(equations, (2.1, 0))
    return x,y

def uv2wd(u, v):
  return np.where(u==0,np.where(v>0,180.0,0.0),180.0 + np.arctan2(u, v) * 180.0 / np.pi)

def _preprocess_wind(ds):
    return ds.reindex(latitude=list(reversed(ds.latitude)))\
             .sel(latitude=slice(min_lat-1,max_lat+1),longitude=slice(min_lon-1,max_lon+1),level=1000.0)
#define focusing area
[min_lon, max_lon, min_lat, max_lat]=[100.,150.,10.,50.]
[min_lon_plot, max_lon_plot, min_lat_plot, max_lat_plot]=[116.,127.5,19.2,27.8]
ERA5_PRS_DIR='/data/dadm1/reanalysis/ERA5/PRS/day'

if __name__=='__main__':
  '''
  p_selected=1000
  yr=2020
  #ERA5 
  #wind field at 1000 mb 
  fList=np.concatenate([np.array(sorted(glob('%s/%s/%04d/*'%(ERA5_PRS_DIR,var,yr)))) for var in ['u','v']])
  ds_ERA5_wind=xr.open_mfdataset(fList,preprocess=_preprocess_wind).sel(time='2020-08-20').squeeze()

  [min_lon_plot, max_lon_plot, min_lat_plot, max_lat_plot]=[116.,127.5,19.2,27.8]
  #plot
  plt.close()
  resample = ds_ERA5_wind.isel(longitude=slice(None, None, 2),latitude=slice(None, None, 2))
  xx,yy = np.meshgrid(resample.longitude.values,resample.latitude.values)
  # Defining the figure
  fig = plt.figure(figsize=(12,8), facecolor='w', edgecolor='k')

  # Axes with Cartopy projection
  ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=180))
  # and extent
  ax.set_extent([min_lon_plot, max_lon_plot, min_lat_plot, max_lat_plot], ccrs.PlateCarree())

  q1=resample.plot.quiver(ax=ax,x='longitude', y='latitude', u='u', v='v', transform=ccrs.PlateCarree(),
                          scale=250,color='salmon',width=0.003,add_guide=False)
  ax.coastlines(resolution='50m',linewidth=0.8,color='royalblue')
  # Plot lat/lon grid 
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=0.1, color='k', alpha=1, 
                    linestyle='--')
  gl.top_labels = False
  gl.right_labels = False
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  gl.xlabel_style = {'size': 12}
  gl.ylabel_style = {'size': 12} 

  #ax.quiverkey(q1,0.9,0.9,veclenght,maxstr,labelpos='E', linewidth=50,coordinates='axes')

  #get ishigaki winds
  u_ish=ds_ERA5_wind.sel(latitude=24.33,longitude=124.16,method='nearest').u.values.item()
  v_ish=ds_ERA5_wind.sel(latitude=24.33,longitude=124.16,method='nearest').v.values.item()
  ws=np.sqrt(u_ish*u_ish+v_ish*v_ish)
  wd=uv2wd(u_ish, v_ish)
  '''
  #local circulation
  num_input_channels=2
  latent_dim=2
  beta=0.01
  dataset='ctrl'
  thd=11
  seed=3
  ts=6
  te=54
  #for write out
  sf='vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,ts,te,seed,thd)
  device = torch.device("cpu")
  # Load model
  model_ae = torch.load('../data/leevortex.%s.pth'%sf,map_location=torch.device('cpu'))
  model_ae.eval()

  #load topo and lat/lon of generated circulation
  ys=150
  ye=451
  xs=20
  xe=321
  lonTW=np.load('../data/tw_s_lon.npy')[xs:xe]
  latTW=np.load('../data/tw_s_lat.npy')[ys:ye]
  topo=np.load('../data/topodata.npy')[ys:ye,xs:xe]
  topo_m=np.copy(topo)
  topo_m[topo_m==0]=np.nan

  xx,yy=np.meshgrid(lonTW[::5],latTW[::5])
  ws=2.5
  wd=150
  print(ws,wd)
  x_temp,y_temp=wind2xy(wd,ws)
  print(x_temp,y_temp)
  local_flow=model_ae.decode(torch.Tensor([x_temp,y_temp]).to(device)).detach().cpu().numpy()[0]*thd
  generated_ws=np.sqrt(local_flow[0,:,:]**2+local_flow[1,:,:]**2)
  fig=plt.figure()
  ax2=plt.subplot(111)
  ax2.contour(lonTW,latTW,topo,levels=[0.01,],colors='k',linewidths=2)
  ax2.contourf(lonTW,latTW,topo_m,cmap='Greys')
  strm=ax2.streamplot(xx,yy,local_flow[0,:,:],local_flow[1,:,:],color=generated_ws,cmap='YlGnBu',
                      norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=2,zorder=3,arrowsize=2.5)
  ax2.set_xlim(117.8,122.2)
  ax2.set_ylim(21.7,25.5)
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2.set_title('$WD=%.0f^{\circ}, WS=%.1f m/s$'%(wd,ws),fontsize=20)
  fig.subplots_adjust(left=0.05,right=0.8)
  ax_cb_ws = fig.add_axes([0.85, 0.1, 0.05, 0.8])
  cbar_ws=plt.colorbar(strm.lines,cax=ax_cb_ws,extend='max')
  cbar_ws.set_label('wind speed (m/s)',fontsize=20)

