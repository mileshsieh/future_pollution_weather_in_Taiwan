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

#define focusing area
[min_lon, max_lon, min_lat, max_lat]=[100.,150.,10.,50.]
[min_lon_plot, max_lon_plot, min_lat_plot, max_lat_plot]=[105.,145.,12.,48.]

if __name__=='__main__':
  dataDir='/data/mileshsieh/CMIP6'
  m='TaiESM1'
  sce='ssp585'

  template='%s/%s/%s/atmos/day/r1i1p1f1/%s_day_%s_%s_r1i1p1f1_gn_%04d0101-*.nc'
  varList=['psl','ua','va','pr']

  yrDict={'ssp585':[2085,'2092-04-13','2092-04-17']} #[start yr of files,stat_start_date,stat_end_date]
  p_selected=1000
  yr_start=yrDict[sce][0]
  #open multiple files
  fList=[glob(template%(dataDir,m,sce,var,m,sce,yr_start))[0] for var in varList]
  #open multiple files
  ds=xr.open_mfdataset(fList).sel(lat=slice(min_lat,max_lat),lon=slice(min_lon,max_lon),
                                        time=slice(yrDict[sce][1], yrDict[sce][2]),plev=p_selected*100)
  #plot
  plt.close()
  fig=plt.figure(figsize=(24,16))
  #fig, axes = plt.subplots(1,5,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},figsize=(24,16),facecolor='w', edgecolor='k')

  #lbl=['a','b','c','d','e']
  # plot East Asia weather map of each day

  slp_levels=np.linspace(980,1040,16)
  slp_lws=[2 if (pp-980)%20==0 else 1 for pp in slp_levels]

  dateList=[]
  tlbl=['a','b','c','d','e']
  for i,dd in enumerate(ds.time.values):
    dateStr=dd.strftime('%Y-%m-%d')
    dateList.append(dateStr)
    print(dateStr)

    ds_plot=ds.sel(time=dateStr).squeeze()
    msl=ds_plot.psl/100
    msl.attrs = ds_plot.psl.attrs
    msl.attrs['units']='hPa'

    prect=ds_plot.pr*1000/998*3600 #mm/hr
    prect.attrs = ds_plot.pr.attrs
    prect.attrs['units']='mm/hr'

    resample = ds_plot.isel(lon=slice(None, None, 2),lat=slice(None, None, 2))
    # Axes with Cartopy projection
    ax=plt.subplot(2,5,i+1,projection=ccrs.PlateCarree(central_longitude=180))
    # and extent
    ax.set_extent([min_lon_plot, max_lon_plot, min_lat_plot, max_lat_plot], ccrs.PlateCarree())

    # Plotting using Matplotlib the mean current
    cs_p = prect.plot.contourf(ax=ax,x='lon', y='lat',levels=[0.01,0.05,0.1,0.5,1.0],extend='max',cmap='Greens',
                               transform=ccrs.PlateCarree(),add_colorbar=False)
    cs = msl.plot.contour(ax=ax,transform=ccrs.PlateCarree(),
                      levels=slp_levels,
                      colors='k',
                      linewidths=slp_lws)
    q1=resample.plot.quiver(ax=ax,x='lon', y='lat', u='ua', v='va', transform=ccrs.PlateCarree(),
                        scale=250,color='salmon',width=0.003,add_guide=False)

    ax.clabel(cs, slp_levels, inline=True, fmt='%.0f', fontsize=8)
    ax.coastlines(resolution='50m',linewidth=0.8,color='royalblue')
    # Plot lat/lon grid 
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.1, color='k', alpha=1, 
                  linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 8} 

    ax.set_title('(%s) %s\n\n'%(tlbl[i],dateStr),fontsize=15)
    ax.set_title('TaiESM1\n%s'%sce,fontsize=12,loc='left')
    ax.set_title('MSLP\nPRECT\nWind@1000 hPa',fontsize=12,loc='right')
    if i==4:
      veclenght = 10
      maxstr = '%3.1f m/s' % veclenght
      ax.quiverkey(q1,1.1,1.1,veclenght,maxstr,labelpos='E', linewidth=50,coordinates='axes')

      cax=fig.add_axes([0.92, 0.615, 0.008, 0.19])
      cbar=plt.colorbar(cs_p, cax=cax,extend='max')
      cbar.set_label('Precipitation(mm/hr)',fontsize=20)

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

  df=pd.read_csv('../data/cold_season_TaiESM1_ssp585_20850101-20941231.indices.csv',header=0,parse_dates=[0]) 
  tlbl=['f','g','h','i','j']
  for i,dd in enumerate(dateList):
    ws=df[df.date==dd].meanWS.values[0]
    wd=df[df.date==dd].meanWD.values[0]
    print(dd,ws,wd)
    x_temp,y_temp=wind2xy(wd,ws)
    local_flow=model_ae.decode(torch.Tensor([x_temp,y_temp]).to(device)).detach().cpu().numpy()[0]*thd
    ax=plt.subplot(2,5,i+6)
    lbl='(%s) %s\n$[WD=%.2f^\circ, WS=%.2fms^{-1}]$'%(tlbl[i],dd,wd,ws)
    ax.contour(lonTW,latTW,topo,levels=[0.01,],colors='k',linewidths=2)
    ax.contourf(lonTW,latTW,topo_m,cmap='Greys')

    generated_ws=np.sqrt(local_flow[0,:,:]**2+local_flow[1,:,:]**2)
    strm=plt.streamplot(xx,yy,local_flow[0,:,:],local_flow[1,:,:],color=generated_ws,cmap='YlGnBu',
                        norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=2,zorder=3,arrowsize=2.5)
    ax.set_xlim(119.8,122.2)
    ax.set_ylim(21.7,25.5)
    ax.set_title(lbl,fontsize=15)
    plt.xticks([])
    plt.yticks([])
  fig.subplots_adjust(bottom=0.1)
  ax_cb_ws = fig.add_axes([0.92, 0.11, 0.008, 0.4])
  cbar_ws=plt.colorbar(strm.lines,cax=ax_cb_ws,extend='max')
  cbar_ws.set_label('Wind Speed (m/s)',fontsize=20)

  plt.suptitle('Episode of Consecutive WD150',y=0.9,fontsize=30)
  plt.savefig('../figures/fig8_consecutive_episode.png',bbox_inches='tight')
