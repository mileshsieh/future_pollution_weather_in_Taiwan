#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import pandas as pd
from glob import glob
import torch
import torch.nn as nn
from VAE import variationalAutoEncoder,Encoder,Decoder,Reshape
from scipy.optimize import fsolve
from scipy.interpolate import griddata
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

def plotTW(ax,lonTW,latTW,topo):
    xx,yy=np.meshgrid(lonTW,latTW)
    #cf=ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    ax.contour(xx,yy,topo,levels=[0.05,1.5],colors=['k'])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
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
    ax.set_title(title,fontsize=20,loc='center')
    cb=plt.colorbar(strm.lines,extend='max')
    cb.set_label('Wind Speed ($ms^{{-1}}$)',fontsize=20)
    return strm

def plotWSWDAxes(ax,clr_WD,clr_WS):
    mu_x=np.linspace(-4,2,100)
    mu_y=np.linspace(-3,3,100)
    grid_x,grid_y=np.meshgrid(mu_x,mu_y)
    c_wd=ax.contour(grid_x,grid_y,wd_h(grid_x,grid_y),levels=np.linspace(40,180,8),colors=[clr_WD])
    ax.clabel(c_wd, levels=[40,80,140], inline=True, fmt='$WD=%.0f\degree$', fontsize=22)

    c_ws=ax.contour(grid_x,grid_y,ws_h(grid_x,grid_y),levels=np.linspace(4,18,8),colors=[clr_WS])
    ax.clabel(c_ws, levels=[6,10,12], inline=True, fmt='WS=%.1f m/s', fontsize=22)
    #plt.xticks([])
    #plt.yticks([])
    return


if __name__=='__main__':
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
  (xstart,xend,ystart,yend)=(20,321,150,451)
  step=1
  lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend:step]
  latTW=np.load('../data/tw_s_lat.npy')[ystart:yend:step]
  topo=np.load('../data/topodata.npy')[ystart:yend:step,xstart:xend:step]
  lon_flow=lonTW[::5]
  lat_flow=latTW[::5]
  xx_TW,yy_TW=np.meshgrid(lonTW, latTW)
  xx_flow,yy_flow=np.meshgrid(lon_flow, lat_flow)

  ws=9
  for i,wd in enumerate(np.arange(50,160,2)):
    print(ws,wd)
    x_temp,y_temp=wind2xy(wd,ws)
    print(x_temp,y_temp)
    local_flow=model_ae.decode(torch.Tensor([x_temp,y_temp]).to(device)).detach().cpu().numpy()[0]*thd
    #interpolate to 2km
    u=griddata((xx_flow.flatten(),yy_flow.flatten()),local_flow[0,:,:].flatten(),\
                (xx_TW,yy_TW),method='cubic')
    v=griddata((xx_flow.flatten(),yy_flow.flatten()),local_flow[1,:,:].flatten(),\
                (xx_TW,yy_TW),method='cubic')
    local_flow_2km=np.array([u,v])
    #plot
    plt.close() 
    fig=plt.figure(figsize=(18,7.5)) 
    ax1=plt.subplot(121)
    ax1.set_title('Upstream FLow RegimesPhase Diagram',fontsize=20)
    plotWSWDAxes(ax1,'darkred','blue')
    ax1.scatter(x_temp,y_temp,c='g',s=200,marker='X',edgecolors='k')

    ax2=plt.subplot(122)
    strm=plotStreamLine(ax2,lonTW,latTW,topo,local_flow_2km,f'WS={ws:.1f}$ms^{{-1}}$,WD={wd:>3}$^\circ$','','')

    plt.savefig(f'../figures/ani/WD{i:03}.png')
