#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mc
import matplotlib
import matplotlib.patches as patches

def plotTW(ax,lonTW,latTW,topo):
    xx,yy=np.meshgrid(lonTW,latTW)
    cf=ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    ax.contour(xx,yy,topo,levels=[0.05,],colors=['k'])
    ax.set_ylabel('Latitude ($^\\circ N$)')
    ax.set_xlabel('Longitude ($^\\circ E$)')
    cb=plt.colorbar(cf,extend='max')
    cb.set_label('Height (km)')
    return

def plotStreamLine(ax,lonTW,latTW,topo,vardata,slon,slat,title,rtitle,ltitle):
#def plotStreamLine(ax,lonTW,latTW,topo,step,vardata,slon,slat,title,rtitle,ltitle):
    #xx_flow,yy_flow=np.meshgrid(lonTW[::step],latTW[::step])
    xx,yy=np.meshgrid(lonTW,latTW)
    plotTW(ax,lonTW,latTW,topo)
    #ax.contourf(xx,yy,topo,levels=np.linspace(0.1,3.5,35),cmap='Greys')
    #ax.contour(xx,yy,topo,levels=[0.05,],colors=['k'])
    #ax.contour(topo,levels=[0.05,0.2],colors=['k'])
    ws=np.sqrt(vardata[0,:,:]**2+vardata[1,:,:]**2)
    #print(ws.min(),ws.max())
    strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
            color='lightblue',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=1.5,density=1.0,zorder=3,arrowsize=1.5)
    #strm=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:],
    #        color=ws,cmap='YlGnBu',norm=mc.Normalize(vmin=0.0,vmax=7.0),linewidth=2,density=0.8,zorder=3,arrowsize=1.5)
    #strm_site=ax.streamplot(xx_flow,yy_flow,vardata[0,:,:],vardata[1,:,:], color='r', linewidth=2,
    #strm_site=ax.streamplot(xx,yy,vardata[0,:,:],vardata[1,:,:], color='r', linewidth=3,
    #                 start_points=np.array([[slon,],[slat,]]).T,integration_direction='forward',maxlength=1,zorder=5)
    #ax.plot([xstart,xend,xend,xstart,xstart],[ystart,ystart,yend,yend,ystart],lw=2,ls='--')
    ax.plot(slon,slat,'rs',zorder=5)
    #ax.contourf(xx,yy,topo,levels=[0.2,5.0],colors=['darkgreen'],zorder=5)
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

#https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot
def add_arrow(line, ax, position, direction='right', color=None, label=''):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    color:      if None, line color is taken.
    label:      label for arrow
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    # find closest index
    start_ind = position
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1
    
    dx = xdata[end_ind] - xdata[start_ind]
    dy = ydata[end_ind] - ydata[start_ind]
    size = 0.1
    x = xdata[start_ind] 
    y = ydata[start_ind] 

    arrow = patches.FancyArrow(x, y, dx, dy, color=color, width=0, 
                               head_width=size, head_length=size, 
                               label=label,length_includes_head=True, 
                               overhang=0.3, zorder=10)
    ax.add_patch(arrow)
    return

if __name__=='__main__':
  #domain and step of the original TaiwanVVM 2km simulation
  (xstart,xend,ystart,yend)=(20,321,150,451)
  #step=5
  topo=np.load('../data/topodata.npy')[ystart:yend,xstart:xend]
  lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend]
  latTW=np.load('../data/tw_s_lat.npy')[ystart:yend]

  #get emission xy index
  lonlat={'TPP':[120.4848,24.20943],'SNC':[120.2058,23.80023]}

  #load historical/ssp-585 upstream flow regimes and local flows
  m='TaiESM1'
  sce='ssp585'
  runDict={'historical':['Historical',2001,2010],}
  for yr in range(2025,2095,10):
    runDict['%04d'%yr]=['SSP-585',yr,yr+9]
  wd_cat=150
  #for run in runDict.keys():
  for run in ['2085']:
    print(run)
    df=pd.read_csv('../data/sites.%s_%s_%04d0101-%04d1231.csv'%\
               (m,runDict[run][0],runDict[run][1],runDict[run][2]),header=0)
    df_cat=df.query(f'meanWD >= {wd_cat-15} and meanWD < {wd_cat+15}')[['date','meanWD','meanWS']]
    #for site in lonlat.keys():
    for site in['TPP',]:
      traj=np.load('../data/traj.%s.%s_%s_%04d0101-%04d1231.npy'%\
                   (site,m,runDict[run][0],runDict[run][1],runDict[run][2]))
      synoptic_wind=df.meanWD.values
      idxList=np.where(np.logical_and(synoptic_wind>=(wd_cat-15),synoptic_wind<(wd_cat+15)))[0]
      traj_cat=traj[idxList]
      nday,_,nt=traj_cat.shape
      plt.close()
      ax=plt.subplot(111)
      plotTW(ax,lonTW,latTW,topo)
      ax.plot(lonlat[site][0],lonlat[site][1],'rs',zorder=5)

      for idx in range(46,46+5):
        #calculate trajectory for 24 hr
        wdstr=f'{site}_WD'
        wsstr=f'{site}_WS'
        datestr=df_cat.date.values[idx][:4]+df_cat.date.values[idx][5:7]+df_cat.date.values[idx][-2:]
        print(site,idx,datestr)
        #plot trajectory
        line=ax.plot(traj_cat[idx,0,:24],traj_cat[idx,1,:24],lw=2,label=datestr,zorder=7)[0]
        add_arrow(line, ax, 4)
      plt.xlim(119,122)
      plt.ylim(23,26)
      plt.legend(loc=2)

      #plt.savefig(f'../figures/emission_winds/{site}.{runDict[run][0]}.{idx:03d}.{datestr}.png')   
 



