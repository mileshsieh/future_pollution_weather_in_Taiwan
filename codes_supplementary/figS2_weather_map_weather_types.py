#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import pyplot as plt
import glob
import matplotlib

#define focusing area
[min_lon_EA, max_lon_EA, min_lat_EA, max_lat_EA]=[100.,150.,12.,48.]
[min_lon_local, max_lon_local, min_lat_local, max_lat_local]=[114.,130.,18.,30.]
[min_lon_tw, max_lon_tw, min_lat_tw, max_lat_tw]=[118,126,22,26]
[min_lon_ish, max_lon_ish, min_lat_ish, max_lat_ish]=[122,126,22,26]

slp_levels=np.linspace(980,1040,16)
slp_lws=[2 if (pp-980)%20==0 else 1 for pp in slp_levels]

m='TaiESM1'
yr_start=2005
yr_end=2014
df=pd.read_csv('../data/cold_season_weather_regimes_with_idx.%s.%d-%d.csv'%(m,yr_start,yr_end),header=0,parse_dates=[0])
dateList={}
for r,grp in df.groupby('regime'):
    print(r,grp.date.size)
    dateList[r]=grp.date.values

ds=xr.open_mfdataset(['../data/regridded_%s_%04d.nc'%(m,yr) for yr in range(yr_start,yr_end+1)])

#put each regime in subplots
plt.close()
fig, axes = plt.subplots(2,3,subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},figsize=(24,16),facecolor='w', edgecolor='k')
axes[1][2].set_visible(False)

lbl=['a','b','c','d','e']
# plot East Asia weather map of each regime
for i,(ax,r) in enumerate(zip(axes.ravel()[:5],['FT','TC','CS','NE','LCD'])):
    ds_plot=ds.sel(time=dateList[r]).mean(dim='time')
    titleStr='%s (%d days)'%(r,dateList[r].shape[0])
    if r=='LCD':
        titleStr='%s (%d days)'%('WS',dateList[r].shape[0])
    resample = ds_plot.isel(lon=slice(None, None, 2),lat=slice(None, None, 2))
    # Defining the figure
    # Axes with Cartopy projection
    #ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    # and extent
    #ax.set_extent([min_lon_local, max_lon_local, min_lat_local, max_lat_local], ccrs.PlateCarree())
    ax.set_extent([min_lon_EA, max_lon_EA, min_lat_EA, max_lat_EA], ccrs.PlateCarree())

    # Plotting using Matplotlib the mean current
    cs_p = ds_plot['precipitation'].plot.contourf(ax=ax,x='lon', y='lat',levels=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,1.0],                                                  extend='max',cmap='Greens',transform=ccrs.PlateCarree(),                                                  #cbar_kwargs=dict(label='Precipitation (mm/hr)'))
                                                  add_colorbar=False)
    #cs_p2 = prect.plot.contour(ax=ax,x='lon', y='lat',levels=[0.01,0.05,0.1,0.5,1.0],cmap='hot_r',linewidths=1.2,transform=ccrs.PlateCarree())
    cs = ds_plot['msl'].plot.contour(ax=ax,transform=ccrs.PlateCarree(),
                      levels=slp_levels,
                      colors='k',
                      linewidths=slp_lws)
    if r=='LCD':
        cs2 = ds_plot['msl'].plot.contour(ax=ax,transform=ccrs.PlateCarree(),
                      levels=[1018,],
                      colors='k',  
                      linestyles=['--',],linewidths=[0.8,])
    q1=resample.plot.quiver(ax=ax,x='lon', y='lat', u='u', v='v', transform=ccrs.PlateCarree(),
                        scale=250,color='coral',width=0.005,add_guide=False)

    ax.clabel(cs, slp_levels, inline=True, fmt='%.0f', fontsize=12)
    #ax.clabel(cs_p2, [0.01,0.05,0.1,0.5,1.0], inline=True, fmt='%.2f', fontsize=12)
    #q1=ds_plot.plot.quiver(x='lon', y='lat', u='ua', v='va', transform=ccrs.PlateCarree(), scale=80,color='g')
    ax.coastlines(resolution='50m',linewidth=2.0,color='royalblue')
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

    ax.plot([min_lon_tw, max_lon_tw,max_lon_tw,min_lon_tw,min_lon_tw],
            [min_lat_tw,min_lat_tw, max_lat_tw,max_lat_tw,min_lat_tw],
            lw=5,color='darkgreen',transform=ccrs.PlateCarree())
    ax.plot([min_lon_ish, max_lon_ish,max_lon_ish,min_lon_ish,min_lon_ish],
            [min_lat_ish,min_lat_ish, max_lat_ish,max_lat_ish,min_lat_ish],
            lw=1.5,color='r',transform=ccrs.PlateCarree())

    ax.set_title(titleStr,fontsize=22)
    ax.set_title('ERA5\nMSLP\nWind@1000 hPa',fontsize=12,loc='left')
    ax.set_title('IMERG\nPrecipitation',fontsize=12,loc='right')
    #ax.text(max_lon_tw+1,max_lat_tw-1.5,'prR=%.1f%%'%(prR*100),color='b',fontsize=15,transform=ccrs.PlateCarree())
    #ax.text(max_lon_tw+1,min_lat_tw+1,'WDCor=%.1f $\degree$'%wdCor,color='r',fontsize=15,transform=ccrs.PlateCarree())
    # Vector options declaration
    ax.text(-0.09,1.07,'(%s)'%lbl[i],fontsize=20,transform=ax.transAxes)
    if r=='LCD':
        veclenght = 10
        maxstr = '%3.1f m/s' % veclenght
        ax.quiverkey(q1,0.95,-0.2,veclenght,maxstr,labelpos='E', linewidth=30,coordinates='axes')

axes[1][0].set_position([0.24,0.225,0.228,0.343])
axes[1][1].set_position([0.55,0.225,0.228,0.343])

matplotlib.rc('xtick',labelsize=14)
cax=fig.add_axes([0.25, 0.2, 0.5, 0.03])
cbar=plt.colorbar(cs_p, cax=cax, orientation='horizontal',extend='max')
cbar.set_label('Precipitation(mm/hr)',fontsize=14)
plt.suptitle('Composite Weather Maps in Different Cold-Season Weather Types(%d-%d)'%(yr_start,yr_end),y=0.90,fontsize=30)
plt.savefig('../figures/figS2_weatherMap_all_types.ERA5.png',bbox_inches='tight',dpi=300)







