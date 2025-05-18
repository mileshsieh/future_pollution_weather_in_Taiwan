###data processing assoicated package
#!/home/mileshsieh/anaconda3/envs/sat/bin/python
import pickle
import numpy as np
import glob
import xarray as xr
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

[min_lon, max_lon, min_lat, max_lat]=[115.,130.,18.,30.]
def _preprocess_wind(ds):
  return ds.reindex(latitude=list(reversed(ds.latitude)))\
           .sel(latitude=slice(min_lat-1,max_lat+1),longitude=slice(min_lon-1,max_lon+1),level=1000.0)
ERA5_PRS_DIR='/data/dadm1/reanalysis/ERA5/PRS/day'

slon,elon,slat,elat=116.0,127.5,19.2,27.8
if __name__=='__main__':
  
  #ERA5 
  #wind field at 1000 mb 
  fList=np.concatenate([np.array(sorted(glob.glob('%s/%s/2020/*'%(ERA5_PRS_DIR,var)))) for var in ['u','v']])
  ds_ERA5_wind=xr.open_mfdataset(fList,preprocess=_preprocess_wind)
  #ds_ERA5_vortex=ds_ERA5_wind.sel(time='2020-08-20').squeeze().isel(longitude=slice(None, None, 4),latitude=slice(None, None, 4))
  ds_ERA5_vortex=ds_ERA5_wind.sel(time='2020-08-20').squeeze()
  lon=ds_ERA5_vortex.longitude.values
  lat=ds_ERA5_vortex.latitude.values
  xx,yy=np.meshgrid(lon,lat)
  uu=ds_ERA5_vortex.u.values
  vv=ds_ERA5_vortex.v.values
  '''
  fList=sorted(glob.glob('../data/sat_data/*_rgb.pkl'))
  for tt,f in enumerate(fList):
    band_date=f[-20:-8]
    ### read satellite data
    with open(f, 'rb') as f:
  '''
  for band_date in ['202008200300',]:
    print(band_date)
    with open('../data/sat_data/'+ band_date +'_rgb.pkl', 'rb') as f:
      enh_data=pickle.load(f)
    ### drawing satellite image
    plt.close()
    fig,ax1 = plt.subplots(1,1,figsize=(9,8),tight_layout=True)
    m = Basemap(llcrnrlon=slon, urcrnrlon=elon, llcrnrlat=slat, urcrnrlat=elat,resolution='i')
    m.drawcoastlines(linewidth=1.2,color='yellow')
    m.drawparallels(np.arange(-10., 61., 20), labels=[1, 0, 0, 0], linewidth=0.2, color='k', fontsize=16)
    m.drawmeridians(np.arange(90., 181., 20), labels=[0, 0, 0, 1], linewidth=0.2, color='k', fontsize=16)
    m.imshow(enh_data)
    q=plt.quiver(xx,yy,uu,vv,color='aqua')
    plt.tight_layout()
    qk = ax1.quiverkey(q, 0.85, 1.05, 5, r'$5 m s^{-1}$', labelpos='E',
                   coordinates='figure',labelcolor='k',fontproperties={'weight':'bold','size':16})
    plt.text(0.45,0.95,'%s/%s/%s %02d:%s LST'%\
           (band_date[0:4],band_date[4:6],band_date[6:8],\
            int(band_date[8:10])+8,band_date[10:12]),\
           color='yellow',fontsize=20,transform=plt.gca().transAxes)
    plt.savefig('../figures/sat/'+band_date+'_wind_0.25.png',dpi=200)
    #plt.savefig('../figures/sat/sat.%02d.png'%tt,dpi=200)

