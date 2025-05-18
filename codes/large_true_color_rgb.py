###data processing assoicated package
import pickle
import numpy as np
import glob
import numpy.ma as ma
import cv2
import math
from pyspectral.rayleigh import Rayleigh
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#### enhancement method
def enh_data_cloud(band_data):
        band_data = band_data*255
        #enh_band = np.zeros((14000,19000,3))
        enh_band = np.zeros(band_data.shape)
        enh_band[:,:] = 255
        x = [0, 25,  55, 100, 255]
        y = [0, 90, 140, 175, 255]
        for  i in range(0,4):
            x1 = x[i]
            x2 = x[i+1]
            y1 = y[i]
            y2 = y[i+1]
            m = (y2-y1)/(x2-x1)
            b = y2-(m*x2)
            mask1 = np.zeros((tlat_ran,tlon_ran,3))
            mask1[band_data>=x1] = 1
            mask2 = np.zeros((tlat_ran,tlon_ran,3))
            mask2[band_data<x2] = 1
            new_mask = mask1*mask2
            enh_band[new_mask>0] = band_data[new_mask>0]*m + b
        enh_band = enh_band/255
        return enh_band
####
### Taiwan area on original himawari domain
x = np.arange(85.0025,205.0025,0.005)
y = np.arange(-59.9975,60.0025,0.005)
tai_lon=x[6700:7700]
tai_lat=y[16200:17200]
print(tai_lon[0],tai_lon[-1])
print(tai_lat[0],tai_lat[-1])
### Download area
lon_s = 0
lon_ran = 19000
lat_s = 10000
lat_ran = 14000
local_lon=x[lon_s:lon_s+lon_ran]
local_lat=y[lat_s:lat_s+lat_ran]
print('Download area')
print(local_lon[0],local_lon[-1])
print(local_lat[0],local_lat[-1])
### Taiwan area on download area
#tlon_ran = 19000
#tlat_ran = 14000
#drawing_lon_s = 0
#drawing_lat_s = 0
slon,elon,slat,elat=116.0,127.5,19.2,27.8
drawing_lon_s=np.where(local_lon>=slon)[0][0]
drawing_lon_e=np.where(local_lon>=elon)[0][0]
drawing_lat_s=np.where(local_lat>=slat)[0][0]
drawing_lat_e=np.where(local_lat>=elat)[0][0]
tlon_ran=drawing_lon_e-drawing_lon_s
tlat_ran=drawing_lat_e-drawing_lat_s
local_tai_lon = local_lon[drawing_lon_s:drawing_lon_s+tlon_ran]
local_tai_lat = local_lat[drawing_lat_s:drawing_lat_s+tlat_ran]
print('Taiwan area')
print(local_tai_lon[0],local_tai_lon[-1])
print(local_tai_lat[0],local_tai_lat[-1])


### loading data
folder = '/data/dadm1/obs/Himawari/vortex_20200820/'
folder_len = len(folder)
print(folder_len)
hima_file = sorted(glob.glob('' + folder + '*202008200520*band_03.pkl'))

for i in range(len(hima_file)):
  print(hima_file[i])
  band_name = hima_file[i]
  band_date = band_name[folder_len:folder_len+12]
## ext
  with open('' + folder + ''+band_date+'_band_03.pkl', 'rb') as f:
        band03 = pickle.load(f)
## vis
  with open('' + folder + ''+band_date+'_band_01.pkl', 'rb') as f:
        band01 = pickle.load(f)
  with open('' + folder + ''+band_date+'_band_02.pkl', 'rb') as f:
        band02 = pickle.load(f)
  with open('' + folder + ''+band_date+'_band_04.pkl', 'rb') as f:
        band04 = pickle.load(f)
## geo
  with open('' + folder + ''+band_date+'_geo.pkl', 'rb') as f:
        hi_geo = pickle.load(f)
## 0.5 km resloution array size
  array_size = band03.shape
### resolution sharping and export selected area
## geo data
  sun_az_map = hi_geo[0,:,:]
  sun_zh_map = hi_geo[1,:,:]
  sat_az_map = hi_geo[2,:,:]
  sat_zh_map = hi_geo[3,:,:]
  fin_sun_az_map = cv2.resize(np.single(sun_az_map), (array_size[1], array_size[0]), interpolation=cv2.INTER_LINEAR)
  fin_sun_zh_map = cv2.resize(np.single(sun_zh_map), (array_size[1], array_size[0]), interpolation=cv2.INTER_LINEAR)
  fin_sat_az_map = cv2.resize(np.single(sat_az_map), (array_size[1], array_size[0]), interpolation=cv2.INTER_LINEAR)
  fin_sat_zh_map = cv2.resize(np.single(sat_zh_map), (array_size[1], array_size[0]), interpolation=cv2.INTER_LINEAR)
  local_sun_az_map = fin_sun_az_map[drawing_lat_s:drawing_lat_s+tlat_ran,drawing_lon_s:drawing_lon_s+tlon_ran]
  local_sun_zh_map = fin_sun_zh_map[drawing_lat_s:drawing_lat_s+tlat_ran,drawing_lon_s:drawing_lon_s+tlon_ran]
  local_sat_az_map = fin_sat_az_map[drawing_lat_s:drawing_lat_s+tlat_ran,drawing_lon_s:drawing_lon_s+tlon_ran]
  local_sat_zh_map = fin_sat_zh_map[drawing_lat_s:drawing_lat_s+tlat_ran,drawing_lon_s:drawing_lon_s+tlon_ran]

  local_sun_sat_diff=local_sun_az_map-local_sat_az_map
  local_rad=np.radians(local_sun_zh_map)
  local_adjust=np.cos(local_rad)
  hima = Rayleigh('Himawari-8', 'ahi')
  del fin_sun_az_map, fin_sun_zh_map, fin_sat_az_map, fin_sat_zh_map
## band data
  bb01 = cv2.resize(band01, (array_size[1], array_size[0]), interpolation=cv2.INTER_CUBIC)
  bb02 = cv2.resize(band02, (array_size[1], array_size[0]), interpolation=cv2.INTER_CUBIC)
  bb03 = band03
  bb04 = cv2.resize(band04, (array_size[1], array_size[0]), interpolation=cv2.INTER_CUBIC)

  b01 = bb01[drawing_lat_s:drawing_lat_s+tlat_ran,drawing_lon_s:drawing_lon_s+tlon_ran]
  b02 = bb02[drawing_lat_s:drawing_lat_s+tlat_ran,drawing_lon_s:drawing_lon_s+tlon_ran]
  b03 = bb03[drawing_lat_s:drawing_lat_s+tlat_ran,drawing_lon_s:drawing_lon_s+tlon_ran]
  b04 = bb04[drawing_lat_s:drawing_lat_s+tlat_ran,drawing_lon_s:drawing_lon_s+tlon_ran]
  del bb01, bb02, bb03, bb04

### adjust to true color
  b01=b01/local_adjust
  b02=b02/local_adjust
  b03=b03/local_adjust
  b04=b04/local_adjust

  refl_cor_band1 = hima.get_reflectance(local_sun_zh_map, local_sat_zh_map, local_sun_sat_diff, 'ch1',b03)
  refl_cor_band2 = hima.get_reflectance(local_sun_zh_map, local_sat_zh_map, local_sun_sat_diff, 'ch2',b03)
  refl_cor_band3 = hima.get_reflectance(local_sun_zh_map, local_sat_zh_map, local_sun_sat_diff, 'ch3',b03)

  cor_bandgreen=0.93*(b02-refl_cor_band2)+0.07*b04
  cor_bandgreen[cor_bandgreen<0]=0
  cor_band011=b01-refl_cor_band1
  cor_band033=b03-refl_cor_band3
  cor_band011[cor_band011<0]=0
  cor_band033[cor_band033<0]=0

  band_data = np.zeros((tlat_ran,tlon_ran,3))
  band_data[:,:,0] = cor_band033
  band_data[:,:,1] = cor_bandgreen
  band_data[:,:,2] = cor_band011
  enh_data = enh_data_cloud(band_data/100)

### output data
  with open('../data/sat_data/'+ band_date +'_rgb.pkl', 'wb') as f:
    pickle.dump(enh_data, f)
### drawing
  plt.close()
  fig,ax1 = plt.subplots(1,1,figsize=(9,8),tight_layout=True)
  m = Basemap(llcrnrlon=slon, urcrnrlon=elon, llcrnrlat=slat, urcrnrlat=elat,resolution='i')
  m.drawcoastlines(linewidth=1.2,color='yellow')
  m.drawparallels(np.arange(-10., 61., 20), labels=[1, 0, 0, 0], linewidth=0.2, color='k', fontsize=16)
  m.drawmeridians(np.arange(90., 181., 20), labels=[0, 0, 0, 1], linewidth=0.2, color='k', fontsize=16)
  m.imshow(enh_data)
  plt.tight_layout()
  plt.savefig('../figures/test_'+band_date+'.png',dpi=200)

### output data
#  with open('temporary_data/'+ band_date +'_r.pkl', 'wb') as f:
#    pickle.dump(enh_data[:,:,0], f)
#  with open('temporary_data/'+ band_date +'_g.pkl', 'wb') as f:
#    pickle.dump(enh_data[:,:,1], f)
#  with open('temporary_data/'+ band_date +'_b.pkl', 'wb') as f:
#    pickle.dump(enh_data[:,:,2], f)


