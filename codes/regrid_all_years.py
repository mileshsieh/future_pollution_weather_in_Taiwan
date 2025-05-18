#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import xarray as xr
import pandas as pd
import numpy as np
import xesmf as xe
import glob
import matplotlib
from dask.diagnostics import ProgressBar
ProgressBar().register()

#file list and template
fileDict=np.load('../data/CMIP6_file_dict.npy',allow_pickle=True).item()

IMERG_DIR='/data/dadm1/obs/GPM_IMERG/GPM_3IMERGHH.07'
ERA5_SFC_DIR='/data/dadm1/reanalysis/ERA5/SFC/day'
ERA5_PRS_DIR='/data/dadm1/reanalysis/ERA5/PRS/day'

#define focusing region
[min_lon, max_lon, min_lat, max_lat]=[100.,150.,10.,50.]
m='TaiESM1'
#m='MPI-ESM1-2-LR'
#for m in fileDict.keys():
for m in ['TaiESM1',]:
  CMIP6_DIR='/data/dadm1/model_output/CMIP6/%s'%m
  if 'pr' in fileDict[m]['historical_var']:
    f_CMIP6=glob.glob('%s/historical/atmos/day/r1i1p1f1/pr*%s*%s.nc'
                    %(CMIP6_DIR,m,fileDict[m]['historical_prd'][-1]))[-1]
  else:
    f_CMIP6=glob.glob('%s/historical/atmos/day/r1i1p1f1/ua*%s*%s.nc'
                    %(CMIP6_DIR,m,fileDict[m]['historical_prd'][-1]))[-1]
  print(m,f_CMIP6)
  #f_CMIP6=CMIP6_DIR+'/historical/atmos/day/r1i1p1f1/pr_day_TaiESM1_historical_r1i1p1f1_gn_20000101-20091231.nc'
  ds_model=xr.open_dataset(f_CMIP6).sel(lat=slice(min_lat,max_lat),lon=slice(min_lon,max_lon)).isel(time=0)
  
  def _preprocess_wind(ds):
      return ds.reindex(latitude=list(reversed(ds.latitude)))\
               .sel(latitude=slice(min_lat-1,max_lat+1),longitude=slice(min_lon-1,max_lon+1),level=1000.0)
  def _preprocess_mslp(ds):
      return ds.reindex(latitude=list(reversed(ds.latitude)))\
               .sel(latitude=slice(min_lat-1,max_lat+1),longitude=slice(min_lon-1,max_lon+1))['msl']/100.0
  def _preprocess_temp(ds):
      return ds.reindex(latitude=list(reversed(ds.latitude)))\
               .sel(latitude=slice(min_lat-1,max_lat+1),longitude=slice(min_lon-1,max_lon+1),level=[700.0,1000.0])
  def _preprocess_IMERG(ds):
      return ds.sel(lat=slice(min_lat-1,max_lat+1),lon=slice(min_lon-1,max_lon+1))
  
  for yr in range(2015,2023):
  
      #ERA5 
      #wind field at 1000 mb 
      fList=np.concatenate([np.array(sorted(glob.glob('%s/%s/%04d/*'%(ERA5_PRS_DIR,var,yr)))) for var in ['u','v']])
      ds_ERA5_wind=xr.open_mfdataset(fList,preprocess=_preprocess_wind)
  
      #Calculate LTS
      #Temp field at 700 and 1000 mb 
      fList=np.array(sorted(glob.glob('%s/t/%04d/*'%(ERA5_PRS_DIR,yr))))
      ds_ERA5_temp=xr.open_mfdataset(fList,preprocess=_preprocess_temp)
      ds_ERA5_temp['LTS']=ds_ERA5_temp.t.sel(level=700)*np.power(1000/700,287/1004)-ds_ERA5_temp.t.sel(level=1000).compute()
  
      #SFC
      fname='%s/%s/%04d/*'%(ERA5_SFC_DIR,'msl',yr)
      ds_ERA5_mslp=xr.open_mfdataset(fname,preprocess=_preprocess_mslp)
  
      #merge ERA5 fields
      ds_ERA5=xr.merge([ds_ERA5_wind,ds_ERA5_mslp,ds_ERA5_temp.LTS])
      ds_ERA5.coords['time'] = ds_ERA5.time.dt.floor('1D')
      #create regridder
      regridder_ERA5=xe.Regridder(ds_ERA5,ds_model,'conservative')
      ds_ERA5.compute()
      #IMERG
      #create regridder
      f_IMERG='/data/dadm1/obs/GPM_IMERG/GPM_3IMERGHH.07/2004/001/3B-HHR.MS.MRG.3IMERG.20040101-S000000-E002959.0000.V07B.HDF5'
      ds_IMERG=xr.open_mfdataset(f_IMERG,group='/Grid',preprocess=_preprocess_IMERG)
      regridder_IMERG=xe.Regridder(ds_IMERG,ds_model,"conservative")
  
      #get file list of year and loop over everyday
      day_path=sorted(glob.glob('/data/dadm1/obs/GPM_IMERG/GPM_3IMERGHH.07/%04d/*'%yr))
      ds_list=[]
  
      for i,p in enumerate(day_path):
          print(p)
          #calcuate daily mean precipitation (mm/hr)
          prec=xr.open_mfdataset(p+'/*.HDF5',group='/Grid',preprocess=_preprocess_IMERG)['precipitation'].resample(time='1D').mean().compute()
          #get daily time label
          prec.coords['time']=prec.time.dt.floor('1D')
          dt_idx=prec.indexes['time'].to_datetimeindex()
          #make sure we merge IMERG and ERA% at the same day
          if dt_idx.values[0]==ds_ERA5.time.values[i]:
              rIMERG=regridder_IMERG(prec.isel(time=0),keep_attrs=True).compute()
              rERA5=regridder_ERA5(ds_ERA5.isel(time=i),keep_attrs=True).compute()
              final=xr.merge([rIMERG,rERA5])
              final=final.assign_coords(time=[pd.to_datetime(ds_ERA5.time.values[i])]).compute()
          else:
              print(dt_idx.values[0],' of IMERG does not match ',ds_ERA5.time.values[i],' of ERA5')
              continue
          ds_list.append(final)
  
      daily_ds=xr.concat(ds_list,dim='time')
      daily_ds.to_netcdf('../data/regridded_%s_%04d.nc'%(m,yr))
