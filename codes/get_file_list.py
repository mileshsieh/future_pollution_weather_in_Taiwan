#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import xarray as xr
import pandas as pd
import numpy as np
import glob

CMIP6_DIR='/data/dadm1/model_output/CMIP6/'
mList=[x.split('/')[-1] for x in glob.glob(CMIP6_DIR+'/*')]
available={}
grid_info=[]
#walk through entire directory tree for models with hist and ssp585 runs
for m in mList: 
  dList=glob.glob('%s%s/*'%(CMIP6_DIR,m)) 
  #print(m,sorted([d.split('/')[-1] for d in dList])) 
  sceList=[d.split('/')[-1] for d in dList] 
  if ('historical' in sceList) and ('ssp585' in sceList):
    available[m]={}
    for sce in ['historical','ssp585']:
      fList=sorted(glob.glob('%s%s/%s/atmos/day/r1i1p1f1/*.nc'%(CMIP6_DIR,m,sce)))
      prd=[x.split('_')[-1][:-3] for x in fList]
      varList=[x.split('/')[-1].split('_')[0] for x in
               sorted(glob.glob('%s%s/%s/atmos/day/r1i1p1f1/*%s*.nc'%(CMIP6_DIR,m,sce,prd[-1])))]
      #print(m,sce,prd[-1],varList)
      available[m]['%s_prd'%sce]=prd
      available[m]['%s_var'%sce]=varList
    #test file availability in history run
    fname=glob.glob('%s%s/%s/atmos/day/r1i1p1f1/%s*%s.nc'%(CMIP6_DIR,m,sce,varList[-1],prd[-1]))[0]
    print(m,fname)
    try:
      ds=xr.open_dataset(fname)
    except:
      print('remove %s'%m)
      del available[m]
      continue
    grid_info.append([m,ds.lon.values[1]-ds.lon.values[0],ds.lon.values.shape[0],\
                      ds.lat.values[1]-ds.lat.values[0],ds.lat.values.shape[0]])
x=pd.DataFrame(grid_info,columns=['model','dx','nx','dy','ny'])
x.to_csv('../data/gird_info.csv',index=False)
np.save('../data/CMIP6_file_dict.npy',available)

