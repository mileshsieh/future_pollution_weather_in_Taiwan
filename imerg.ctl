dset /data/C.shaoyu/data/GPM_IMERG/all_rename/imerg_%y4%m2%d2_%h2%n200.HDF5
dtype hdf5_grid
title imerg
options template
undef -9999.9
xdef 3600 linear -179.95 0.1
ydef 1800 linear -89.95 0.1
zdef 1    levels 1000
tdef 333073   linear 00:00Z01JAN2001 30mn
vars 2
/Grid/precipitation=>rain  0 t,x,y prec
/Grid/precipitationQualityIndex=>qlt 9 t,x,y prec quality
endvars


!! h5ls -r all_rename/imerg_20010101_120000.HDF5
!! /                        Group
!! /Grid                    Group
!! /Grid/Intermediate       Group
!! /Grid/Intermediate/IRinfluence Dataset {1/Inf, 3600, 1800}
!! /Grid/Intermediate/IRprecipitation Dataset {1/Inf, 3600, 1800}
!! /Grid/Intermediate/MWobservationTime Dataset {1/Inf, 3600, 1800}
!! /Grid/Intermediate/MWprecipSource Dataset {1/Inf, 3600, 1800}
!! /Grid/Intermediate/MWprecipitation Dataset {1/Inf, 3600, 1800}
!! /Grid/Intermediate/precipitationUncal Dataset {1/Inf, 3600, 1800}
!! /Grid/lat                Dataset {1800}
!! /Grid/lat_bnds           Dataset {1800, 2}
!! /Grid/latv               Dataset {2}
!! /Grid/lon                Dataset {3600}
!! /Grid/lon_bnds           Dataset {3600, 2}
!! /Grid/lonv               Dataset {2}
!! /Grid/nv                 Dataset {2}
!! /Grid/precipitation      Dataset {1/Inf, 3600, 1800}
!! /Grid/precipitationQualityIndex Dataset {1/Inf, 3600, 1800}
!! /Grid/probabilityLiquidPrecipitation Dataset {1/Inf, 3600, 1800}
!! /Grid/randomError        Dataset {1/Inf, 3600, 1800}
!! /Grid/time               Dataset {1/Inf}
!! /Grid/time_bnds          Dataset {1/Inf, 2}

