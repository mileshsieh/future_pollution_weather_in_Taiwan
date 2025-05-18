#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd

if __name__=='__main__':
    #load historical/ssp-585 upstream flow regimes and local flows
    m='TaiESM1'
    sce='ssp585'
    runDict={'historical':['Historical',2001,2010],}
    for yr in range(2021,2100,10):
        runDict['%04d'%yr]=['SSP-585',yr,yr+9]

    #for run in ['historical','2025','2085']:
    for run in ['2021','2091']:
        flow=np.load('../data/local_flow.%s_%s_%s0101-%s1231.npy'%\
                     (m,runDict[run][0],runDict[run][1],runDict[run][2]))
        df=pd.read_csv('../data/LVD.%s_%s_%04d0101-%04d1231.csv'%\
               (m,runDict[run][0],runDict[run][1],runDict[run][2]),header=0)
        no2_24hr=np.array([np.load('/data/mileshsieh/future_daily_NO2/local_NO2.%s_%s.%s.npy'%\
                (m,runDict[run][0],dd))[-1,:,:] for dd in df.date])    
        print(flow.shape,df.date.size,no2_24hr.shape)
        np.save('../data/no2_transport_24hr.%s_%s_%s0101-%s1231.npy'%\
                     (m,runDict[run][0],runDict[run][1],runDict[run][2]),no2_24hr)
