#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from glob import glob
import torch
import torch.nn as nn
from VAE import variationalAutoEncoder,Encoder,Decoder,Reshape
from scipy.optimize import fsolve
from scipy.interpolate import griddata

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
    m='TaiESM1'
    sce='ssp585'
    #local circulation
    #get lat/lon for interpolate generated 10-km flow into 2-km resolution
    (xstart,xend,ystart,yend)=(20,321,150,451)
    lonTW=np.load('../data/tw_s_lon.npy')[xstart:xend]
    latTW=np.load('../data/tw_s_lat.npy')[ystart:yend]
    step=5
    lon_flow=lonTW[::step]
    lat_flow=latTW[::step]
    xx_TW,yy_TW=np.meshgrid(lonTW, latTW)
    xx_flow,yy_flow=np.meshgrid(lon_flow, lat_flow)

    #load AI-TaiwanVVM model
    num_input_channels=2
    latent_dim=2
    beta=0.01
    dataset='ctrl'
    thd=11
    seed=3
    ts=6
    te=54
    sf='vae61x61_ldim%d_b%.4f_%s_t%dto%d_seed%d_norm%d'%(latent_dim,beta,dataset,ts,te,seed,thd)
    device = torch.device("cpu")
    # Load model
    model_ae = torch.load('../data/leevortex.%s.pth'%sf,map_location=torch.device('cpu'))
    model_ae.eval()

    #load TaiESM synoptic flow
    #runDict={'historical':['Historical',2001,2010],}
    runDict={}

    for yr in range(2021,2100,10):
    #for yr in [2021,]:
        runDict['%04d'%yr]=['SSP-585',yr,yr+9]
    for run in runDict.keys():
        if run=='historical':
            df=pd.read_csv('../data/cold_season_%s_%s.indices.csv'%(m,run),header=0,parse_dates=[0])
        else:
            df=pd.read_csv('../data/cold_season_%s_%s_%04d0101-%04d1231.indices.csv'\
                                         %(m,sce,runDict[run][1],runDict[run][2]),header=0,parse_dates=[0])

        df=df[(df.prRatio<=0.3)&(df.meanWS>=4.0)&(df.meanWS<=12.0)&(df.meanWD>=40)&(df.meanWD<=180)]
        print(m,run,runDict[run][1],df.date.size)

        temp_xy=np.array([wind2xy(df.meanWD.values[k],df.meanWS.values[k]) for k in range(df.date.size)])
        local_flow=model_ae.decode(torch.Tensor(temp_xy).to(device)).detach().cpu().numpy()*thd
        #interpolate
        u=np.array([griddata((xx_flow.flatten(),yy_flow.flatten()),local_flow[k,0,:,:].flatten(),\
                   (xx_TW,yy_TW),method='cubic') for k in range(len(local_flow))])
        v=np.array([griddata((xx_flow.flatten(),yy_flow.flatten()),local_flow[k,1,:,:].flatten(),\
                   (xx_TW,yy_TW),method='cubic') for k in range(len(local_flow))])
        local_flow_2km=np.swapaxes(np.array([u,v]),0,1)

        np.save('../data/local_flow.%s_%s_%04d0101-%04d1231.npy'%\
                 (m,runDict[run][0],runDict[run][1],runDict[run][2]),local_flow_2km)
        df.to_csv('../data/LVD.%s_%s_%04d0101-%04d1231.csv'%\
                 (m,runDict[run][0],runDict[run][1],runDict[run][2]),index=False)

