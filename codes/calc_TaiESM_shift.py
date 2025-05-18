from scipy.optimize import fsolve
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mc
import matplotlib,glob
import matplotlib.ticker as ticker
matplotlib.rc('xtick',labelsize=12)
matplotlib.rc('ytick',labelsize=12)

mu_x=np.linspace(-2,4,100)
mu_y=np.linspace(-3,3,100)
grid_x,grid_y=np.meshgrid(mu_x,mu_y)

[a,b,x0,y0]=[0.08303117,0.95491014,8.25855574,0.11692898]
[a0,a1,wd0]=[2.93885019,-1.29016647,103.75438081]
def ws_h(x, y):
    return a * (-x - x0)**2 + b * (-y - y0)**2

def wd_h(x, y):
    return a1*np.arctan(-y/(-x-a0))/np.pi*180+wd0

def wind2xy(wd,ws):
    def equations(vars):
        x, y = vars
        alpha=np.tan((wd-wd0)*np.pi/180/a1)
        eq1 = x-a0-y/alpha
        eq2 = a*(x-x0)**2+b*(y-y0)**2-ws
        return [eq1, eq2]

    x, y =  fsolve(equations, (2.1, 0))
    return -x,-y

def plotWSWDAxes(ax,clr_WD,clr_WS):
    mu_x=np.linspace(-2,4,100)
    mu_y=np.linspace(-3,3,100)
    grid_x,grid_y=np.meshgrid(mu_x,mu_y)
    c_wd=ax.contour(grid_x,grid_y,wd_h(grid_x,grid_y),levels=np.linspace(40,180,8),colors=[clr_WD])
    ax.clabel(c_wd, levels=[40,80,140], inline=True, fmt='$WD=%.0f\degree$', fontsize=12)

    c_ws=ax.contour(grid_x,grid_y,ws_h(grid_x,grid_y),levels=np.linspace(4,18,8),colors=[clr_WS])
    ax.clabel(c_ws, levels=[6,10,12], inline=True, fmt='WS=%.1f m/s', fontsize=12)
    #plt.xticks([])
    #plt.yticks([])
    return

def calcDensity(xy,xbins,ybins):
    #xtk=ytk=0.5*(bins[1:]+bins[:-1])
    xbin=np.digitize(xy[:,0],xbins)
    ybin=np.digitize(xy[:,1],ybins)
    cnt=np.zeros((ybins.shape[0]-1,xbins.shape[0]-1))

    for i in range(xbins.shape[0]-1):
        for j in range(ybins.shape[0]-1):
            #print(i,j,xbin[(xbin==i)&(ybin==j)].shape[0])
            cnt[j,i]=xbin[(xbin==i+1)&(ybin==j+1)].shape[0]
    return cnt

if __name__=='__main__':
    #load TaiESM synoptic flow
    runDict={'historical':['Historical',2001,2010],
             }
    
    xbins=np.linspace(-2,4,7)
    ybins=np.linspace(-3,3,7)
    m='TaiESM1'
    sce='ssp585'
    for yr in range(2025,2085,10):
      runDict['%04d'%yr]=['SSP-585',yr,yr+9] 
    for run in runDict.keys():
        if run=='historical':
            df=pd.read_csv('../data/cold_season_%s_%s.indices.csv'%(m,run),header=0,parse_dates=[0])
        else:
            df=pd.read_csv('../data/cold_season_%s_%s_%04d0101-%04d1231.indices.csv'\
                                         %(m,sce,runDict[run][1],runDict[run][2]),header=0,parse_dates=[0])


        df=df[(df.prRatio<=0.3)&(df.meanWS>=4.0)&(df.meanWS<=12.0)&(df.meanWD>=40)&(df.meanWD<=180)]
        print(m,run,runDict[run][1],df.date.size)
        wsList=df.meanWS.values
        wdList=df.meanWD.values

        xy=np.array([wind2xy(wd,ws) for wd,ws in zip(wdList,wsList)])
        cnt=calcDensity(xy,xbins,ybins)
        df=df.assign(**{'latent_x': xy[:,0], 'latent_y': xy[:,1],'xbin':np.digitize(xy[:,0],xbins),'ybin':np.digitize(xy[:,1],ybins)})
        runDict[run].append(df)
        runDict[run].append(cnt)

    for yr in range(2025,2085,10):
        yrStr='%04d'%yr
        #plot the flow regime change between current climate and future
        plt.close()
        plt.figure(figsize=(16,5))
        for i,run in enumerate(['historical',yrStr]):
            ax=plt.subplot(1,3,i+1)
            plotWSWDAxes(ax,'darkgoldenrod','skyblue')
            cs=ax.pcolormesh(xbins,ybins,runDict[run][4],vmin=0,vmax=100)
            for (jj,ii),days in np.ndenumerate(runDict[run][4]):
                ax.text(0.5*(xbins[ii]+xbins[ii+1]),0.5*(ybins[jj]+ybins[jj+1]),'%d'%days,color='crimson',ha='center',va='center')

            ax.set_title('%s %s\n%d-%d\n(%d days)'%(m,runDict[run][0],runDict[run][1],runDict[run][2],runDict[run][4].sum()),fontsize=12,loc='left')
            cb=plt.colorbar(cs,extend='max')
            cb.set_label('Days',fontsize=12)
        ax=plt.subplot(133)
        plotWSWDAxes(ax,'darkgoldenrod','skyblue')
        diff=runDict[yrStr][4]-runDict['historical'][4]
        cs=ax.pcolormesh(xbins,ybins,diff,cmap='bwr',vmin=-30,vmax=30)
        for (jj,ii),days in np.ndenumerate(diff):
            ax.text(0.5*(xbins[ii]+xbins[ii+1]),0.5*(ybins[jj]+ybins[jj+1]),'%d'%days,color='snow',ha='center',va='center')
        cb=plt.colorbar(cs,extend='both')
        cb.set_label('Days',fontsize=12)
        ax.set_title('%s [%s]-[Historical]\nSynoptic Flow Regime Change'%(m,runDict[run][0]),fontsize=12,loc='left')
        plt.savefig('../figures/%s.flow_regimes_changes_in_%s_%04dto%04d.png'%(m,sce,runDict[yrStr][1],runDict[yrStr][2]))

        np.save('../data/days_in_latent_space.%s_%s_%04dto%04d.npy'%(m,sce,runDict[yrStr][1],runDict[yrStr][2])\
            ,np.array([runDict['historical'][4],runDict[yrStr][4]]))


