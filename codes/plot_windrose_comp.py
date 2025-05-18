#!/home/mileshsieh/anaconda3/envs/dask/bin/python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter,FixedLocator,FixedFormatter
matplotlib.rc('xtick',labelsize=12)
matplotlib.rc('ytick',labelsize=12)
#windrose citiation:
#https://gist.github.com/phobson/41b41bdd157a2bcf6e14
def speed_labels(bins, units):   
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if np.isinf(right):
            labels.append('>{} {}'.format(left, units))
        else:
            labels.append('{} - {} {}'.format(left, right, units))
        print(left,right,labels[-1])
    return list(labels)

def _convert_dir(directions, N=None):
    if N is None:
        N = directions.shape[0]
    barDir = directions * np.pi/180. - np.pi/N
    barWidth = 2 * np.pi / N
    return barDir, barWidth

def wind_rose(rosedata, wind_dirs, ax=None,palette=None):
    if palette is None:
        #palette = seaborn.color_palette("Spectral", n_colors=rosedata.shape[1])
        #palette = seaborn.color_palette("inferno", n_colors=rosedata.shape[1])
        palette = seaborn.color_palette("Spectral", n_colors=rosedata.shape[1])

    bar_dir, bar_width = _convert_dir(wind_dirs)

    if ax==None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')

    for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
        print(n,c1,c2)
        if n == 0:
            # first column only
            ax.bar(bar_dir, rosedata[c1].values, 
                   width=bar_width,
                   color=palette[0],
                   edgecolor='none',
                   label=c1,
                   linewidth=0)

        # all other columns
        rects=ax.bar(bar_dir, rosedata[c2].values, 
               width=bar_width, 
               bottom=rosedata.cumsum(axis=1)[c1].values,
               color=palette[n+1],
               edgecolor='none',
               label=c2,
               linewidth=0)
    #ax.bar_label(rects,fmt='%d',padding=10,fontsize=12)
    #xtl = ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    #ax.xaxis.set_major_locator(FixedLocator(np.array([20.,50.,80.,110.,140.,170.0,200.,230.,260.,290.,320.,350.,])*np.pi/180.0))
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(0,330,12)*np.pi/180.0))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(FixedLocator(np.array([0.,90.,180.,270.])*np.pi/180.0))
    ax.xaxis.set_minor_formatter(FixedFormatter(['N','E','S','W']))
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(axis='x', which='minor', labelsize=30)
    ax.grid(which='minor', axis='x',linewidth=3,color='k')
    ax.set_ylim(0,350)
    ax.set_xlim(0/180*np.pi,210/180*np.pi)
    return ax,rosedata

if __name__=='__main__':
    m='TaiESM1'
    runDict={#'historical':['Historical','2001','2010'],
             'near-ssp585':['SSP-585','2021','2030'],
             'far-ssp585':['SSP-585','2091','2100'],
             }

    #date parser
    parser = lambda date: pd.datetime.strptime(date, '%Y%m%d')
    #create ws bins    
    spd_bins = [4,5,6,7,8,9,10,11,np.inf]
    spd_labels = speed_labels(spd_bins, units='m/s')
    spd_dict={}
    for i,lbl in enumerate(spd_labels):
        spd_dict['%d'%i]=lbl
    #create wd bins
    dir_bins = np.arange(-15, 380, 30)
    #dir_bins = np.arange(15, 196, 30)
    dir_labels = (dir_bins[:-1] + dir_bins[1:]) / 2    
    
    dfDict={}
    for run in runDict.keys():
        df=pd.read_csv('../data/LVD.%s_%s_%s0101-%s1231.csv'%(m,runDict[run][0],runDict[run][1],runDict[run][2]))
        #df.dropna(how='any',inplace=True)
        #f = interpolate.interp2d(df.WDsfc_479.values, epa.WS.values, epa['PM2.5'].values, kind='cubic')
        dfDict[run]=df
    runList=list(runDict.keys())
    nRun=len(runList)
    [col,ws_col]=['meanWD','meanWS']
    #wind rose
    plt.close()
    #plt.figure(figsize=(32,16))
    fig,axes=plt.subplots(1,nRun,figsize=(8*nRun, 8), subplot_kw=dict(polar=True))
    iplt=0
        
    for ax,run in zip(axes,runList):
        #df_for_bin=dfDict[run][dfDict[run].month.isin([2,3,4,])]
        df_for_bin=dfDict[run]
        total_count=df_for_bin.shape[0]
        
        print(run)
        print(f'total observations: {total_count}')
        wscut = pd.cut(df_for_bin[ws_col], bins=spd_bins, labels=list(spd_dict.keys()), right=True)
        wdcut = pd.cut(df_for_bin[col], bins=dir_bins, labels=dir_labels, right=False)
        df_bin = df_for_bin.assign(
            WindSpd_bins=lambda df: pd.cut(df[ws_col], bins=spd_bins, labels=list(spd_dict.keys()), right=True))\
            .assign(WindDir_bins=lambda df:pd.cut(df[col], bins=dir_bins, labels=dir_labels, right=False))\
            .replace({'WindDir_bins': {0: 360}})
                
        size_2D=df_bin.groupby(by=['WindSpd_bins', 'WindDir_bins']).size()
        rose=(size_2D.reindex(pd.MultiIndex.from_product([wscut.cat.categories, wdcut.cat.categories]))
              .unstack()
              .fillna(0.0)
              .sort_index(axis=1)).T
              #.applymap(lambda x: x / total_count * 100)).T
        rose.drop([0.0],inplace=True)
        rose.rename(columns=spd_dict,inplace=True)
        #print(rose.index.unique())
    
        #directions = np.arange(0, 360, 15)
        directions = dir_labels[1:]+15
        #ax=plt.subplot(1,2,iplt+1,projection='polar')
        ax,rosedata = wind_rose(rose, directions,ax)
        print(rosedata)
        #if iplt==0:
        #    leg = ax.legend(loc=(0.4, 0.1), ncol=3,fontsize=30)
    
        #ax.set_title('%s-%s\n(%d days)'%(runDict[run][1],runDict[run][2],total_count),fontsize=14)
        if iplt==0:
            plt.annotate('(a)\n%s-%s\n(%d days)'%(runDict[run][1],runDict[run][2],total_count),\
              xy=(0.08, 0.82), xytext=(0.08, 0.82),xycoords='figure fraction',fontsize=14)
        else:
            plt.annotate('(b)\n%s-%s\n(%d days)'%(runDict[run][1],runDict[run][2],total_count),\
              xy=(0.52, 0.82), xytext=(0.52, 0.82),xycoords='figure fraction',fontsize=14)
        iplt=iplt+1
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',fontsize=12,ncol=4)
    #plt.annotate('(a)', xy=(0.08, 0.88), xytext=(0.08, 0.88),xycoords='figure fraction',fontsize=14)
    #plt.annotate('(b)', xy=(0.52, 0.88), xytext=(0.52, 0.88),xycoords='figure fraction',fontsize=14)
    plt.suptitle(f'Upstream Flow Regimes of Lee Vortex Days\n in {m} {runDict[run][0][:3]}{runDict[run][0][-3:]} Projection',fontsize=20)
    plt.savefig('../figures/wr.%s.comp.png'%m)
