#!/home/mileshsieh/anaconda3/envs/python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('xtick',labelsize=12)
matplotlib.rc('ytick',labelsize=12)

if __name__=='__main__':
  df=pd.read_csv('../data/wd150.csv',sep=r'\s+',header=0)
  #plot count with prRatio criteria
  x=np.arange(df.ndays.size)  # the label locations
  plt.close()
  fig,ax=plt.subplots(figsize=(10,6))
  ax.plot(x,df.ndays,lw=2,zorder=5)
  ax.set_ylim(0,90 )
  ax.set_ylabel('Occurrence of WD150 Days',fontsize=15)
  ax.set_xticks(x,df.Decade,rotation=70)
  #ax.set_xticklabels(df.Decade)
  ax.grid(axis='y')
  ax2=ax.twinx()

  rects=ax2.bar(x-0.25,df['3p_episode'],0.5,align='edge',color='orange',zorder=4)
  #ax2.bar_label(rects,fmt='%d',padding=3,fontsize=10,zorder=6)

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax2.set_ylabel('Occurrence of\nConsecutive WD150 days\n(3+ days)',fontsize=15)

  ax.set_title('Trends of WD150 Days and Consecutive Episodes',fontsize=18)

  ax.set_zorder(4) 
  ax.set_frame_on(False)
  ax2.set_ylim(0,5.5)
  ax2.bar_label(rects,fmt='%d',padding=3,fontsize=12,zorder=6)
  fig.subplots_adjust(top=0.93,bottom=0.2)
  plt.savefig('../figures/fig8_WD150_count.png',dpi=300)



