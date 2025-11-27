import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import copy,math,os
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.arrayprint import dtype_is_implied
#import subprocess

xl=100; yl=100; nx=100; ny=100; u=0.5; v=0.0
nx1=nx+1; ny1=ny+1; nx2=nx+2; ny2=ny+2
xp=50;yp=50;rb=10;fp=0.5
xsize=10.;ysize=xsize/xl*yl
etime=60;dt=0.5
ims=[]


x = np.linspace(0, xl, nx1)
y = np.linspace(0, yl, ny1)
z = np.round(np.linspace(0,fp,11),2)
X,Y=np.meshgrid(x,y)
dx=xl/nx; dy=yl/ny

f=np.zeros([ny1,nx1]);fn=np.zeros([ny1,nx1])
u2d=u*np.ones([ny1,nx1])
u2d[:,:nx1//2]=-u
v2d=v*np.ones([ny1,nx1])

# Initial f profile
for i in np.arange(0,nx):
    for j in np.arange(0,ny):
        r=math.sqrt((xp-x[i])**2+(yp-y[j])**2)
        if r<rb:
            f[j,i]=(rb-r)/rb*fp
        else:
            f[j,i]=0.

fn=copy.copy(f)

levels=np.arange(0.,fp,fp/5.)
time=0.;icount=0;fskip=5
nfile=0

while time<=etime:
    #element-wise
    #for i in np.arange(0,nx):
    #    for j in np.arange(0,ny): 
    #        fn[i,j]=f[i,j]-(u*(f[i,j]-f[i-1,j])/dx+v*(f[i,j]-f[i,j-1])/dy)*dt
    #array operation
    #fn=f-u2d*dt*(f-np.roll(f,1,axis=0))/dx-v2d*dt*(f-np.roll(f,1,axis=1))/dy

    #upwind
    dfdx_p=f-np.roll(f,1,axis=1)
    dfdx_n=np.roll(f,-1,axis=1)-f
    dfdy_p=f-np.roll(f,1,axis=0)
    dfdy_n=np.roll(f,-1,axis=0)-f
    fn=f-u2d*dt/dx*np.where(u2d>0,dfdx_p,np.where(u2d<0,dfdx_n,0))\
        -v2d*dt/dy*np.where(v2d>0,dfdy_p,np.where(v2d<0,dfdy_n,0))
    

    f=copy.copy(fn)
    if (icount % fskip) == 0:
        nfile=nfile+1
        fig = plt.figure(figsize=(25, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        text1=ax.text(xl*.6,yl*.9,0.25,"Time="+str(np.round(time,3))+"sec",size='25')
        ax.set_title('2D Backward 3D-Plot',size='32')
        ax.set_xlabel('x',size='20')
        ax.set_ylabel('y',size='20')
        ax.set_xticklabels(x,size='20')
        ax.set_yticklabels(y,size='20')
        ax.set_zlabel('f',size='20')
        ax.set_zticklabels(z,size='20')
        ax.set_zlim(0, 0.3) 
        srf=ax.plot_surface(X,Y,f,cmap="plasma")
#        fig.colorbar(srf)
        fname=f'test.adv.{nfile:03d}.png'
        print(fname)
        im=plt.savefig(fname)
        plt.clf()
        plt.close()

#  
    time=time+dt
    icount=icount+1

#subprocess.call('ffmpeg -framerate 30 -i png/f%4d.png -r 60 -an -vcodec libx264 -pix_fmt yuv420p animation.mp4', shell=True)
#os.system("ffmpeg -i animation.mp4 animation.gif -loop 0")
