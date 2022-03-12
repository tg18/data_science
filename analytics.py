import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
from scipy import interpolate 
from numpy import where
from math import sin

# plt.get_current_fig_manager().window.raise_()
LNWDT=2;FNT=15
plt.rcParams['lines.linewidth']=LNWDT; plt.rcParams['font.size']=FNT

init_func=1

if(init_func==0):
    def f(x):
        f=np.zeros_like(x)
        f[np.where(x<=0.1)]=1.0
        return f
elif(init_func==1):
    def f(x):
        f=np.zeros_like(x)
        x_left=0.25
        x_right=0.75
        xm=(x_right-x_left)/2.0
        f=where((x>x_left) & (x<x_right),np.sin(np.pi*(x-x_left)/(x_right-x_left))**4,f)
        return f
def ftbs(u):
    u[1:-1]=(1-c)*u[1:-1]+c*u[:-2]
    return u[1:-1]
def lax_wendroff(u):
    u[1:-1]=c/2.0*(1+c)*u[:-2]+(1-c**2)*u[1:-1]-c/2.0*(1-c)*u[2:]
    return u[1:-1]

# constants

a=0.5
tmin,tmax=0.0,6.0
xmin,xmax=0.0,4.0
Nx=320
c=0.5

# discretize

x=np.linspace(xmin,xmax,Nx+1)
dx=float((xmax-xmin)/Nx)
dt=c/a*dx
Nt=int((tmax-tmin)/dt)
time=np.linspace(tmin,tmax,Nt)

solvers=[ftbs,lax_wendroff]

u_solutions=np.zeros((len(solvers),len(time),len(x)))
uanalytical=np.zeros((len(time),len(x)))

for k,solver in enumerate(solvers):
    u=f(x)
    un=np.zeros((len(time),len(x)))
    for i,t in enumerate(time[1:]):
        if k==0:
            uanalytical[i,:]=f(x-a*t)
        u_bc=interpolate.interp1d(x[-2:],u[-2:])
        u[1:-1]=solver(u[:])
        u[-1]=u_bc(x[-1]-a*dt)
        un[i,:]=u[:]
    u_solutions[k,:,:]=un

# Animation

fig=plt.figure()
ax=plt.axes(xlim=(xmin,xmax),ylim=(np.min(un),np.max(un)*1.1))
lines=[]
legends=[]
for solver in solvers:
    line, = ax.plot([],[])
    lines.append(line)
    legends.append(solver.__name__)
line, = ax.plot([],[])
lines.append(line)
legends.append('Analytical')
plt.xlabel('x-coordinate')
plt.ylabel('Amplitude')
plt.legend(legends,loc=1,frameon=False)

def init():
    for line in lines:
        line.set_data([],[])
    return lines,
def animate_alt(i):
    for k,line in enumerate(lines):
        if(k==len(lines)-1):
            line.set_data(x,uanalytical[i,:])
        else:
            line.set_data(x,u_solutions[k,i,:])
    return lines,

anim=animation.FuncAnimation(fig,animate_alt,init_func=init,frames=Nt,interval=100,blit=False)
plt.show()



