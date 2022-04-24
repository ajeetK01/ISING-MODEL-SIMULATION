""" ISING MODEL using monte carlo,metropolis algorithm"""
"""
Created on Tue Apr 12 18:43:21 2022
AJEET KUMAR(1810019)
INT.MSC.PHYSICS
"""

import numpy as np
import matplotlib.pyplot as plt
#plt.style.use(['science','notebook','grid'])
import numba
from numba import njit
#from scipy.ndimage import convolve,generate_binary_structure
N=50
init_random=np.random.random((N,N)) #1st make random 2-D lattice
lattice_n=np.zeros((N,N))
#making negative lattice
lattice_n[init_random>=0.75]=1
lattice_n[init_random<.75]=-1
#next part
init_random=np.random.random((N,N))
lattice_p=np.zeros((N,N))
lattice_p[init_random>=.25]=1     
lattice_p[init_random<.25]=-1
#print(lattice_n)
plt.imshow(lattice_p)    #IMAGE OF LATTICE
#print(lattice_n)
#we have to modify this function
#print("net lattice energy=",get_energy(lattice_p))

@numba.njit("UniTuple(f8[:],2)(f8[:,:],i8,f8,f8)",nopython=True,nogil=True)
def metropolis(spin_arr,times,T,energy):
    spin_arr=spin_arr.copy()
    net_spins=np.zeros(times-1)
    net_energy=np.zeros(times-1)
    for t in range(0,times-1):
        #co-ordinate of randomly selected spin
        x=np.random.randint(0,N)
        y=np.random.randint(0,N)
        spin_i=spin_arr[x,y]
        spin_f=-1*spin_arr[x,y]
        e_i=0
        e_f=0
        #boundary conditions
        if(x>0):
            e_i+=-spin_i*spin_arr[x-1,y]
            e_f+=-spin_f*spin_arr[x-1,y]
        if(x<N-1):
            e_i+=-spin_i*spin_arr[x+1,y]
            e_f+=-spin_f*spin_arr[x+1,y]
        if(y>0):
            e_i+=-spin_i*spin_arr[x,y-1]
            e_f+=-spin_f*spin_arr[x,y-1]
        if(x<N-1):
            e_i+=-spin_i*spin_arr[x,y+1]
            e_f+=-spin_f*spin_arr[x,y+1]
        de=e_f-e_i
        """MOST IMPORTANT STEP MONTE CARLO (MATROPOLIS)ALGORITHM"""
        if(de>0 and np.random.random()<np.exp(-de/T)): """IF DE IS POSITIVE ,GENERATE RANDOM NUMBER BW 0-1 AND COMPARE WITH BOLTZMANN PORBABILITY.AND DECIDE FLIP OR REJECT  """
            spin_arr[x,y]=spin_f
            energy+=de
        elif(de<=0):      """SPIN SHOULD FLIP IF de IS NEGATIVE . i.e SPIN FLIP FAVOUR LOWERING ENERGY"""
            
            spin_arr[x,y]=spin_f   
            energy+=de
        net_spins[t]=spin_arr.sum()
        net_energy[t]=energy
    return(net_spins,net_energy)
#CALLING METROPOLIS ALGO FUNCTION BY GIVING RELEVANT PARAMETER 
spins,energies=metropolis(lattice_n,1000000,1.42,get_energy(lattice_n))
fig,axes=plt.subplots(1,2,figsize=(12,4)) # 1 row and 2 coulumn plots
ax=axes[0]
ax.plot(spins/N**2)
ax.set_xlabel("time step")
ax.set_ylabel("average spin")
ax.grid
ax=axes[1]
ax.plot(energies)
ax.set_xlabel("time step")
ax.set_ylabel("energy")
ax.grid()
fig.tight_layout()
#function to get initial energy
def get_energy(lattice):
    ener=0
    for i in range(1,N-1):
        for j in range(1,N-1):
            ener=ener-lattice[i,j]*(lattice[i,j+1]+lattice[i,j-1]+lattice[i+1,j]+lattice[i-1,j])
    return(ener)
print("total initial energy=",get_energy(lattice_n))
# FUNCTION FOR VARIATION OF MAGNETIZATION VS TEMPERATURE
def magnetization(lattice,T,carlo_step,energy):
    
    av_magnetization=np.zeros(carlo_step)
    temp=np.zeros(carlo_step)
    f_energy=np.zeros(carlo_step)
    for stp in range(carlo_step):
        temp[stp]=T
        for i in range(1,N-1):
            for j in range(1,N-1):
                ei=-lattice[i,j]*(lattice[i,j+1]+lattice[i,j-1]+lattice[i+1,j]+lattice[i-1,j])

                ef=lattice[i,j]*(lattice[i,j+1]+lattice[i,j-1]+lattice[i+1,j]+lattice[i-1,j])

                de=ef-ei
                if(de<0):
                    lattice[i,j]=-1*lattice[i,j]
                    energy+=de
                elif(de>0 and np.random.random()<np.exp(-de/T)):
                    lattice[i,j]=-lattice[i,j]
                    energy+=de
        av_magnetization[stp]=lattice.sum()/N**2
        f_energy[stp]=energy
        T=T+(4)/1000
    return(av_magnetization,temp,f_energy)
res_magnetization,TEMP,ENERGY=magnetization(lattice_n,1,1000,get_energy(lattice_n))
#print("mag as times",res_magnetization)

plt.figure(figsize=(8,4))
plt.xlabel("TEMPERATURE")
plt.ylabel("MAGNETIZATION")
plt.plot(TEMP,res_magnetization)
#plt.plot(TEMP)
plt.figure(figsize=(8,4))
plt.plot(TEMP,ENERGY)
plt.xlabel("TEMPERATURE")
plt.ylabel("ENERGY")
#CORRELATION FUNCTION
def correlation(lattice,T,m,n,dist,steps,energy):
    corr=np.zeros(steps)
    temp=np.zeros(steps)
    for stp in range(steps):
        temp[stp]=T
        corr[stp]=lattice[m,n]*(lattice[m,n+dist]+lattice[m,n-dist]+lattice[m+dist,n]+lattice[m-dist,n])
        for i in range(1,N-1):
            for j in range(1,N-1):
                
                ei=-lattice[i,j]*(lattice[i,j+1]+lattice[i,j-1]+lattice[i+1,j]+lattice[i-1,j])

                ef=lattice[i,j]*(lattice[i,j+1]+lattice[i,j-1]+lattice[i+1,j]+lattice[i-1,j])

                de=ef-ei
                if(de<0):
                    lattice[i,j]=-1*lattice[i,j]
                    energy+=de
                elif(de>0 and np.random.random()<np.exp(-de/T)):
                    lattice[i,j]=-lattice[i,j]
                    energy+=de
        T=T+4/steps
    return(corr,temp)
CORR,TEMPC=correlation(lattice_n,1,8,8,5,10000,get_energy(lattice_n))
plt.figure(figsize=(8,4))
plt.ylabel("CORRELATION")
plt.xlabel("TEMPERATURE")
#plt.plot(TEMPC,CORR/4)
#FUNCTION TO GET FLUCTUATION IN ENERGY AND BEHAVIOUR OF HEAT CAPACITY AS FUNCTN OF TEMPERATURE
def fluctuation(lattice,T,steps):
    av_energy=np.zeros(steps)
    e_std=np.zeros(steps)
    temp=np.zeros(steps)
    
#    temp=T
    for stp in range(steps):
        temp[stp]=T
        spins,energies=metropolis(lattice,100000,T,get_energy(lattice))
        av_energy[stp]=energies[-10000:].mean()   #AVERAGE ENERGY IS MEAN OF LAST 10000  MONTE CARLO STEPS
        e_std[stp]=energies[-10000:].std()        #standard deviation is calculated by taking last 10000 steps deviation
        T=T+4/steps
    return(av_energy,e_std,temp)
AVG_ENERGY,E_DEVIATION,TEMP=fluctuation(lattice_n,1,1000)  
#print(TEMP)   
plt.figure(figsize=(8,4))
plt.xlabel("TEMPERATURE")
plt.ylabel("ENERGY FLUCTUATION")  
plt.plot(TEMP,E_DEVIATION)
        
                
                
    
    
 


























