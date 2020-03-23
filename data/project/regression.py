


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import statistics


# In[8]:


"""def linear(l,W_p,TV_p,sig,s,n):
 Mob=np.zeros((2,s))
 d_2=np.zeros((1,s))#Distance between WiFi and Mobile
 d=np.zeros((1,s)) #Distance between mobile and Tv
 Mob[:,0]=[1,2]#Initial Coordinates
 for i in range(s-1):
    Mob[0,i+1]=Mob[0,i]+ random.random()*(sig)
    Mob[1,i+1]=Mob[1,i]+ random.random()*(sig)

 for i in range(s):
    d_2[0,i]=math.sqrt((Mob[0,i]-W_p[0,0])**2 + (Mob[1,i]-W_p[1,0])**2)
    d[0,i]=math.sqrt((Mob[0,i]-TV_p[0,0])**2 + (Mob[1,i]-TV_p[1,0])**2)

  return Mob,d_2,d ;"""


"""def planar(l,wp,tvp,sig,s,n):
    mob=np.zeros((2,s,n))
    d_2=np.zeros((1,s,n))
    d =np.zeros((1,s,n))

    for j in range(n):
        mob[0,0,j]=1
        mob[1,0,j]=2
        for i in range(s-1):
            if i<6:
                mob[0,i+1,j]=mob[0,i,j]+random.random()*0.7
                mob[1,i+1,j]=mob[1,i,j]-random.random()*0.2
            elif i<14 and i>=6:
                mob[0,i+1,j]=mob[0,i,j]-random.random()*0.2
                mob[1,i+1,j]=mob[1,i,j]+random.random()*0.7
            elif i<23 and i>=14:
                mob[0,i+1,j]=mob[0,i,j]-random.random()*0.7
                mob[1,i+1,j]=mob[1,i,j]-random.random()*0.2
            else:
                mob[0,i+1,j]=mob[0,i,j]+random.random()*0.3
                mob[1,i+1,j]=mob[1,i,j]-random.random()*1.2
            
    for j in range(n) :
        for i in range(s):
            d_2[0,i,j]=math.sqrt((mob[0,i,j]-wp[0,0])**2+(mob[1,i,j]-wp[1,0])**2)
            d[0,i,j]=math.sqrt((mob[0,i,j]-tvp[0,0])**2+(mob[1,i,j]-tvp[1,0])**2)
                        
    return mob,d_2,d;"""


# In[9]:


#lnsm_generate
"""def lnsm_generate(pref,dref,n,d):
 rss=pref-10*n*(math.log10(d/dref))
 return rss


#noise adder
def noise_adder(rss,sd):
 temp = rss+sd*np.random.normal(0,1)
 return temp"""


#lnsm_comput
def lnsm_comput (pref , dref , N , rss ):
 d = dref* 10**((pref-rss)/(10*N))
 return d

#lateration
def lateration(d,d2,d1,TV_p):
    alpha=math.acos((d**2+d1**2-d2**2)/(2*d*d1))
    y=TV_p[0]+(d*math.sin(alpha))
    x=TV_p[1]+(d*math.cos(alpha))
    return x,y;

def regressionmodel(beta,rss):
    y  = beta[0] + (beta[1]*(math.log10(-rss))) + (beta[2]*(math.log10(math.log10(-rss))))
    return 10**y

# In[10]:


#Kalman Filter
def KalmanSmoother(x,P,z,F,H,Q,R):
 Y=np.dot(F,x)
 Ft=F.transpose()
 Pupd=np.dot(np.dot(F,P),Ft) + Q
             #Update step
 IM=np.dot(H,Y)
 IS =R+(np.dot(np.dot(H,Pupd),H.transpose()))
 K= np.dot(Pupd,H.transpose())/IS
 K2=np.zeros((2,1))
 K2[0,0]=K[0]
 K2[1,0]=K[1]
 Y=Y+ K2*(z-IM)
 Pupd=Pupd-(np.dot(np.dot(K2,IS),K2.transpose()))
 #print(Y)
 return Y,Pupd;

def dist(refx,refy,posx,posy):
 d=math.sqrt((posx-refx)**2+(posy-refy)**2)
 return d;


# In[11]:
pts=[[1.95,1.19],[1.95,1.75],[1.95,2.3],[1.95,3.57],[1.95,4.76],[1.95,6.545],[1.95,7.6],[0.81,7.6],[-0.453,4.76],[-0.453,3.57],[-0.453,1.19]] #List of all coordinates

betaw= [-13.7343,15.0700,-48.8996]
betab=[-24.0820,29.0001,-108.6417]
Pref_bt=-50
Pref_w=-26 #RSSI of WiFi at 1 meter
dref=1  #Reference distance for both WiFi and BLE
#Pref_bt=-66  #Refrence RSSI of BLE at 1meter
N=4
TV_p= np.zeros((2,1)) #Assumption of origin at TV
W_p=np.zeros((2,1))
W_p[0][0]=1.95 #WiFi position on X-axis
l = 10 #Room dimension (l x l)
n = 1 #no of trajectories
s = 11#no of samples in a trajectory
dw=np.zeros((1,s))
dbt=np.zeros((1,s))
#sig_traj = 0.3 #noise variation for trajectory generation
#RSSIw_2 = np.zeros((1,s)) #RSSI of WiFi
#RSSIbt_2 = np.zeros((1,s)) #RSSI of TV
#noise = 2.5 # Noise variance to be added in RSSI value

for i in range(s):
 dw[0][i]=dist(1.95,0,pts[i][0],pts[i][1])
 dbt[0][i]=dist(0,0,pts[i][0],pts[i][1])

#mob_coord , d_2 , d = planar(l,W_p,TV_p,sig_traj,s,n) # generating mobile coordination


d_1 = math.sqrt((W_p[0][0]-TV_p[0][0])**2 + (W_p[1][0]-TV_p[1][0])**2)
#RSSIw_1 = lnsm_generate(Pref_w , dref , N , d_1 )

#for i in  range(s):
#    RSSIw_2[0,i] = lnsm_generate(Pref_w , dref , N ,d_2[0,i])
#   RSSIbt_2[0,i] = lnsm_generate(Pref_bt , dref , N ,d[0,i]) 
    
rw = np.zeros((1,s))   # to store the difference between consecutive RSSI values
rbt = np.zeros((1,s))
x = np.zeros((2,s))   # to store the 2*s matrix of ideal RSSI and rw
xb = np.zeros((2,s))
#z = np.zeros((2,s))   #noise added RSSIs
#zb = np.zeros((2,s))
pos = np.zeros((2,s))  #position estimated from filtered RSSI values
npos = np.zeros((2,s))

d2f = np.zeros((1,s)) # distance obtained from RSSI values
d2n = np.zeros((1,s))
df = np.zeros((1,s))
dn = np.zeros((1,s))
xs = np.zeros((2,s))  # filtered RSSI values
xsb = np.zeros((2,s))


#x[0,0] = RSSIw_2[0,0] 
#xb[0,0] = RSSIbt_2[0,0]

# storing ideal RSSI value 
"""for i in range(1,s):
    rw[0,i] = RSSIw_2[0,i] - RSSIw_2[0,i-1]
    rbt[0,i] = RSSIbt_2[0,i] - RSSIbt_2[0,i-1]
    x[0,i] = RSSIw_2[0,i]
    x[1,i]=rw[0,i]
    xb[0,i]=RSSIbt_2[0,i]
    xb[1,i]=rbt[0,i]  """
#storing noisy RSSI value or measured values
#for i in range(s):
#   z[0,i] = noise_adder(RSSIw_2[0,i],noise)
#   zb[0,i] = noise_adder(RSSIbt_2[0,i],noise)
zb = []
zb.append(-50)
##Reading input from the exel file of BLE Data
df1 = pd.read_excel(r"/Users/shubhendrasingh/Desktop/Realtime/ble_data.xlsx")
for i in range(s):
    arr = []
    for x in list(df1.iloc[:,i]) :
        arr.append(int(x[0:-1]))
    zb.append(statistics.median(arr))


z = []
z.append(-26)
##Reading Input from the Exel file of Wifi Data
df1 = pd.read_excel(r"/Users/shubhendrasingh/Desktop/Realtime/wifi_data.xlsx")
for i in range(s):
    arr = []
    for x in list(df1.iloc[:,i]) :
        arr.append(x)
    z.append(statistics.median(arr))
print(z)
print(zb)
    
#Kalman Filter Parameters
dt = 1 #As per the variation in the environment
F = np.array([(1,dt),(0,1)])
H = np.array([1,0])
R = 50
Q = 1/R * np.eye(2,2)
Pw = 100*np.eye(2,2)
Pbt = 100*np.eye(2,2)

m=np.zeros((2,1))
mb=np.zeros((2,1))
xsb[0,0] = -50
xsb[0,1]=0
xs[0,0] =-26
m[0,0] = -26
m[1,0]=z[1]+26
mb[0,0] = -50
mb[1,0]=zb[1]+50

#d1f = lnsm_comput (Pref_w , dref , N , RSSIw_1 ) #distance between TV and Wifi
d1f = 1.95

d2f[0,0] = regressionmodel(betaw,xs[0,0] )
df[0,0] = regressionmodel(betab,xsb[0,0] )

d2n[0,0] = regressionmodel(betaw, z[0] )
dn[0,0] = regressionmodel(betab, zb[0] )


pos[:,0] = lateration(df[0,0],d2f[0,0],d1f,TV_p)
npos[:,0] = pos[:,0]

m,Pw = KalmanSmoother(m,Pw,z[1],F,H,Q,R)
mb,Pbt = KalmanSmoother(mb,Pbt,zb[1],F,H,Q,R)
xs[0,0] = m[0,0]
xs[1,0]=m[1,0]
xsb[0,0] = mb[0,0]
xsb[1,0] = mb[1,0]
d2f[0,i] = regressionmodel(betaw,xs[0,0] )
df[0,i] = regressionmodel(betab,xsb[0,0] )
d2n[0,i] = regressionmodel(betaw, z[1] )
dn[0,i] = regressionmodel(betab,zb[1] )
pos[0,0],pos[1,0] = lateration(df[0,0],d2f[0,0],d1f,TV_p)
npos[0,0],npos[1,0] = lateration(dn[0,0],d2n[0,0],d_1,TV_p)

for i in range(1,s):
    #when both wifi and ble are available
    if z[i] != 0 and zb[i] != 0 :
         
        m,Pw = KalmanSmoother(m,Pw,z[i+1],F,H,Q,R)
        mb,Pbt = KalmanSmoother(mb,Pbt,zb[i+1],F,H,Q,R)
        xs[0,i] = m[0,0]
        xs[1,i]=m[1,0]
        xsb[0,i] = mb[0,0]
        xsb[1,i] = mb[1,0]
        d2f[0,i] = regressionmodel(betaw,xs[0,i] )
        df[0,i] = regressionmodel(betab,xsb[0,i] )
        print(df[0,i])
        d2n[0,i] = regressionmodel(betaw, z[i] )
        dn[0,i] = regressionmodel(betab,zb[i] )

        
    elif zb[i] != 0 and z[i] == 0 :
        temp = m[0,0] + m[1,0]
        m,Pw = KalmanSmoother(m,Pw,temp,F,H,Q,R)
        mb,Pbt = KalmanSmoother(mb,Pbt,zb[i],F,H,Q,R)
        xs[0,i] = m[0,0]
        xs[1,i]=m[1,0]
        xsb[0,i] = mb[0,0]
        xsb[1,i] = mb[1,0]
        d2f[0,i] = regressionmodel(betaw,xs[0,i] )
        df[0,i] = regressionmodel(betab,xsb[0,i] )
        
        d2n[0,i] = regressionmodel(betaw, z[i] )
        dn[0,i] = regressionmodel(betab,zb[i] )


#pos[0,i],pos[1,i] = lateration(df[0,i],d2f[0,i],d1f,TV_p)
#npos[0,i],npos[1,i] = lateration(dn[0,i],d2n[0,i],d_1,TV_p)
    z1=[]
z1b=[]
t= []
errw=[]
errb=[]
n=["1","2","3","3a","3b","4","5","6","9a","10","11"]
for i in range(s):
    t.append(i)
    z1.append(z[i+1])
    z1b.append(zb[i+1])
    errb.append(abs(1-df[0,i]/dbt[0,i])*100)
    errw.append(abs(1-d2f[0,i]/dw[0,i])*100)
plt.figure(1)
plt.plot(t,z1)
plt.plot(t,xs[0,:])
plt.scatter(t,z1,label="Noisy")
plt.scatter(t,xs[0,:],label="Filtered")
#plt.plot(t,RSSIw_2[0,:],label="ideal")
plt.xlabel("Sample points")
plt.ylabel("Wifi RSSI values")
plt.legend()
plt.figure(2)
plt.plot(t,z1b)
plt.plot(t,xsb[0,:])
plt.scatter(t,z1b,label ="Noisy")
plt.scatter(t,xsb[0,:],label="Filtered")
#plt.plot(t,RSSIbt_2[0,:],label="ideal")
plt.xlabel("Sample points")
plt.ylabel("BLE RSSI values")
plt.legend()
plt.figure(3)
#plt.plot(t,d[0,:],label="actual")
plt.plot(t,dn[0,:])
plt.plot(t,df[0,:])
plt.scatter(t,dn[0,:],label="Noisy")
plt.scatter(t,df[0,:],label="Filtered")
plt.xlabel("Samples point")
plt.ylabel("BLE-Distances")
plt.legend()
plt.figure(4)
plt.plot(t,d2n[0,:])
plt.plot(t,d2f[0,:])
plt.scatter(t,d2n[0,:],label="Noisy")
plt.scatter(t,d2f[0,:],label="Filtered")
plt.xlabel("Sample point")
plt.ylabel("Wifi-Distances")
plt.legend()
plt.figure(5)
plt.plot(npos[0,:],npos[1,:])
plt.plot(pos[0,:],pos[1,:])
plt.scatter(npos[0,:],npos[1,:],label="Noisy")
plt.scatter(pos[0,:],pos[1,:],label="Filtered")
plt.xlabel("x - coordinates")
plt.ylabel("y - coordinates")
plt.legend()
plt.figure(6)
plt.plot(t,errw,label="")
plt.plot(t,errb,label="")
plt.scatter(t,errw,label="% error wifi")
plt.scatter(t,errb,label="% error ble")
plt.xlabel("Point")
plt.ylabel("Error")
plt.legend()
for i, txt in enumerate(n):
    plt.annotate(txt,(t[i],errb[i]))
    plt.annotate(txt,(t[i],errw[i]))
plt.show()


# In[ ]:




