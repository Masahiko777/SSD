import numpy as np
import matplotlib.pyplot as plt

def f1(x,t):
    f = [x[1],-3*x[0]]
    return f

# Time and interval
T = 50
n = 500
dt = T / n

x = np.zeros((n+1,2))
t = np.zeros(n+1)
x[0,0:2] = [1.0,0.0]#‰Šú’l
t[0] = 0.0#‰Šú’l

# RungeKutta
for i in range(n):
    k_1 = dt * np.array(f1(x[i,:],t[i]))
    k_2 = dt * np.array(f1(x[i,:] + k_1 /2 , t[i] + dt/2))
    k_3 = dt * np.array(f1(x[i,:] + k_2 /2 , t[i] + dt/2))
    k_4 = dt * np.array(f1(x[i,:] + k_3 , t[i] + dt))      
    x[i+1,:] = x[i,:] + 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4 )
    t[i+1] = t[i] + dt

# ‰ğÍ‰ğ
xa = np.zeros((501,2))
ta = np.zeros((501))
dta = 0.1
xa[0,0:2] = [1.0,0.0]
ta[0] = 0.0
for i in range(0,500):
    xa[i+1] = np.sin(np.sqrt(3)*ta[i]+np.arcsin(1.0))
    ta[i+1] = ta[i] + dta
    
# create figure window
fig = plt.figure(figsize=(12, 8), facecolor='w')
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] ='Arial'
plt.plot(t[:],x[:,0],lw = 1.5, marker="o", label='numerical')
#plt.plot(t[:],x[:,1],lw = 1.5, marker="*", label='numerical')
plt.plot(ta[:],xa[:,0],lw = 1.5, color="red", label='analytical')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
plt.xlabel("t")
plt.ylabel("x")
plt.show()