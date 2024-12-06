import numpy as np
import matplotlib.pyplot as plt

def f1(x, t):
    f = np.exp(-x) * np.cos(t)
    return f

x0 = np.log(2) # Initial value
t0 = 0.0 # Initial time

# Time and interval
T = 50
n = 500
dt = T / n

x = np.zeros(n+1)
t = np.zeros(n+1)
x[0] = x0
t[0] = t0

# Runge-Kutta
for i in range(n):
    k_1 = dt * f1(x[i], t[i])
    k_2 = dt * f1(x[i] + k_1 / 2, t[i] + dt / 2)
    k_3 = dt * f1(x[i] + k_2 / 2, t[i] + dt / 2)
    k_4 = dt * f1(x[i] + k_3, t[i] + dt)        
    x[i+1] = x[i] + 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
    t[i+1] = t[i] + dt

# Analytical solution
xa = np.zeros((501))
ta = np.zeros((501))
dta = 0.1
xa[0] = x0
ta[0] = t0
for i in range(0, 500):
    xa[i+1] = np.log(2 + np.sin(ta[i]))
    ta[i+1] = ta[i] + dta
     
# Create figure window
fig = plt.figure(figsize=(12, 8), facecolor='w')
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Arial'
plt.plot(t[:], x[:], lw=1.5, marker="o", label='numerical')
plt.plot(ta[:], xa[:], lw=1.5, color="red", label='analytical')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
plt.xlabel("t")
plt.ylabel("x")
plt.show()