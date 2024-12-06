import numpy as np
import matplotlib.pyplot as plt

def f1(x,t,s_f):
    
    epsi = 0.888
    sig  = 5.669*10**(-8)
    A    = 0.06   # m^2
    Fu   = 0.8
    m    = 1.0    # kg
    c    = 905    # J/kg�EK 
    alp  = 0.659
    Es   = 1353   # W/m^2 ���z���G�l���M
    Ea   = 0.3*Es # W/m^2 �A���x�h�G�l���M
    Ee   = 237    # W/m^2 �n���ԊO�G�l���M
    Fs   = 0.25   # ���z�Ƃ̌`�ԌW�� �܂��̓x�[�^�p����\�����W���i�\�ʐς̓�1/4�����犄�ƑÓ��j
    Fa   = 0.25   # �n���i�A���x�h�j�Ƃ̌`�ԌW��
    Fe   = 0.25   # �n���i�ԊO�j�Ƃ̌`�ԌW��
    Q   = 0.0    # W �������M
    
    if s_f==0:
        f = (- epsi*sig*A*Fu*x**4)/(m*c)
    elif s_f==1:
        f = (Q+alp*Es*Fs*A+alp*Ea*Fa*A+alp*Ee*Fe*A - epsi*sig*A*Fu*x**4)/(m*c)
    return f

x0 = 273.15#�����l
t0 = 0.0#�����l

# Time and interval
T = 20000
n = 20000
dt = T / n

# sunflag
s_f = np.zeros(n+1)
s_t = 0
sunny_time = 60*60
shade_time = 30*60
    
x = np.zeros(n+1)
t = np.zeros(n+1)
x[0] = x0
t[0] = t0

# RungeKutta
for i in range(n):
    
    #sun-shade flag
    if s_t>shade_time:
        s_f = 1
        if s_t>sunny_time + shade_time:
            s_f = 0
            s_t = 0
    else:
        s_f = 0
    s_t = s_t + 1
    
    #Runge-kutta
    k_1 = dt * f1(x[i],t[i], s_f)
    k_2 = dt * f1(x[i] + k_1 /2 , t[i] + dt/2, s_f)
    k_3 = dt * f1(x[i] + k_2 /2 , t[i] + dt/2, s_f)
    k_4 = dt * f1(x[i] + k_3 , t[i] + dt, s_f)
    x[i+1] = x[i] + 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4 )
    t[i+1] = t[i] + dt
     
# create figure window
fig = plt.figure(figsize=(12, 8), facecolor='w')
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] ='Arial'
plt.plot(t[:],x[:],lw = 1.5, marker="o", label='numerical')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
plt.xlabel("time [s]")
plt.ylabel("T [K]")
plt.show()