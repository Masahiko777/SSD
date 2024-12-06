import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# Function to compute the derivative of the quaternion
def quaternion_derivative(quaternion, omega):
    q = quaternion
    q_dot = 0.5 * np.array([
        q[3] * omega[0] + q[1] * omega[2] - q[2] * omega[1],
        q[3] * omega[1] - q[0] * omega[2] + q[2] * omega[0],
        q[3] * omega[2] + q[0] * omega[1] - q[1] * omega[0],
        -q[0] * omega[0] - q[1] * omega[1] - q[2] * omega[2]
    ])
    return q_dot

# Function to solve Euler's equations with feedback control
def euler_equations_with_feedback(state, I, kd):
    quaternion, omega = state[:4], state[4:]
    # Feedback control to reduce angular velocity
    control_torque = -kd * omega
    omega_dot = np.linalg.inv(I).dot(control_torque + np.cross(-omega, np.dot(I, omega)))
    q_dot = quaternion_derivative(quaternion, omega)
    return np.concatenate([q_dot, omega_dot])

# Fourth-order Runge-Kutta integration step with feedback control
def rk4_step_with_feedback(y, t, dt, I, kd):
    k1 = euler_equations_with_feedback(y, I, kd)
    k2 = euler_equations_with_feedback(y + 0.5 * k1 * dt, I, kd)
    k3 = euler_equations_with_feedback(y + 0.5 * k2 * dt, I, kd)
    k4 = euler_equations_with_feedback(y + k3 * dt, I, kd)
    y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return y_new

# Initial conditions
I = np.diag([0.2235, 0.1342, 0.1600])  # Moment of inertia (units: kg*m^2)
omega_initial = np.array([0.0, 0.0, 0.8])  # Initial angular velocity (units: rad/s)
quaternion_initial = R.from_euler('XYZ', [0, 0, 0]).as_quat()  # Initial attitude (quaternion)

# Initial state vector
initial_state = np.concatenate([quaternion_initial, omega_initial])

# Time parameters
t_start = 0
t_end = 10
dt = 0.01
num_steps = int((t_end - t_start) / dt)
times = np.linspace(t_start, t_end, num_steps)

# Control gain for reducing angular velocity
kd = 0.1

# Runge-Kutta integration with feedback control
states = np.zeros((num_steps, len(initial_state)))
states[0] = initial_state
for i in range(1, num_steps):
    states[i] = rk4_step_with_feedback(states[i-1], times[i-1], dt, I, kd)
    states[i, :4] /= np.linalg.norm(states[i, :4])  # Normalize the quaternion

# Extracting quaternions and angular velocities
quaternions = states[:, :4]
omegas = states[:, 4:]

# Extracting Euler angles in degrees
euler_angles_deg = np.degrees(np.array([R.from_quat(q).as_euler('XYZ') for q in quaternions]))

# Plotting the quaternion
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(times, quaternions)
plt.title("Quaternion")
plt.xlabel("Time (seconds)")
plt.ylabel("Quaternion Components")
plt.legend(['q0', 'q1', 'q2', 'q3'])

# Plotting the angular velocity in degrees per second
plt.subplot(1, 3, 2)
plt.plot(times, omegas)
plt.title("Angular Velocity")
plt.xlabel("Time (seconds)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend(['wx', 'wy', 'wz'])

# Plotting the Euler angles in degrees
plt.subplot(1, 3, 3)
plt.plot(times, euler_angles_deg)
plt.title("Euler Angles")
plt.xlabel("Time (seconds)")
plt.ylabel("Euler Angles (degrees)")
plt.legend(['Roll', 'Pitch', 'Yaw'])

plt.tight_layout()
plt.show()
