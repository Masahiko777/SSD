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

# Function to compute quaternion error for feedback control
def quaternion_error(target_quaternion, current_quaternion):
    # Convert to Rotation objects
    target_rot = R.from_quat(target_quaternion)
    current_rot = R.from_quat(current_quaternion)
    # Compute the conjugate of the current quaternion
    conj_current = current_rot.inv()
    # Compute the error quaternion (difference between target and current)
    error_rot = conj_current * target_rot
    # Convert back to quaternion
    error_quat = error_rot.as_quat()
    return error_quat

# Function to solve Euler's equations with feedback control
def euler_equations_with_feedback_2(state, I, kd, kp, target_quaternion):
    quaternion, omega = state[:4], state[4:]
    # Quaternion error for attitude control
    error_quat = quaternion_error(target_quaternion, quaternion)
    # Control torque based on quaternion error and angular velocity
    control_torque = -kd * omega - kp * error_quat[:3]
    omega_dot = np.linalg.inv(I).dot(control_torque + np.cross(-omega, np.dot(I, omega)))
    q_dot = quaternion_derivative(quaternion, omega)
    return np.concatenate([q_dot, omega_dot])

# Fourth-order Runge-Kutta integration step with feedback control
def rk4_step_with_feedback(y, t, dt, I, kd, kp, target_quaternion):
    k1 = euler_equations_with_feedback_2(y, I, kd, kp, target_quaternion)
    k2 = euler_equations_with_feedback_2(y + 0.5 * k1 * dt, I, kd, kp, target_quaternion)
    k3 = euler_equations_with_feedback_2(y + 0.5 * k2 * dt, I, kd, kp, target_quaternion)
    k4 = euler_equations_with_feedback_2(y + k3 * dt, I, kd, kp, target_quaternion)
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

# Target Euler angles and corresponding quaternion
target_euler = np.radians([-20, 0, 30])  # Convert to radians
target_quaternion = R.from_euler('XYZ', target_euler).as_quat()

# Control gains
kd = 0.1  # Damping gain for angular velocity control
kp = 0.1  # Proportional gain for quaternion error control

# Runge-Kutta integration with feedback control
states = np.zeros((num_steps, len(initial_state)))
states[0] = initial_state
for i in range(1, num_steps):
    states[i] = rk4_step_with_feedback(states[i-1], times[i-1], dt, I, kd, kp, target_quaternion)
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
plt.legend(['É÷x', 'É÷y', 'É÷z'])

# Plotting the Euler angles in degrees
plt.subplot(1, 3, 3)
plt.plot(times, euler_angles_deg)
plt.title("Euler Angles")
plt.xlabel("Time (seconds)")
plt.ylabel("Euler Angles (degrees)")
plt.legend(['Roll', 'Pitch', 'Yaw'])

plt.tight_layout()
plt.show()