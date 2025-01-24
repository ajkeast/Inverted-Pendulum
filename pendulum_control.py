import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are
from matplotlib.animation import FuncAnimation

# System parameters
m = 0.1     # mass of the pendulum (kg)
M = 0.8     # mass of the cart (kg)
L = 0.5     # length of the pendulum (m)
g = -9.81   # gravitational acceleration (m/s^2)
d = 1       # dampening of cart (kg/s)

# Equilibrium point (upright position)
theta_eq = 0
x_eq = 0

A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, -m*g/M, 0],
    [0, 0, 0, 1],
    [0, -d/(M*L), -(M + m)*g/(M*L), 0]
])

B = np.array([[0], [1/M], [0], [-1/(M*L)]])

# Define Q and R matrices for LQR
Q = np.diag([10, 1, 100, 1])  # State weighting matrix
R = np.array([[5]])            # Control weighting matrix

# Solve the Continuous Algebraic Riccati Equation (CARE)
P = solve_continuous_are(A, B, Q, R)

# Compute the LQR gain matrix K
K = np.linalg.inv(R) @ B.T @ P

# Simulation parameters
t_span = np.linspace(0, 5, 500) 
initial_state = [0, 0, np.pi/8, 0]  # [x, x_dot, theta, theta_dot]

# Linearized system dynamics with LQR control
def linearized_system(state, t):
    F = -np.dot(K, state)  # LQR control law
    dstate = np.dot(A, state) + np.dot(B, F).flatten()  # Ensure output is 1D
    return dstate

# Simulate the system
solution = odeint(linearized_system, initial_state, t_span)

# Extract states
x = solution[:, 0]
theta = solution[:, 2] + theta_eq  # Add equilibrium point to get actual angle

# Create the animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.grid(True)

cart_width = 0.2
cart_height = 0.1
pendulum_line, = ax.plot([], [], 'o-', lw=2)
cart_rect = plt.Rectangle((0, 0), cart_width, cart_height, fc='r')

time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    ax.add_patch(cart_rect)
    time_text.set_text('')
    return pendulum_line, cart_rect, time_text

def animate(i):
    cart_x = x[i]
    pendulum_x = cart_x + L * np.sin(theta[i])
    pendulum_y = L * np.cos(theta[i])
    
    pendulum_line.set_data([cart_x, pendulum_x], [0, pendulum_y])
    cart_rect.set_xy((cart_x - cart_width/2, -cart_height/2))
    time_text.set_text(f'Time: {t_span[i]:.2f} s')
    return pendulum_line, cart_rect, time_text

anim = FuncAnimation(
                    fig, 
                    animate, 
                    init_func=init, 
                    frames=len(t_span), 
                    interval=10, 
                    blit=True, 
                    repeat=False
    )

plt.title('Pendulum on a Cart with LQR Control')
plt.xlabel('Position (m)')
plt.ylabel('Height (m)')
plt.show()

# Uncomment the following line to save the animation as a gif
anim.save('pendulum_cart_lqr_animation.gif', writer='pillow', fps=24)