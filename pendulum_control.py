import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

# System parameters
m = 1.0     # mass of the pendulum (kg)
M = 10.0     # mass of the cart (kg)
L = 2.0     # length of the pendulum (m)
g = 9.81   # gravitational acceleration (m/s^2)
d = 0       # dampening of cart (m/s)

# Equilibrium point (upright position)
theta_eq = np.pi
x_eq = 0

# Linearization
A = np.array([
    [0, 1, 0, 0],
    [0, -d/M, m*g/M, 0],
    [0, 0, 0, 1],
    [0, -d/(M*L), -(M+m)*g/(M*L), 0]
])

B = np.array([
    [0],
    [1/M],
    [0],
    [1/(M*L)]
])

# Simulation parameters
t_span = np.linspace(0, 10, 1000) 
initial_state = [0, 0, np.pi, 0.0]  # [x, x_dot, theta, theta_dot]

# Linearized system dynamics
def linearized_system(state, t, F):
    return np.dot(A, state) + np.dot(B, [F])

# Simulate the system
F = 0  # No external force
solution = odeint(linearized_system, initial_state, t_span, args=(F,))

# Extract states
x = solution[:, 0]
theta = solution[:, 2] + theta_eq  # Add equilibrium point to get actual angle

# Create the animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 8)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.grid(True)

cart_width = 0.4
cart_height = 0.2
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

plt.title('Pendulum on a Cart')
plt.xlabel('Position (m)')
plt.ylabel('Height (m)')
plt.show()

# Uncomment the following line to save the animation as a gif
anim.save('pendulum_cart_animation.gif', writer='pillow', fps=24)