import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm


def create_animated_trajectory(trajectory):
    """
    Create animated plots for controller performance from trajectory data
    """
    trajectory = trajectory[::4]
    num_points = len(trajectory)
    t = np.linspace(0, 7, num_points)
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    yaw = trajectory[:, 2]

    x_start, y_start, yaw_start = x[0], y[0], yaw[0]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].set_xlim(0, 7)
    axs[0, 1].set_xlim(0, 7)
    axs[1, 0].set_xlim(0, 7)
    axs[1, 1].set_xlim(-6, 7)
    axs[0, 0].set_ylim(-6, 7)
    axs[0, 1].set_ylim(-6, 7)
    axs[1, 0].set_ylim(-3.2, 3.2)
    axs[1, 1].set_ylim(-6, 7)

    axs[0, 0].set_xlabel('time')
    axs[0, 1].set_xlabel('time')
    axs[1, 0].set_xlabel('time')
    axs[1, 1].set_xlabel('x [m]')
    axs[0, 0].set_ylabel('x [m]')
    axs[0, 1].set_ylabel('y [m]')
    axs[1, 0].set_ylabel('yaw [rad]')
    axs[1, 1].set_ylabel('y [m]')

    axs[0, 0].set_title(r'$x(t)$')
    axs[0, 1].set_title(r'$y(t)$')
    axs[1, 0].set_title(r'$\theta(t)$')
    axs[1, 1].set_title('2d trajectory')
    axs[0, 0].plot(t, [0] * len(t), 'g--', label=r'$x_{des}$')
    axs[0, 1].plot(t, [0] * len(t), 'g--', label=r'$y_{des}$')
    axs[1, 0].plot(t, [0] * len(t), 'g--', label=r'$\theta_{des}$')
    axs[1, 1].scatter([0], [0], marker='X')
    axs[0, 0].legend(loc='upper right')
    axs[0, 1].legend(loc='upper right')
    axs[1, 0].legend(loc='upper right')
    axs[1, 1].legend(loc='upper right')
    fig.tight_layout()

    line1, = axs[0, 0].plot([], [])
    line2, = axs[0, 1].plot([], [])
    line3, = axs[1, 0].plot([], [])
    line4, = axs[1, 1].plot([], [])

    def animate(i):
        line1.set_data(t[:i], x[:i])
        line2.set_data(t[:i], y[:i])
        line3.set_data(t[:i], yaw[:i])
        line4.set_data(x[:i], y[:i])
        return line1, line2, line3, line4

    ani = FuncAnimation(fig, animate, frames=num_points, interval=500)
    ani.save(f'x={x_start} y={y_start} theta={yaw_start}.gif', writer=PillowWriter(fps=30))


if __name__ == '__main__':
    trajectories = np.array(
        [
            np.load(f"ModelTraining/jackal/data/trajectories/x=-5.000 y=5.000 theta=0.785.npy"),
        ]
    )
    for trajectory in tqdm(trajectories):
        create_animated_trajectory(trajectory)
