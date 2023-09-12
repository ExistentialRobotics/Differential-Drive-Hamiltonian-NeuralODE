import os.path

import numpy as np
import matplotlib.pyplot as plt
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data/"


def create_joint_trajectory_plots(trajectories):
    """
    create a single plot showing controller performance for movement from different poses to origin
    """
    print(len(trajectories))
    colors = np.random.randint(0, 255, (len(trajectories), 3)) / 255
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['text.usetex'] = True
    fontsize=36
    legend_handles = [
        plt.Line2D([0], [0], color='magenta', label='yaw'),
        plt.Line2D([0], [0], marker='x', color='red', label='start position'),
        plt.Line2D([0], [0], marker='o', color='green', label='end position', markersize=5),
        plt.Line2D([0], [0], marker='*', color='blue', label='desired position', markersize=5),
    ]
    for it, trajectory in enumerate(trajectories):
        color = tuple(colors[it])
        N = len(trajectory)
        print(color)
        legend_handles.append(
            plt.Line2D([0], [0], color=color, label=f'trajectory {it + 1}')
        )
        plt.scatter(trajectory[:, 0], trajectory[:, 1], s=0.5, color=color, label=f'trajectory {it}')
        arrow_x = np.append(trajectory[:, 0][50:N - 800:200], trajectory[:, 0][-100].reshape(1), axis=0)
        arrow_y = np.append(trajectory[:, 1][50:N - 800:200], trajectory[:, 1][-100].reshape(1), axis=0)
        arrow_dx = np.cos(np.append(trajectory[:, 2][50:N - 800:200], trajectory[:, 2][-100].reshape(1), axis=0)) * 1
        arrow_dy = np.sin(np.append(trajectory[:, 2][50:N - 800:200], trajectory[:, 2][-100].reshape(1), axis=0)) * 1
        print(arrow_x.shape, arrow_y.shape)
        print(arrow_dx.shape, arrow_dy.shape)
        plt.quiver(arrow_x, arrow_y, arrow_dx, arrow_dy, scale=1.5, scale_units='xy', angles='xy', color='magenta')
        plt.xlabel("X coordinate", fontsize=fontsize)
        plt.ylabel("Y Coordinate", fontsize=fontsize)
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'rx', markersize=15)
        plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'go', markersize=15)
        plt.plot(0, 0, 'b*', markersize=15)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])

    plt.tick_params(axis='x', labelsize=fontsize)
    plt.tick_params(axis='y', labelsize=fontsize)

    plt.legend(handles=legend_handles, loc="best", fontsize=22)
    plt.savefig(
        f"{THIS_DIR}png/pose_stabilization.pdf",
        dpi=500, format='pdf')
    plt.show(block=True)


if __name__ == '__main__':
    trajectories = np.array(
        [
            np.load(f"{THIS_DIR}x=5.000 y=5.000 theta=-2.094.npy"),
            np.load(f"{THIS_DIR}x=5.000 y=-5.000 theta=2.094.npy"),
            np.load(f"{THIS_DIR}x=6.000 y=0.001 theta=1.571.npy"),
            np.load(f"{THIS_DIR}x=-5.000 y=5.000 theta=0.785.npy"),
            np.load(f"{THIS_DIR}x=-5.000 y=-5.000 theta=1.047.npy"),
        ]
    )
    print(trajectories.shape)
    create_joint_trajectory_plots(trajectories)
