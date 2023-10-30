import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.interpolate import make_interp_spline, BSpline
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/data/trajectories/"


def create_pose_stabilization_plot(trajectories):
    width = 12
    height = 7.8
    font_size = 30
    pdf = PdfPages("./plots/pose_stabilization.pdf")

    # Create a colormap to generate colors
    radius = 0.5
    transparent = 0.6
    scaling_factor = 0.5

    colors = ['#6dbce9', '#f3dda2', '#7ba848', '#db6e6a', '#dfa16c']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(width, height))
    # Create the grid
    gs = gridspec.GridSpec(1, 1)
    legend_handles = []
    for it, traj in enumerate(trajectories):
        color = colors[it % len(colors)]
        legend_handles.append(
            plt.Line2D([0], [0], color=color, label=f'Trajectory {it + 1}')
        )
        x = traj[:, 0]
        y = traj[:, 1]
        print(x[0], y[0], x[-1], y[-1])
        theta = traj[:, 2]
        t = np.arange(0, len(traj)) * 0.05
        spl = make_interp_spline(t, np.c_[x, y], k=3)
        smooth_trajectory = spl(t)
        ax = plt.subplot(gs[0, 0])
        ax.plot(smooth_trajectory[:, 0], smooth_trajectory[:, 1], color=color)

        initial_pose = Circle((x[0], y[0]), radius, facecolor=color, fill=True, alpha=transparent, edgecolor='black',
                              linewidth=0.5)
        initial_heading = FancyArrow(x[0], y[0], np.cos(theta[0]) * scaling_factor,
                                     np.sin(theta[0]) * scaling_factor,
                                     color='black', width=0.1, alpha=transparent, head_length=0.5)

        final_pose = Circle((x[-1], y[-1]), radius, facecolor=color, fill=True, alpha=1, edgecolor='black',
                            linewidth=0.5)
        final_heading = FancyArrow(x[-1], y[-1], np.cos(theta[-1] * np.pi / 180) * scaling_factor,
                                   np.sin(theta[-1] * np.pi / 180) * scaling_factor,
                                   color='black', width=0.1, alpha=1, head_length=0.5)

        # Specify the range and number of ticks for x and y axes
        x_ticks = np.linspace(start=-6, stop=7, num=14)  # 11 ticks from -5 to 5
        y_ticks = np.linspace(start=-6, stop=7, num=14)  # 11 ticks from -5 to 5

        ax.add_patch(initial_pose)
        ax.add_patch(initial_heading)
        ax.add_patch(final_pose)
        ax.add_patch(final_heading)

        # Set the ticks on the x and y axes
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        ax.grid(True)
        ax.set_xlabel("x [m]", fontsize=font_size)
        ax.set_ylabel("y [m]", fontsize=font_size)
        ax.set_xlim([-7, 7.5])
        ax.set_ylim([-7, 7])
    plt.legend(loc="upper center", fontsize=font_size - 8, handles=legend_handles, framealpha=0.2)

    # Save the figure to the PDF
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0)

    plt.close('all')
    pdf.close()


if __name__ == '__main__':
    trajectories = np.array(
        [
            np.load(f"{THIS_DIR}x=5.000 y=-5.000 theta=2.094.npy"),
            np.load(f"{THIS_DIR}x=5.000 y=-5.000 theta=-2.094.npy"),
            np.load(f"{THIS_DIR}x=6.000 y=0.001 theta=1.571.npy"),
            np.load(f"{THIS_DIR}x=-5.000 y=5.000 theta=0.785.npy"),
            np.load(f"{THIS_DIR}x=-5.000 y=-5.000 theta=1.047.npy"),
        ]
    )
    create_pose_stabilization_plot(trajectories)
