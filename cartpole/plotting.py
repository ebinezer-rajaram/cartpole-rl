import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os

def _make_output_dirs(base_dir):
    dirs = {
        "png": os.path.join(base_dir, "png"),
        "pdf": os.path.join(base_dir, "pdf")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def plot_time_series(traj, name, base_dir):
    dirs = _make_output_dirs(base_dir)
    t = np.arange(len(traj))
    labels = ["x", "x_dot", "theta", "theta_dot"]
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for i in range(4):
        axs[i].plot(t, traj[:, i])
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

    axs[-1].set_xlabel("Time step")
    fig.suptitle(f"Time Series - {name}")
    fig.tight_layout()
    plt.show()

    fig.savefig(os.path.join(dirs["png"], f"{name}_timeseries.png"))
    fig.savefig(os.path.join(dirs["pdf"], f"{name}_timeseries.pdf"))
    plt.close(fig)

def plot_phase_portraits(traj, name, base_dir):
    dirs = _make_output_dirs(base_dir)
    pairs = [(0, 1), (2, 3)]
    labels = [("x", "x_dot"), ("theta", "theta_dot")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (i, j), (label_x, label_y) in zip(axs, pairs, labels):
        ax.plot(traj[:, i], traj[:, j])
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.grid(True)

    fig.suptitle(f"Phase Portraits - {name}")
    fig.tight_layout()
    plt.show()

    fig.savefig(os.path.join(dirs["png"], f"{name}_phase.png"))
    fig.savefig(os.path.join(dirs["pdf"], f"{name}_phase.pdf"))
    plt.close(fig)

def animate_cartpole(traj):
    x = traj[:, 0]
    theta = traj[:, 2]

    cart_width = 1.0
    cart_height = 0.2
    pole_length = 0.5

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1, 1.5)
    ax.set_aspect('equal')

    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, color='black')
    pole_line, = ax.plot([], [], lw=3, color='blue')
    ax.add_patch(cart_patch)

    def init():
        cart_patch.set_xy((x[0] - cart_width / 2, 0))
        pole_x = x[0] + pole_length * np.sin(theta[0])
        pole_y = cart_height + pole_length * np.cos(theta[0])
        pole_line.set_data([x[0], pole_x], [cart_height, pole_y])
        return cart_patch, pole_line

    def animate(i):
        cart_patch.set_xy((x[i] - cart_width / 2, 0))
        pole_x = x[i] + pole_length * np.sin(theta[i])
        pole_y = cart_height + pole_length * np.cos(theta[i])
        pole_line.set_data([x[i], pole_x], [cart_height, pole_y])
        return cart_patch, pole_line

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(x), interval=50, blit=True
    )

    plt.show()
