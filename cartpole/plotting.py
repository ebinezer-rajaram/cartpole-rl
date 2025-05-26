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
    
    
def plot_predicted_vs_true_deltas(Y_true, Y_pred, labels, out_dir):
    for j in range(4):
        plt.figure()
        plt.scatter(Y_true[:, j], Y_pred[:, j], alpha=0.5)
        plt.plot([Y_true[:, j].min(), Y_true[:, j].max()],
                 [Y_true[:, j].min(), Y_true[:, j].max()], 'k--')
        plt.xlabel(f"True Δ{labels[j]}")
        plt.ylabel(f"Pred Δ{labels[j]}")
        plt.title(f"Δ{labels[j]}: Predicted vs True")
        plt.grid(True)

        fname = f"predicted_vs_true_delta_{labels[j]}"
        plt.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"))
        plt.savefig(os.path.join(out_dir["png"], f"{fname}.png"))
        plt.close()

def plot_deltas_vs_inputs(X, Y_true, Y_pred, labels, out_dir):
    for i in range(4):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axs = axs.flatten()
        for j in range(4):
            axs[j].plot(X[:, i], Y_true[:, j], 'o', label='true', alpha=0.4)
            axs[j].plot(X[:, i], Y_pred[:, j], '.', label='pred', alpha=0.4)
            axs[j].set_ylabel(f"Δ{labels[j]}")
            axs[j].set_title(f"Δ{labels[j]} vs {labels[i]}")
            axs[j].legend()
            axs[j].grid(True)
        axs[-1].set_xlabel(labels[i])
        fig.suptitle(f"All Δs vs {labels[i]} (true vs predicted)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f"combined_deltas_vs_{labels[i]}"
        plt.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"))
        plt.savefig(os.path.join(out_dir["png"], f"{fname}.png"))
        plt.close()
        
def plot_all_deltas_vs_inputs(X, Y_true, Y_pred, labels, out_dir):
    d = X.shape[1]
    for i in range(d):
        input_label = labels[i] if i < 4 else "action"
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axs = axs.flatten()

        for j in range(4):
            axs[j].plot(X[:, i], Y_true[:, j], 'o', label="true", alpha=0.5)
            axs[j].plot(X[:, i], Y_pred[:, j], '.', label="pred", alpha=0.5)
            axs[j].set_ylabel(f"Δ{labels[j]}")
            axs[j].set_title(f"Δ{labels[j]} vs {input_label}")
            axs[j].grid(True)
            axs[j].legend()

        axs[-1].set_xlabel(input_label)
        fig.suptitle(f"All Δs vs {input_label} (true vs predicted)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        fname = f"combined_deltas_vs_{input_label}_true_vs_pred"
        fig.savefig(os.path.join(out_dir["pdf"], f"{fname}.pdf"))
        fig.savefig(os.path.join(out_dir["png"], f"{fname}.png"))
        plt.show()
        plt.close(fig)

