import numpy as np
import matplotlib.pyplot as plt
from cartpole.simulation import rollout
from cartpole.plotting import _make_output_dirs

def trajectory_loss(traj, X0=None, sigma_l=0.5):
    if X0 is None:
        X0 = np.zeros(4)
    sigma_l = np.asarray(sigma_l)
    diffs = (traj - X0) / sigma_l
    sq_norms = np.sum(diffs**2, axis=-1)
    losses = 1 - np.exp(-0.5 * sq_norms)
    return np.sum(losses), losses

def linear_policy(p):
    return lambda t, state: np.dot(p, state)

def scan_1d(idx=0, p0=None, scan_range=(-10, 10), num=100, x0=None, T=50, sigma_l=0.5):
    if p0 is None:
        p0 = np.zeros(4)
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.1, 0.0])
    scan_vals = np.linspace(*scan_range, num)
    losses = []
    for val in scan_vals:
        p = p0.copy()
        p[idx] = val
        traj = rollout(x0, T=T, action_fn=linear_policy(p), remap=True)
        total_loss, _ = trajectory_loss(traj, sigma_l=sigma_l)
        losses.append(total_loss)
    return scan_vals, np.array(losses)

def scan_2d(i=0, j=1, p0=None, scan_range_i=(-10, 10), scan_range_j=(-10, 10), num=30, x0=None, T=50, sigma_l=0.5):
    if p0 is None:
        p0 = np.zeros(4)
    if x0 is None:
        x0 = np.array([0.0, 0.0, 0.1, 0.0])
    vals_i = np.linspace(*scan_range_i, num)
    vals_j = np.linspace(*scan_range_j, num)
    grid_i, grid_j = np.meshgrid(vals_i, vals_j)
    losses = np.zeros_like(grid_i)
    for idx_i in range(num):
        for idx_j in range(num):
            p = p0.copy()
            p[i] = grid_i[idx_i, idx_j]
            p[j] = grid_j[idx_i, idx_j]
            traj = rollout(x0, T=T, action_fn=linear_policy(p), remap=True)
            total_loss, _ = trajectory_loss(traj, sigma_l=sigma_l)
            losses[idx_i, idx_j] = total_loss
    return grid_i, grid_j, losses

def main():
    out_dir = _make_output_dirs("figures/task_3.2/scan")
    x0 = np.array([0.0, 0.0, 0.1, 0.0])  # initial condition
    T = 50
    sigma_l = 0.5
    p0 = np.zeros(4)
    labels = ["p[0]", "p[1]", "p[2]", "p[3]"]

    # 1D scan for each p_i
    for idx in range(4):
        scan_vals, losses = scan_1d(idx=idx, p0=p0, scan_range=(-10, 10), num=100, x0=x0, T=T, sigma_l=sigma_l)
        plt.figure()
        plt.plot(scan_vals, losses)
        plt.xlabel(labels[idx])
        plt.ylabel("Trajectory Loss")
        plt.title(f"1D Loss Scan: {labels[idx]}")
        plt.grid(True)
        fname = f"loss_scan_1d_{labels[idx]}"
        plt.savefig(f"{out_dir['pdf']}/{fname}.pdf")
        plt.savefig(f"{out_dir['png']}/{fname}.png")
        plt.close()

    # 2D scan for selected pairs
    pairs = [(0, 1), (2, 3)]
    for i, j in pairs:
        grid_i, grid_j, losses_2d = scan_2d(i=i, j=j, p0=p0, scan_range_i=(-10, 10), scan_range_j=(-10, 10), num=40, x0=x0, T=T, sigma_l=sigma_l)
        plt.figure(figsize=(7, 5))
        cp = plt.contourf(grid_i, grid_j, losses_2d, levels=40)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.title(f"2D Loss Scan: {labels[i]} vs {labels[j]}")
        plt.colorbar(cp, label="Trajectory Loss")
        fname = f"loss_scan_2d_{labels[i]}_{labels[j]}"
        plt.savefig(f"{out_dir['pdf']}/{fname}.pdf")
        plt.savefig(f"{out_dir['png']}/{fname}.png")
        plt.close()

    print("Loss scan figures saved to:", out_dir)

if __name__ == "__main__":
    main()
