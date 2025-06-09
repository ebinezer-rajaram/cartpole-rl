import numpy as np
import matplotlib.pyplot as plt
from cartpole.simulation import rollout
from cartpole.plotting import _make_output_dirs

# Set global plotting parameters for better readability in reports
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.titlesize': 16
})

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
    state_labels = ["x", "x_dot", "theta", "theta_dot"]
    
    # Store all 1D scan results for combined plot
    all_scans = []

    # 1D scan for each p_i
    for idx in range(4):
        scan_vals, losses = scan_1d(idx=idx, p0=p0, scan_range=(-10, 10), num=100, x0=x0, T=T, sigma_l=sigma_l)
        all_scans.append((scan_vals, losses))
        
        plt.figure(figsize=(8, 6))
        plt.plot(scan_vals, losses, linewidth=2.0)
        plt.xlabel(f"{labels[idx]} (state: {state_labels[idx]})")
        plt.ylabel("Trajectory Loss")
        plt.title(f"Loss landscape for policy coefficient {labels[idx]}")
        plt.grid(True)
        plt.tight_layout()
        
        # Find and mark the minimum point
        min_idx = np.argmin(losses)
        min_val = scan_vals[min_idx]
        min_loss = losses[min_idx]
        plt.plot(min_val, min_loss, 'ro', markersize=8)
        plt.annotate(f'Min: ({min_val:.2f}, {min_loss:.2f})', 
                     xy=(min_val, min_loss),
                     xytext=(min_val+1, min_loss),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        fname = f"loss_scan_1d_{labels[idx].replace('[', '').replace(']', '')}"
        plt.savefig(f"{out_dir['pdf']}/{fname}.pdf", bbox_inches='tight')
        plt.savefig(f"{out_dir['png']}/{fname}.png", bbox_inches='tight')
        plt.close()

    # Combined view of all 1D scans
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'green', 'red', 'purple']
    
    for idx, (scan_vals, losses) in enumerate(all_scans):
        plt.plot(scan_vals, losses, color=colors[idx], linewidth=2.0, 
                 label=f"{labels[idx]} ({state_labels[idx]})")
        
        # Mark minimum for each curve
        min_idx = np.argmin(losses)
        min_val = scan_vals[min_idx]
        min_loss = losses[min_idx]
        plt.plot(min_val, min_loss, 'o', color=colors[idx], markersize=8)
        
    plt.xlabel("Policy Coefficient Value")
    plt.ylabel("Trajectory Loss")
    plt.title("Loss landscape comparison across all policy coefficients")
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    
    fname = "loss_scan_1d_combined"
    plt.savefig(f"{out_dir['pdf']}/{fname}.pdf", bbox_inches='tight')
    plt.savefig(f"{out_dir['png']}/{fname}.png", bbox_inches='tight')
    plt.close()

    # 2D scan for selected pairs with improved visualization
    pairs = [(0, 1), (2, 3)]
    pair_results = []
    
    for i, j in pairs:
        grid_i, grid_j, losses_2d = scan_2d(i=i, j=j, p0=p0, 
                                           scan_range_i=(-10, 10), 
                                           scan_range_j=(-10, 10), 
                                           num=40, x0=x0, T=T, sigma_l=sigma_l)
        pair_results.append((grid_i, grid_j, losses_2d))
        
        plt.figure(figsize=(9, 7))
        cp = plt.contourf(grid_i, grid_j, losses_2d, levels=40, cmap='viridis')
        plt.xlabel(f"{labels[i]} ({state_labels[i]})")
        plt.ylabel(f"{labels[j]} ({state_labels[j]})")
        plt.title(f"Loss landscape for policy coefficients {labels[i]} vs {labels[j]}")
        
        # Add a colorbar with better formatting
        cbar = plt.colorbar(cp)
        cbar.set_label("Trajectory Loss", rotation=270, labelpad=15)
        
        # Find and mark the minimum loss point
        min_idx = np.unravel_index(np.argmin(losses_2d), losses_2d.shape)
        min_i_val = grid_i[min_idx]
        min_j_val = grid_j[min_idx]
        min_loss = losses_2d[min_idx]
        
        plt.plot(min_i_val, min_j_val, 'r*', markersize=12)
        plt.annotate(f'Min: ({min_i_val:.2f}, {min_j_val:.2f})\nLoss: {min_loss:.2f}', 
                     xy=(min_i_val, min_j_val),
                     xytext=(min_i_val+2, min_j_val+2),
                     arrowprops=dict(facecolor='white', shrink=0.05, width=1.5),
                     color='white',
                     fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))
        
        # Add contour lines for better readability of levels
        contour_lines = plt.contour(grid_i, grid_j, losses_2d, levels=10, colors='white', alpha=0.5, linewidths=0.8)
        plt.clabel(contour_lines, inline=True, fontsize=10)
        
        plt.tight_layout()
        fname = f"loss_scan_2d_{labels[i].replace('[', '').replace(']', '')}_{labels[j].replace('[', '').replace(']', '')}"
        plt.savefig(f"{out_dir['pdf']}/{fname}.pdf", bbox_inches='tight')
        plt.savefig(f"{out_dir['png']}/{fname}.png", bbox_inches='tight')
        plt.close()
    
    # Combined 2D plots side by side
    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    
    for idx, ((grid_i, grid_j, losses_2d), (i, j)) in enumerate(zip(pair_results, pairs)):
        cp = axs[idx].contourf(grid_i, grid_j, losses_2d, levels=40, cmap='viridis')
        axs[idx].set_xlabel(f"{labels[i]} ({state_labels[i]})")
        axs[idx].set_ylabel(f"{labels[j]} ({state_labels[j]})")
        axs[idx].set_title(f"Loss landscape for {labels[i]} vs {labels[j]}")
        
        # Add contour lines
        contour_lines = axs[idx].contour(grid_i, grid_j, losses_2d, levels=10, colors='white', alpha=0.5, linewidths=0.8)
        axs[idx].clabel(contour_lines, inline=True, fontsize=10)
        
        # Mark minimum point
        min_idx = np.unravel_index(np.argmin(losses_2d), losses_2d.shape)
        min_i_val = grid_i[min_idx]
        min_j_val = grid_j[min_idx]
        min_loss = losses_2d[min_idx]
        
        axs[idx].plot(min_i_val, min_j_val, 'r*', markersize=12)
        axs[idx].annotate(f'Min: ({min_i_val:.2f}, {min_j_val:.2f})\nLoss: {min_loss:.2f}', 
                         xy=(min_i_val, min_j_val),
                         xytext=(min_i_val+2, min_j_val+2),
                         arrowprops=dict(facecolor='white', shrink=0.05, width=1.5),
                         color='white', 
                         fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))
        
        fig.colorbar(cp, ax=axs[idx], label="Trajectory Loss")
    
    plt.suptitle("Loss Landscapes for Policy Parameter Pairs", fontsize=18)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    fname = "loss_scan_2d_combined"
    plt.savefig(f"{out_dir['pdf']}/{fname}.pdf", bbox_inches='tight')
    plt.savefig(f"{out_dir['png']}/{fname}.png", bbox_inches='tight')
    plt.close()

    print("Loss scan figures saved to:", out_dir)

if __name__ == "__main__":
    main()
