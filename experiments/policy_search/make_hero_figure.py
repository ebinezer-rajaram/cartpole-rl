import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from cartpole.simulation import rollout
from experiments.policy_search.optimise_policy import objective
from experiments.policy_search.optimise_on_action_model import model_rollout_jax, trajectory_loss_jax

# Requires models/policy_search/model_kernel_optimized_sincos_action.npz
# (experiments.policy_search.dataset, then experiments.policy_search.action_model).

PARAMS = dict(
    pole_length=0.5,
    pole_mass=0.5,
    cart_mass=0.5,
    mu_c=0.001,
    mu_p=0.001,
    gravity=9.8,
    sim_steps=50,
    delta_time=0.1,
    max_force=20.0,
)
SIGMA_L = jnp.array([0.5, 0.4, 0.05, 0.25])
X0 = jnp.array([0.0, 0.0, 0.1, 0.0])
T_LOSS = 20       # loss window on the simulator, as in optimise_policy
T_MODEL = 5       # optimisation horizon inside the learned model, as in optimise_on_action_model
T_PLOT = 40       # 4 s at delta_time = 0.1
MAX_FORCE = 20.0

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#808080"
SHADE = "#f3f2ee"
GRID = "#e4e3df"
MUTED = "#52514e"

def lbfgs(loss_fn):
    obj_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    return minimize(
        lambda p: float(obj_and_grad(p)[0]),
        np.zeros(4),
        method="L-BFGS-B",
        jac=lambda p: np.array(obj_and_grad(p)[1]),
        options={"maxiter": 100}
    )

def simulate(policy, T):
    """Replay a linear policy on the reference simulator; returns (T+1, 4) including x0."""
    if policy is None:
        action_fn = None
    else:
        def action_fn(t, state):
            return np.clip(np.dot(policy, state), -MAX_FORCE, MAX_FORCE)
    traj = rollout(np.array(X0), T=T, action_fn=action_fn, remap=True)
    return np.vstack([np.array(X0), traj])

def simulator_loss(traj):
    # Same loss as optimise_policy: sum over the T_LOSS states after x0
    return float(trajectory_loss_jax(jnp.array(traj[1:T_LOSS + 1]), X0=jnp.zeros(4), sigma_l=SIGMA_L))

def plot_panel(ax, t, series, idx, ylabel, ylim):
    ax.axvspan(0, T_LOSS * PARAMS["delta_time"], color=SHADE, zorder=0, lw=0)
    ax.text(0.33, ylim[0] + 0.04 * (ylim[1] - ylim[0]), f"loss window, T = {T_LOSS} steps",
            color=MUTED, fontsize=12, va="bottom")
    ax.axhline(0, color=GRID, lw=2, zorder=1)
    for traj, style in series:
        ax.plot(t, traj[:, idx], **style)
    ax.set_xlim(0, T_PLOT * PARAMS["delta_time"])
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

def main():
    start = time.time()

    res_true = lbfgs(lambda p: objective(p, X0, T_LOSS, PARAMS, SIGMA_L))

    model = np.load("models/policy_search/model_kernel_optimized_sincos_action.npz")
    X_basis = jnp.array(model["X_basis"])
    alpha = jnp.array(model["alpha"])
    lengthscales = jnp.array(model["lengthscales"])
    res_model = lbfgs(lambda p: trajectory_loss_jax(
        model_rollout_jax(X0, p, T_MODEL, X_basis, alpha, lengthscales, max_force=MAX_FORCE),
        X0=jnp.zeros(4), sigma_l=SIGMA_L))

    traj_true = simulate(res_true.x, T_PLOT)
    traj_model = simulate(res_model.x, T_PLOT)
    traj_free = simulate(None, T_PLOT)

    loss_true = simulator_loss(traj_true)
    loss_model = simulator_loss(traj_model)
    # Largest |theta| once the first control step has been applied (t >= 0.1 s)
    max_theta_model = float(np.max(np.abs(traj_model[1:, 2])))

    print("Policy optimised on true dynamics:", res_true.x)
    print(f"  loss on simulator (T={T_LOSS}): {loss_true:.4f} / {T_LOSS}")
    print(f"Policy optimised on learned model (T={T_MODEL}):", res_model.x)
    print(f"  loss inside the model: {res_model.fun:.4f}")
    print(f"  loss replayed on simulator (T={T_LOSS}): {loss_model:.4f} / {T_LOSS}")
    print(f"  max |theta| for t >= 0.1 s over {T_PLOT} steps: {max_theta_model:.4f} rad")
    print(f"No control: |theta| at t = {T_PLOT * PARAMS['delta_time']:.1f} s: {abs(traj_free[-1, 2]):.4f} rad")

    t = np.arange(T_PLOT + 1) * PARAMS["delta_time"]
    theta_lim = (-0.06, 0.22)
    # Draw the uncontrolled pole only until it leaves the axes (it then wraps round through pi)
    exit_idx = int(np.argmax(traj_free[:, 2] > theta_lim[1]))
    traj_fall = traj_free.copy()
    traj_fall[exit_idx + 1:] = np.nan
    controlled = [
        (traj_true, dict(color=BLUE, lw=2.2, zorder=3)),
        (traj_model, dict(color=ORANGE, lw=2.2, zorder=3)),
    ]
    fall = (traj_fall, dict(color=GREY, lw=2.5, ls=(0, (4, 2)), zorder=2))

    with plt.rc_context({
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }):
        fig, (ax_theta, ax_x) = plt.subplots(1, 2, figsize=(16, 6.2))
        plot_panel(ax_theta, t, [fall] + controlled, 2, "Pole angle θ (rad)", theta_lim)
        plot_panel(ax_x, t, controlled, 0, "Cart position x (m)", (-0.35, 0.35))

        ax_theta.annotate(
            "no control: pole falls (θ → π)",
            xy=(t[exit_idx - 1] + 0.02, 0.205), xytext=(0.45, 0.19),
            color=MUTED, fontsize=12,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1),
        )

        handles = [
            plt.Line2D([], [], color=BLUE, lw=2.2,
                       label=f"Policy optimised on true dynamics (loss {loss_true:.2f} / {T_LOSS})"),
            plt.Line2D([], [], color=ORANGE, lw=2.2,
                       label=f"Policy optimised on learned model, replayed on simulator (loss {loss_model:.2f} / {T_LOSS})"),
            plt.Line2D([], [], color=GREY, lw=2.5, ls=(0, (4, 2)), label="No control"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
        fig.suptitle("Stabilising the cart-pole from a tilted start (θ₀ = 0.1 rad)",
                     x=0.06, ha="left", fontsize=17)
        fig.subplots_adjust(left=0.077, right=0.975, top=0.826, bottom=0.21, wspace=0.2)

        os.makedirs("assets", exist_ok=True)
        fig.savefig("assets/policy_stabilisation.png", dpi=100)
        plt.close(fig)

    print("Saved assets/policy_stabilisation.png")
    print(f"Done in {time.time() - start:.1f} s")

if __name__ == "__main__":
    main()
