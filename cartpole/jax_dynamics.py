import jax.numpy as jnp
import jax

def cartpole_dynamics(state, action, params=None):
    if params is None:
        params = dict(
            pole_length=0.5,
            pole_mass=0.5,
            cart_mass=0.5,
            mu_c=0.001,
            mu_p=0.001,
            gravity=9.8,
            sim_steps=50,
            delta_time=0.1,
            max_force=10.0,
        )
    x, x_dot, theta, theta_dot = state

    force = params["max_force"] * jnp.tanh(action / params["max_force"])

    dt = params["delta_time"] / params["sim_steps"]
    for _ in range(params["sim_steps"]):
        s = jnp.sin(theta)
        c = jnp.cos(theta)
        m = 4.0 * (params["cart_mass"] + params["pole_mass"]) - 3.0 * params["pole_mass"] * (c ** 2)
        cart_accel = (
            2.0 * (
                params["pole_length"] * params["pole_mass"] * (theta_dot ** 2) * s +
                2.0 * (force - params["mu_c"] * x_dot)
            )
            - 3.0 * params["pole_mass"] * params["gravity"] * c * s +
            6.0 * params["mu_p"] * theta_dot * c / params["pole_length"]
        ) / m
        pole_accel = (
            -3.0 * c * (2.0 / params["pole_length"]) * (
                params["pole_length"] / 2.0 * params["pole_mass"] * (theta_dot ** 2) * s +
                force - params["mu_c"] * x_dot
            )
            + 6.0 * (params["cart_mass"] + params["pole_mass"]) / (params["pole_mass"] * params["pole_length"]) * (
                params["pole_mass"] * params["gravity"] * s -
                2.0 / params["pole_length"] * params["mu_p"] * theta_dot
            )
        ) / m
        x_dot = x_dot + dt * cart_accel
        theta_dot = theta_dot + dt * pole_accel
        theta = theta + dt * theta_dot
        x = x + dt * x_dot

    theta = ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    return jnp.array([x, x_dot, theta, theta_dot])

def rollout_jax(x0, policy, T, params):
    def step(state, _):
        action = jnp.dot(policy, state)
        next_state = cartpole_dynamics(state, action, params)
        return next_state, next_state
    _, traj = jax.lax.scan(step, x0, None, length=T)
    return traj
