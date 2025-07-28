# CartPole Dynamics and Control — SF3 Machine Learning Project

## Overview

This project explores the modeling and control of a classic mechanical system: the inverted pendulum on a cart, commonly referred to as the **CartPole**. The goal is to understand, simulate, model, and control this nonlinear dynamical system using data-driven techniques rooted in **machine learning**.

The project is part of the SF3 Machine Learning course at the University of Cambridge and is designed to blend theory from control systems, reinforcement learning, and statistical regression with practical implementation in Python.

---

## System Description

The CartPole system consists of a cart that can move along a horizontal axis and a pendulum attached to the cart that can swing freely. The objective is to balance the pendulum in its upright, unstable equilibrium position by applying appropriate horizontal forces to the cart.

The system is described by four continuous state variables:

- `x`: cart position
- `ẋ`: cart velocity
- `θ`: pole angle (measured from the upright position)
- `θ̇`: angular velocity of the pole

The dynamics are nonlinear and affected by gravity, friction, and input force. Importantly, the angle θ is treated as a periodic variable.

---

## Project Phases

### **Week 1: Simulation & Linear Modeling**
- Simulate the system's free dynamics using numerical integration.
- Collect data from rollouts.
- Analyze how the state evolves over time.
- Fit a linear model to predict state transitions using regression.
- Evaluate model accuracy and simulate future states by iterative prediction.

### **Week 2: Nonlinear Modeling**
- Replace linear models with nonlinear kernel-based regression using Gaussian kernels.
- Handle angular periodicity in the kernel function.
- Introduce regularization to address ill-conditioning.
- Evaluate model generalization, and test how well the model captures complex dynamics over time.
- Later, optimize kernel hyperparameters using gradient-based methods with JAX for efficiency.

### **Week 3: Control**
- Develop linear control policies to stabilize the pole near the upright position.
- Define a trajectory loss function to measure how well the system stays near a desired target state.
- Optimize policy parameters to minimize this loss using direct policy search (a form of reinforcement learning).
- Extend control to use learned (model-predictive) dynamics rather than real dynamics for planning.

### **Week 4: Robustness & Sensitivity**
- Introduce observation and process noise.
- Assess the robustness of both learned models and controllers under noisy conditions.
- Refit models in noisy regimes and evaluate stability degradation.

---

## Learning Objectives

- Understand and simulate nonlinear physical systems using differential equations.
- Develop data-driven models for unknown dynamics using regression (linear and nonlinear).
- Implement and compare models based on Gaussian processes and kernel methods.
- Design control strategies that leverage learned dynamics to achieve robust performance.
- Use modern tools like JAX for efficient differentiation and optimization.

---

## Deliverables

- **Interim Report**: Focused on modeling up to linear dynamics (Week 1).
- **Final Report**: Full analysis, modeling, control experiments, and robustness evaluation (Weeks 1–4).
- All code used in simulation, modeling, and control.
- Visualizations including time series, phase portraits, error plots, and policy rollouts.

---

## Supervisors

- **Jose Miguel Hernandez Lobato**
- **Carl E. Rasmussen**

---

## Acknowledgements

This project is based on classical examples from control theory and leverages insights from Gaussian process modeling and reinforcement learning.

