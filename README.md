# Autonomous Drifting Control: JAX-based NMPC 
   ## Preliminary Feasibility study and Demo
This project is a preliminary investigation and feasibility study. The goal is to test whether a JAX-native NMPC framework can be used effectively for autonomous drift tracking, and whether this approach is suitable for deeper research and future development.

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/42b4ccc5-6d13-4260-9851-fbab32ccd9cb" />

# Introduction 

Autonomous drifting is a nonlinear control problem in which a vehicle must intentionally operate near tire-force saturation while maintaining path tracking, sideslip regulation, yaw-rate regulation, and actuator feasibility. Unlike normal path-following control, drifting requires the vehicle to maintain a nonzero sideslip angle while following a reference trajectory.

This repository studies circular drifting using nonlinear model predictive control. At every control step, the solver predicts the vehicle trajectory over a finite horizon, optimizes a sequence of steering-rate and rear-force inputs, applies the first control input, and repeats the process.

The project focuses on three goals:

- Build a fully differentiable NMPC solver in JAX.
- Compare several JAX-based solver variants under the same drifting benchmark.
- Compare the JAX solver family with ACADOS using closed-loop tracking and timing metrics.

# Main Features
- Dynamic single-track vehicle model with seven states.
- Steering-rate and rear longitudinal-force inputs.
- Simplified nonlinear Pacejka-type lateral tire model.
- Rear lateral tire capacity coupled with rear longitudinal force.
- Fourth-order Runge–Kutta integration.
- Circular drift reference generation.
- Feasibility search for drift equilibrium.
- Condensed single-shooting NMPC formulation.
- JAX automatic differentiation for gradients and Jacobians.
- JAX jit compilation for fixed-structure solver kernels.
- JAX vmap for vectorized multi-candidate control evaluation.
- Gauss–Newton, matrix-free, first-order, Optax, and multiple-shooting solver variants.
- ACADOS SQP-RTI and ACADOS SQP comparison.
- Closed-loop benchmark with tracking, robustness, and timing metrics.
  

## Vehicle Model

<img width="946" height="781" alt="Screenshot From 2026-06-10 01-46-16" src="https://github.com/user-attachments/assets/9814c5b4-c5cb-4d7b-8224-441b389240f9" />


The vehicle state is

```math
x =
\begin{bmatrix}
X & Y & \psi & v_x & v_y & r & \delta
\end{bmatrix}^T
```

where:

| Symbol   | Description             |
| -------- | ----------------------- |
| `X, Y`   | Global vehicle position |
| `\psi`   | Yaw angle               |
| `v_x`    | Longitudinal velocity   |
| `v_y`    | Lateral velocity        |
| `r`      | Yaw rate                |
| `\delta` | Steering angle          |

The control input is

```math
u =
\begin{bmatrix}
\dot{\delta}_{cmd} & F_{x,r}
\end{bmatrix}^T
```

where:

| Symbol               | Description             |
| -------------------- | ----------------------- |
| `\dot{\delta}_{cmd}` | Steering-rate command   |
| `F_{x,r}`            | Rear longitudinal force |

### Continuous-Time Dynamics

The continuous-time dynamics are

```math
\dot{X} = v_x \cos\psi - v_y \sin\psi
```

```math
\dot{Y} = v_x \sin\psi + v_y \cos\psi
```

```math
\dot{\psi} = r
```

```math
\dot{v}_x =
\frac{
F_{x,r} - F_{y,f}\sin\delta - F_{drag}(v_x)
}{m}
+ r v_y
```

```math
\dot{v}_y =
\frac{
F_{y,f}\cos\delta + F_{y,r}
}{m}
- r v_x
```

```math
\dot{r} =
\frac{
l_f F_{y,f}\cos\delta - l_r F_{y,r}
}{I_z}
```

```math
\dot{\delta} =
\mathrm{sat}(\dot{\delta}_{cmd})
```

### Drag Force

The drag force is

```math
F_{drag}(v_x) = c_2 v_x^2 + c_1 v_x
```

### Discrete-Time Model

The model is discretized using fourth-order Runge--Kutta integration:

```math
x_{k+1} = f_d(x_k, u_k, \mu_k)
```

## Tire Model

<img width="759" height="728" alt="Screenshot From 2026-06-10 02-05-04" src="https://github.com/user-attachments/assets/5c52e0db-c0dc-45ac-95ac-00bc76d9c007" />


The front and rear slip angles are

```math
\alpha_f =
\delta -
\arctan2(v_y + l_f r, v_x)
```

```math
\alpha_r =
-\arctan2(v_y - l_r r, v_x)
```

The lateral tire forces use a simplified Pacejka-type saturation model:

```math
F_{y,f}
=
D_f
\sin
\left(
C_f \arctan(B_f \alpha_f)
\right)
```

```math
F_{y,r}
=
D_r
\sin
\left(
C_r \arctan(B_r \alpha_r)
\right)
```

The front tire capacity is

```math
D_f = \mu F_{z,f}
```
The rear lateral capacity is reduced by rear longitudinal force:

```math
D_r =
10^{-4}
\mathrm{softplus}
\left(
10^{-4}
\left[
(\mu F_{z,r})^2 - F_{x,r}^2
\right]
\right)
+
10^{-6}
```

This smooth expression is used so that the tire-capacity model remains compatible with automatic differentiation.

## Vehicle and Solver Parameters

| Parameter              |               Symbol |         Value |
| ---------------------- | -------------------: | ------------: |
| Mass                   |                  `m` |     `1800 kg` |
| Yaw inertia            |                `I_z` | `2800 kg m^2` |
| Front axle distance    |                `l_f` |       `1.2 m` |
| Rear axle distance     |                `l_r` |       `1.6 m` |
| Gravity                |                  `g` |  `9.81 m/s^2` |
| Friction coefficient   |                `\mu` |        `0.95` |
| Maximum steering angle |       `\delta_{max}` |     `0.6 rad` |
| Maximum steering rate  | `\dot{\delta}_{max}` |   `1.5 rad/s` |
| Rear force minimum     |      `F_{x,r}^{min}` |     `-4000 N` |
| Rear force maximum     |      `F_{x,r}^{max}` |      `5000 N` |
| State dimension        |                `n_x` |           `7` |
| Control dimension      |                `n_u` |           `2` |
| Horizon length         |                  `N` |          `20` |
| Sampling time          |           `\Delta t` |      `0.08 s` |
| Prediction horizon     |      `T = N\Delta t` |       `1.6 s` |


## Normalized Control Variables

The solver uses normalized control variables internally. This is important because the two control inputs have very different physical units and numerical scales. Steering rate is measured in `rad/s` and has magnitude on the order of `1`, while rear longitudinal force is measured in newtons and has magnitude on the order of thousands. Directly optimizing these two variables without scaling can make the optimization problem poorly conditioned.

Let the normalized decision variable at stage `k` be

```math
z_k =
\begin{bmatrix}
z_{\dot{\delta},k} \\
z_{F,k}
\end{bmatrix}
```

The physical steering-rate command is mapped from the normalized variable using

```math
\dot{\delta}_k =
\dot{\delta}_{max} z_{\dot{\delta},k}
```

If

```math
z_{\dot{\delta},k} \in [-1, 1]
```

then the physical steering-rate command automatically satisfies

```math
-\dot{\delta}_{max}
\leq
\dot{\delta}_k
\leq
\dot{\delta}_{max}
```

The rear longitudinal force is mapped using a midpoint and half-range scaling. Let

```math
F_{mid}
=
\frac{
F_{x,r}^{max} + F_{x,r}^{min}
}{2}
```

and

```math
F_{half}
=
\frac{
F_{x,r}^{max} - F_{x,r}^{min}
}{2}
```

Then

```math
F_{x,r,k}
=
F_{mid}
+
F_{half} z_{F,k}
```

If

```math
z_{F,k} \in [-1, 1]
```

then

```math
F_{x,r}^{min}
\leq
F_{x,r,k}
\leq
F_{x,r}^{max}
```

Using the notebook bounds,

```math
F_{x,r}^{min} = -4000 \ \mathrm{N}
```

```math
F_{x,r}^{max} = 5000 \ \mathrm{N}
```

Therefore,

```math
F_{mid}
=
\frac{
5000 + (-4000)
}{2}
=
500 \ \mathrm{N}
```

and

```math
F_{half}
=
\frac{
5000 - (-4000)
}{2}
=
4500 \ \mathrm{N}
```

Thus,

```math
F_{x,r,k}
=
500 + 4500 z_{F,k}
```

This normalization improves numerical conditioning because both decision variables are dimensionless and generally lie in the same range. It also makes trust-region limits and gradient-based updates more meaningful because the update vector is not dominated by the force variable purely due to units.

The full normalized decision sequence is

```math
Z =
\{z_0, z_1, \dots, z_{N-1}\}
```

The physical control sequence is obtained through the mapping

```math
U = T(Z)
```

where

```math
u_k =
T(z_k)
=
\begin{bmatrix}
\dot{\delta}_{max} z_{\dot{\delta},k} \\
F_{mid} + F_{half} z_{F,k}
\end{bmatrix}
```

The condensed optimization problem can therefore be written in terms of normalized variables:

```math
\min_Z
\frac{1}{2}
\left\|
r(Z)
\right\|_2^2
```

The rollout is

```math
x_{k+1}
=
f_d
\left(
x_k,
T(z_k),
\mu_k
\right)
```

After each update, the normalized variables can be clipped to remain in the feasible range:

```math
z_k
\leftarrow
\mathrm{clip}
\left(
z_k,
-1,
1
\right)
```

This guarantees that the physical control inputs remain within the defined actuator bounds.





## Drift Reference

The benchmark uses circular drifting.

| Quantity                   |            Value |
| -------------------------- | ---------------: |
| Circle radius              |           `18 m` |
| Reference speed            |          `8 m/s` |
| Desired sideslip           |       `0.35 rad` |
| Selected sideslip          |       `0.08 rad` |
| Reference yaw rate         | `0.444444 rad/s` |
| Equilibrium steering angle |   `0.183127 rad` |
| Rear longitudinal force    |      `553.948 N` |
| Equilibrium residual norm  |       `1.605758` |

The reference yaw rate is

```math
r_{ref} = \frac{V}{R}
```

Using `V = 8 m/s` and `R = 18 m`,

```math
r_{ref} = \frac{8}{18} = 0.444444 \ \mathrm{rad/s}
```

Although the desired sideslip is `0.35 rad`, the feasibility search selects `0.08 rad`.

Higher sideslip candidates require saturated steering and rear longitudinal force and produce larger equilibrium residuals.


## NMPC Objective

The controller solves a finite-horizon nonlinear least-squares problem:

```math
\min_Z \ \frac{1}{2} \left\| r(Z) \right\|_2^2
```

where `Z` is the normalized control sequence and `r(Z)` is the residual vector.

The residual includes:

* Position tracking error
* Heading error
* Velocity error
* Sideslip error
* Yaw-rate error
* Steering error
* Control effort
* Control smoothness
* Radial corridor penalty
* Sideslip envelope penalty
* Yaw-rate envelope penalty
* Speed-limit penalty
* Terminal tracking penalty

The predicted trajectory is generated by

```math
x_{k+1} = f_d(x_k, u_k, \mu_k)
```

for

```math
k = 0, \dots, N-1
```

The objective can be written as

```math
J(Z)
=
\frac{1}{2}
\sum_{k=0}^{N-1}
\left\| r_k(Z) \right\|_2^2
+
\frac{1}{2}
\left\| r_N(Z) \right\|_2^2
```

### 

## NMPC Formulation

The controller is formulated as a finite-horizon nonlinear model predictive control problem. At each time step, the controller receives the current vehicle state, predicts future motion over a horizon of `N = 20` steps, optimizes a sequence of control inputs, applies the first input, and repeats the process at the next time step.

Let the current state be

```math
x_0
```

The predicted state sequence is

```math
X =
\{x_0, x_1, \dots, x_N\}
```

and the control sequence is

```math
U =
\{u_0, u_1, \dots, u_{N-1}\}
```

The discrete-time dynamics are

```math
x_{k+1}
=
f_d(x_k, u_k, \mu_k),
\qquad
k = 0, \dots, N-1
```

The general NMPC problem can be written as

```math
\min_U \ J(X, U)
```

subject to

```math
x_{k+1}
=
f_d(x_k, u_k, \mu_k)
```

```math
u_{min}
\leq
u_k
\leq
u_{max}
```

```math
\delta_{min}
\leq
\delta_k
\leq
\delta_{max}
```

```math
v_x^{min}
\leq
v_{x,k}
\leq
v_x^{max}
```

---

## Condensed Single-Shooting Formulation

In the condensed single-shooting formulation, the optimizer does not treat the states as independent decision variables. Instead, only the control sequence is optimized.

The state trajectory is generated by forward rollout from the initial state:

```math
x_1 =
f_d(x_0, u_0, \mu_0)
```

```math
x_2 =
f_d(x_1, u_1, \mu_1)
```

and generally,

```math
x_{k+1}
=
f_d(x_k, u_k, \mu_k)
```

Thus, the cost becomes a function of the control sequence only:

```math
J(U)
=
J(X(U), U)
```

 The NMPC cost as a nonlinear least-squares objective. The residual vector is

```math
r(U)
=
\begin{bmatrix}
r_0(U) \\
r_1(U) \\
\vdots \\
r_N(U)
\end{bmatrix}
```

and the objective is

```math
J(U)
=
\frac{1}{2}
r(U)^T r(U)
```

Equivalently,

```math
J(U)
=
\frac{1}{2}
\left\| r(U) \right\|_2^2
```

This least-squares structure is important because it allows Gauss--Newton methods to be applied. Instead of solving a general nonlinear program using arbitrary second-order derivatives, the solver approximates the Hessian using the residual Jacobian.

---

## Stage Residuals

For each stage `k`, the residual includes several tracking and penalty terms.

### Position-Tracking Residual

```math
r_{p,k}
=
\sqrt{w_p}
\begin{bmatrix}
X_k - X_{ref,k} \\
Y_k - Y_{ref,k}
\end{bmatrix}
```

### Heading Residual

```math
r_{\psi,k}
=
\sqrt{w_{\psi}}
\mathrm{wrap}
\left(
\psi_k - \psi_{ref,k}
\right)
```

### Velocity Residual

```math
r_{v,k}
=
\begin{bmatrix}
\sqrt{w_{v_x}}
\left(
v_{x,k} - v_{x,ref,k}
\right) \\
\sqrt{w_{v_y}}
\left(
v_{y,k} - v_{y,ref,k}
\right)
\end{bmatrix}
```

### Sideslip Residual

```math
r_{\beta,k}
=
\sqrt{w_{\beta}}
\left(
\beta_k - \beta_{ref}
\right)
```

### Yaw-Rate Residual

```math
r_{r,k}
=
\sqrt{w_r}
\left(
r_k - r_{ref}
\right)
```

### Steering Residual

```math
r_{\delta,k}
=
\sqrt{w_{\delta}}
\left(
\delta_k - \delta_{eq}
\right)
```

### Control-Effort Residual

```math
r_{u,k}
=
\begin{bmatrix}
\sqrt{w_{\dot{\delta}}}
\left(
\dot{\delta}_k - \dot{\delta}_{ref}
\right) \\
\sqrt{w_F}
\left(
F_{x,r,k} - F_{x,r,eq}
\right)
\end{bmatrix}
```

Since

```math
\dot{\delta}_{ref} = 0
```

the first term penalizes steering-rate magnitude.

---

## Control-Smoothness Residual

The control-smoothness residual penalizes input increments:

```math
\Delta u_k
=
u_k - u_{k-1}
```

For `k > 0`,

```math
r_{\Delta u,k}
=
\begin{bmatrix}
\sqrt{w_{\Delta \dot{\delta}}}
\left(
\dot{\delta}_k - \dot{\delta}_{k-1}
\right) \\
\sqrt{w_{\Delta F}}
\left(
F_{x,r,k} - F_{x,r,k-1}
\right)
\end{bmatrix}
```

For the first input, the increment is computed relative to the previously applied control:

```math
\Delta u_0
=
u_0 - u_{prev}
```

This term helps produce smooth control commands and reduces abrupt steering or force changes.

---

## Radial Corridor Penalty

The radial error is

```math
e_{\rho,k}
=
\rho_k - R
```

where

```math
\rho_k
=
\sqrt{
X_k^2 + (Y_k - R)^2
}
```

The corridor violation is defined using a hinge function:

```math
h_{corr,k}
=
\max
\left(
0,
|e_{\rho,k}| - w_{corr}
\right)
```

The corresponding residual is

```math
r_{corr,k}
=
\sqrt{w_{corr,p}}
h_{corr,k}
```

In implementation, a smooth hinge may be used to preserve differentiability. A generic smooth hinge can be written as

```math
\mathrm{softplus}(a)
=
\log
\left(
1 + \exp(a)
\right)
```

or in scaled form,

```math
h_{\epsilon}(a)
=
\epsilon
\log
\left(
1 + \exp(a / \epsilon)
\right)
```

Then the smooth corridor violation becomes

```math
h_{corr,k}
=
h_{\epsilon}
\left(
|e_{\rho,k}| - w_{corr}
\right)
```

---

## Drift-State Envelope Penalties

The sideslip-envelope violation is defined as

```math
h_{\beta,k}
=
\max
\left(
0,
|\beta_k - \beta_{ref}| - w_{\beta,band}
\right)
```

The corresponding residual is

```math
r_{\beta,env,k}
=
\sqrt{w_{\beta,env}}
h_{\beta,k}
```

The yaw-rate-envelope violation is

```math
h_{r,k}
=
\max
\left(
0,
|r_k - r_{ref}| - w_{r,band}
\right)
```

The corresponding residual is

```math
r_{r,env,k}
=
\sqrt{w_{r,env}}
h_{r,k}
```

---

## Speed-Limit Penalties

The speed-limit penalties are based on the longitudinal speed bounds.

The lower-speed violation is

```math
h_{v,min,k}
=
\max
\left(
0,
v_x^{min} - v_{x,k}
\right)
```

and the upper-speed violation is

```math
h_{v,max,k}
=
\max
\left(
0,
v_{x,k} - v_x^{max}
\right)
```

The speed-limit residual is

```math
r_{v,lim,k}
=
\sqrt{w_{v,lim}}
\begin{bmatrix}
h_{v,min,k} \\
h_{v,max,k}
\end{bmatrix}
```

---

## Terminal Residual

The terminal residual penalizes the final predicted state at the end of the horizon. It is generally written as

```math
r_N
=
\sqrt{w_N}
\left(
x_N - x_{ref,N}
\right)
```

or, more specifically, using the same relevant components as the stage residual:

```math
r_N
=
\begin{bmatrix}
\sqrt{w_{p,N}}
\left(
X_N - X_{ref,N}
\right) \\
\sqrt{w_{p,N}}
\left(
Y_N - Y_{ref,N}
\right) \\
\sqrt{w_{\psi,N}}
\mathrm{wrap}
\left(
\psi_N - \psi_{ref,N}
\right) \\
\sqrt{w_{v_x,N}}
\left(
v_{x,N} - v_{x,ref,N}
\right) \\
\sqrt{w_{v_y,N}}
\left(
v_{y,N} - v_{y,ref,N}
\right) \\
\sqrt{w_{\beta,N}}
\left(
\beta_N - \beta_{ref}
\right) \\
\sqrt{w_{r,N}}
\left(
r_N - r_{ref}
\right)
\end{bmatrix}
```


## Proposed JAX Solver Architecture

The main contribution of this project is a JAX-native differentiable NMPC solver architecture.

The solver is designed around the following pipeline:

```text
Current state
     ↓
Shift previous solution
     ↓
Generate candidate control sequences
     ↓
Vectorized rollout and cost evaluation
     ↓
Select best candidate
     ↓
Apply fixed-iteration solver update
     ↓
Clip/control-normalize solution
     ↓
Apply first control input
     ↓
Warm-start next MPC step
```

The solver uses normalized controls so that steering rate and rear force have comparable numerical scales. This improves numerical conditioning because steering rate is measured in radians per second while rear force is measured in newtons.

The normalized control vector is

```math
z_k =
\begin{bmatrix}
z_{\dot{\delta},k} \\
z_{F,k}
\end{bmatrix}
```

The physical control is recovered using

```math
\dot{\delta}_k =
\dot{\delta}_{max} z_{\dot{\delta},k}
```

```math
F_{x,r,k}
=
F_{mid}
+
F_{half} z_{F,k}
```

where

```math
F_{mid}
=
\frac{
F_{x,r}^{max} + F_{x,r}^{min}
}{2}
```

and

```math
F_{half}
=
\frac{
F_{x,r}^{max} - F_{x,r}^{min}
}{2}
```

Using the force bounds,

```math
F_{mid} = 500 \ \mathrm{N}
```

```math
F_{half} = 4500 \ \mathrm{N}
```

so

```math
F_{x,r,k}
=
500 + 4500 z_{F,k}
```


## Algorithm : Vectorized JAX NMPC Solver

### Input

* Current state `x_0`
* Previous solution `Z_{prev}`
* Previous applied control `u_{prev}`
* Reference trajectory `X_{ref}`
* Friction sequence `\mu`
* Solver configuration

### Output

* First control input `u_0^\star`
* Optimized control sequence `Z^\star`

### Procedure

```text id="3ltdky"
1. Warm-start
   Shift the previous solution:
       Z_shift = shift(Z_prev)

2. Candidate generation
   Construct candidate control sequences:
       - Shifted candidate
       - Reference-control candidate
       - Blended candidates
       - Steering-pulse candidates
       - Rear-force perturbation candidate
       - Randomized perturbation candidate

3. Vectorized rollout
   For each candidate Z_i:
       simulate x_0 over the horizon using RK4
       compute residual vector r(Z_i)
       compute objective J_i = 0.5 ||r(Z_i)||^2

4. Candidate selection
   Select the candidate with the minimum objective:
       Z_0 = argmin_i J_i

5. Solver update
   Apply a fixed number of optimization iterations.

   Depending on the selected method, use:
       - Gauss-Newton update
       - Lineax Gauss-Newton update
       - Matrix-free Gauss-Newton update
       - First-order update
       - Optax update

6. Acceptance
   Accept the update only if the new objective improves.

7. Projection
   Clip normalized controls to feasible bounds.

8. Control extraction
   Convert z_0^* to physical control u_0^*.

9. Return
   Apply u_0^* and store Z^* for the next MPC step.
```

## Gauss--Newton Update

For the least-squares objective

```math id="r4ljf8"
J(Z)
=
\frac{1}{2}
\left\| r(Z) \right\|_2^2
```

the residual Jacobian is

```math id="4kk97w"
J_r(Z)
=
\frac{\partial r}{\partial Z}
```

The damped Gauss--Newton system is

```math id="hu0h22"
\left(
J_r^T J_r + \lambda I
\right)
\Delta Z
=
-
J_r^T r
```

The candidate update is

```math id="puuisp"
Z^+ = Z + \Delta Z
```

A trust-radius scaling is applied to limit overly large updates:

```math id="7x8url"
\Delta Z
\leftarrow
\Delta Z
\min
\left(
1,
\frac{
\Delta_{trust}
}{
\left\| \Delta Z \right\| + 10^{-9}
}
\right)
```

The update is accepted only if it improves the objective:

```math id="4o2hbr"
J(Z + \Delta Z) \leq J(Z)
```

This improves robustness when nonlinear tire saturation makes the local linearization unreliable.


## Solver Variants

| ID    | Method             | Description                                            |
| ----- | ------------------ | ------------------------------------------------------ |
| `M01` | JAX SS-GN          | Single-shooting Gauss--Newton                          |
| `M02` | JAX VM-GN          | Vectorized multi-candidate Gauss--Newton, 3 iterations |
| `M03` | JAX VM-GN          | Vectorized multi-candidate Gauss--Newton, 8 iterations |
| `M04` | JAX VM-GN + Lineax | Vectorized Gauss--Newton with Lineax solve             |
| `M05` | JAX Matrix-Free GN | Matrix-free Gauss--Newton using JVP/VJP                |
| `M06` | JAX First-Order    | Gradient-based first-order solver                      |
| `M07` | JAX Optax          | Optax-based gradient solver                            |
| `M08` | JAX MS-SQP         | JAX-native multiple-shooting SQP-style solver          |
| `M09` | JAX MS-Lineax      | Multiple-shooting solver with Lineax                   |
| `M10` | ACADOS SQP-RTI     | ACADOS real-time iteration                             |
| `M11` | ACADOS SQP         | ACADOS SQP with 3 iterations                           |


## Multiple-Shooting Formulation

The JAX multiple-shooting solver optimizes both states and controls.

The decision vector is

```math id="fikd59"
Y =
\begin{bmatrix}
x_1, \dots, x_N, z_0, \dots, z_{N-1}
\end{bmatrix}
```

The dynamic defect is

```math id="o4avul"
d_k =
x_{k+1}
-
f_d(x_k, u_k, \mu_k)
```

A perfectly dynamically feasible trajectory satisfies

```math id="z3jblt"
d_k = 0
```

The multiple-shooting objective includes weighted defect residuals:

```math id="ns3eaq"
r_{d,k}
=
\sqrt{w_d} d_k
```

and the full objective is

```math id="ingywe"
J_{MS}(Y)
=
\frac{1}{2}
\left\|
r_{MS}(Y)
\right\|_2^2
```

This formulation is closer to SQP-style optimal-control solvers such as ACADOS, but it has a larger decision vector than condensed single shooting.


## Benchmark Results

The benchmark uses a `40`-step closed-loop circular drift simulation.

### Tracking Performance

| Method          | Mean position error `[m]` | Max position error `[m]` | Mean radial error `[m]` | Max corridor violation `[m]` | Success fraction |
| --------------- | ------------------------: | -----------------------: | ----------------------: | ---------------------------: | ---------------: |
| JAX First-Order |                `0.019630` |               `0.053098` |              `0.005960` |                   `0.000000` |            `1.0` |
| JAX SS-GN       |                 `0.02198` |                `0.04901` |                     `—` |                   `0.000000` |            `1.0` |
| JAX VM-GN       |                  `0.0221` |                 `0.0498` |                     `—` |                   `0.000000` |            `1.0` |
| JAX MS-SQP      |                 `0.02256` |                `0.05246` |                     `—` |                   `0.000000` |            `1.0` |
| ACADOS SQP-RTI  |                 `1.01196` |                `3.06727` |               `0.98667` |                    `0.41935` |            `0.9` |
| ACADOS SQP      |                 `1.01196` |                `3.06727` |               `0.98667` |                    `0.41935` |            `0.9` |

The JAX methods achieve lower tracking error and zero corridor violation in this benchmark. ACADOS achieves much faster solve time but larger closed-loop tracking errors.

### Timing Performance

| Method             | Mean solve time `[ms]` | p95 solve time `[ms]` | Max solve time `[ms]` |
| ------------------ | ---------------------: | --------------------: | --------------------: |
| ACADOS SQP-RTI     |              `0.66819` |             `0.75412` |             `0.90134` |
| ACADOS SQP         |              `0.78972` |             `1.04535` |             `1.34253` |
| JAX First-Order    |             `43.79811` |            `18.90490` |          `1128.94064` |
| JAX SS-GN          |             `53.46737` |            `13.22111` |          `1716.46886` |
| JAX Matrix-Free GN |             `69.26162` |            `18.94278` |          `2161.54073` |
| JAX VM-GN, 3 iters |            `143.73028` |                   `—` |                   `—` |
| JAX VM-GN, 8 iters |                 `~288` |                   `—` |                   `—` |

ACADOS is significantly faster in raw online solve time. The JAX methods provide stronger closed-loop tracking in this benchmark but are slightly slower, especially when first-step timing spikes are included.



# Results

<img width="1894" height="1429" alt="image" src="https://github.com/user-attachments/assets/9e43e008-6aea-481a-bff6-4de2ec0a7741" />

Caption: Closed-loop circular drifting trajectories for all solvers. JAX methods remain close to the reference trajectory, while ACADOS trajectories show larger radial deviation in this benchmark.

<img width="1926" height="916" alt="image" src="https://github.com/user-attachments/assets/44beb038-eb7b-4f8b-aff5-9827cbe13e98" />

Caption: Per-step position tracking error over the 40-step benchmark. JAX methods maintain low tracking error, while ACADOS accumulates larger error.

<img width="1926" height="916" alt="image" src="https://github.com/user-attachments/assets/24bc2fc9-fd33-4184-8e65-e8df7c62c5b2" />

Caption: Radial error relative to the circular reference. JAX methods remain within the corridor, while ACADOS violates the corridor near the end of the run.

<img width="1904" height="954" alt="image" src="https://github.com/user-attachments/assets/f05f0985-c347-4d99-a018-c0f2c48332c5" />
Caption: Sideslip tracking during circular drifting. All methods remain inside the sideslip envelope; ACADOS has smaller mean sideslip error in this benchmark.

<img width="1904" height="954" alt="image" src="https://github.com/user-attachments/assets/ddb436bb-e9bf-4b32-be28-e7faddab3872" />
Yaw-rate tracking during circular drifting. JAX methods remain close to the reference yaw rate, while ACADOS shows larger yaw-rate error.

<img width="1904" height="954" alt="image" src="https://github.com/user-attachments/assets/5023fee9-a27d-4161-a4d8-a767cf16d89c" />
Per-step NMPC solve time. ACADOS is much faster, while JAX methods show higher solve time and first-step timing spikes.

<img width="1893" height="909" alt="image" src="https://github.com/user-attachments/assets/dff16174-4c62-4cc4-be65-fb9404282ffd" />
Per-step NMPC solve time. ACADOS is  faster, while JAX methods show higher solve time and first-step timing spikes.

<img width="1866" height="1214" alt="image" src="https://github.com/user-attachments/assets/3e513a65-21b0-4067-a8b6-753e9fca5cfb" />
Time–quality Pareto comparison. JAX methods achieve lower tracking error at higher computational cost, while ACADOS achieves lower solve time with larger tracking error.

<img width="1868" height="1319" alt="image" src="https://github.com/user-attachments/assets/a3f22b27-6d35-4060-978c-761a87240d30" />

The closed-loop sideslip–yaw-rate phase-plane figure shows the evolution of each controller in the (β,r) state space, where β is the vehicle sideslip angle and r is the yaw rate. This phase-plane representation is useful because drifting is not defined only by path position. A vehicle may remain near the geometric path while still having an incorrect drift state, or it may maintain sideslip while rotating too quickly or too slowly. Therefore, plotting sideslip and yaw rate together provides a direct assessment of whether the controller stabilizes the intended drift equilibrium.

In this figure, the vertical reference line corresponds to the desired sideslip reference,

```math
\beta_{ref} = 0.08 \ \mathrm{rad}
```
and the horizontal reference line corresponds to the desired yaw-rate reference,

```math
r_{ref} = 0.444444 \ \mathrm{rad/s}
```
The shaded or bounded region represents the permitted drift-state envelope. This envelope is defined by the sideslip tolerance and yaw-rate tolerance. A trajectory that remains inside this region is considered to maintain an acceptable drift state, whereas a trajectory outside the envelope indicates drift-state deviation.

The JAX-based methods remain close to the desired yaw-rate region and stay within the allowable sideslip--yaw-rate envelope. This means that the JAX controllers  track the circular path and also regulate the internal drift dynamics of the vehicle.

Their trajectories converge toward the neighborhood of the reference drift state rather than moving away from it. In particular, the yaw-rate behavior of the JAX methods is close to the reference value, which is important because yaw rate determines whether the vehicle rotates consistently with the circular path curvature.


---
## Summary 

### JAX vs ACADOS Benchmark Analysis

The  results demonstrate that the JAX-based solver architecture produces strong closed-loop drift-tracking performance on the circular benchmark:

*  Across the nine JAX methods,
* - the **mean position error** is approximately `0.02 m`.
* - **Maximum position error** stays below `~0.062 m`.
* - **Maximum corridor violation** is `0`.

These metrics indicate that JAX solvers maintain the vehicle close to the reference circular path while satisfying the radial corridor constraint.

### Key Observations

1. **JAX First-Order Method**

   * Achieves the strongest tracking performance.
   * Simpler than Gauss–Newton methods, using gradient-based updates.
   * Lowest mean position and radial errors.
   * Suggests that residual design, input normalization, warm start, and candidate selection are critical, more so than repeated Newton-type iterations.

2. **Gauss–Newton Methods**

   * Additional iterations beyond three do not significantly improve tracking.
   * Eight iterations increase computational cost without measurable closed-loop benefit.
   * Highlights that in receding-horizon control, “good enough” solutions quickly may outperform higher-accuracy but slower solutions.

3. **Matrix-Free Gauss–Newton**

   * Avoids explicit residual Jacobian formation using JVP/VJP.
   * Useful for larger-scale problems with memory constraints.
   * On this benchmark, not faster due to JVP/VJP runtime overhead.

4. **Multiple-Shooting Methods**

   * Introduce predicted states as decision variables with defect residuals.
   * Closer to standard SQP formulations (like ACADOS).
   * Computationally more expensive due to larger decision vectors.
   * For `N=20`, `n_x=7`, `n_u=2`, condensed shooting has 40 variables; multiple-shooting has ~180.

### Comparison with ACADOS

* **ACADOS**:

  * Significantly faster (`<1 ms` mean solve time for SQP-RTI and SQP with 3 iterations).
  * Larger tracking errors: mean position error ~1.01 m, max position error ~3.07 m.
  * Mean absolute radial error ~0.9867 m; max corridor violation 0.419 m.
  * Success fraction: 0.9.

* **JAX Methods**:

  * Slower (`>40 ms` mean solve time), with first-step timing spikes.
  * Mean position error ~0.02 m, zero corridor violation.
  * Success fraction: 1.0.

 ACADOS is faster but less accurate for this benchmark, while JAX provides higher closed-loop tracking accuracy and robustness. These results are specific to this project setup and do not imply general superiority of one framework.

### Insights

**JAX Advantages**:
  * Automatic differentiation simplifies implementing new residuals, objectives, and solver variants.
  * Vectorized candidate evaluation improves local initialization and convergence.
  * JIT compilation allows repeated numerical kernels for dynamics, residuals, and updates.
    
  **Considerations**:

  * First-step or compilation overhead can inflate apparent solve time.
  * Timing results should separate compilation, first-call, and steady-state solve times for real-time assessment.
  * Solver choice depends on application priorities:
    *  Strict real-time (<1 ms) → ACADOS.
    * Research, flexibility, and high accuracy → JAX.

This benchmark highlights the trade-offs between speed, accuracy, and flexibility for NMPC solvers in drift-tracking tasks.


# References 

1] Meijer, S., Bertipaglia, A., & Shyrokau, B. (2024, July). A nonlinear model predictive control for automated drifting with a standard passenger vehicle. In 2024 IEEE International Conference on Advanced Intelligent Mechatronics (AIM) (pp. 284-289). IEEE.

2] Hindiyeh, Rami Y., and J. Christian Gerdes. "A controller framework for autonomous drifting: Design, stability, and experimental validation." Journal of Dynamic Systems, Measurement, and Control 136.5 (2014): 051015.












