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

The front and rear slip angles are

```math id="phxu0q"
\alpha_f =
\delta -
\arctan2(v_y + l_f r, v_x)
```

```math id="pyls2b"
\alpha_r =
-\arctan2(v_y - l_r r, v_x)
```

The lateral tire forces use a simplified Pacejka-type saturation model:

```math id="p6zmdu"
F_{y,f}
=
D_f
\sin
\left(
C_f \arctan(B_f \alpha_f)
\right)
```

```math id="rbyzll"
F_{y,r}
=
D_r
\sin
\left(
C_r \arctan(B_r \alpha_r)
\right)
```

The front tire capacity is

```math id="c3yqot"
D_f = \mu F_{z,f}
```

The rear lateral capacity is reduced by rear longitudinal force:

```math id="3hhtff"
D_r =
10^{-4}
\operatorname{softplus}
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


## Algorithm 1: Vectorized JAX NMPC Solver

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
---
---

# References 

1] Meijer, S., Bertipaglia, A., & Shyrokau, B. (2024, July). A nonlinear model predictive control for automated drifting with a standard passenger vehicle. In 2024 IEEE International Conference on Advanced Intelligent Mechatronics (AIM) (pp. 284-289). IEEE.

2] Hindiyeh, Rami Y., and J. Christian Gerdes. "A controller framework for autonomous drifting: Design, stability, and experimental validation." Journal of Dynamic Systems, Measurement, and Control 136.5 (2014): 051015.












