<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/42b4ccc5-6d13-4260-9851-fbab32ccd9cb" />

# Introductioin 

Autonomous drifting is a challenging benchmark for advanced vehicle control because it intentionally operates the vehicle beyond the conventional stable-handling region. Unlike typical autonomous-driving controllers, which aim to limit tire slip and preserve lateral stability within near-linear tire-force regimes, drifting requires the deliberate generation and regulation of large sideslip angles while maintaining path tracking, yaw-rate stability, and actuator feasibility. Thus, drifting is relevant not only for aggressive maneuvering but also as a stress test for safety-critical control under emergency conditions, where vehicles may operate near or beyond tire saturation. Understanding and controlling vehicle motion in these regimes can improve robustness during extreme maneuvers, low-friction driving, obstacle avoidance, and loss-of-control recovery [1].

From a control-theoretic perspective, drifting is challenging because the vehicle dynamics are strongly nonlinear, practically underactuated, and highly sensitive to tire–road interactions. At large sideslip angles, tire forces become nonlinear and coupled through friction limits, load transfer, and saturation. So small changes in steering, throttle, braking, or road friction can cause large variations in yaw moment and sideslip dynamics. Moreover, steady-state drift conditions may correspond to unstable, saddle-type equilibria, requiring continuous feedback stabilization rather than simple reference tracking [2]. This fundamentally distinguishes autonomous drifting from conventional path-following control, which typically assumes operation near stable nominal motion.


Within this research direction, nonlinear model predictive control is appealing because it can integrate nonlinear vehicle dynamics, state and input constraints, actuator limits, path-tracking objectives, and stability-related penalties within a single receding-horizon optimization framework. However, this expressiveness comes with substantial computational cost, as NMPC must solve a nonlinear optimal control problem at each sampling instant with sufficient speed and accuracy for closed-loop operation. This creates a key challenge for autonomous drifting: the controller must be both dynamically expressive and numerically reliable in a highly sensitive regime.

Consequently, autonomous drifting is  a spectacular aggressive-driving maneuver. And, also it is a rigorous benchmark for nonlinear predictive control. A successful controller must stabilize an intrinsically unstable high-sideslip motion, manage nonlinear tire-force saturation, resolve competing path-tracking and drift-stabilization objectives, and do so under real-time computational constraints. These properties make drifting an ideal testbed for evaluating NMPC algorithms, solver warm-start strategies. Real time embedded optimal-control software/solver such as acados implementats with fast multiple-shooting NMPC with sensitivities and RTI-class solvers for embedded execution and focuses on efficient SQP/RTI implementations for general NMPC.This work studies a narrower question: whether a drifting NMPC solve can be reorganized as a compiled differentiable program that supports batched multi-start exploration and low-latency Gauss–Newton real-time iteration in JAX. 


Therefore, this works contributes on a solver organization for autonomous drifting NMPC. Specifically, the project formulates the drifting NMPC solve as a compiled differentiable program in JAX, combining batched candidate generation with a Gauss–Newton real-time iteration update. This organization makes the compile/runtime trade-off explicit, which is often hidden in classical external-solver workflows. It further exploits JAX transformations such as : `jit` for compiling fused numerical kernels, `vmap` for batched candidate evaluation, `scan` for compact fixed-horizon rollout, and checkpointing for memory–computation trade-offs during reverse-mode differentiation.


# Vechile Model 


The reference implementation uses a seven-state dynamic single-track vehicle model:

```math
x =
\begin{bmatrix}
X & Y & \psi & v_x & v_y & r & \delta
\end{bmatrix}^{T},
\qquad
and control input vector is 
u =
\begin{bmatrix}
\dot{\delta} & F_{x,r}
\end{bmatrix}^{T}

````

where:

| Symbol | Meaning | Unit |
|---|---|---|
| `X` | Global x-position of the vehicle center of mass | `m` |
| `Y` | Global y-position of the vehicle center of mass | `m` |
| `\psi` | Yaw angle / heading angle | `rad` |
| `v_x` | Longitudinal velocity in the body frame | `m/s` |
| `v_y` | Lateral velocity in the body frame | `m/s` |
| `r` | Yaw rate | `rad/s` |
| `\delta` | Front steering angle | `rad` |

and for control input: 
where:

| Symbol | Meaning | Unit |
|---|---|---|
| `\dot{\delta}_{cmd}` | Commanded steering rate | `rad/s` |
| `F_{x,r}` | Rear longitudinal tire force | `N` |

---

## Continuous-Time Dynamics

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
F_{x,r} - F_{y,f}\sin\delta - F_{\mathrm{drag}}(v_x)
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
\ell_f F_{y,f}\cos\delta - \ell_r F_{y,r}
}{I_z}
```

```math
\dot{\delta} =
\operatorname{sat}(\dot{\delta}_{cmd})
```

---

## Compact State-Space Form

The nonlinear system can be written as

```math
\dot{x} = f(x,u)
```

where

```math
\dot{x}
=
\begin{bmatrix}
\dot{X} \\
\dot{Y} \\
\dot{\psi} \\
\dot{v}_x \\
\dot{v}_y \\
\dot{r} \\
\dot{\delta}
\end{bmatrix}
```

and

```math
f(x,u)
=
\begin{bmatrix}
v_x \cos\psi - v_y \sin\psi \\

v_x \sin\psi + v_y \cos\psi \\

r \\

\dfrac{
F_{x,r} - F_{y,f}\sin\delta - F_{\mathrm{drag}}(v_x)
}{m}
+ r v_y \\

\dfrac{
F_{y,f}\cos\delta + F_{y,r}
}{m}
- r v_x \\

\dfrac{
\ell_f F_{y,f}\cos\delta - \ell_r F_{y,r}
}{I_z} \\

\operatorname{sat}(\dot{\delta}_{cmd})
\end{bmatrix}
```

---

| Symbol | Meaning | Unit |
|---|---|---|
| `x` | State vector | — |
| `u` | Control input vector | — |
| `X` | Global x-position | `m` |
| `Y` | Global y-position | `m` |
| `\psi` | Yaw angle | `rad` |
| `v_x` | Longitudinal body-frame velocity | `m/s` |
| `v_y` | Lateral body-frame velocity | `m/s` |
| `r` | Yaw rate | `rad/s` |
| `\delta` | Front steering angle | `rad` |
| `\dot{\delta}_{cmd}` | Commanded steering rate | `rad/s` |
| `F_{x,r}` | Rear longitudinal tire force | `N` |
| `F_{y,f}` | Front lateral tire force | `N` |
| `F_{y,r}` | Rear lateral tire force | `N` |
| `F_{\mathrm{drag}}(v_x)` | Drag force as a function of longitudinal velocity | `N` |
| `m` | Vehicle mass | `kg` |
| `I_z` | Yaw moment of inertia | `kg m^2` |
| `\ell_f` | Distance from center of mass to front axle | `m` |
| `\ell_r` | Distance from center of mass to rear axle | `m` |
| `\operatorname{sat}(\cdot)` | Saturation function | — |

---

## Coordinate Transformation

The body-frame velocity vector is

```math
v_b =
\begin{bmatrix}
v_x \\
v_y
\end{bmatrix}
```

The global-frame velocity is obtained using the yaw rotation matrix

```math
R(\psi)
=
\begin{bmatrix}
\cos\psi & -\sin\psi \\
\sin\psi & \cos\psi
\end{bmatrix}
```

Therefore,

```math
\begin{bmatrix}
\dot{X} \\
\dot{Y}
\end{bmatrix}
=
R(\psi)
\begin{bmatrix}
v_x \\
v_y
\end{bmatrix}
```

which gives

```math
\dot{X} = v_x \cos\psi - v_y \sin\psi
```

```math
\dot{Y} = v_x \sin\psi + v_y \cos\psi
```

---

## Longitudinal Dynamics

The longitudinal acceleration equation is

```math
\dot{v}_x =
\frac{
F_{x,r} - F_{y,f}\sin\delta - F_{\mathrm{drag}}(v_x)
}{m}
+ r v_y
```

The terms have the following physical meanings:

| Term | Meaning |
|---|---|
| `F_{x,r}` | Driving or braking force at the rear tire |
| `-F_{y,f}\sin\delta` | Longitudinal projection of the front lateral tire force |
| `-F_{\mathrm{drag}}(v_x)` | Resistive drag force |
| `r v_y` | Coupling term due to the rotating body frame |
| `m` | Vehicle mass |

---

## Lateral Dynamics

The lateral acceleration equation is

```math
\dot{v}_y =
\frac{
F_{y,f}\cos\delta + F_{y,r}
}{m}
- r v_x
```

The terms have the following physical meanings:

| Term | Meaning |
|---|---|
| `F_{y,f}\cos\delta` | Lateral projection of the front tire force |
| `F_{y,r}` | Rear lateral tire force |
| `-r v_x` | Coupling term due to the rotating body frame |
| `m` | Vehicle mass |

---

## Yaw Dynamics

The yaw acceleration equation is

```math
\dot{r} =
\frac{
\ell_f F_{y,f}\cos\delta - \ell_r F_{y,r}
}{I_z}
```

This equation follows from the rotational equation of motion

```math
I_z \dot{r} = M_z
```

where the net yaw moment is

```math
M_z =
\ell_f F_{y,f}\cos\delta - \ell_r F_{y,r}
```

The terms have the following physical meanings:

| Term | Meaning |
|---|---|
| `\ell_f F_{y,f}\cos\delta` | Yaw moment generated by the front lateral tire force |
| `-\ell_r F_{y,r}` | Yaw moment generated by the rear lateral tire force |
| `I_z` | Yaw moment of inertia |

---

## Steering Dynamics

The steering angle evolves according to

```math
\dot{\delta} =
\operatorname{sat}(\dot{\delta}_{cmd})
```

where `sat` limits the commanded steering rate to the actuator limits.

A typical saturation function is

```math
\operatorname{sat}(\dot{\delta}_{cmd})
=
\begin{cases}
\dot{\delta}_{max}, &
\dot{\delta}_{cmd} > \dot{\delta}_{max}
\\

\dot{\delta}_{cmd}, &
-\dot{\delta}_{max}
\leq
\dot{\delta}_{cmd}
\leq
\dot{\delta}_{max}
\\

-\dot{\delta}_{max}, &
\dot{\delta}_{cmd} < -\dot{\delta}_{max}
\end{cases}
```

where `\dot{\delta}_{max}` is the maximum physically achievable steering rate.

---

## Tire Force Model

The lateral tire forces are typically modeled as functions of the front and rear slip angles:

```math
F_{y,f} = f_f(\alpha_f)
```

```math
F_{y,r} = f_r(\alpha_r)
```

where:

| Symbol | Meaning |
|---|---|
| `\alpha_f` | Front tire slip angle |
| `\alpha_r` | Rear tire slip angle |
| `f_f(\cdot)` | Front tire lateral-force model |
| `f_r(\cdot)` | Rear tire lateral-force model |

For a linear tire model,

```math
F_{y,f} = C_f \alpha_f
```

```math
F_{y,r} = C_r \alpha_r
```

where:

| Symbol | Meaning | Unit |
|---|---|---|
| `C_f` | Front cornering stiffness | `N/rad` |
| `C_r` | Rear cornering stiffness | `N/rad` |

Common slip-angle definitions are

```math
\alpha_f =
\delta -
\tan^{-1}
\left(
\frac{v_y + \ell_f r}{v_x}
\right)
```

```math
\alpha_r =
-
\tan^{-1}
\left(
\frac{v_y - \ell_r r}{v_x}
\right)
```

---

## Drag Force Model

The drag force is represented as

```math
F_{\mathrm{drag}}(v_x)
```

A common aerodynamic drag model is

```math
F_{\mathrm{drag}}(v_x)
=
\frac{1}{2}
\rho C_d A_f v_x^2
```

where:

| Symbol | Meaning | Unit |
|---|---|---|
| `\rho` | Air density | `kg/m^3` |
| `C_d` | Aerodynamic drag coefficient | — |
| `A_f` | Frontal area of the vehicle | `m^2` |

Rolling resistance may also be included:

```math
F_{\mathrm{drag}}(v_x)
=
\frac{1}{2}
\rho C_d A_f v_x^2
+
C_{rr}mg
```

where:

| Symbol | Meaning | Unit |
|---|---|---|
| `C_{rr}` | Rolling resistance coefficient | — |
| `g` | Gravitational acceleration | `m/s^2` |

---

## Model Properties

The model is nonlinear because it contains:

```math
\sin\psi,\quad
\cos\psi,\quad
\sin\delta,\quad
\cos\delta
```

and nonlinear force functions such as

```math
F_{y,f} = f_f(\alpha_f)
```

```math
F_{y,r} = f_r(\alpha_r)
```

```math
F_{\mathrm{drag}} = F_{\mathrm{drag}}(v_x)
```

The complete model combines:

- planar rigid-body kinematics
- body-frame translational dynamics
- yaw rotational dynamics
- tire-road interaction
- steering actuator saturation
- aerodynamic and rolling resistance


## 

The seven-state dynamic single-track vehicle model is a nonlinear system of the form

```math
\dot{x} = f(x,u)
```

with

```math
x =
\begin{bmatrix}
X & Y & \psi & v_x & v_y & r & \delta
\end{bmatrix}^{T}
```

and

```math
u =
\begin{bmatrix}
\dot{\delta}_{cmd} & F_{x,r}
\end{bmatrix}^{T}
```



# References 

1] Meijer, S., Bertipaglia, A., & Shyrokau, B. (2024, July). A nonlinear model predictive control for automated drifting with a standard passenger vehicle. In 2024 IEEE International Conference on Advanced Intelligent Mechatronics (AIM) (pp. 284-289). IEEE.

2] Hindiyeh, Rami Y., and J. Christian Gerdes. "A controller framework for autonomous drifting: Design, stability, and experimental validation." Journal of Dynamic Systems, Measurement, and Control 136.5 (2014): 051015.












