<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/42b4ccc5-6d13-4260-9851-fbab32ccd9cb" />

# Introductioin 

Autonomous drifting is a challenging benchmark for advanced vehicle control because it intentionally operates the vehicle beyond the conventional stable-handling region. Unlike typical autonomous-driving controllers, which aim to limit tire slip and preserve lateral stability within near-linear tire-force regimes, drifting requires the deliberate generation and regulation of large sideslip angles while maintaining path tracking, yaw-rate stability, and actuator feasibility. Thus, drifting is relevant not only for aggressive maneuvering but also as a stress test for safety-critical control under emergency conditions, where vehicles may operate near or beyond tire saturation. Understanding and controlling vehicle motion in these regimes can improve robustness during extreme maneuvers, low-friction driving, obstacle avoidance, and loss-of-control recovery [1].

From a control-theoretic perspective, drifting is challenging because the vehicle dynamics are strongly nonlinear, practically underactuated, and highly sensitive to tire–road interactions. At large sideslip angles, tire forces become nonlinear and coupled through friction limits, load transfer, and saturation. So small changes in steering, throttle, braking, or road friction can cause large variations in yaw moment and sideslip dynamics. Moreover, steady-state drift conditions may correspond to unstable, saddle-type equilibria, requiring continuous feedback stabilization rather than simple reference tracking [2]. This fundamentally distinguishes autonomous drifting from conventional path-following control, which typically assumes operation near stable nominal motion.


Within this research direction, nonlinear model predictive control is appealing because it can integrate nonlinear vehicle dynamics, state and input constraints, actuator limits, path-tracking objectives, and stability-related penalties within a single receding-horizon optimization framework. However, this expressiveness comes with substantial computational cost, as NMPC must solve a nonlinear optimal control problem at each sampling instant with sufficient speed and accuracy for closed-loop operation. This creates a key challenge for autonomous drifting: the controller must be both dynamically expressive and numerically reliable in a highly sensitive regime.

Consequently, autonomous drifting is  a spectacular aggressive-driving maneuver. And, also it is a rigorous benchmark for nonlinear predictive control. A successful controller must stabilize an intrinsically unstable high-sideslip motion, manage nonlinear tire-force saturation, resolve competing path-tracking and drift-stabilization objectives, and do so under real-time computational constraints. These properties make drifting an ideal testbed for evaluating NMPC algorithms, solver warm-start strategies. Real time embedded optimal-control software/solver such as acados implementats with fast multiple-shooting NMPC with sensitivities and RTI-class solvers for embedded execution and focuses on efficient SQP/RTI implementations for general NMPC.This work studies a narrower question: whether a drifting NMPC solve can be reorganized as a compiled differentiable program that supports batched multi-start exploration and low-latency Gauss–Newton real-time iteration in JAX. 


Therefore, this works contributes on a solver organization for autonomous drifting NMPC. Specifically, the project formulates the drifting NMPC solve as a compiled differentiable program in JAX, combining batched candidate generation with a Gauss–Newton real-time iteration update. This organization makes the compile/runtime trade-off explicit, which is often hidden in classical external-solver workflows. It further exploits JAX transformations such as : `jit` for compiling fused numerical kernels, `vmap` for batched candidate evaluation, `scan` for compact fixed-horizon rollout, and checkpointing for memory–computation trade-offs during reverse-mode differentiation.


# Vehicle Model 

| Symbol | Description | Unit |
|---|---|---|
| `x` | State vector | — |
| `u` | Control input vector | — |
| `X` | Global x-position of the vehicle center of mass | m |
| `Y` | Global y-position of the vehicle center of mass | m |
| `\dot{X}` | Global x-direction velocity | m/s |
| `\dot{Y}` | Global y-direction velocity | m/s |
| `\psi` | Yaw angle of the vehicle measured in the inertial frame | rad |
| `\dot{\psi}` | Time derivative of yaw angle | rad/s |
| `v_x` | Longitudinal velocity in the vehicle body-fixed frame | m/s |
| `v_y` | Lateral velocity in the vehicle body-fixed frame | m/s |
| `\dot{v}_x` | Longitudinal acceleration in the body-fixed frame | m/s² |
| `\dot{v}_y` | Lateral acceleration in the body-fixed frame | m/s² |
| `r` | Yaw rate of the vehicle | rad/s |
| `\dot{r}` | Yaw acceleration | rad/s² |
| `\delta` | Front steering angle | rad |
| `\dot{\delta}` | Actual steering rate | rad/s |
| `\dot{\delta}_{cmd}` | Commanded steering rate | rad/s |
| `\dot{\delta}_{max}` | Maximum allowable steering rate | rad/s |
| `F_{x,r}` | Rear longitudinal tire force | N |
| `F_{y,f}` | Front lateral tire force | N |
| `F_{y,r}` | Rear lateral tire force | N |
| `F_{\mathrm{drag}}(v_x)` | Longitudinal drag force as a function of forward speed | N |
| `m` | Vehicle mass | kg |
| `I_z` | Yaw moment of inertia about the vertical axis | kg m² |
| `\ell_f` | Distance from vehicle center of mass to front axle | m |
| `\ell_r` | Distance from vehicle center of mass to rear axle | m |
| `M_z` | Net yaw moment about the vehicle center of mass | N m |
| `R(\psi)` | Rotation matrix from body frame to inertial frame | — |
| `v_b` | Body-frame velocity vector | m/s |
| `\alpha_f` | Front tire slip angle | rad |
| `\alpha_r` | Rear tire slip angle | rad |
| `C_f` | Front tire cornering stiffness | N/rad |
| `C_r` | Rear tire cornering stiffness | N/rad |
| `f_f(\alpha_f)` | Front tire lateral-force model | N |
| `f_r(\alpha_r)` | Rear tire lateral-force model | N |
| `\rho` | Air density | kg/m³ |
| `C_d` | Aerodynamic drag coefficient | — |
| `A_f` | Vehicle frontal area | m² |
| `C_{rr}` | Rolling resistance coefficient | — |
| `g` | Gravitational acceleration | m/s² |
| `\mathrm{sat}(\cdot)` | Steering-rate saturation function | — |




The continuous-time equations of motion for a nonlinear seven-state dynamic single-track vehicle model are described as below.The model describes planar vehicle motion using global position, yaw orientation, body-frame translational velocities, yaw rate, and front steering angle.

The system is written in nonlinear state-space form as

```math
\dot{x} = f(x,u)
```

where `x` is the state vector and `u` is the control input vector.


The state vector is defined as

```math
x =
\begin{bmatrix}
X & Y & \psi & v_x & v_y & r & \delta
\end{bmatrix}^{T}
```
The control input vector is

```math
u =
\begin{bmatrix}
\dot{\delta}_{cmd} & F_{x,r}
\end{bmatrix}^{T}
```

The continuous-time dynamics are given by

```math
\dot{X}
=
v_x \cos \psi
-
v_y \sin \psi
```

```math
\dot{Y}
=
v_x \sin \psi
+
v_y \cos \psi
```

```math
\dot{\psi}
=
r
```

```math
\dot{v}_x
=
\frac{
F_{x,r}
-
F_{y,f}\sin \delta
-
F_{\mathrm{drag}}(v_x)
}{m}
+
r v_y
```

```math
\dot{v}_y
=
\frac{
F_{y,f}\cos \delta
+
F_{y,r}
}{m}
-
r v_x
```

```math
\dot{r}
=
\frac{
\ell_f F_{y,f}\cos \delta
-
\ell_r F_{y,r}
}{I_z}
```

```math
\dot{\delta}
=
\mathrm{sat}
\left(
\dot{\delta}_{cmd}
\right)
```

The complete nonlinear state-space model can be written as

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
=
\begin{bmatrix}
v_x \cos \psi - v_y \sin \psi \\

v_x \sin \psi + v_y \cos \psi \\

r \\

\dfrac{
F_{x,r}
-
F_{y,f}\sin \delta
-
F_{\mathrm{drag}}(v_x)
}{m}
+
r v_y \\

\dfrac{
F_{y,f}\cos \delta
+
F_{y,r}
}{m}
-
r v_x \\

\dfrac{
\ell_f F_{y,f}\cos \delta
-
\ell_r F_{y,r}
}{I_z} \\

\mathrm{sat}
\left(
\dot{\delta}_{cmd}
\right)
\end{bmatrix}
```


The velocity components `v_x` and `v_y` are expressed in the vehicle body-fixed frame.

The body-frame velocity vector is

```math
v_b
=
\begin{bmatrix}
v_x \\
v_y
\end{bmatrix}
```

The rotation matrix from the vehicle body frame to the global inertial frame is

```math
R(\psi)
=
\begin{bmatrix}
\cos \psi & -\sin \psi \\
\sin \psi & \cos \psi
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
\dot{X}
=
v_x \cos \psi
-
v_y \sin \psi
```

```math
\dot{Y}
=
v_x \sin \psi
+
v_y \cos \psi
```

---

The longitudinal body-frame dynamics are

```math
\dot{v}_x
=
\frac{
F_{x,r}
-
F_{y,f}\sin \delta
-
F_{\mathrm{drag}}(v_x)
}{m}
+
r v_y
```

The term `F_{x,r}` is the rear longitudinal force generated by propulsion or braking.

The term `-F_{y,f}\sin\delta` is the longitudinal projection of the front lateral tire force due to the steering angle.

The term `-F_{\mathrm{drag}}(v_x)` represents aerodynamic drag and other speed-dependent resistive forces.

The term `r v_y` is a body-frame coupling term caused by the rotating vehicle coordinate frame.


The lateral body-frame dynamics are

```math
\dot{v}_y
=
\frac{
F_{y,f}\cos \delta
+
F_{y,r}
}{m}
-
r v_x
```

The term `F_{y,f}\cos\delta` is the lateral projection of the front tire force.

The term `F_{y,r}` is the rear lateral tire force.

The term `-r v_x` is the lateral inertial coupling term caused by expressing the translational dynamics in the rotating body-fixed frame.


The yaw dynamics are obtained from the rotational equation of motion about the vertical axis through the vehicle center of mass:

```math
I_z \dot{r}
=
M_z
```

where the yaw moment is

```math
M_z
=
\ell_f F_{y,f}\cos \delta
-
\ell_r F_{y,r}
```

Thus,

```math
\dot{r}
=
\frac{
\ell_f F_{y,f}\cos \delta
-
\ell_r F_{y,r}
}{I_z}
```


The steering angle is treated as a dynamic state. Its evolution is governed by

```math
\dot{\delta}
=
\mathrm{sat}
\left(
\dot{\delta}_{cmd}
\right)
```

where `\dot{\delta}_{cmd}` is the commanded steering rate and `\mathrm{sat}(\cdot)` is a saturation function that enforces the physical steering-rate limit.

A symmetric steering-rate saturation model is

```math
\mathrm{sat}
\left(
\dot{\delta}_{cmd}
\right)
=
\begin{cases}
\dot{\delta}_{max},
&
\dot{\delta}_{cmd}
>
\dot{\delta}_{max}
\\

\dot{\delta}_{cmd},
&
-\dot{\delta}_{max}
\leq
\dot{\delta}_{cmd}
\leq
\dot{\delta}_{max}
\\

-\dot{\delta}_{max},
&
\dot{\delta}_{cmd}
<
-\dot{\delta}_{max}
\end{cases}
```

---

The lateral tire forces are nonlinear functions of the front and rear slip angles:

```math
F_{y,f}
=
f_f(\alpha_f)
```

```math
F_{y,r}
=
f_r(\alpha_r)
```

A common linear tire approximation is

```math
F_{y,f}
=
C_f \alpha_f
```

```math
F_{y,r}
=
C_r \alpha_r
```

---
The front and rear slip angles may be defined as

```math
\alpha_f
=
\delta
-
\tan^{-1}
\left(
\frac{
v_y + \ell_f r
}{
v_x
}
\right)
```

```math
\alpha_r
=
-
\tan^{-1}
\left(
\frac{
v_y - \ell_r r
}{
v_x
}
\right)
```

The front slip angle depends on the steering angle `\delta`, the lateral velocity `v_y`, and the yaw-rate contribution `\ell_f r`.

The rear slip angle depends on the lateral velocity `v_y` and the yaw-rate contribution `-\ell_r r`.

---

The longitudinal resistive force is written as

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

If rolling resistance is included, the drag model may be written as

```math
F_{\mathrm{drag}}(v_x)
=
\frac{1}{2}
\rho C_d A_f v_x^2
+
C_{rr} m g
```

---






# References 

1] Meijer, S., Bertipaglia, A., & Shyrokau, B. (2024, July). A nonlinear model predictive control for automated drifting with a standard passenger vehicle. In 2024 IEEE International Conference on Advanced Intelligent Mechatronics (AIM) (pp. 284-289). IEEE.

2] Hindiyeh, Rami Y., and J. Christian Gerdes. "A controller framework for autonomous drifting: Design, stability, and experimental validation." Journal of Dynamic Systems, Measurement, and Control 136.5 (2014): 051015.












