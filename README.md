<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/42b4ccc5-6d13-4260-9851-fbab32ccd9cb" />

# Introductioin 

Autonomous drifting is a challenging benchmark for advanced vehicle control because it intentionally operates the vehicle beyond the conventional stable-handling region. Unlike typical autonomous-driving controllers, which aim to limit tire slip and preserve lateral stability within near-linear tire-force regimes, drifting requires the deliberate generation and regulation of large sideslip angles while maintaining path tracking, yaw-rate stability, and actuator feasibility. Thus, drifting is relevant not only for aggressive maneuvering but also as a stress test for safety-critical control under emergency conditions, where vehicles may operate near or beyond tire saturation. Understanding and controlling vehicle motion in these regimes can improve robustness during extreme maneuvers, low-friction driving, obstacle avoidance, and loss-of-control recovery [1].

From a control-theoretic perspective, drifting is challenging because the vehicle dynamics are strongly nonlinear, practically underactuated, and highly sensitive to tire–road interactions. At large sideslip angles, tire forces become nonlinear and coupled through friction limits, load transfer, and saturation. So small changes in steering, throttle, braking, or road friction can cause large variations in yaw moment and sideslip dynamics. Moreover, steady-state drift conditions may correspond to unstable, saddle-type equilibria, requiring continuous feedback stabilization rather than simple reference tracking [2]. This fundamentally distinguishes autonomous drifting from conventional path-following control, which typically assumes operation near stable nominal motion.


Within this research direction, nonlinear model predictive control is appealing because it can integrate nonlinear vehicle dynamics, state and input constraints, actuator limits, path-tracking objectives, and stability-related penalties within a single receding-horizon optimization framework. However, this expressiveness comes with substantial computational cost, as NMPC must solve a nonlinear optimal control problem at each sampling instant with sufficient speed and accuracy for closed-loop operation. This creates a key challenge for autonomous drifting: the controller must be both dynamically expressive and numerically reliable in a highly sensitive regime.

Consequently, autonomous drifting is  a spectacular aggressive-driving maneuver. And, also it is a rigorous benchmark for nonlinear predictive control. A successful controller must stabilize an intrinsically unstable high-sideslip motion, manage nonlinear tire-force saturation, resolve competing path-tracking and drift-stabilization objectives, and do so under real-time computational constraints. These properties make drifting an ideal testbed for evaluating NMPC algorithms, solver warm-start strategies. Real time embedded optimal-control software/solver such as acados implementats with fast multiple-shooting NMPC with sensitivities and RTI-class solvers for embedded execution and focuses on efficient SQP/RTI implementations for general NMPC.This work studies a narrower question: whether a drifting NMPC solve can be reorganized as a compiled differentiable program that supports batched multi-start exploration and low-latency Gauss–Newton real-time iteration in JAX. 


Therefore, this works contributes on a solver organization for autonomous drifting NMPC. Specifically, the project formulates the drifting NMPC solve as a compiled differentiable program in JAX, combining batched candidate generation with a Gauss–Newton real-time iteration update. This organization makes the compile/runtime trade-off explicit, which is often hidden in classical external-solver workflows. It further exploits JAX transformations such as : `jit` for compiling fused numerical kernels, `vmap` for batched candidate evaluation, `scan` for compact fixed-horizon rollout, and checkpointing for memory–computation trade-offs during reverse-mode differentiation.


# Vechile Model 



# References 

1] Meijer, S., Bertipaglia, A., & Shyrokau, B. (2024, July). A nonlinear model predictive control for automated drifting with a standard passenger vehicle. In 2024 IEEE International Conference on Advanced Intelligent Mechatronics (AIM) (pp. 284-289). IEEE.

2] Hindiyeh, Rami Y., and J. Christian Gerdes. "A controller framework for autonomous drifting: Design, stability, and experimental validation." Journal of Dynamic Systems, Measurement, and Control 136.5 (2014): 051015.












