# time_integrator.py
# =============================================================
# Newmark-Beta transient solver for structural dynamics
# Solves: M*a + C*v + K*u = F(t)
# All units in IPS system: inches, lbf, seconds
# =============================================================

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from config import NEWMARK_BETA, NEWMARK_GAMMA, N_STEPS, DT

# -------------------------------------------------------------
# UNIT NOTE
# u  : displacement  (in)
# v  : velocity      (in/s)
# a  : acceleration  (in/s²)
# F  : force         (lbf)
# K  : lbf/in
# M  : lbf·s²/in
# C  : lbf·s/in
# -------------------------------------------------------------

def newmark_beta_solver(M_free, C_free, K_free, F_global,
                        dt=DT, n_steps=N_STEPS,
                        beta=NEWMARK_BETA, gamma=NEWMARK_GAMMA):
    """
    Newmark-Beta transient solver (average acceleration method).

    Solves M*a + C*v + K*u = F(t) over n_steps timesteps.

    Inputs:
        M_free   : mass matrix (n_free x n_free) sparse
        C_free   : damping matrix (n_free x n_free) sparse
        K_free   : stiffness matrix (n_free x n_free) sparse
        F_global : force matrix (n_free x n_steps)
        dt       : timestep size (s)
        n_steps  : number of timesteps
        beta     : Newmark beta parameter (0.25)
        gamma    : Newmark gamma parameter (0.50)

    Output:
        accel_history : acceleration at every free DOF
                        shape (n_free, n_steps)  in/s²
    """
    n_free = M_free.shape[0]

    # ----------------------------------------------------------
    # Initialize displacement, velocity, acceleration
    # Beam starts completely at rest
    # ----------------------------------------------------------
    u = np.zeros(n_free)   # displacement (in)
    v = np.zeros(n_free)   # velocity     (in/s)
    a = np.zeros(n_free)   # acceleration (in/s²)

    # Compute initial acceleration from equation of motion
    # M*a0 = F(0) - C*v0 - K*u0
    # Since u0=0, v0=0: a0 = M^-1 * F(0)
    rhs_0 = F_global[:, 0] - C_free.dot(v) - K_free.dot(u)
    a     = spsolve(csr_matrix(M_free), rhs_0)

    # ----------------------------------------------------------
    # Precompute effective stiffness matrix K_eff
    # K_eff = K + (gamma/beta*dt)*C + (1/beta*dt²)*M
    # Computed ONCE before the loop — stays constant
    # ----------------------------------------------------------
    c1 = 1.0  / (beta * dt**2)
    c2 = gamma / (beta * dt)

    K_eff = K_free + c2 * C_free + c1 * M_free
    K_eff = csr_matrix(K_eff)

    # ----------------------------------------------------------
    # Storage for acceleration history
    # ----------------------------------------------------------
    accel_history = np.zeros((n_free, n_steps))
    accel_history[:, 0] = a

    # ----------------------------------------------------------
    # Time stepping loop
    # ----------------------------------------------------------
    for t in range(1, n_steps):

        # External force at this timestep
        F_ext = F_global[:, t]

        # --- Predictor step ---
        # Predict displacement and velocity before solving
        u_pred = u + dt * v + dt**2 * (0.5 - beta) * a
        v_pred = v + dt * (1.0 - gamma) * a

        # --- Effective force vector ---
        F_eff = F_ext + M_free.dot(c1 * u_pred) + C_free.dot(c2 * u_pred - v_pred)

        # --- Solve for new displacement ---
        # K_eff * u_new = F_eff
        u_new = spsolve(K_eff, F_eff)

        # --- Corrector step ---
        # Update acceleration and velocity
        a_new = c1 * (u_new - u_pred)
        v_new = v_pred + gamma * dt * a_new

        # --- Store acceleration ---
        accel_history[:, t] = a_new

        # --- Update state for next step ---
        u = u_new
        v = v_new
        a = a_new

    return accel_history