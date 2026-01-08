import math
import os
import sys

import numpy as np

import warp as wp
import warp.optim

# Grid dimensions
N_GRID = 64
DH = 1.0 / N_GRID

# Simulation parameters
DT = 0.005  # Reduced further
DENSITY_INIT_RADIUS = 0.1
CENTER_X = 0.5
CENTER_Y = 0.5


@wp.func
def cyclic_index(idx: wp.int32):
    """Helper function to index with periodic boundary conditions."""
    ret_idx = idx % N_GRID
    if ret_idx < 0:
        ret_idx += N_GRID
    return ret_idx


@wp.kernel
def initialize_fields(
    density: wp.array2d(dtype=float),
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float)
):
    i, j = wp.tid()
    
    # Initialize density as a circle in the center
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    
    dx = x - CENTER_X
    dy = y - CENTER_Y
    dist_sq = dx*dx + dy*dy
    
    if dist_sq < DENSITY_INIT_RADIUS * DENSITY_INIT_RADIUS:
        density[i, j] = 1.0
    else:
        density[i, j] = 0.0

    u[i, j] = 0.0
    v[i, j] = 0.0


@wp.kernel
def apply_forces(
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
    force_basis_x: wp.array3d(dtype=float),
    force_basis_y: wp.array3d(dtype=float),
    weights: wp.array(dtype=float),
    dt: float
):
    """
    Apply weighted sum of basis force fields to the velocity field.
    vx += dt * sum(w_k * F_k_x)
    """
    i, j = wp.tid()
    
    fx_sum = float(0.0)
    fy_sum = float(0.0)
    
    num_bases = force_basis_x.shape[0]
    
    for k in range(num_bases):
        w = weights[k]
        fx_sum += w * force_basis_x[k, i, j]
        fy_sum += w * force_basis_y[k, i, j]
        
    vx[i, j] = vx[i, j] + fx_sum * dt
    vy[i, j] = vy[i, j] + fy_sum * dt


@wp.kernel
def advect(
    dt: float,
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
    f0: wp.array2d(dtype=float),
    f1: wp.array2d(dtype=float),
):
    """Move field f0 according to vx and vy velocities using an implicit Euler integrator."""

    i, j = wp.tid()

    # Backtrace
    center_xs = float(i) - vx[i, j] * dt / DH # working in grid units for interpolation
    center_ys = float(j) - vy[i, j] * dt / DH

    # Compute indices of source cells.
    left_idx = wp.int32(wp.floor(center_xs))
    bot_idx = wp.int32(wp.floor(center_ys))

    s1 = center_xs - float(left_idx)  # Relative weight of right cell
    s0 = 1.0 - s1
    t1 = center_ys - float(bot_idx)  # Relative weight of top cell
    t0 = 1.0 - t1

    i0 = cyclic_index(left_idx)
    i1 = cyclic_index(left_idx + 1)
    j0 = cyclic_index(bot_idx)
    j1 = cyclic_index(bot_idx + 1)

    # Perform bilinear interpolation
    f1[i, j] = s0 * (t0 * f0[i0, j0] + t1 * f0[i0, j1]) + s1 * (t0 * f0[i1, j0] + t1 * f0[i1, j1])


@wp.kernel
def divergence(wx: wp.array2d(dtype=float), wy: wp.array2d(dtype=float), div: wp.array2d(dtype=float)):
    """Compute div(w)."""
    i, j = wp.tid()
    
    dx = (wx[cyclic_index(i + 1), j] - wx[cyclic_index(i - 1), j]) * 0.5 / DH
    dy = (wy[i, cyclic_index(j + 1)] - wy[i, cyclic_index(j - 1)]) * 0.5 / DH
    
    div[i, j] = dx + dy


@wp.kernel
def jacobi_iter(div: wp.array2d(dtype=float), p0: wp.array2d(dtype=float), p1: wp.array2d(dtype=float)):
    """Calculate a single Jacobi iteration for solving the pressure Poisson equation."""
    i, j = wp.tid()

    p1[i, j] = 0.25 * (
        -DH * DH * div[i, j]
        + p0[cyclic_index(i - 1), j]
        + p0[cyclic_index(i + 1), j]
        + p0[i, cyclic_index(j - 1)]
        + p0[i, cyclic_index(j + 1)]
    )


@wp.kernel
def update_velocities(
    p: wp.array2d(dtype=float),
    wx: wp.array2d(dtype=float),
    wy: wp.array2d(dtype=float),
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
):
    """Given p and (wx, wy), compute an 'incompressible' velocity field (vx, vy)."""
    i, j = wp.tid()

    vx[i, j] = wx[i, j] - 0.5 * (p[cyclic_index(i + 1), j] - p[cyclic_index(i - 1), j]) / DH
    vy[i, j] = wy[i, j] - 0.5 * (p[i, cyclic_index(j + 1)] - p[i, cyclic_index(j - 1)]) / DH


@wp.kernel
def compute_velocity_loss(
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
    target_vx: wp.array2d(dtype=float),
    target_vy: wp.array2d(dtype=float),
    loss: wp.array(dtype=float)
):
    i, j = wp.tid()
    
    diff_x = vx[i, j] - target_vx[i, j]
    diff_y = vy[i, j] - target_vy[i, j]
    
    # Mean squared error
    val = (diff_x * diff_x + diff_y * diff_y) / float(N_GRID * N_GRID)
    
    wp.atomic_add(loss, 0, val)


class FluidOptimizer:
    def __init__(self, num_basis_fields=5, sim_steps=50, pressure_iterations=20, device=None):
        self.device = device if device else wp.get_device()
        self.num_basis_fields = num_basis_fields
        self.sim_steps = sim_steps
        self.pressure_iterations = pressure_iterations
        self.dt = DT
        
        # ---------------------------------------------------------------------
        # Create Data Arrays
        # ---------------------------------------------------------------------
        
        # We need to store full trajectory for backprop
        # States: vx, vy, density, pressure
        # Intermediate: wx, wy, div, pressure_iters
        
        # Velocity fields (u, v)
        self.vx_arrays = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps + 1)]
        self.vy_arrays = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps + 1)]
        
        # Density (passive scalar)
        self.density_arrays = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps + 1)]
        
        # Intermediate Advected Velocities (w)
        self.wx_arrays = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps)]
        self.wy_arrays = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps)]
        
        # Divergence
        self.div_arrays = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps)]
        
        # Pressure (multiple iterations per step)
        # Store all pressure iterations to backprop through the solver
        self.pressure_arrays = []
        for _ in range(sim_steps):
            step_pressures = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device, requires_grad=True) for _ in range(self.pressure_iterations + 1)]
            self.pressure_arrays.append(step_pressures)
            
        # ---------------------------------------------------------------------
        # Basis Fields and Parameters
        # ---------------------------------------------------------------------
        
        # Generate random basis fields for this example
        rng = np.random.default_rng(42)
        
        # Create persistent basis fields on device
        self.basis_fx_np = np.zeros((num_basis_fields, N_GRID, N_GRID), dtype=np.float32)
        self.basis_fy_np = np.zeros((num_basis_fields, N_GRID, N_GRID), dtype=np.float32)
        
        for k in range(num_basis_fields):
            # Generate smooth random fields using low freq sines
            freq_x = rng.uniform(1.0, 3.0)
            freq_y = rng.uniform(1.0, 3.0)
            phase_x = rng.uniform(0, 2*np.pi)
            phase = rng.uniform(0, 2*np.pi)
            
            for y in range(N_GRID):
                for x in range(N_GRID):
                    yf = y * DH
                    xf = x * DH
                    # Simple divergence-free-ish vortex or wave
                    self.basis_fx_np[k, y, x] = np.sin(xf * freq_x + phase_x) * np.cos(yf * freq_y)
                    self.basis_fy_np[k, y, x] = np.cos(xf * freq_x) * np.sin(yf * freq_y + phase)

        self.basis_fx = wp.array(self.basis_fx_np, dtype=float, device=self.device)
        self.basis_fy = wp.array(self.basis_fy_np, dtype=float, device=self.device)
        
        # Learnable Weights
        self.weights = wp.zeros(num_basis_fields, dtype=float, device=self.device, requires_grad=True)

        # Target Velocity (placeholder, will set later)
        self.target_vx_arrays = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device) for _ in range(sim_steps)]
        self.target_vy_arrays = [wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device) for _ in range(sim_steps)]

        self.loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)
        self.optimizer = warp.optim.Adam([self.weights], lr=0.01) # Raised LR slightly as gradients will be more stable

    def run_step(self, t):
        """Execute one simulation step t -> t+1."""
        
        # 1. Self-Advection
        wp.launch(advect, (N_GRID, N_GRID), inputs=[self.dt, self.vx_arrays[t], self.vy_arrays[t], self.vx_arrays[t]], outputs=[self.wx_arrays[t]])
        wp.launch(advect, (N_GRID, N_GRID), inputs=[self.dt, self.vx_arrays[t], self.vy_arrays[t], self.vy_arrays[t]], outputs=[self.wy_arrays[t]])
        
        # 2. Apply Forces (Modify w inplace)
        wp.launch(
            apply_forces, 
            (N_GRID, N_GRID), 
            inputs=[
                self.wx_arrays[t], 
                self.wy_arrays[t], 
                self.basis_fx, 
                self.basis_fy, 
                self.weights, 
                self.dt
            ]
        )
        
        # 3. Pressure Projection
        # 3a. Divergence
        wp.launch(divergence, (N_GRID, N_GRID), inputs=[self.wx_arrays[t], self.wy_arrays[t]], outputs=[self.div_arrays[t]])
        
        # 3b. Solve Pressure (Jacobi)
        # Initialize p0 to 0 or previous step? (previous step is better but keeping 0 for simplicity/independence)
        self.pressure_arrays[t][0].zero_()
        
        for k in range(self.pressure_iterations):
            wp.launch(
                jacobi_iter, 
                (N_GRID, N_GRID), 
                inputs=[self.div_arrays[t], self.pressure_arrays[t][k]], 
                outputs=[self.pressure_arrays[t][k+1]]
            )
            
        # 3c. Subtract Gradient
        final_p_idx = self.pressure_iterations
        wp.launch(
            update_velocities,
            (N_GRID, N_GRID),
            inputs=[self.pressure_arrays[t][final_p_idx], self.wx_arrays[t], self.wy_arrays[t]],
            outputs=[self.vx_arrays[t+1], self.vy_arrays[t+1]]
        )
        
        # 4. Advect Density (Passive Scalar)
        wp.launch(
            advect,
            (N_GRID, N_GRID),
            inputs=[self.dt, self.vx_arrays[t+1], self.vy_arrays[t+1], self.density_arrays[t]],
            outputs=[self.density_arrays[t+1]]
        )

    def clear_all_gradients(self):
        """Manually zero gradients of all state arrays to prevent accumulation."""
        self.weights.grad.zero_()
        self.loss.grad.zero_()
        
        # Helper to zero a list of arrays
        def zero_list(arr_list):
            for arr in arr_list:
                if arr.requires_grad:
                    arr.grad.zero_()
        
        zero_list(self.vx_arrays)
        zero_list(self.vy_arrays)
        zero_list(self.density_arrays)
        zero_list(self.wx_arrays)
        zero_list(self.wy_arrays)
        zero_list(self.div_arrays)
        
        # Pressure is a list of lists
        for step_list in self.pressure_arrays:
            zero_list(step_list)

    def forward(self):
        # Reset state (only t=0 is fixed)
        # We assume vx[0], vy[0], density[0] are set.

        self.loss.zero_()

        for t in range(self.sim_steps):
            self.run_step(t)
            wp.launch(
                compute_velocity_loss,
                (N_GRID, N_GRID),
                inputs=[
                    self.vx_arrays[t+1],      # Current simulation state
                    self.vy_arrays[t+1], 
                    self.target_vx_arrays[t], # Corresponding target frame
                    self.target_vy_arrays[t], 
                    self.loss
                ]
            )
        
    def step_optimization(self):
        self.clear_all_gradients()
        self.weights.grad.zero_()
        self.tape = wp.Tape()
        self.loss.zero_()
        
        with self.tape:
            self.forward()
        
        self.tape.backward(self.loss)
        
        # Gradient Processing
        grad_np = self.weights.grad.numpy()
        
        if not np.all(np.isfinite(grad_np)):
            print("  [Warning] Gradients contain NaN/Inf! Zeroing grads.")
            self.weights.grad.zero_()
            return self.loss.numpy()[0], float('nan')

        grad_norm = np.linalg.norm(grad_np)
        
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            grad_np = grad_np * scale
            self.weights.grad = wp.array(grad_np, dtype=float, device=self.device)
            grad_norm = max_grad_norm
        
        self.optimizer.step([self.weights.grad])
        
        return self.loss.numpy()[0], grad_norm
