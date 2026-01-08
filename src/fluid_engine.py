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


@wp.func
def sample_field(field: wp.array2d(dtype=float), x: float, y: float):
    # Bilinear interpolation
    # field is defined at integer indices 0..N-1
    # clamp coords
    x = wp.max(0.0, wp.min(x, float(N_GRID) - 1.0))
    y = wp.max(0.0, wp.min(y, float(N_GRID) - 1.0))

    x0 = wp.int32(wp.floor(x))
    y0 = wp.int32(wp.floor(y))
    
    wx1 = x - float(x0)
    wy1 = y - float(y0)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    
    x1 = wp.min(x0 + 1, N_GRID - 1)
    y1 = wp.min(y0 + 1, N_GRID - 1)
    
    val = (wx0 * wy0 * field[x0, y0] + 
           wx1 * wy0 * field[x1, y0] + 
           wx0 * wy1 * field[x0, y1] + 
           wx1 * wy1 * field[x1, y1])
    return val

@wp.kernel
def advect_mac_u(
    dt: float,
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    u_new: wp.array2d(dtype=float)
):
    # Advect u component. u lives at (i, j+0.5) -- wait, standard MAC:
    # u[i,j] is at face (i-1/2, j)? Or (i+1/2, j)?
    # Convention: u[i, j] is flow across CONSTANT-X face between cell (i-1, j) and (i, j).
    # Coordinate: x = i * DH, y = (j + 0.5) * DH.
    
    i, j = wp.tid()
    
    # World pos of this u-face
    x = float(i) * DH
    y = (float(j) + 0.5) * DH
    
    # Velocity at this point
    # u is exact at this point
    vel_x = u[i, j]
    
    # v needs interpolation. v lives at (x+0.5, y-0.5).
    # We want v at (i, j+0.5)
    # v grid coords: x_v = i - 0.5, y_v = j + 0.5
    # To get world (i, j+0.5), we sample v at grid coord (i-0.5, j+0.5).
    grid_v_x = float(i) - 0.5
    grid_v_y = float(j) + 0.5
    vel_y = sample_field(v, grid_v_x, grid_v_y)
    
    # Backtrace
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample u at src. u lives at grid (i, j+0.5).
    # grid u coords:
    grid_src_x = src_x / DH
    grid_src_y = (src_y / DH) - 0.5
    
    u_new[i, j] = sample_field(u, grid_src_x, grid_src_y)


@wp.kernel
def advect_mac_v(
    dt: float,
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    v_new: wp.array2d(dtype=float)
):
    # Advect v component. v lives at (i+0.5, j).
    # Convention: v[i, j] is flow across CONSTANT-Y face between cell (i, j-1) and (i, j).
    
    i, j = wp.tid()
    
    # World pos of this v-face
    x = (float(i) + 0.5) * DH
    y = float(j) * DH
    
    # Velocity
    vel_y = v[i, j]
    
    # u needs interpolation. u lives at (i, j+0.5).
    # We want u at (i+0.5, j).
    # u grid coords: x_u = i + 0.5, y_u = j - 0.5
    grid_u_x = float(i) + 0.5
    grid_u_y = float(j) - 0.5
    vel_x = sample_field(u, grid_u_x, grid_u_y)
    
    # Backtrace
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample v at src. v lives at grid (i+0.5, j).
    grid_src_x = (src_x / DH) - 0.5
    grid_src_y = (src_y / DH)
    
    v_new[i, j] = sample_field(v, grid_src_x, grid_src_y)


@wp.kernel
def advect_density(
    dt: float,
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    rho_old: wp.array2d(dtype=float),
    rho_new: wp.array2d(dtype=float)
):
    # Advect scalar at cell centers (i+0.5, j+0.5)
    i, j = wp.tid()
    
    # World pos
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    
    # Velocity at center
    # u at (i, j+0.5) and (i+1, j+0.5). Avg to get center.
    # We need u[i, j] and u[i+1, j].
    # But u indices are 0..N. u[N] is boundary?
    # Assume u has shape (N+1, N) or we use cyclic/clamp in sampler?
    # Let's assume u, v are (N, N) and we use cyclic indices implicitly in kernel or valid ranges.
    # Current codebase uses cyclic_index. Let's use that for neighbor access.
    
    idx_i = i
    idx_i1 = cyclic_index(i + 1)
    
    idx_j = j
    idx_j1 = cyclic_index(j + 1)
    
    vel_x = (u[idx_i, j] + u[idx_i1, j]) * 0.5
    vel_y = (v[i, idx_j] + v[i, idx_j1]) * 0.5
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample rho (centered)
    grid_src_x = (src_x / DH) - 0.5
    grid_src_y = (src_y / DH) - 0.5
    
    rho_new[i, j] = sample_field(rho_old, grid_src_x, grid_src_y)


@wp.kernel
def divergence(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), div: wp.array2d(dtype=float)):
    """Compute div(u) at cell centers."""
    i, j = wp.tid()
    
    # u[i, j] is left face, u[i+1, j] is right face
    # v[i, j] is bottom face, v[i, j+1] is top face
    
    u_right = u[cyclic_index(i + 1), j]
    u_left = u[i, j]
    
    v_top = v[i, cyclic_index(j + 1)]
    v_bot = v[i, j]
    
    div[i, j] = (u_right - u_left + v_top - v_bot) / DH


@wp.kernel
def jacobi_iter(div: wp.array2d(dtype=float), p0: wp.array2d(dtype=float), p1: wp.array2d(dtype=float)):
    """Solve Laplacian P = div."""
    i, j = wp.tid()
    
    # Standard 5-point Laplacian
    # 4 * p[i, j] - neighbors = -div * dx^2
    # p[i,j] = (neighbors - div*dx^2) / 4
    
    sum_neighbors = (p0[cyclic_index(i - 1), j] +
                     p0[cyclic_index(i + 1), j] +
                     p0[i, cyclic_index(j - 1)] +
                     p0[i, cyclic_index(j + 1)])
                     
    p1[i, j] = 0.25 * (sum_neighbors - div[i, j] * DH * DH)


@wp.kernel
def update_velocities(
    p: wp.array2d(dtype=float),
    u_in: wp.array2d(dtype=float),
    v_in: wp.array2d(dtype=float),
    u_out: wp.array2d(dtype=float),
    v_out: wp.array2d(dtype=float),
):
    """Subtract pressure gradient."""
    i, j = wp.tid()

    # Update u at (i, j+0.5) (left face of cell i,j)
    # Grad P x component at face: (P[i,j] - P[i-1,j]) / DH
    grad_p_x = (p[i, j] - p[cyclic_index(i - 1), j]) / DH
    u_out[i, j] = u_in[i, j] - grad_p_x
    
    # Update v at (i+0.5, j) (bottom face of cell i,j)
    # Grad P y component at face: (P[i,j] - P[i,j-1]) / DH
    grad_p_y = (p[i, j] - p[i, cyclic_index(j - 1)]) / DH
    v_out[i, j] = v_in[i, j] - grad_p_y


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
    def __init__(self, num_basis_fields=5, sim_steps=50, pressure_iterations=100, device=None):
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
        
        # 1. Advect Velocities (Self-Advection)
        # u -> w_u, v -> w_v
        wp.launch(advect_mac_u, (N_GRID, N_GRID), inputs=[self.dt, self.vx_arrays[t], self.vy_arrays[t], self.wx_arrays[t]])
        wp.launch(advect_mac_v, (N_GRID, N_GRID), inputs=[self.dt, self.vx_arrays[t], self.vy_arrays[t], self.wy_arrays[t]])
        
        # 2. Apply Forces
        # We need to sample basis at face centers.
        # For efficiency, let's just use existing kernels but we should strictly use face coordinates.
        # The basis functions are smooth so using standard grid indices is ok approximation, 
        # but let's stick to the structure.
        # Since apply_forces assumes collocated, let's just reuse it but acknowledge positions are slightly shifted?
        # Actually `apply_forces` uses basis arrays which are N_GRID x N_GRID.
        # We can just apply them directly to u and v arrays.
        # u[i,j] corresponds to (i, j) in basis array?
        # Let's trust the optimizer to learn the weights regardless of sub-grid phase shift.
        # But we need to make sure we operate on w (intermediate)
        
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
        wp.launch(divergence, (N_GRID, N_GRID), inputs=[self.wx_arrays[t], self.wy_arrays[t], self.div_arrays[t]])
        
        # 3b. Solve Pressure (Jacobi)
        self.pressure_arrays[t][0].zero_()
        
        for k in range(self.pressure_iterations):
            wp.launch(
                jacobi_iter, 
                (N_GRID, N_GRID), 
                inputs=[self.div_arrays[t], self.pressure_arrays[t][k], self.pressure_arrays[t][k+1]]
            )
            
        # 3c. Subtract Gradient
        final_p_idx = self.pressure_iterations
        wp.launch(
            update_velocities,
            (N_GRID, N_GRID),
            inputs=[self.pressure_arrays[t][final_p_idx], self.wx_arrays[t], self.wy_arrays[t], self.vx_arrays[t+1], self.vy_arrays[t+1]]
        )
        
        # 4. Advect Density (Passive Scalar)
        wp.launch(
            advect_density,
            (N_GRID, N_GRID),
            inputs=[self.dt, self.vx_arrays[t+1], self.vy_arrays[t+1], self.density_arrays[t], self.density_arrays[t+1]]
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
