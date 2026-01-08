
import math
import os
import sys

import numpy as np

import warp as wp
import warp.optim

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Grid dimensions
N_GRID = 128
DH = 1.0 / N_GRID

# Simulation parameters
DT = 0.01  # Reduced for stability
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
    def __init__(self, num_basis_fields=5, sim_steps=50, device=None):
        self.device = device if device else wp.get_device()
        self.num_basis_fields = num_basis_fields
        self.sim_steps = sim_steps
        self.pressure_iterations = 20
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

        # Loss
        self.loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)
        
        # Target Velocity (placeholder, will set later)
        self.target_vx = wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device)
        self.target_vy = wp.zeros((N_GRID, N_GRID), dtype=float, device=self.device)

        # Optimizer
        self.optimizer = warp.optim.Adam([self.weights], lr=0.1)
        self.tape = wp.Tape()

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

    def forward(self):
        # Reset state (only t=0 is fixed)
        # We assume vx[0], vy[0], density[0] are set.
        
        with self.tape:
            for t in range(self.sim_steps):
                self.run_step(t)
                
            # Compute Loss
            wp.launch(
                compute_velocity_loss,
                (N_GRID, N_GRID),
                inputs=[self.vx_arrays[-1], self.vy_arrays[-1], self.target_vx, self.target_vy],
                outputs=[self.loss]
            )
            
    def step_optimization(self):
        self.tape.zero()
        self.loss.zero_()
        
        self.forward()
        
        self.tape.backward(self.loss)
        
        # Gradient Clipping
        # We need to copy grad to host to check/modify or use a kernel
        # Since weights array is small (8), numpy is fine
        grad_np = self.weights.grad.numpy()
        grad_norm = np.linalg.norm(grad_np)
        
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            # Apply scaling
            grad_np = grad_np * scale
            # Write back
            self.weights.grad = wp.array(grad_np, dtype=float, device=self.device)
            # Update norm for logging
            grad_norm = max_grad_norm
        
        self.optimizer.step([self.weights.grad])
        
        return self.loss.numpy()[0], grad_norm

def main():
    # Setup
    wp.init()
    device = wp.get_device()
    print(f"Running on device: {device}")

    num_steps = 40
    num_bases = 8
    
    # Create Optimizer/Sim
    sim = FluidOptimizer(num_basis_fields=num_bases, sim_steps=num_steps, device=device)
    
    # -------------------------------------------------------------------------
    # 1. Generate a "Target" by running with secret weights
    # -------------------------------------------------------------------------
    print("Generating target trajectory...")
    # Scale down the weights to keep simulation stable (CFL condition)
    true_weights_np = np.random.uniform(-0.2, 0.2, size=(num_bases,)).astype(np.float32)
    true_weights_wp = wp.array(true_weights_np, dtype=float, device=device)
    
    # Assign true weights temporarily
    wp.copy(sim.weights, true_weights_wp)
    
    # Initialize state
    wp.launch(initialize_fields, (N_GRID, N_GRID), inputs=[sim.density_arrays[0], sim.vx_arrays[0], sim.vy_arrays[0]])
    
    # Run forward (no tape needed for target gen, but class does it)
    sim.forward()
    
    # Copy result to target
    wp.copy(sim.target_vx, sim.vx_arrays[-1])
    wp.copy(sim.target_vy, sim.vy_arrays[-1])
    
    print(f"Target generated with weights: {true_weights_np}")
    
    # -------------------------------------------------------------------------
    # 2. Reset and Optimize
    # -------------------------------------------------------------------------
    print("\nStarting Optimization...")
    
    # Reset weights to zero
    sim.weights.zero_()
    # Reset Optimizer explicitly if needed or creating new one? 
    sim.optimizer = warp.optim.Adam([sim.weights], lr=0.01) # Reduced LR
    
    # Training Loop
    iterations = 200
    for i in range(iterations):
        # Reset State for training
        wp.launch(initialize_fields, (N_GRID, N_GRID), inputs=[sim.density_arrays[0], sim.vx_arrays[0], sim.vy_arrays[0]])

        loss_val, grad_norm = sim.step_optimization()
        
        if math.isnan(loss_val):
            print(f"Iter {i:03d} | Loss is NaN! Stopping.")
            break

        if i % 10 == 0:
            current_weights = sim.weights.numpy()
            w_err = np.linalg.norm(current_weights - true_weights_np)
            print(f"Iter {i:03d} | Loss: {loss_val:.6f} | Weights Error: {w_err:.4f} | Grad Norm: {grad_norm:.4f}")

    print("\nOptimization Complete.")
    final_weights = sim.weights.numpy()
    print(f"True Weights:  {true_weights_np}")
    print(f"Found Weights: {final_weights}")
    
    # verify
    if MATPLOTLIB_AVAILABLE:
        # Run one last updated forward
        wp.launch(initialize_fields, (N_GRID, N_GRID), inputs=[sim.density_arrays[0], sim.vx_arrays[0], sim.vy_arrays[0]])
        sim.forward()
        v_final_x = sim.vx_arrays[-1].numpy()
        v_final_y = sim.vy_arrays[-1].numpy()
        v_final_norm = np.sqrt(v_final_x**2 + v_final_y**2)
        
        try:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 2)
            plt.title("Optimized Velocity Magnitude")
            plt.imshow(v_final_norm, origin="lower")
            plt.savefig("optimization_result.png")
            print("Result saved to optimization_result.png")
        except Exception as e:
            print(f"Failed to save plot: {e}")

if __name__ == "__main__":
    main()
