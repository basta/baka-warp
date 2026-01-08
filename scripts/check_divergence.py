
import warp as wp
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from fluid_engine import FluidOptimizer, N_GRID, DH, cyclic_index

@wp.kernel
def compute_divergence(
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
    div_out: wp.array2d(dtype=float)
):
    i, j = wp.tid()
    # Centered difference
    dx = (vx[cyclic_index(i + 1), j] - vx[cyclic_index(i - 1), j]) * 0.5 / DH
    dy = (vy[i, cyclic_index(j + 1)] - vy[i, cyclic_index(j - 1)]) * 0.5 / DH
    div_out[i, j] = dx + dy

def run_simulation(iterations):
    fluid = FluidOptimizer(sim_steps=10, pressure_iterations=iterations)
    
    # Manually set weights
    w_data = np.ones(fluid.num_basis_fields, dtype=np.float32)
    wp.copy(fluid.weights, wp.array(w_data, dtype=float, device=fluid.device))
    
    # Run a few steps
    fluid.forward()
    
    # Check divergence of the last step
    last_vx = fluid.vx_arrays[-1]
    last_vy = fluid.vy_arrays[-1]
    
    div_field = wp.zeros((N_GRID, N_GRID), dtype=float, device=fluid.device)
    
    wp.launch(
        compute_divergence,
        (N_GRID, N_GRID),
        inputs=[last_vx, last_vy, div_field]
    )
    
    div_np = div_field.numpy()
    max_div = np.max(np.abs(div_np))
    avg_div = np.mean(np.abs(div_np))
    return max_div, avg_div

def check():
    wp.init()
    
    print(f"Checking divergence across different pressure iterations:")
    for iters in [20, 100, 500, 2000]:
        max_d, avg_d = run_simulation(iters)
        print(f"Iterations: {iters:4d} | Max Div: {max_d:.6f} | Avg Div: {avg_d:.6f}")

if __name__ == "__main__":
    check()
