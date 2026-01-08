
import sys
import os
import math
import numpy as np
import warp as wp
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fluid_engine import FluidOptimizer, initialize_fields, N_GRID, DH, BoundaryCondition
from src.export_utils import export_simulation_sequence

@wp.kernel
def set_diagonal_target(
    target_vx: wp.array2d(dtype=float),
    target_vy: wp.array2d(dtype=float)
):
    i, j = wp.tid()
    
    # Normalized coordinates 0..1
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    
    # Center (0.5, 0.5)
    dx = x - 0.5
    dy = y - 0.5
    
    # Gaussian falloff
    # width of the stream
    sigma = 0.15
    
    # Compute Gaussian intensity
    # exp(-r^2 / (2*sigma^2))
    dist_sq = dx*dx + dy*dy
    val = wp.exp(-dist_sq / (2.0 * sigma * sigma))
    
    # Target Flow: Diagonal (1, 1)
    # We set the magnitude to be roughly 1.0 at peak
    target_vx[i, j] = val * 1.0
    target_vy[i, j] = val * 1.0

def main():
    wp.init()
    device = wp.get_device()
    print(f"Running diagonal flow experiment on device: {device}")
    
    # Set up optimizer
    num_steps = 20
    num_bases = 8
    
    sim = FluidOptimizer(num_basis_fields=num_bases, sim_steps=num_steps, device=device, bc_type=BoundaryCondition.WALL)
    
    # ---------------------------------------------------------
    # Define Target: Diagonal flow in center for ALL time steps
    # ---------------------------------------------------------
    print("Setting up diagonal target field...")
    for t in range(sim.sim_steps):
        wp.launch(
            set_diagonal_target, 
            (N_GRID, N_GRID), 
            inputs=[sim.target_vx_arrays[t], sim.target_vy_arrays[t]]
        )
        
    # Visualize the target once to be sure
    target_vx_np = sim.target_vx_arrays[0].numpy()
    target_vy_np = sim.target_vy_arrays[0].numpy()
    
    # Optional: Save target visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Target VX")
    plt.imshow(target_vx_np, origin="lower")
    plt.subplot(1, 2, 2)
    plt.title("Target VY")
    plt.imshow(target_vy_np, origin="lower")
    plt.savefig("target_field_viz.png")
    plt.close()

    # ---------------------------------------------------------
    # Optimization Loop
    # ---------------------------------------------------------
    print("Starting Optimization...")
    
    # Reset weights
    sim.weights.zero_()
    # Optimizer settings
    sim.optimizer = wp.optim.Adam([sim.weights], lr=0.05)

    iterations = 200
    for i in range(iterations):
        # Reset init state (density circle, 0 velocity)
        wp.launch(
            initialize_fields, 
            sim.shape, 
            inputs=[
                sim.density_arrays[0], 
                sim.vx_arrays[0], 
                sim.vy_arrays[0],
                sim.n_grid,
                sim.dh
            ]
        )
        loss_val, grad_norm = sim.step_optimization()
        
        if math.isnan(grad_norm):
            print("Optimization unstable (NaN gradients).")
            break
            
        if i % 10 == 0:
            print(f"Iter {i:03d} | Loss: {loss_val:.6f} | Grad: {grad_norm:.4f}")

    print("Optimization Complete.")
    print(f"Final Weights: {sim.weights.numpy()}")

    # ---------------------------------------------------------
    # Verification Run & Plot
    # ---------------------------------------------------------
    # Run with optimized weights
    wp.launch(
        initialize_fields, 
        sim.shape, 
        inputs=[
            sim.density_arrays[0], 
            sim.vx_arrays[0], 
            sim.vy_arrays[0],
            sim.n_grid,
            sim.dh
        ]
    )
    sim.forward()
    
    # Get final velocity
    vx_final = sim.vx_arrays[-1].numpy()
    vy_final = sim.vy_arrays[-1].numpy()
    speed = np.sqrt(vx_final**2 + vy_final**2)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # 1. Target (just one component or speed)
    plt.subplot(1, 3, 1)
    target_speed = np.sqrt(target_vx_np**2 + target_vy_np**2)
    plt.title("Target Speed (Center Diagonal)")
    plt.imshow(target_speed, origin="lower", vmin=0, vmax=1.0)
    plt.colorbar()
    
    # 2. Optimized Result
    plt.subplot(1, 3, 2)
    plt.title("Optimized Result Speed")
    plt.imshow(speed, origin="lower", vmin=0, vmax=1.0)
    plt.colorbar()
    
    # 3. Quiver / Vector field
    plt.subplot(1, 3, 3)
    plt.title("Optimized Vector Field")
    # Subsample for clearer quiver
    step = 4
    Y, X = np.mgrid[0:N_GRID:step, 0:N_GRID:step]
    plt.quiver(X, Y, vx_final[::step, ::step], vy_final[::step, ::step], scale=10)
    plt.xlim(0, N_GRID/step)
    plt.ylim(0, N_GRID/step)
    
    plt.tight_layout()
    plt.savefig("experiment_diagonal_result.png")
    print("Saved result to experiment_diagonal_result.png")

    # ---------------------------------------------------------
    # Visualize Optimized Force Field
    # ---------------------------------------------------------
    
    # Use the new helper method
    (total_fx, total_fy), basis_forces = sim.get_force_fields()
    
    total_force_mag = np.sqrt(total_fx**2 + total_fy**2)
    
    plt.figure(figsize=(10, 5))
    
    # Magnitude
    plt.subplot(1, 2, 1)
    plt.title("Optimized Force Field Magnitude")
    plt.imshow(total_force_mag, origin="lower", cmap='inferno')
    plt.colorbar()
    
    # Quiver
    plt.subplot(1, 2, 2)
    plt.title("Optimized Force Field Vectors")
    step = 4

    
    plt.imshow(total_force_mag.T, origin="lower", cmap='gray', alpha=0.3)
    
    Y, X = np.mgrid[0:N_GRID:step, 0:N_GRID:step]
    # Arrays are [x, y]. We want to plot Vectors at (X, Y).
    plt.quiver(X, Y, total_fx[::step, ::step], total_fy[::step, ::step], color='r')
    
    plt.xlim(0, N_GRID)
    plt.ylim(0, N_GRID)
    
    plt.tight_layout()
    plt.savefig("optimized_force_field.png")
    print("Saved optimized force field to optimized_force_field.png")

    # ---------------------------------------------------------
    # Export to VTK
    # ---------------------------------------------------------
    print("Exporting simulation to VTK...")
    
    # Prepare extra fields
    # Forces are constant in time, so we pass single frames which export_utils will repeat
    extra_vectors = {
        "force_total": (total_fx, total_fy)
    }
    
    # Add individual basis forces
    for k, (bfx, bfy) in enumerate(basis_forces):
        extra_vectors[f"force_basis_{k:02d}"] = (bfx, bfy)

    export_simulation_sequence(
        output_dir="vtk_output",
        prefix="diagonal_flow",
        density_list=sim.density_arrays,
        vx_list=sim.vx_arrays,
        vy_list=sim.vy_arrays,
        extra_vector_fields=extra_vectors
    )

if __name__ == "__main__":
    main()
