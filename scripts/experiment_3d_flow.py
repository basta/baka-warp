
import sys
import os
import math
import numpy as np
import warp as wp
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fluid_engine import FluidOptimizer, initialize_fields_3d, N_GRID, DH, BoundaryCondition
from src.export_utils import export_simulation_sequence

@wp.kernel
def set_diagonal_target_3d(
    target_vx: wp.array3d(dtype=float),
    target_vy: wp.array3d(dtype=float),
    target_vz: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()
    
    # Normalized coordinates 0..1
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    z = (float(k) + 0.5) * DH
    
    # Center (0.5, 0.5, 0.5)
    dx = x - 0.5
    dy = y - 0.5
    dz = z - 0.5
    
    # Gaussian falloff
    # width of the stream
    sigma = 0.15
    
    # Compute Gaussian intensity
    dist_sq = dx*dx + dy*dy + dz*dz
    val = wp.exp(-dist_sq / (2.0 * sigma * sigma))
    
    # Target Flow: Diagonal (1, 1, 1)
    target_vx[i, j, k] = val * 1.0
    target_vy[i, j, k] = val * 1.0
    target_vz[i, j, k] = val * 1.0

def main():
    wp.init()
    device = wp.get_device()
    print(f"Running 3D diagonal flow experiment on device: {device}")
    
    # Set up optimizer
    num_steps = 20
    num_bases = 8
    
    # Initialize optimization wrapper with dim=3
    sim = FluidOptimizer(
        num_basis_fields=num_bases, 
        sim_steps=num_steps, 
        device=device, 
        bc_type=BoundaryCondition.WALL, 
        dim=3
    )
    
    # ---------------------------------------------------------
    # Define Target: Diagonal flow in center for ALL time steps
    # ---------------------------------------------------------
    print("Setting up 3D diagonal target field...")
    for t in range(sim.sim_steps):
        wp.launch(
            set_diagonal_target_3d, 
            sim.shape, 
            inputs=[sim.target_vx_arrays[t], sim.target_vy_arrays[t], sim.target_vz_arrays[t]]
        )
        
    # Visualize the target once (Slice at z = N/2)
    mid_z = N_GRID // 2
    target_vx_slice = sim.target_vx_arrays[0].numpy()[:, :, mid_z]
    target_vy_slice = sim.target_vy_arrays[0].numpy()[:, :, mid_z]
    
    # Optional: Save target slice visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Target VX Slice (z={mid_z})")
    plt.imshow(target_vx_slice, origin="lower")
    plt.subplot(1, 2, 2)
    plt.title(f"Target VY Slice (z={mid_z})")
    plt.imshow(target_vy_slice, origin="lower")
    plt.savefig("target_field_3d_slice.png")
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
        # Reset init state (density sphere, 0 velocity) using 3D kernel
        wp.launch(
            initialize_fields_3d, 
            sim.shape, 
            inputs=[
                sim.density_arrays[0], 
                sim.vx_arrays[0], 
                sim.vy_arrays[0], 
                sim.vz_arrays[0],
                sim.n_grid,
                sim.dh
            ]
        )
        wp.synchronize()
        
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
        initialize_fields_3d, 
        sim.shape, 
        inputs=[
            sim.density_arrays[0], 
            sim.vx_arrays[0], 
            sim.vy_arrays[0], 
            sim.vz_arrays[0],
            sim.n_grid,
            sim.dh
        ]
    )
    sim.forward()
    wp.synchronize()
    
    # Get final velocity slices
    vx_final = sim.vx_arrays[-1].numpy()
    vy_final = sim.vy_arrays[-1].numpy()
    vz_final = sim.vz_arrays[-1].numpy()
    
    vx_slice = vx_final[:, :, mid_z]
    vy_slice = vy_final[:, :, mid_z]
    vz_slice = vz_final[:, :, mid_z]
    
    speed_slice = np.sqrt(vx_slice**2 + vy_slice**2 + vz_slice**2)
    
    # Plotting Slice
    plt.figure(figsize=(12, 5))
    
    # 1. Target speed slice
    plt.subplot(1, 3, 1)
    target_speed_slice = np.sqrt(target_vx_slice**2 + target_vy_slice**2) # Z component was target 1.0 too, but just showing XY slice mag roughly
    # Actually calculate full target speed slice for comparison
    target_vz_slice = sim.target_vz_arrays[0].numpy()[:, :, mid_z]
    target_speed_slice_full = np.sqrt(target_vx_slice**2 + target_vy_slice**2 + target_vz_slice**2)
    
    plt.title("Target Speed Slice (Center)")
    plt.imshow(target_speed_slice_full, origin="lower", vmin=0, vmax=math.sqrt(3))
    plt.colorbar()
    
    # 2. Optimized Result Slice
    plt.subplot(1, 3, 2)
    plt.title("Optimized Result Speed Slice")
    plt.imshow(speed_slice, origin="lower", vmin=0, vmax=math.sqrt(3))
    plt.colorbar()
    
    # 3. Quiver / Vector field (XY projection on slice)
    plt.subplot(1, 3, 3)
    plt.title("Optimized Vector Field (XY Slice)")
    step = 4
    Y, X = np.mgrid[0:N_GRID:step, 0:N_GRID:step]
    plt.quiver(X, Y, vx_slice[::step, ::step], vy_slice[::step, ::step], scale=10)
    plt.xlim(0, N_GRID/step)
    plt.ylim(0, N_GRID/step)
    
    plt.tight_layout()
    plt.savefig("experiment_3d_diagonal_result.png")
    print("Saved result slice to experiment_3d_diagonal_result.png")

    # ---------------------------------------------------------
    # Visualize Optimized Force Field
    # ---------------------------------------------------------
    
    (total_fx, total_fy, total_fz), basis_forces = sim.get_force_fields()
    
    total_force_mag = np.sqrt(total_fx**2 + total_fy**2 + total_fz**2)
    total_mag_slice = total_force_mag[:, :, mid_z]
    
    plt.figure(figsize=(10, 5))
    
    # Magnitude Slice
    plt.subplot(1, 2, 1)
    plt.title("Optimized Force Field Mag Slice")
    plt.imshow(total_mag_slice, origin="lower", cmap='inferno')
    plt.colorbar()
    
    # Quiver Slice (XY)
    plt.subplot(1, 2, 2)
    plt.title("Optimized Force Field Vectors (XY Slice)")
    
    plt.imshow(total_mag_slice.T, origin="lower", cmap='gray', alpha=0.3)
    
    Y, X = np.mgrid[0:N_GRID:step, 0:N_GRID:step]
    
    fx_slice = total_fx[:, :, mid_z]
    fy_slice = total_fy[:, :, mid_z]
    
    plt.quiver(X, Y, fx_slice[::step, ::step], fy_slice[::step, ::step], color='r')
    
    plt.xlim(0, N_GRID)
    plt.ylim(0, N_GRID)
    
    plt.tight_layout()
    plt.savefig("optimized_force_field_3d_slice.png")
    print("Saved optimized force field slice to optimized_force_field_3d_slice.png")

    # ---------------------------------------------------------
    # Export to VTK
    # ---------------------------------------------------------
    print("Exporting simulation to VTK (vtk_output_3d_exp)...")
    
    extra_vectors = {
        "force_total": (total_fx, total_fy, total_fz)
    }
    
    # Add individual basis forces
    for k, (bfx, bfy, bfz) in enumerate(basis_forces):
        extra_vectors[f"force_basis_{k:02d}"] = (bfx, bfy, bfz)

    export_simulation_sequence(
        output_dir="vtk_output_3d_exp",
        prefix="diagonal_flow_3d",
        density_list=sim.density_arrays,
        vx_list=sim.vx_arrays,
        vy_list=sim.vy_arrays,
        vz_list=sim.vz_arrays,
        extra_vector_fields=extra_vectors
    )

if __name__ == "__main__":
    main()
