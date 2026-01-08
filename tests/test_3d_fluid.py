
import sys
import os
import math
import numpy as np
import warp as wp

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fluid_engine import FluidOptimizer, initialize_fields_3d, N_GRID, DH, BoundaryCondition
from src.export_utils import export_simulation_sequence

def main():
    wp.init()
    device = wp.get_device()
    print(f"Running 3D fluid test on device: {device}")
    
    # Set up optimizer in 3D
    num_steps = 5
    num_bases = 5
    
    print("Initializing FluidOptimizer (3D)...")
    sim = FluidOptimizer(
        num_basis_fields=num_bases, 
        sim_steps=num_steps,
        pressure_iterations=10, 
        device=device, 
        bc_type=BoundaryCondition.WALL,
        dim=3
    )
    
    # Initialize fields
    print("Initializing fields...")
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
    
    # Run simulation forward
    print("Running simulation forward...")
    sim.forward()
    wp.synchronize()
    print("Simulation complete.")
    
    # Export to VTK
    print("Exporting results to 'vtk_output_3d'...")
    
    output_dir = "vtk_output_3d"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get forces for export
    (total_fx, total_fy, total_fz), basis_forces = sim.get_force_fields()
    
    extra_vectors = {
        "force_total": (total_fx, total_fy, total_fz)
    }
    
    # Export
    export_simulation_sequence(
        output_dir=output_dir,
        prefix="fluid_3d",
        density_list=sim.density_arrays,
        vx_list=sim.vx_arrays,
        vy_list=sim.vy_arrays,
        vz_list=sim.vz_arrays,
        extra_vector_fields=extra_vectors
    )
    
    print("Test complete.")

if __name__ == "__main__":
    main()
