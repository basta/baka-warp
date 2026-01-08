import warp as wp
import numpy as np
import pytest
import os
import sys

# Assume project root is in pythonpath, or src is installable.
# If not, we might need sys.path hack, but let's try to assume pytest runs from root.
# Just in case, keep the sys.path append for now or rely on conftest.
# The previous script had:
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from fluid_engine import FluidOptimizer, N_GRID, DH

def test_verify_masking_2d():
    wp.init()
    
    # Create optimizer
    fluid = FluidOptimizer(dim=2, sim_steps=50)
    
    # Set circular mask
    radius = 0.4
    fluid.set_mask_circle(radius=radius)
    
    # Run simulation
    fluid.forward()
    
    # Check step 10
    density_np = fluid.density_arrays[10].numpy()
    vx_np = fluid.vx_arrays[10].numpy()
    vy_np = fluid.vy_arrays[10].numpy()
    mask_np = fluid.mask.numpy()
    
    outside_indices = (mask_np == 0)
    
    max_rho_outside = np.max(np.abs(density_np[outside_indices]))
    max_v_outside = np.max(np.sqrt(vx_np[outside_indices]**2 + vy_np[outside_indices]**2))
    
    # Assertions
    assert max_rho_outside < 1e-4, f"Density leaked outside mask: {max_rho_outside}"
    assert max_v_outside < 1e-4, f"Velocity non-zero outside mask: {max_v_outside}"

def test_verify_masking_3d():
    # Helper to test 3D path as well
    wp.init()
    fluid = FluidOptimizer(dim=3, sim_steps=5)
    
    fluid.set_mask_circle(radius=0.3) # Will use sphere logic internally
    
    fluid.forward()
    
    vx_np = fluid.vx_arrays[5].numpy()
    vy_np = fluid.vy_arrays[5].numpy()
    vz_np = fluid.vz_arrays[5].numpy()
    mask_np = fluid.mask.numpy()
    
    outside_indices = (mask_np == 0)
    
    v_mag = np.sqrt(vx_np**2 + vy_np**2 + vz_np**2)
    max_v_outside = np.max(v_mag[outside_indices])
    
    assert max_v_outside < 1e-4, f"3D Velocity non-zero outside mask: {max_v_outside}"
