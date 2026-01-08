import warp as wp
import numpy as np
import pytest
import os
import sys

# Assume project root is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from fluid_engine import FluidOptimizer

def test_ngrid_configurability():
    wp.init()
    
    # Test 1: Standard
    n_grid_1 = 32
    sim1 = FluidOptimizer(dim=2, n_grid=n_grid_1, sim_steps=5)
    assert sim1.n_grid == 32
    assert sim1.shape == (32, 32)
    assert np.isclose(sim1.dh, 1.0/32)
    
    # Run a few steps
    sim1.forward()
    assert sim1.vx_arrays[0].shape == (32, 32)
    
    # Test 2: Different size
    n_grid_2 = 16
    sim2 = FluidOptimizer(dim=2, n_grid=n_grid_2, sim_steps=5)
    assert sim2.n_grid == 16
    assert sim2.shape == (16, 16)
    assert np.isclose(sim2.dh, 1.0/16)
    
    sim2.forward()
    assert sim2.vx_arrays[0].shape == (16, 16)
    
def test_ngrid_3d():
    wp.init()
    n_grid = 10 # Small for speed
    sim = FluidOptimizer(dim=3, n_grid=n_grid, sim_steps=2)
    assert sim.n_grid == 10
    assert sim.shape == (10, 10, 10)
    
    sim.forward()
    assert sim.vx_arrays[0].shape == (10, 10, 10)
