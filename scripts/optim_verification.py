
import sys
import os
import math
import numpy as np
import warp as wp
import matplotlib.pyplot as plt

# Add project root to path to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.fluid_engine import FluidOptimizer, initialize_fields, N_GRID

def main():
    # Setup
    wp.init()
    device = wp.get_device()
    print(f"Running on device: {device}")

    # Reduce horizon for stability
    num_steps = 15
    num_bases = 8
    
    # Create Optimizer/Sim
    sim = FluidOptimizer(num_basis_fields=num_bases, sim_steps=num_steps, device=device)
    
    # -------------------------------------------------------------------------
    # 1. Generate a "Target" by running with secret weights
    # -------------------------------------------------------------------------
    print("Generating target trajectory...")
    true_weights_np = np.random.uniform(-0.1, 0.1, size=(8,)).astype(np.float32)
    wp.copy(sim.weights, wp.array(true_weights_np, dtype=float, device=device))
    
    # Init state
    wp.launch(initialize_fields, sim.shape, inputs=[sim.density_arrays[0], sim.vx_arrays[0], sim.vy_arrays[0], sim.n_grid, sim.dh])
    
    # Run forward manually to capture frames
    for t in range(sim.sim_steps):
        sim.run_step(t)
        # Copy THIS step's result to the target buffer
        wp.copy(sim.target_vx_arrays[t], sim.vx_arrays[t+1])
        wp.copy(sim.target_vy_arrays[t], sim.vy_arrays[t+1])
        
    print(f"Target generated.")
    
    # -------------------------------------------------------------------------
    # 2. Reset and Optimize
    # -------------------------------------------------------------------------
    print("\nStarting Optimization...")
    
    # Reset weights to zero
    sim.weights.zero_()
    # Lower LR
    sim.optimizer = wp.optim.Adam([sim.weights], lr=0.001)
    
    # Training Loop
    iterations = 200
    for i in range(iterations):
        # Reset State for training
        wp.launch(initialize_fields, sim.shape, inputs=[sim.density_arrays[0], sim.vx_arrays[0], sim.vy_arrays[0], sim.n_grid, sim.dh])
        
        loss_val, grad_norm = sim.step_optimization()
        
        if math.isnan(grad_norm):
            print("Optimization unstable.")
            break
            
        if i % 10 == 0:
            print(f"Iter {i:03d} | Loss: {loss_val:.6f} | Grad: {grad_norm:.4f}")

    print("\nOptimization Complete.")
    final_weights = sim.weights.numpy()
    print(f"True Weights:  {true_weights_np}")
    print(f"Found Weights: {final_weights}")
    
    # verify
    # Run one last updated forward
    wp.launch(initialize_fields, sim.shape, inputs=[sim.density_arrays[0], sim.vx_arrays[0], sim.vy_arrays[0], sim.n_grid, sim.dh])
    sim.forward()
    v_final_x = sim.vx_arrays[-1].numpy()
    v_final_y = sim.vy_arrays[-1].numpy()
    v_final_norm = np.sqrt(v_final_x**2 + v_final_y**2)
    
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 2)
        plt.title("Optimized Velocity Magnitude")
        plt.imshow(v_final_norm, origin="lower")
        # Save to experiment_result.png in current directory
        plt.savefig("experiment_result.png")
        print("Result saved to experiment_result.png")
    except Exception as e:
        print(f"Failed to save plot: {e}")

if __name__ == "__main__":
    main()
