import os
import numpy as np

def save_vtk_frame(filename, x_coords, y_coords, data_arrays, vector_arrays=None):
    """
    Export a 2D grid frame to Legacy VTK format (.vtk) for ParaView.
    
    Args:
        filename (str): Output filename (e.g., 'frame_001.vtk').
        x_coords (int or np.array): Number of grid points in X or array of coords. 
                                    Currently supports uniform grid defined by number of points (N_GRID).
        y_coords (int): Number of grid points in Y.
        data_arrays (dict): Dictionary of scalar fields {'name': numpy_array_2d}. 
                            Array shape should be (N_GRID, N_GRID) where arr[i,j] is (x=i, y=j).
        vector_arrays (dict): Dictionary of vector fields {'name': (vx_array, vy_array)}.
    """
    
    # Assuming uniform grid for now based on the fluid engine
    # Inputs are usually just N_GRID
    if isinstance(x_coords, int):
        nx = x_coords
    else:
        nx = len(x_coords)
        
    if isinstance(y_coords, int):
        ny = y_coords
    else:
        ny = len(y_coords)
        
    nz = 1
    
    # Total points
    n_points = nx * ny * nz
    
    with open(filename, 'w') as f:
        # Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Fluid Simulation Frame {os.path.basename(filename)}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write("SPACING 1 1 1\n") # We can adjust spacing if needed, but for viz 1.0 is fine or DH
        
        f.write(f"POINT_DATA {n_points}\n")
        
        # Write Scalars
        if data_arrays:
            for name, data in data_arrays.items():
                f.write(f"SCALARS {name} float 1\n")
                f.write("LOOKUP_TABLE default\n")
                
                flat_data = data.flatten(order='F')
                np.savetxt(f, flat_data, fmt='%.6f', newline=' ')
                f.write("\n")
                
        # Write Vectors
        if vector_arrays:
            for name, (vx, vy) in vector_arrays.items():
                f.write(f"VECTORS {name} float\n")
                
                flat_vx = vx.flatten(order='F')
                flat_vy = vy.flatten(order='F')
                flat_vz = np.zeros_like(flat_vx)
                
                # We need to interleave them: vx0 vy0 vz0 vx1 vy1 vz1 ...
                # shape (N, 3)
                vectors = np.column_stack((flat_vx, flat_vy, flat_vz))
                np.savetxt(f, vectors, fmt='%.6f', newline='\n')


def export_simulation_sequence(output_dir, prefix, density_list, vx_list, vy_list):
    """
    Export a full sequence of frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_steps = len(density_list)
    
    # Assume all have same shape
    nx, ny = density_list[0].shape
    
    print(f"Exporting {num_steps} frames to {output_dir}...")
    
    for t in range(num_steps):
        filename = os.path.join(output_dir, f"{prefix}_{t:03d}.vtk")
        
        # Convert warp arrays to numpy if they aren't already
        # We handle this by checking type
        d = density_list[t]
        u = vx_list[t]
        v = vy_list[t]
        
        # Helper to get numpy
        if hasattr(d, 'numpy'): d = d.numpy()
        if hasattr(u, 'numpy'): u = u.numpy()
        if hasattr(v, 'numpy'): v = v.numpy()
            
        save_vtk_frame(
            filename,
            nx, ny,
            data_arrays={"density": d},
            vector_arrays={"velocity": (u, v)}
        )
    print("Export complete.")
