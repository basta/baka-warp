import os
import numpy as np

def save_vtk_frame(filename, x_coords, y_coords, z_coords, data_arrays, vector_arrays=None):
    """
    Export a 2D or 3D grid frame to Legacy VTK format (.vtk) for ParaView.
    
    Args:
        filename (str): Output filename (e.g., 'frame_001.vtk').
        x_coords (int or np.array): Number of grid points in X or array of coords.
        y_coords (int): Number of grid points in Y.
        z_coords (int): Number of grid points in Z.
        data_arrays (dict): Dictionary of scalar fields {'name': numpy_array}. 
                            Array shape should be (N, N) for 2D or (N, N, N) for 3D.
        vector_arrays (dict): Dictionary of vector fields {'name': (vx, vy)} or {'name': (vx, vy, vz)}.
    """
    
    if isinstance(x_coords, int):
        nx = x_coords
    else:
        nx = len(x_coords)
        
    if isinstance(y_coords, int):
        ny = y_coords
    else:
        ny = len(y_coords)

    if isinstance(z_coords, int):
        nz = z_coords
    else:
        nz = len(z_coords)
    
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
                
                # Flatten order F is column-major (Fortran style), which matches 
                # how VTK expects structured points if we treat indices as (x, y, z)
                # with x changing fastest? 
                # Actually VTK STRUCTURED_POINTS usually iterates x then y then z.
                # If our array is [x, y, z], then flatten('F') effectively iterates x first.
                flat_data = data.flatten(order='F')
                np.savetxt(f, flat_data, fmt='%.6f', newline=' ')
                f.write("\n")
                
        # Write Vectors
        if vector_arrays:
            for name, components in vector_arrays.items():
                f.write(f"VECTORS {name} float\n")
                
                vx = components[0]
                vy = components[1]
                if len(components) > 2:
                    vz = components[2]
                else:
                    vz = np.zeros_like(vx)
                
                flat_vx = vx.flatten(order='F')
                flat_vy = vy.flatten(order='F')
                flat_vz = vz.flatten(order='F')
                
                # We need to interleave them: vx0 vy0 vz0 vx1 vy1 vz1 ...
                # shape (N, 3)
                vectors = np.column_stack((flat_vx, flat_vy, flat_vz))
                np.savetxt(f, vectors, fmt='%.6f', newline='\n')


def export_simulation_sequence(output_dir, prefix, density_list, vx_list, vy_list, vz_list=None, extra_scalar_fields=None, extra_vector_fields=None):
    """
    Export a full sequence of frames.
    
    Args:
        vz_list (list): Optional list of z-velocity frames.
        extra_scalar_fields (dict): Dict of name -> list of Frames OR single Frame (if static)
        extra_vector_fields (dict): Dict of name -> list of (u, v) or (u, v, w) tuples OR single tuple (if static)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_steps = len(density_list)
    
    # Check shape
    d0 = density_list[0]
    if hasattr(d0, 'shape'):
        shape = d0.shape
    else:
        # fallback if scalar (unlikely for density field)
        shape = (1, 1)

    nx = shape[0]
    ny = shape[1]
    nz = 1
    if len(shape) > 2:
        nz = shape[2]
    
    print(f"Exporting {num_steps} frames to {output_dir} (Grid: {nx}x{ny}x{nz})...")
    
    for t in range(num_steps):
        filename = os.path.join(output_dir, f"{prefix}_{t:03d}.vtk")
        
        # Convert warp arrays to numpy if they aren't already
        # We handle this by checking type
        d = density_list[t]
        u = vx_list[t]
        v = vy_list[t]
        
        if hasattr(d, 'numpy'): d = d.numpy()
        if hasattr(u, 'numpy'): u = u.numpy()
        if hasattr(v, 'numpy'): v = v.numpy()
        
        # Prepare data dicts
        data_arrays = {"density": d}
        
        if vz_list is not None:
            w = vz_list[t]
            if hasattr(w, 'numpy'): w = w.numpy()
            vector_arrays = {"velocity": (u, v, w)}
        else:
            vector_arrays = {"velocity": (u, v)}
        
        # Add extra scalars
        if extra_scalar_fields:
            for name, val in extra_scalar_fields.items():
                if isinstance(val, list) and len(val) == num_steps:
                    item = val[t]
                else:
                    item = val # Assume static
                    
                if hasattr(item, 'numpy'): item = item.numpy()
                data_arrays[name] = item
                
        # Add extra vectors
        if extra_vector_fields:
            for name, val in extra_vector_fields.items():
                if isinstance(val, list) and len(val) == num_steps:
                    comps = val[t]
                else:
                    comps = val # Assume static
                
                # comps is tuple (vx, vy) or (vx, vy, vz)
                comps_np = []
                for c in comps:
                    if hasattr(c, 'numpy'):
                        comps_np.append(c.numpy())
                    else:
                        comps_np.append(c)
                
                vector_arrays[name] = tuple(comps_np)
                
        save_vtk_frame(
            filename,
            nx, ny, nz,
            data_arrays=data_arrays,
            vector_arrays=vector_arrays
        )
    print("Export complete.")
