
import os
import sys
import numpy as np
import tempfile
import pytest

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.export_utils import save_vtk_frame

def test_save_vtk_frame_basic():
    """Test if save_vtk_frame creates a file with correct header and dimensions."""
    
    # Setup dummy data
    N = 10
    x_coords = N
    y_coords = N
    
    # Create fields
    density = np.zeros((N, N), dtype=np.float32)
    density[5, 5] = 1.0 # arbitrary value
    
    vx = np.ones((N, N), dtype=np.float32)
    vy = np.zeros((N, N), dtype=np.float32)
    
    data_arrays = {'density': density}
    vector_arrays = {'velocity': (vx, vy)}
    
    # Use a temporary file
    with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as tmp:
        tmp_filename = tmp.name
        
    try:
        save_vtk_frame(
            tmp_filename, 
            x_coords, 
            y_coords, 
            data_arrays=data_arrays, 
            vector_arrays=vector_arrays
        )
        
        # Check if file exists
        assert os.path.exists(tmp_filename)
        assert os.path.getsize(tmp_filename) > 0
        
        # Read content and check header/structure
        with open(tmp_filename, 'r') as f:
            lines = f.readlines()
            
        # Basic sanity checks on VTK format
        assert lines[0].strip() == "# vtk DataFile Version 3.0"
        assert "DATASET STRUCTURED_POINTS" in lines[3]
        assert lines[4].strip() == f"DIMENSIONS {N} {N} 1"
        
        # Check for SCALARS
        scalar_line_idx = -1
        for i, line in enumerate(lines):
            if "SCALARS density float 1" in line:
                scalar_line_idx = i
                break
        assert scalar_line_idx != -1, "Scalar header not found"
        
        # Check for VECTORS
        vector_line_idx = -1
        for i, line in enumerate(lines):
            if "VECTORS velocity float" in line:
                vector_line_idx = i
                break
        assert vector_line_idx != -1, "Vector header not found"
        
    finally:
        # Cleanup
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
