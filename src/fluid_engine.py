import math
import os
import sys

import numpy as np

import warp as wp
import warp.optim
from enum import IntEnum

# Grid dimensions
N_GRID = 64
DH = 1.0 / N_GRID

# Simulation parameters
DT = 0.005  # Reduced further
DENSITY_INIT_RADIUS = 0.1
CENTER_X = 0.5
CENTER_Y = 0.5

class BoundaryCondition(IntEnum):
    PERIODIC = 0
    WALL = 1

BC_PERIODIC = 0
BC_WALL = 1


@wp.kernel
def init_mask_circle(
    mask: wp.array2d(dtype=int),
    radius: float,
    center_x: float,
    center_y: float,
    dh: float
):
    i, j = wp.tid()
    x = (float(i) + 0.5) * dh
    y = (float(j) + 0.5) * dh
    
    dx = x - center_x
    dy = y - center_y
    dist_sq = dx*dx + dy*dy
    
    if dist_sq <= radius * radius:
        mask[i, j] = 1 # Fluid
    else:
        mask[i, j] = 0 # Solid

@wp.kernel
def init_mask_sphere(
    mask: wp.array3d(dtype=int),
    radius: float,
    center_x: float,
    center_y: float,
    center_z: float,
    dh: float
):
    i, j, k = wp.tid()
    x = (float(i) + 0.5) * dh
    y = (float(j) + 0.5) * dh
    z = (float(k) + 0.5) * dh
    
    dx = x - center_x
    dy = y - center_y
    dz = z - center_z
    dist_sq = dx*dx + dy*dy + dz*dz
    
    if dist_sq <= radius * radius:
        mask[i, j, k] = 1 # Fluid
    else:
        mask[i, j, k] = 0 # Solid

@wp.func
def cyclic_index(idx: wp.int32, dim_size: int):
    """Helper function to index with periodic boundary conditions."""
    ret_idx = idx % dim_size
    if ret_idx < 0:
        ret_idx += dim_size
    return ret_idx


@wp.kernel
def initialize_fields(
    density: wp.array2d(dtype=float),
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    dh: float
):
    i, j = wp.tid()
    
    # Initialize density as a circle in the center
    x = (float(i) + 0.5) * dh
    y = (float(j) + 0.5) * dh
    
    dx = x - CENTER_X
    dy = y - CENTER_Y
    dist_sq = dx*dx + dy*dy
    
    if dist_sq < DENSITY_INIT_RADIUS * DENSITY_INIT_RADIUS:
        density[i, j] = 1.0
    else:
        density[i, j] = 0.0

    u[i, j] = 0.0
    v[i, j] = 0.0


@wp.kernel
def apply_forces(
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
    force_basis_x: wp.array3d(dtype=float),
    force_basis_y: wp.array3d(dtype=float),
    weights: wp.array(dtype=float),
    dt: float
):
    """
    Apply weighted sum of basis force fields to the velocity field.
    vx += dt * sum(w_k * F_k_x)
    """
    i, j = wp.tid()
    
    fx_sum = float(0.0)
    fy_sum = float(0.0)
    
    num_bases = force_basis_x.shape[0]
    
    for k in range(num_bases):
        w = weights[k]
        fx_sum += w * force_basis_x[k, i, j]
        fy_sum += w * force_basis_y[k, i, j]
        
    vx[i, j] = vx[i, j] + fx_sum * dt
    vy[i, j] = vy[i, j] + fy_sum * dt


@wp.func
def sample_field(field: wp.array2d(dtype=float), x: float, y: float, nx: int, ny: int):
    # Bilinear interpolation
    # field is defined at integer indices 0..N-1
    # clamp coords
    x = wp.max(0.0, wp.min(x, float(nx) - 1.0))
    y = wp.max(0.0, wp.min(y, float(ny) - 1.0))

    x0 = wp.int32(wp.floor(x))
    y0 = wp.int32(wp.floor(y))
    
    wx1 = x - float(x0)
    wy1 = y - float(y0)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    
    val = (wx0 * wy0 * field[x0, y0] + 
           wx1 * wy0 * field[x1, y0] + 
           wx0 * wy1 * field[x0, y1] + 
           wx1 * wy1 * field[x1, y1])
    return val

@wp.kernel
def advect_mac_u(
    dt: float,
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    u_new: wp.array2d(dtype=float),
    nx: int,
    ny: int,
    dh: float
):
    # Advect u component. u lives at (i, j+0.5):
    # Convention: u[i, j] is flow across CONSTANT-X face between cell (i-1, j) and (i, j).
    # Coordinate: x = i * DH, y = (j + 0.5) * DH.
    
    i, j = wp.tid()
    
    # World pos of this u-face
    x = float(i) * dh
    y = (float(j) + 0.5) * dh
    
    # Velocity at this point
    # u is exact at this point
    vel_x = u[i, j]
    
    # v needs interpolation. v lives at (x+0.5, y-0.5).
    # We want v at (i, j+0.5)
    # v grid coords: x_v = i - 0.5, y_v = j + 0.5
    # To get world (i, j+0.5), we sample v at grid coord (i-0.5, j+0.5).
    grid_v_x = float(i) - 0.5
    grid_v_y = float(j) + 0.5
    vel_y = sample_field(v, grid_v_x, grid_v_y, nx, ny)
    
    # Backtrace
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample u at src. u lives at grid (i, j+0.5).
    # grid u coords:
    grid_src_x = src_x / dh
    grid_src_y = (src_y / dh) - 0.5
    
    u_new[i, j] = sample_field(u, grid_src_x, grid_src_y, nx, ny)


@wp.kernel
def advect_mac_v(
    dt: float,
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    v_new: wp.array2d(dtype=float),
    nx: int,
    ny: int,
    dh: float
):
    # Advect v component. v lives at (i+0.5, j).
    # Convention: v[i, j] is flow across CONSTANT-Y face between cell (i, j-1) and (i, j).
    
    i, j = wp.tid()
    
    # World pos of this v-face
    x = (float(i) + 0.5) * dh
    y = float(j) * dh
    
    # Velocity
    vel_y = v[i, j]
    
    # u needs interpolation. u lives at (i, j+0.5).
    # We want u at (i+0.5, j).
    # u grid coords: x_u = i + 0.5, y_u = j - 0.5
    grid_u_x = float(i) + 0.5
    grid_u_y = float(j) - 0.5
    vel_x = sample_field(u, grid_u_x, grid_u_y, nx, ny)
    
    # Backtrace
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample v at src. v lives at grid (i+0.5, j).
    grid_src_x = (src_x / dh) - 0.5
    grid_src_y = (src_y / dh)
    
    v_new[i, j] = sample_field(v, grid_src_x, grid_src_y, nx, ny)


@wp.kernel
def advect_density(
    dt: float,
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    rho_old: wp.array2d(dtype=float),
    rho_new: wp.array2d(dtype=float),
    bc_type: int,
    nx: int,
    ny: int,
    dh: float
):
    # Advect scalar at cell centers (i+0.5, j+0.5)
    i, j = wp.tid()
    
    # World pos
    x = (float(i) + 0.5) * dh
    y = (float(j) + 0.5) * dh
    
    # Velocity at center
    # u at (i, j+0.5) and (i+1, j+0.5). Avg to get center.
    
    u_idx = u[i, j]
    u_idx1 = float(0.0)
    
    v_idx = v[i, j]
    v_idx1 = float(0.0)
    
    if bc_type == BC_PERIODIC:
        u_idx1 = u[cyclic_index(i + 1, nx), j]
        v_idx1 = v[i, cyclic_index(j + 1, ny)]
    else:
        # BC_WALL
        if i == nx - 1:
            u_idx1 = 0.0
        else:
            u_idx1 = u[i + 1, j]
            
        if j == ny - 1:
            v_idx1 = 0.0
        else:
            v_idx1 = v[i, j + 1]
    
    vel_x = (u_idx + u_idx1) * 0.5
    vel_y = (v_idx + v_idx1) * 0.5
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample rho (centered)
    grid_src_x = (src_x / dh) - 0.5
    grid_src_y = (src_y / dh) - 0.5
    
    rho_new[i, j] = sample_field(rho_old, grid_src_x, grid_src_y, nx, ny)


@wp.kernel
def divergence(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), mask: wp.array2d(dtype=int), div: wp.array2d(dtype=float), nx: int, ny: int, dh: float):
    """Compute div(u) at cell centers."""
    i, j = wp.tid()
    
    if mask[i, j] == 0:
        div[i, j] = 0.0
        return

    # u[i, j] is left face, u[i+1, j] is right face
    # v[i, j] is bottom face, v[i, j+1] is top face
    
    u_right = float(0.0)
    u_left = float(0.0)
    v_top = float(0.0)
    v_bot = float(0.0)

    
    # Left face
    u_left = u[i, j]
    
    # Right face
    if i == nx - 1:
        u_right = 0.0
    else:
        u_right = u[i + 1, j]
        
    # Bottom face
    v_bot = v[i, j]
    
    # Top face
    if j == ny - 1:
        v_top = 0.0
    else:
        v_top = v[i, j + 1]

    div[i, j] = (u_right - u_left + v_top - v_bot) / dh


@wp.kernel
def jacobi_iter(div: wp.array2d(dtype=float), p0: wp.array2d(dtype=float), mask: wp.array2d(dtype=int), p1: wp.array2d(dtype=float), nx: int, ny: int, dh: float):
    """Solve Laplacian P = div with mask support."""
    i, j = wp.tid()
    
    if mask[i, j] == 0:
        p1[i, j] = 0.0
        return
    
    # Standard 5-point Laplacian
    # For solid neighbors, we use Neumann BC: dP/dn = 0 => P_neighbor = P_center
    
    val_left = float(0.0)
    val_right = float(0.0)
    val_down = float(0.0)
    val_up = float(0.0)
    
    # Left (i-1)
    if i == 0:
        val_left = p0[i, j]
    elif mask[i-1, j] == 0:
        val_left = p0[i, j] # Solid neighbor -> Neumann
    else:
        val_left = p0[i-1, j]
        
    # Right (i+1)
    if i == nx - 1:
        val_right = p0[i, j]
    elif mask[i+1, j] == 0:
        val_right = p0[i, j]
    else:
        val_right = p0[i+1, j]
        
    # Down (j-1)
    if j == 0:
        val_down = p0[i, j]
    elif mask[i, j-1] == 0:
        val_down = p0[i, j]
    else:
        val_down = p0[i, j-1]
        
    # Up (j+1)
    if j == ny - 1:
        val_up = p0[i, j]
    elif mask[i, j+1] == 0:
        val_up = p0[i, j]
    else:
        val_up = p0[i, j+1]

    sum_neighbors = val_left + val_right + val_down + val_up
                     
    p1[i, j] = 0.25 * (sum_neighbors - div[i, j] * dh * dh)


@wp.kernel
def update_velocities(
    p: wp.array2d(dtype=float),
    u_in: wp.array2d(dtype=float),
    v_in: wp.array2d(dtype=float),
    u_out: wp.array2d(dtype=float),
    v_out: wp.array2d(dtype=float),
    bc_type: int,
    nx: int,
    ny: int,
    dh: float
):
    """Subtract pressure gradient."""
    i, j = wp.tid()

    # Update u at (i, j+0.5) (left face of cell i,j)
    # Grad P x component at face: (P[i,j] - P[i-1,j]) / DH
    
    if bc_type == BC_PERIODIC:
        grad_p_x = (p[i, j] - p[cyclic_index(i - 1, nx), j]) / dh
        u_out[i, j] = u_in[i, j] - grad_p_x
        
        grad_p_y = (p[i, j] - p[i, cyclic_index(j - 1, ny)]) / dh
        v_out[i, j] = v_in[i, j] - grad_p_y
    else:
        # BC_WALL
        # For u (x-velocity on vertical faces)
        # u[i, j] is on left face of cell i.
        # If i=0 (left domain boundary), u MUST be 0.
        if i == 0:
            u_out[i, j] = 0.0
        else:
            grad_p_x = (p[i, j] - p[i - 1, j]) / dh
            u_out[i, j] = u_in[i, j] - grad_p_x
            
        # For v (y-velocity on horizontal faces)
        # v[i, j] is on bottom face of cell j.
        # If j=0 (bottom domain boundary), v MUST be 0.
        if j == 0:
            v_out[i, j] = 0.0
        else:
            grad_p_y = (p[i, j] - p[i, j - 1]) / dh
            v_out[i, j] = v_in[i, j] - grad_p_y


@wp.kernel
def compute_velocity_loss(
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
    target_vx: wp.array2d(dtype=float),
    target_vy: wp.array2d(dtype=float),
    loss: wp.array(dtype=float),
    nx: int,
    ny: int
):
    i, j = wp.tid()
    
    diff_x = vx[i, j] - target_vx[i, j]
    diff_y = vy[i, j] - target_vy[i, j]
    
    # Mean squared error
    val = (diff_x * diff_x + diff_y * diff_y) / float(nx * ny)
    
    wp.atomic_add(loss, 0, val)

@wp.kernel
def enforce_noslip_velocity(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    nx: int,
    ny: int
):
    """Explicitly zero out boundaries for wall BC."""
    i, j = wp.tid()
    
    # Left/Right walls (x=0, x=N)
    if i == 0:
        u[i, j] = 0.0 # No penetration Left
        v[i, j] = 0.0 # No slip Left
        
    if i == nx - 1:
        v[i, j] = 0.0 # No slip Right
        # u[N-1] is interior face, u[N] is wall. 
        # But for 'box' behavior, zeroing tangential near wall is good.
    
    # Bottom/Top walls (y=0, y=N)
    if j == 0:
        v[i, j] = 0.0 # No penetration Bottom
        u[i, j] = 0.0 # No slip Bottom
        
    if j == ny - 1:
        u[i, j] = 0.0 # No slip Top

@wp.kernel
def enforce_mask_velocity(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    mask: wp.array2d(dtype=int),
    nx: int,
    ny: int
):
    i, j = wp.tid()
    
    # Check if cell is solid
    if mask[i, j] == 0:
        
        # Left face
        u[i, j] = 0.0
        # Right face
        if i < nx - 1:
             u[i+1, j] = 0.0
        
        # Bottom face
        v[i, j] = 0.0
        # Top face
        if j < ny - 1:
            v[i, j+1] = 0.0
            
    # Also enforce domain boundaries
    if i == 0: u[i, j] = 0.0
    if i == nx - 1: u[i+1, j] = 0.0 # Wait, array size is (N, N). u has shape (N,N)? 

    # if mask[i, j] == 0, zero its own stored faces (Left/Bottom)
    if mask[i, j] == 0:
        u[i, j] = 0.0
        v[i, j] = 0.0
        
    # Check neighbors to zero shared faces
    # u[i,j] is shared with i-1. 
    if i > 0:
        if mask[i-1, j] == 0:
            u[i, j] = 0.0
            
    # v[i,j] is shared with j-1
    if j > 0:
        if mask[i, j-1] == 0:
            v[i, j] = 0.0
            
    # Domain boundaries
    if i == 0:
        u[i, j] = 0.0
    if j == 0:
        v[i, j] = 0.0



CENTER_Z = 0.5 

@wp.kernel
def initialize_fields_3d(
    density: wp.array3d(dtype=float),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    dh: float
):
    i, j, k = wp.tid()
    
    # Initialize density as a sphere in the center
    x = (float(i) + 0.5) * dh
    y = (float(j) + 0.5) * dh
    z = (float(k) + 0.5) * dh
    
    dx = x - CENTER_X
    dy = y - CENTER_Y
    dz = z - CENTER_Z
    dist_sq = dx*dx + dy*dy + dz*dz
    
    if dist_sq < DENSITY_INIT_RADIUS * DENSITY_INIT_RADIUS:
        density[i, j, k] = 1.0
    else:
        density[i, j, k] = 0.0

    u[i, j, k] = 0.0
    v[i, j, k] = 0.0
    w[i, j, k] = 0.0

@wp.kernel
def apply_forces_3d(
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    force_basis_x: wp.array4d(dtype=float),
    force_basis_y: wp.array4d(dtype=float),
    force_basis_z: wp.array4d(dtype=float),
    weights: wp.array(dtype=float),
    dt: float
):
    i, j, k = wp.tid()
    
    fx_sum = float(0.0)
    fy_sum = float(0.0)
    fz_sum = float(0.0)
    
    num_bases = force_basis_x.shape[0]
    
    for b in range(num_bases):
        w = weights[b]
        fx_sum += w * force_basis_x[b, i, j, k]
        fy_sum += w * force_basis_y[b, i, j, k]
        fz_sum += w * force_basis_z[b, i, j, k]
        
    vx[i, j, k] = vx[i, j, k] + fx_sum * dt
    vy[i, j, k] = vy[i, j, k] + fy_sum * dt
    vz[i, j, k] = vz[i, j, k] + fz_sum * dt

@wp.func
def sample_field_3d(field: wp.array3d(dtype=float), x: float, y: float, z: float, nx: int, ny: int, nz: int):
    # Trilinear interpolation
    x = wp.max(0.0, wp.min(x, float(nx) - 1.0))
    y = wp.max(0.0, wp.min(y, float(ny) - 1.0))
    z = wp.max(0.0, wp.min(z, float(nz) - 1.0))

    x0 = wp.int32(wp.floor(x))
    y0 = wp.int32(wp.floor(y))
    z0 = wp.int32(wp.floor(z))
    
    wx1 = x - float(x0)
    wy1 = y - float(y0)
    wz1 = z - float(z0)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1
    
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)
    
    # 8 corners
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]

    # Interpolate x
    c00 = c000*wx0 + c100*wx1
    c10 = c010*wx0 + c110*wx1
    c01 = c001*wx0 + c101*wx1
    c11 = c011*wx0 + c111*wx1
    
    # Interpolate y
    c0 = c00*wy0 + c10*wy1
    c1 = c01*wy0 + c11*wy1
    
    # Interpolate z
    val = c0*wz0 + c1*wz1
    return val

@wp.kernel
def advect_mac_u_3d(
    dt: float,
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    u_new: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float
):
    # u at (i, j+0.5, k+0.5)
    
    i, j, k = wp.tid()
    
    x = float(i) * dh
    y = (float(j) + 0.5) * dh
    z = (float(k) + 0.5) * dh
    
    vel_x = u[i, j, k]
    
    # v needs sample at x,y,z
    # v is at (i+0.5, j, k+0.5).
    # u is at (i, j+0.5, k+0.5).
    # Coords of u node: (i, j+0.5, k+0.5)
    # v grid coords: x_v = i - 0.5, y_v = j + 0.5, z_v = k
    grid_v_x = float(i) - 0.5
    grid_v_y = float(j) + 0.5
    grid_v_z = float(k)
    vel_y = sample_field_3d(v, grid_v_x, grid_v_y, grid_v_z, nx, ny, nz)
    
    # w is at (i+0.5, j+0.5, k).
    # w grid coords: x_w = i - 0.5, y_w = j, z_w = k + 0.5
    grid_w_x = float(i) - 0.5
    grid_w_y = float(j)
    grid_w_z = float(k) + 0.5
    vel_z = sample_field_3d(w, grid_w_x, grid_w_y, grid_w_z, nx, ny, nz)
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    src_z = z - vel_z * dt
    
    # Sample u at src
    grid_src_x = src_x / dh
    grid_src_y = (src_y / dh) - 0.5
    grid_src_z = (src_z / dh) - 0.5
    
    u_new[i, j, k] = sample_field_3d(u, grid_src_x, grid_src_y, grid_src_z, nx, ny, nz)

@wp.kernel
def advect_mac_v_3d(
    dt: float,
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    v_new: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float
):
    # v at (i+0.5, j, k+0.5)
    
    i, j, k = wp.tid()
    
    x = (float(i) + 0.5) * dh
    y = float(j) * dh
    z = (float(k) + 0.5) * dh
    
    vel_y = v[i, j, k]
    
    # u sample. u is at (i, j+0.5, k+0.5)
    # grid_u = (i+0.5, j-0.5, k)
    grid_u_x = float(i) + 0.5
    grid_u_y = float(j) - 0.5
    grid_u_z = float(k)
    vel_x = sample_field_3d(u, grid_u_x, grid_u_y, grid_u_z, nx, ny, nz)
    
    # w sample. w is at (i+0.5, j+0.5, k)
    # grid_w = (i, j-0.5, k+0.5)
    grid_w_x = float(i)
    grid_w_y = float(j) - 0.5
    grid_w_z = float(k) + 0.5
    vel_z = sample_field_3d(w, grid_w_x, grid_w_y, grid_w_z, nx, ny, nz)
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    src_z = z - vel_z * dt
    
    grid_src_x = (src_x / dh) - 0.5
    grid_src_y = src_y / dh
    grid_src_z = (src_z / dh) - 0.5
    
    v_new[i, j, k] = sample_field_3d(v, grid_src_x, grid_src_y, grid_src_z, nx, ny, nz)

@wp.kernel
def advect_mac_w_3d(
    dt: float,
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    w_new: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int,
    dh: float
):
    # w at (i+0.5, j+0.5, k)
    
    i, j, k = wp.tid()
    
    x = (float(i) + 0.5) * dh
    y = (float(j) + 0.5) * dh
    z = float(k) * dh
    
    vel_z = w[i, j, k]
    
    # u sample. u is at (i, j+0.5, k+0.5)
    # grid_u = (i+0.5, j, k-0.5)
    grid_u_x = float(i) + 0.5
    grid_u_y = float(j)
    grid_u_z = float(k) - 0.5
    vel_x = sample_field_3d(u, grid_u_x, grid_u_y, grid_u_z, nx, ny, nz)
    
    # v sample. v is at (i+0.5, j, k+0.5)
    # grid_v = (i, j+0.5, k-0.5)
    grid_v_x = float(i)
    grid_v_y = float(j) + 0.5
    grid_v_z = float(k) - 0.5
    vel_y = sample_field_3d(v, grid_v_x, grid_v_y, grid_v_z, nx, ny, nz)
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    src_z = z - vel_z * dt
    
    grid_src_x = (src_x / dh) - 0.5
    grid_src_y = (src_y / dh) - 0.5
    grid_src_z = src_z / dh
    
    w_new[i, j, k] = sample_field_3d(w, grid_src_x, grid_src_y, grid_src_z, nx, ny, nz)

@wp.kernel
def advect_density_3d(
    dt: float,
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    rho_old: wp.array3d(dtype=float),
    rho_new: wp.array3d(dtype=float),
    bc_type: int,
    nx: int,
    ny: int,
    nz: int,
    dh: float
):
    i, j, k = wp.tid()
    
    # Cell center x,y,z
    x = (float(i) + 0.5) * dh
    y = (float(j) + 0.5) * dh
    z = (float(k) + 0.5) * dh
    
    u_c = float(0.0)
    v_c = float(0.0)
    w_c = float(0.0)
    
    if bc_type == BC_PERIODIC:
        u_c = (u[i, j, k] + u[cyclic_index(i+1, nx), j, k]) * 0.5
        v_c = (v[i, j, k] + v[i, cyclic_index(j+1, ny), k]) * 0.5
        w_c = (w[i, j, k] + w[i, j, cyclic_index(k+1, nz)]) * 0.5
    else:
        # BC_WALL
        # Check bounds
        u1 = float(0.0)
        if i == nx-1:
            u1 = 0.0 
        else:
            u1 = u[i+1, j, k]
            
        v1 = float(0.0)
        if j == ny-1:
            v1 = 0.0
        else:
            v1 = v[i, j+1, k]
            
        w1 = float(0.0)
        if k == nz-1:
            w1 = 0.0 
        else:
            w1 = w[i, j, k+1]
        
        u_c = (u[i, j, k] + u1) * 0.5
        v_c = (v[i, j, k] + v1) * 0.5
        w_c = (w[i, j, k] + w1) * 0.5
        
    src_x = x - u_c * dt
    src_y = y - v_c * dt
    src_z = z - w_c * dt
    
    grid_src_x = (src_x / dh) - 0.5
    grid_src_y = (src_y / dh) - 0.5
    grid_src_z = (src_z / dh) - 0.5
    
    rho_new[i, j, k] = sample_field_3d(rho_old, grid_src_x, grid_src_y, grid_src_z, nx, ny, nz)

@wp.kernel
def divergence_3d(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), mask: wp.array3d(dtype=int), div: wp.array3d(dtype=float), nx: int, ny: int, nz: int, dh: float):
    i, j, k = wp.tid()
    
    if mask[i, j, k] == 0:
        div[i, j, k] = 0.0
        return
        
    u_r = float(0.0)
    u_l = u[i, j, k]
    v_t = float(0.0)
    v_b = v[i, j, k]
    w_f = float(0.0)
    w_k = w[i, j, k]
    
    if i == nx-1:
        u_r = 0.0 
    else:
        u_r = u[i+1, j, k]
        
    if j == ny-1:
        v_t = 0.0 
    else:
        v_t = v[i, j+1, k]
        
    if k == nz-1:
        w_f = 0.0 
    else:
        w_f = w[i, j, k+1]
    
    div[i, j, k] = (u_r - u_l + v_t - v_b + w_f - w_k) / dh


@wp.kernel
def jacobi_iter_3d(div: wp.array3d(dtype=float), p0: wp.array3d(dtype=float), mask: wp.array3d(dtype=int), p1: wp.array3d(dtype=float), nx: int, ny: int, nz: int, dh: float):
    i, j, k = wp.tid()
    
    if mask[i, j, k] == 0:
        p1[i, j, k] = 0.0
        return
    
    val_l = float(0.0)
    val_r = float(0.0)
    val_d = float(0.0)
    val_u = float(0.0)
    val_b = float(0.0)
    val_f = float(0.0)
    
    # Left
    if i == 0:
        val_l = p0[i, j, k]
    elif mask[i-1, j, k] == 0:
        val_l = p0[i, j, k]
    else:
        val_l = p0[i-1, j, k]
        
    # Right
    if i == nx-1:
        val_r = p0[i, j, k]
    elif mask[i+1, j, k] == 0:
        val_r = p0[i, j, k]
    else:
        val_r = p0[i+1, j, k]
        
    # Down
    if j == 0:
        val_d = p0[i, j, k]
    elif mask[i, j-1, k] == 0:
        val_d = p0[i, j, k]
    else:
        val_d = p0[i, j-1, k]
        
    # Up
    if j == ny-1:
        val_u = p0[i, j, k]
    elif mask[i, j+1, k] == 0:
        val_u = p0[i, j, k]
    else:
        val_u = p0[i, j+1, k]
        
    # Back
    if k == 0:
        val_b = p0[i, j, k]
    elif mask[i, j, k-1] == 0:
        val_b = p0[i, j, k]
    else:
        val_b = p0[i, j, k-1]
        
    # Front
    if k == nz-1:
        val_f = p0[i, j, k]
    elif mask[i, j, k+1] == 0:
        val_f = p0[i, j, k]
    else:
        val_f = p0[i, j, k+1]
        
    sum_neighbors = val_l + val_r + val_d + val_u + val_b + val_f
    p1[i, j, k] = (1.0/6.0) * (sum_neighbors - div[i, j, k] * dh * dh)

@wp.kernel
def update_velocities_3d(
    p: wp.array3d(dtype=float),
    u_in: wp.array3d(dtype=float),
    v_in: wp.array3d(dtype=float),
    w_in: wp.array3d(dtype=float),
    u_out: wp.array3d(dtype=float),
    v_out: wp.array3d(dtype=float),
    w_out: wp.array3d(dtype=float),
    bc_type: int,
    nx: int,
    ny: int,
    nz: int,
    dh: float
):
    i, j, k = wp.tid()
    
    if bc_type == BC_PERIODIC:
        grad_p_x = (p[i, j, k] - p[cyclic_index(i-1, nx), j, k]) / dh
        u_out[i, j, k] = u_in[i, j, k] - grad_p_x
        
        grad_p_y = (p[i, j, k] - p[i, cyclic_index(j-1, ny), k]) / dh
        v_out[i, j, k] = v_in[i, j, k] - grad_p_y
        
        grad_p_z = (p[i, j, k] - p[i, j, cyclic_index(k-1, nz)]) / dh
        w_out[i, j, k] = w_in[i, j, k] - grad_p_z
    else:
        if i == 0:
            u_out[i, j, k] = 0.0
        else:
            grad_p_x = (p[i, j, k] - p[i-1, j, k]) / dh
            u_out[i, j, k] = u_in[i, j, k] - grad_p_x
            
        if j == 0:
            v_out[i, j, k] = 0.0
        else:
            grad_p_y = (p[i, j, k] - p[i, j-1, k]) / dh
            v_out[i, j, k] = v_in[i, j, k] - grad_p_y
            
        if k == 0:
            w_out[i, j, k] = 0.0
        else:
            grad_p_z = (p[i, j, k] - p[i, j, k-1]) / dh
            w_out[i, j, k] = w_in[i, j, k] - grad_p_z

@wp.kernel
def compute_velocity_loss_3d(
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    target_vx: wp.array3d(dtype=float),
    target_vy: wp.array3d(dtype=float),
    target_vz: wp.array3d(dtype=float),
    loss: wp.array(dtype=float),
    nx: int,
    ny: int,
    nz: int
):
    i, j, k = wp.tid()
    
    diff_x = vx[i, j, k] - target_vx[i, j, k]
    diff_y = vy[i, j, k] - target_vy[i, j, k]
    diff_z = vz[i, j, k] - target_vz[i, j, k]
    
    val = (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) / float(nx * ny * nz)
    wp.atomic_add(loss, 0, val)

@wp.kernel
def enforce_noslip_velocity_3d(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    nx: int,
    ny: int,
    nz: int
):
    i, j, k = wp.tid()
    
    # Left/Right (i=0, i=N-1)
    if i == 0: 
        u[i, j, k] = 0.0 # No pen
        v[i, j, k] = 0.0 # No slip
        w[i, j, k] = 0.0 # No slip
    elif i == nx - 1:
        v[i, j, k] = 0.0 # No slip
        w[i, j, k] = 0.0 # No slip
        
    # Bottom/Top (j=0, j=N-1)
    if j == 0:
        v[i, j, k] = 0.0 # No pen
        u[i, j, k] = 0.0 # No slip
        w[i, j, k] = 0.0 # No slip
    elif j == ny - 1:
        u[i, j, k] = 0.0 # No slip
        w[i, j, k] = 0.0 # No slip
        
    # Back/Front (k=0, k=N-1)
    if k == 0:
        w[i, j, k] = 0.0 # No pen
        u[i, j, k] = 0.0 # No slip
        v[i, j, k] = 0.0 # No slip
    elif k == nz - 1:
        u[i, j, k] = 0.0 # No slip
        v[i, j, k] = 0.0 # No slip

@wp.kernel
def enforce_mask_velocity_3d(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    mask: wp.array3d(dtype=int),
    nx: int,
    ny: int,
    nz: int
):
    i, j, k = wp.tid()
    
    # Zero self faces if solid
    if mask[i, j, k] == 0:
        u[i, j, k] = 0.0
        v[i, j, k] = 0.0
        w[i, j, k] = 0.0
        
    # Zero faces shared with solid neighbors (prevent flow into/out of solid)
    # Check left neighbor
    if i > 0:
        if mask[i-1, j, k] == 0:
            u[i, j, k] = 0.0
    # Check bottom neighbor
    if j > 0:
        if mask[i, j-1, k] == 0:
            v[i, j, k] = 0.0
    # Check back neighbor
    if k > 0:
        if mask[i, j, k-1] == 0:
            w[i, j, k] = 0.0
            
    # Domain boundaries
    if i == 0: u[i, j, k] = 0.0
    if j == 0: v[i, j, k] = 0.0
    if k == 0: w[i, j, k] = 0.0


class FluidOptimizer:
    def __init__(self, num_basis_fields=5, sim_steps=50, pressure_iterations=100, device=None, bc_type=BoundaryCondition.PERIODIC, dim=2, n_grid=64):
        self.device = device if device else wp.get_device()
        self.bc_type = bc_type
        self.num_basis_fields = num_basis_fields
        self.sim_steps = sim_steps
        self.pressure_iterations = pressure_iterations
        self.dt = DT 
        self.dim = dim
        
        # Parse n_grid
        if isinstance(n_grid, int):
            self.nx = n_grid
            self.ny = n_grid
            self.nz = n_grid if dim == 3 else 1
        else:
            # Tuple
            if dim == 2:
                if len(n_grid) != 2:
                    raise ValueError("n_grid must be len 2 for 2D")
                self.nx = n_grid[0]
                self.ny = n_grid[1]
                self.nz = 1
            else:
                if len(n_grid) != 3:
                     raise ValueError("n_grid must be len 3 for 3D")
                self.nx = n_grid[0]
                self.ny = n_grid[1]
                self.nz = n_grid[2]

        self.n_grid = n_grid # Keep for ref but logic should use nx, ny, nz
        
        # DH derived from X dim. Assuming domain width X=1.0.
        self.dh = 1.0 / self.nx
        
        # Grid shape
        if self.dim == 2:
            self.shape = (self.nx, self.ny)
        else:
            self.shape = (self.nx, self.ny, self.nz)
            
        # Mask
        self.mask = wp.ones(self.shape, dtype=int, device=self.device) # Default all 1 (Fluid)
            

        # We need to store full trajectory for backprop        
        self.vx_arrays = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps + 1)]
        self.vy_arrays = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps + 1)]
        
        # 3D: vz arrays
        if self.dim == 3:
            self.vz_arrays = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps + 1)]
        else:
            self.vz_arrays = []

        self.density_arrays = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps + 1)]
        
        self.wx_arrays = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps)]
        self.wy_arrays = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps)]
        
        if self.dim == 3:
            self.wz_arrays = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps)]
        else:
            self.wz_arrays = []
        
        self.div_arrays = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(sim_steps)]
        
        self.pressure_arrays = []
        for _ in range(sim_steps):
            step_pressures = [wp.zeros(self.shape, dtype=float, device=self.device, requires_grad=True) for _ in range(self.pressure_iterations + 1)]
            self.pressure_arrays.append(step_pressures)
            
        # Basis Fields 
        rng = np.random.default_rng(42)
        
        # We'll use 4D array [num_basis, nx, ny, nz] for 3D basis fields or [num_basis, nx, ny] for 2D
        
        if self.dim == 2:
             basis_shape_np = (num_basis_fields, self.nx, self.ny)
        else:
             basis_shape_np = (num_basis_fields, self.nx, self.ny, self.nz)

        self.basis_fx_np = np.zeros(basis_shape_np, dtype=np.float32)
        self.basis_fy_np = np.zeros(basis_shape_np, dtype=np.float32)
        
        if self.dim == 3:
            self.basis_fz_np = np.zeros(basis_shape_np, dtype=np.float32)
        else:
            self.basis_fz_np = None # Not used

        for k in range(num_basis_fields):
            freq_x = rng.uniform(1.0, 3.0)
            freq_y = rng.uniform(1.0, 3.0)
            if self.dim == 3:
                freq_z = rng.uniform(1.0, 3.0)
            
            phase = rng.uniform(0, 2*np.pi)
             
            if self.dim == 2:
                 # existing 2D generation
                phase_x = rng.uniform(0, 2*np.pi)
                for y in range(self.ny):
                    for x in range(self.nx):
                        yf = y * self.dh
                        xf = x * self.dh
                        self.basis_fx_np[k, y, x] = np.sin(xf * freq_x + phase_x) * np.cos(yf * freq_y)
                        self.basis_fy_np[k, y, x] = np.cos(xf * freq_x) * np.sin(yf * freq_y + phase)
            else:
                # 3D generation
                phase_x = rng.uniform(0, 2*np.pi)
                for z in range(self.nz):
                    for y in range(self.ny):
                        for x in range(self.nx):
                            zf = z * self.dh
                            yf = y * self.dh
                            xf = x * self.dh
                            # Simple divergence-free-ish
                            self.basis_fx_np[k, z, y, x] = np.sin(xf * freq_x + phase_x) * np.cos(yf * freq_y) * np.cos(zf * freq_z)
                            self.basis_fy_np[k, z, y, x] = np.cos(xf * freq_x) * np.sin(yf * freq_y + phase) * np.cos(zf * freq_z)
                            self.basis_fz_np[k, z, y, x] = np.cos(xf * freq_x) * np.cos(yf * freq_y) * np.sin(zf * freq_z)

        self.basis_fx = wp.array(self.basis_fx_np, dtype=float, device=self.device)
        self.basis_fy = wp.array(self.basis_fy_np, dtype=float, device=self.device)
        if self.dim == 3:
            self.basis_fz = wp.array(self.basis_fz_np, dtype=float, device=self.device)
        else:
            self.basis_fz = wp.zeros((1,1,1,1), dtype=float) # Dummy

        self.weights = wp.zeros(num_basis_fields, dtype=float, device=self.device, requires_grad=True)

        self.target_vx_arrays = [wp.zeros(self.shape, dtype=float, device=self.device) for _ in range(sim_steps)]
        self.target_vy_arrays = [wp.zeros(self.shape, dtype=float, device=self.device) for _ in range(sim_steps)]
        if self.dim == 3:
            self.target_vz_arrays = [wp.zeros(self.shape, dtype=float, device=self.device) for _ in range(sim_steps)]
        else:
            self.target_vz_arrays = []

        self.loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)
        self.optimizer = warp.optim.Adam([self.weights], lr=0.01)

    def set_mask_circle(self, radius=0.4):
        if self.dim != 2:
            print("Warning: set_mask_circle called for 3D simulation. Using sphere?")
        
        cx = 0.5
        cy = 0.5
        
        if self.dim == 2:
            wp.launch(init_mask_circle, self.shape, inputs=[self.mask, radius, cx, cy, self.dh])
        else:
            # Sphere
            cz = 0.5
            wp.launch(init_mask_sphere, self.shape, inputs=[self.mask, radius, cx, cy, cz, self.dh])


    def run_step(self, t):
        """Execute one simulation step t -> t+1."""
        if self.dim == 2:
            # 1. Advect Velocities (Self-Advection)
            wp.launch(advect_mac_u, self.shape, inputs=[self.dt, self.vx_arrays[t], self.vy_arrays[t], self.wx_arrays[t], self.nx, self.ny, self.dh])
            wp.launch(advect_mac_v, self.shape, inputs=[self.dt, self.vx_arrays[t], self.vy_arrays[t], self.wy_arrays[t], self.nx, self.ny, self.dh])
            
            # Enforce mask velocity (zero solid faces)
            wp.launch(enforce_mask_velocity, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.mask, self.nx, self.ny])
            
            if self.bc_type == BoundaryCondition.WALL:
                wp.launch(enforce_noslip_velocity, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.nx, self.ny])
                
            # 2. Apply Forces
            wp.launch(apply_forces, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.basis_fx, self.basis_fy, self.weights, self.dt])
            
            # Enforce mask velocity again? Forces might add velocity in solid.
            wp.launch(enforce_mask_velocity, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.mask, self.nx, self.ny])

            # 3. Pressure
            wp.launch(divergence, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.mask, self.div_arrays[t], self.nx, self.ny, self.dh])
            
            self.pressure_arrays[t][0].zero_()
            for k in range(self.pressure_iterations):
                wp.launch(jacobi_iter, self.shape, inputs=[self.div_arrays[t], self.pressure_arrays[t][k], self.mask, self.pressure_arrays[t][k+1], self.nx, self.ny, self.dh])
                
            final_p_idx = self.pressure_iterations
            wp.launch(update_velocities, self.shape, inputs=[self.pressure_arrays[t][final_p_idx], self.wx_arrays[t], self.wy_arrays[t], self.vx_arrays[t+1], self.vy_arrays[t+1], self.bc_type.value, self.nx, self.ny, self.dh])
            
            # Enforce mask velocity final
            wp.launch(enforce_mask_velocity, self.shape, inputs=[self.vx_arrays[t+1], self.vy_arrays[t+1], self.mask, self.nx, self.ny])
            
            # 4. Advect Density
            wp.launch(advect_density, self.shape, inputs=[self.dt, self.vx_arrays[t+1], self.vy_arrays[t+1], self.density_arrays[t], self.density_arrays[t+1], self.bc_type.value, self.nx, self.ny, self.dh])
            
        else:
            # 3D
            inputs_uvw = [self.dt, self.vx_arrays[t], self.vy_arrays[t], self.vz_arrays[t]]
            wp.launch(advect_mac_u_3d, self.shape, inputs=inputs_uvw + [self.wx_arrays[t], self.nx, self.ny, self.nz, self.dh])
            wp.launch(advect_mac_v_3d, self.shape, inputs=inputs_uvw + [self.wy_arrays[t], self.nx, self.ny, self.nz, self.dh])
            wp.launch(advect_mac_w_3d, self.shape, inputs=inputs_uvw + [self.wz_arrays[t], self.nx, self.ny, self.nz, self.dh])
            
            # Enforce mask
            wp.launch(enforce_mask_velocity_3d, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.mask, self.nx, self.ny, self.nz])
            
            if self.bc_type == BoundaryCondition.WALL:
                wp.launch(enforce_noslip_velocity_3d, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.nx, self.ny, self.nz])
                
            wp.launch(apply_forces_3d, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.basis_fx, self.basis_fy, self.basis_fz, self.weights, self.dt])
            
            # Enforce mask again
            wp.launch(enforce_mask_velocity_3d, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.mask, self.nx, self.ny, self.nz])

            wp.launch(divergence_3d, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.mask, self.div_arrays[t], self.nx, self.ny, self.nz, self.dh])
            
            self.pressure_arrays[t][0].zero_()
            for k in range(self.pressure_iterations):
                wp.launch(jacobi_iter_3d, self.shape, inputs=[self.div_arrays[t], self.pressure_arrays[t][k], self.mask, self.pressure_arrays[t][k+1], self.nx, self.ny, self.nz, self.dh])
                
            final_p_idx = self.pressure_iterations
            wp.launch(update_velocities_3d, self.shape, inputs=[self.pressure_arrays[t][final_p_idx], self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.vx_arrays[t+1], self.vy_arrays[t+1], self.vz_arrays[t+1], self.bc_type.value, self.nx, self.ny, self.nz, self.dh])
            
            # Enforce mask final
            wp.launch(enforce_mask_velocity_3d, self.shape, inputs=[self.vx_arrays[t+1], self.vy_arrays[t+1], self.vz_arrays[t+1], self.mask, self.nx, self.ny, self.nz])
            
            wp.launch(advect_density_3d, self.shape, inputs=[self.dt, self.vx_arrays[t+1], self.vy_arrays[t+1], self.vz_arrays[t+1], self.density_arrays[t], self.density_arrays[t+1], self.bc_type.value, self.nx, self.ny, self.nz, self.dh])

    def clear_all_gradients(self):
        """Manually zero gradients of all state arrays to prevent accumulation."""
        self.weights.grad.zero_()
        self.loss.grad.zero_()
        
        # Helper to zero a list of arrays
        def zero_list(arr_list):
            for arr in arr_list:
                if arr.requires_grad:
                    arr.grad.zero_()
        
        zero_list(self.vx_arrays)
        zero_list(self.vy_arrays)
        if self.dim == 3:
             zero_list(self.vz_arrays)
             zero_list(self.wz_arrays)

        zero_list(self.density_arrays)
        zero_list(self.wx_arrays)
        zero_list(self.wy_arrays)
        zero_list(self.div_arrays)
        
        # Pressure is a list of lists
        for step_list in self.pressure_arrays:
            zero_list(step_list)

    def forward(self):
        # Reset state (only t=0 is fixed)
        # We assume vx[0], vy[0], density[0] are set.

        self.loss.zero_()

        for t in range(self.sim_steps):
            self.run_step(t)
            if self.dim == 2:
                wp.launch(
                    compute_velocity_loss,
                    self.shape,
                    inputs=[
                        self.vx_arrays[t+1],      # Current simulation state
                        self.vy_arrays[t+1], 
                        self.target_vx_arrays[t], # Corresponding target frame
                        self.target_vy_arrays[t], 
                        self.loss,
                        self.nx,
                        self.ny
                    ]
                )
            else:
                 wp.launch(
                    compute_velocity_loss_3d,
                    self.shape,
                    inputs=[
                        self.vx_arrays[t+1],
                        self.vy_arrays[t+1],
                        self.vz_arrays[t+1],
                        self.target_vx_arrays[t],
                        self.target_vy_arrays[t],
                        self.target_vz_arrays[t],
                        self.loss,
                        self.nx,
                        self.ny,
                        self.nz
                    ]
                )
        
    def step_optimization(self):
        self.clear_all_gradients()
        self.weights.grad.zero_()
        self.tape = wp.Tape()
        self.loss.zero_()
        
        with self.tape:
            self.forward()
        
        self.tape.backward(self.loss)
        
        # Gradient Processing
        grad_np = self.weights.grad.numpy()
        
        if not np.all(np.isfinite(grad_np)):
            print("  [Warning] Gradients contain NaN/Inf! Zeroing grads.")
            self.weights.grad.zero_()
            return self.loss.numpy()[0], float('nan')

        grad_norm = np.linalg.norm(grad_np)
        
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            scale = max_grad_norm / (grad_norm + 1e-6)
            grad_np = grad_np * scale
            self.weights.grad = wp.array(grad_np, dtype=float, device=self.device)
            grad_norm = max_grad_norm
        
        self.optimizer.step([self.weights.grad])
        
        return self.loss.numpy()[0], grad_norm

    def get_force_fields(self):
        """
        Compute the current force fields based on learned weights.
        Returns:
            total_force: tuple (fx, fy) numpy arrays of shape (N, N)
            basis_forces: list of tuples (fx, fy) numpy arrays of shape (N, N), weighted by current weights
        """
        weights_np = self.weights.numpy()
        
        if self.dim == 2:
             basis_shape = (self.nx, self.ny)
        else:
             basis_shape = (self.nx, self.ny, self.nz)

        # basis arrays are always loaded in __init__
        basis_fx_np = self.basis_fx_np
        basis_fy_np = self.basis_fy_np
        basis_fz_np = self.basis_fz_np
        
        total_fx = np.zeros(basis_shape, dtype=np.float32)
        total_fy = np.zeros(basis_shape, dtype=np.float32)
        if self.dim == 3:
             total_fz = np.zeros(basis_shape, dtype=np.float32)
        else:
             total_fz = None
        
        basis_forces = []
        
        num_bases = len(weights_np)
        for k in range(num_bases):
            w = weights_np[k]
            # Weighted basis
            fx_k = w * basis_fx_np[k]
            fy_k = w * basis_fy_np[k]
            
            if self.dim == 3:
                 fz_k = w * basis_fz_np[k]
                 basis_forces.append((fx_k, fy_k, fz_k))
                 total_fx += fx_k
                 total_fy += fy_k
                 total_fz += fz_k
            else:
                 basis_forces.append((fx_k, fy_k))
                 total_fx += fx_k
                 total_fy += fy_k
             
        if self.dim == 3:
             return (total_fx, total_fy, total_fz), basis_forces
        else:
             return (total_fx, total_fy), basis_forces
