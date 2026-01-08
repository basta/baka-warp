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


@wp.func
def cyclic_index(idx: wp.int32):
    """Helper function to index with periodic boundary conditions."""
    ret_idx = idx % N_GRID
    if ret_idx < 0:
        ret_idx += N_GRID
    return ret_idx


@wp.kernel
def initialize_fields(
    density: wp.array2d(dtype=float),
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float)
):
    i, j = wp.tid()
    
    # Initialize density as a circle in the center
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    
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
def sample_field(field: wp.array2d(dtype=float), x: float, y: float):
    # Bilinear interpolation
    # field is defined at integer indices 0..N-1
    # clamp coords
    x = wp.max(0.0, wp.min(x, float(N_GRID) - 1.0))
    y = wp.max(0.0, wp.min(y, float(N_GRID) - 1.0))

    x0 = wp.int32(wp.floor(x))
    y0 = wp.int32(wp.floor(y))
    
    wx1 = x - float(x0)
    wy1 = y - float(y0)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    
    x1 = wp.min(x0 + 1, N_GRID - 1)
    y1 = wp.min(y0 + 1, N_GRID - 1)
    
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
    u_new: wp.array2d(dtype=float)
):
    # Advect u component. u lives at (i, j+0.5):
    # Convention: u[i, j] is flow across CONSTANT-X face between cell (i-1, j) and (i, j).
    # Coordinate: x = i * DH, y = (j + 0.5) * DH.
    
    i, j = wp.tid()
    
    # World pos of this u-face
    x = float(i) * DH
    y = (float(j) + 0.5) * DH
    
    # Velocity at this point
    # u is exact at this point
    vel_x = u[i, j]
    
    # v needs interpolation. v lives at (x+0.5, y-0.5).
    # We want v at (i, j+0.5)
    # v grid coords: x_v = i - 0.5, y_v = j + 0.5
    # To get world (i, j+0.5), we sample v at grid coord (i-0.5, j+0.5).
    grid_v_x = float(i) - 0.5
    grid_v_y = float(j) + 0.5
    vel_y = sample_field(v, grid_v_x, grid_v_y)
    
    # Backtrace
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample u at src. u lives at grid (i, j+0.5).
    # grid u coords:
    grid_src_x = src_x / DH
    grid_src_y = (src_y / DH) - 0.5
    
    u_new[i, j] = sample_field(u, grid_src_x, grid_src_y)


@wp.kernel
def advect_mac_v(
    dt: float,
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    v_new: wp.array2d(dtype=float)
):
    # Advect v component. v lives at (i+0.5, j).
    # Convention: v[i, j] is flow across CONSTANT-Y face between cell (i, j-1) and (i, j).
    
    i, j = wp.tid()
    
    # World pos of this v-face
    x = (float(i) + 0.5) * DH
    y = float(j) * DH
    
    # Velocity
    vel_y = v[i, j]
    
    # u needs interpolation. u lives at (i, j+0.5).
    # We want u at (i+0.5, j).
    # u grid coords: x_u = i + 0.5, y_u = j - 0.5
    grid_u_x = float(i) + 0.5
    grid_u_y = float(j) - 0.5
    vel_x = sample_field(u, grid_u_x, grid_u_y)
    
    # Backtrace
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample v at src. v lives at grid (i+0.5, j).
    grid_src_x = (src_x / DH) - 0.5
    grid_src_y = (src_y / DH)
    
    v_new[i, j] = sample_field(v, grid_src_x, grid_src_y)


@wp.kernel
def advect_density(
    dt: float,
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float),
    rho_old: wp.array2d(dtype=float),
    rho_new: wp.array2d(dtype=float),
    bc_type: int
):
    # Advect scalar at cell centers (i+0.5, j+0.5)
    i, j = wp.tid()
    
    # World pos
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    
    # Velocity at center
    # u at (i, j+0.5) and (i+1, j+0.5). Avg to get center.
    
    u_idx = u[i, j]
    u_idx1 = float(0.0)
    
    v_idx = v[i, j]
    v_idx1 = float(0.0)
    
    if bc_type == BC_PERIODIC:
        u_idx1 = u[cyclic_index(i + 1), j]
        v_idx1 = v[i, cyclic_index(j + 1)]
    else:
        # BC_WALL
        if i == N_GRID - 1:
            u_idx1 = 0.0
        else:
            u_idx1 = u[i + 1, j]
            
        if j == N_GRID - 1:
            v_idx1 = 0.0
        else:
            v_idx1 = v[i, j + 1]
    
    vel_x = (u_idx + u_idx1) * 0.5
    vel_y = (v_idx + v_idx1) * 0.5
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    
    # Sample rho (centered)
    grid_src_x = (src_x / DH) - 0.5
    grid_src_y = (src_y / DH) - 0.5
    
    rho_new[i, j] = sample_field(rho_old, grid_src_x, grid_src_y)


@wp.kernel
def divergence(u: wp.array2d(dtype=float), v: wp.array2d(dtype=float), div: wp.array2d(dtype=float), bc_type: int):
    """Compute div(u) at cell centers."""
    i, j = wp.tid()
    
    # u[i, j] is left face, u[i+1, j] is right face
    # v[i, j] is bottom face, v[i, j+1] is top face
    
    u_right = float(0.0)
    u_left = float(0.0)
    v_top = float(0.0)
    v_bot = float(0.0)

    if bc_type == BC_PERIODIC:
        u_right = u[cyclic_index(i + 1), j]
        u_left = u[i, j]
        v_top = v[i, cyclic_index(j + 1)]
        v_bot = v[i, j]
    else:
        # BC_WALL
        
        # Left face
        u_left = u[i, j]
        
        # Right face
        if i == N_GRID - 1:
            u_right = 0.0
        else:
            u_right = u[i + 1, j]
            
        # Bottom face
        v_bot = v[i, j]
        
        # Top face
        if j == N_GRID - 1:
            v_top = 0.0
        else:
            v_top = v[i, j + 1]

    div[i, j] = (u_right - u_left + v_top - v_bot) / DH


@wp.kernel
def jacobi_iter(div: wp.array2d(dtype=float), p0: wp.array2d(dtype=float), p1: wp.array2d(dtype=float), bc_type: int):
    """Solve Laplacian P = div."""
    i, j = wp.tid()
    
    # Standard 5-point Laplacian
    # 4 * p[i, j] - neighbors = -div * dx^2
    # p[i,j] = (neighbors - div*dx^2) / 4
    
    val_left = float(0.0)
    val_right = float(0.0)
    val_down = float(0.0)
    val_up = float(0.0)
    
    if bc_type == BC_PERIODIC:
        val_left = p0[cyclic_index(i - 1), j]
        val_right = p0[cyclic_index(i + 1), j]
        val_down = p0[i, cyclic_index(j - 1)]
        val_up = p0[i, cyclic_index(j + 1)]
    else:
        # BC_WALL: Neumann BC (dP/dn = 0) => P_boundary = P_inside
        if i == 0:
            val_left = p0[i, j]
        else:
            val_left = p0[i - 1, j]
            
        if i == N_GRID - 1:
            val_right = p0[i, j]
        else:
            val_right = p0[i + 1, j]
            
        if j == 0:
            val_down = p0[i, j]
        else:
            val_down = p0[i, j - 1]
            
        if j == N_GRID - 1:
            val_up = p0[i, j]
        else:
            val_up = p0[i, j + 1]

    sum_neighbors = val_left + val_right + val_down + val_up
                     
    p1[i, j] = 0.25 * (sum_neighbors - div[i, j] * DH * DH)


@wp.kernel
def update_velocities(
    p: wp.array2d(dtype=float),
    u_in: wp.array2d(dtype=float),
    v_in: wp.array2d(dtype=float),
    u_out: wp.array2d(dtype=float),
    v_out: wp.array2d(dtype=float),
    bc_type: int
):
    """Subtract pressure gradient."""
    i, j = wp.tid()

    # Update u at (i, j+0.5) (left face of cell i,j)
    # Grad P x component at face: (P[i,j] - P[i-1,j]) / DH
    
    if bc_type == BC_PERIODIC:
        grad_p_x = (p[i, j] - p[cyclic_index(i - 1), j]) / DH
        u_out[i, j] = u_in[i, j] - grad_p_x
        
        grad_p_y = (p[i, j] - p[i, cyclic_index(j - 1)]) / DH
        v_out[i, j] = v_in[i, j] - grad_p_y
    else:
        # BC_WALL
        # For u (x-velocity on vertical faces)
        # u[i, j] is on left face of cell i.
        # If i=0 (left domain boundary), u MUST be 0.
        if i == 0:
            u_out[i, j] = 0.0
        else:
            grad_p_x = (p[i, j] - p[i - 1, j]) / DH
            u_out[i, j] = u_in[i, j] - grad_p_x
            
        # For v (y-velocity on horizontal faces)
        # v[i, j] is on bottom face of cell j.
        # If j=0 (bottom domain boundary), v MUST be 0.
        if j == 0:
            v_out[i, j] = 0.0
        else:
            grad_p_y = (p[i, j] - p[i, j - 1]) / DH
            v_out[i, j] = v_in[i, j] - grad_p_y


@wp.kernel
def compute_velocity_loss(
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
    target_vx: wp.array2d(dtype=float),
    target_vy: wp.array2d(dtype=float),
    loss: wp.array(dtype=float)
):
    i, j = wp.tid()
    
    diff_x = vx[i, j] - target_vx[i, j]
    diff_y = vy[i, j] - target_vy[i, j]
    
    # Mean squared error
    val = (diff_x * diff_x + diff_y * diff_y) / float(N_GRID * N_GRID)
    
    wp.atomic_add(loss, 0, val)

@wp.kernel
def enforce_noslip_velocity(
    u: wp.array2d(dtype=float),
    v: wp.array2d(dtype=float)
):
    """Explicitly zero out boundaries for wall BC."""
    i, j = wp.tid()
    
    # Left/Right walls (x=0, x=N)
    if i == 0:
        u[i, j] = 0.0 # No penetration Left
        v[i, j] = 0.0 # No slip Left
        
    if i == N_GRID - 1:
        v[i, j] = 0.0 # No slip Right
        # u[N-1] is interior face, u[N] is wall. 
        # But for 'box' behavior, zeroing tangential near wall is good.
    
    # Bottom/Top walls (y=0, y=N)
    if j == 0:
        v[i, j] = 0.0 # No penetration Bottom
        u[i, j] = 0.0 # No slip Bottom
        
    if j == N_GRID - 1:
        u[i, j] = 0.0 # No slip Top



CENTER_Z = 0.5 

@wp.kernel
def initialize_fields_3d(
    density: wp.array3d(dtype=float),
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()
    
    # Initialize density as a sphere in the center
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    z = (float(k) + 0.5) * DH
    
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
def sample_field_3d(field: wp.array3d(dtype=float), x: float, y: float, z: float):
    # Trilinear interpolation
    x = wp.max(0.0, wp.min(x, float(N_GRID) - 1.0))
    y = wp.max(0.0, wp.min(y, float(N_GRID) - 1.0))
    z = wp.max(0.0, wp.min(z, float(N_GRID) - 1.0))

    x0 = wp.int32(wp.floor(x))
    y0 = wp.int32(wp.floor(y))
    z0 = wp.int32(wp.floor(z))
    
    wx1 = x - float(x0)
    wy1 = y - float(y0)
    wz1 = z - float(z0)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1
    
    x1 = wp.min(x0 + 1, N_GRID - 1)
    y1 = wp.min(y0 + 1, N_GRID - 1)
    z1 = wp.min(z0 + 1, N_GRID - 1)
    
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
    u_new: wp.array3d(dtype=float)
):
    # u at (i, j+0.5, k+0.5)
    
    i, j, k = wp.tid()
    
    x = float(i) * DH
    y = (float(j) + 0.5) * DH
    z = (float(k) + 0.5) * DH
    
    vel_x = u[i, j, k]
    
    # v needs sample at x,y,z
    # v is at x-0.5, y, z-0.5 relative to u node? No. 
    # v is at (i+0.5, j, k+0.5).
    # u is at (i, j+0.5, k+0.5).
    # Coords of u node: (i, j+0.5, k+0.5)
    # v grid coords: x_v = i - 0.5, y_v = j + 0.5, z_v = k
    grid_v_x = float(i) - 0.5
    grid_v_y = float(j) + 0.5
    grid_v_z = float(k)
    vel_y = sample_field_3d(v, grid_v_x, grid_v_y, grid_v_z)
    
    # w is at (i+0.5, j+0.5, k).
    # w grid coords: x_w = i - 0.5, y_w = j, z_w = k + 0.5
    grid_w_x = float(i) - 0.5
    grid_w_y = float(j)
    grid_w_z = float(k) + 0.5
    vel_z = sample_field_3d(w, grid_w_x, grid_w_y, grid_w_z)
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    src_z = z - vel_z * dt
    
    # Sample u at src
    grid_src_x = src_x / DH
    grid_src_y = (src_y / DH) - 0.5
    grid_src_z = (src_z / DH) - 0.5
    
    u_new[i, j, k] = sample_field_3d(u, grid_src_x, grid_src_y, grid_src_z)

@wp.kernel
def advect_mac_v_3d(
    dt: float,
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    v_new: wp.array3d(dtype=float)
):
    # v at (i+0.5, j, k+0.5)
    
    i, j, k = wp.tid()
    
    x = (float(i) + 0.5) * DH
    y = float(j) * DH
    z = (float(k) + 0.5) * DH
    
    vel_y = v[i, j, k]
    
    # u sample. u is at (i, j+0.5, k+0.5)
    # grid_u = (i+0.5, j-0.5, k)
    grid_u_x = float(i) + 0.5
    grid_u_y = float(j) - 0.5
    grid_u_z = float(k)
    vel_x = sample_field_3d(u, grid_u_x, grid_u_y, grid_u_z)
    
    # w sample. w is at (i+0.5, j+0.5, k)
    # grid_w = (i, j-0.5, k+0.5)
    grid_w_x = float(i)
    grid_w_y = float(j) - 0.5
    grid_w_z = float(k) + 0.5
    vel_z = sample_field_3d(w, grid_w_x, grid_w_y, grid_w_z)
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    src_z = z - vel_z * dt
    
    grid_src_x = (src_x / DH) - 0.5
    grid_src_y = src_y / DH
    grid_src_z = (src_z / DH) - 0.5
    
    v_new[i, j, k] = sample_field_3d(v, grid_src_x, grid_src_y, grid_src_z)

@wp.kernel
def advect_mac_w_3d(
    dt: float,
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    w_new: wp.array3d(dtype=float)
):
    # w at (i+0.5, j+0.5, k)
    
    i, j, k = wp.tid()
    
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    z = float(k) * DH
    
    vel_z = w[i, j, k]
    
    # u sample. u is at (i, j+0.5, k+0.5)
    # grid_u = (i+0.5, j, k-0.5)
    grid_u_x = float(i) + 0.5
    grid_u_y = float(j)
    grid_u_z = float(k) - 0.5
    vel_x = sample_field_3d(u, grid_u_x, grid_u_y, grid_u_z)
    
    # v sample. v is at (i+0.5, j, k+0.5)
    # grid_v = (i, j+0.5, k-0.5)
    grid_v_x = float(i)
    grid_v_y = float(j) + 0.5
    grid_v_z = float(k) - 0.5
    vel_y = sample_field_3d(v, grid_v_x, grid_v_y, grid_v_z)
    
    src_x = x - vel_x * dt
    src_y = y - vel_y * dt
    src_z = z - vel_z * dt
    
    grid_src_x = (src_x / DH) - 0.5
    grid_src_y = (src_y / DH) - 0.5
    grid_src_z = src_z / DH
    
    w_new[i, j, k] = sample_field_3d(w, grid_src_x, grid_src_y, grid_src_z)

@wp.kernel
def advect_density_3d(
    dt: float,
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    rho_old: wp.array3d(dtype=float),
    rho_new: wp.array3d(dtype=float),
    bc_type: int
):
    i, j, k = wp.tid()
    
    # Cell center x,y,z
    x = (float(i) + 0.5) * DH
    y = (float(j) + 0.5) * DH
    z = (float(k) + 0.5) * DH
    
    u_c = float(0.0)
    v_c = float(0.0)
    w_c = float(0.0)
    
    if bc_type == BC_PERIODIC:
        u_c = (u[i, j, k] + u[cyclic_index(i+1), j, k]) * 0.5
        v_c = (v[i, j, k] + v[i, cyclic_index(j+1), k]) * 0.5
        w_c = (w[i, j, k] + w[i, j, cyclic_index(k+1)]) * 0.5
    else:
        # BC_WALL
        # Check bounds
        u1 = float(0.0)
        if i == N_GRID-1:
            u1 = 0.0 
        else:
            u1 = u[i+1, j, k]
            
        v1 = float(0.0)
        if j == N_GRID-1:
            v1 = 0.0
        else:
            v1 = v[i, j+1, k]
            
        w1 = float(0.0)
        if k == N_GRID-1:
            w1 = 0.0 
        else:
            w1 = w[i, j, k+1]
        
        u_c = (u[i, j, k] + u1) * 0.5
        v_c = (v[i, j, k] + v1) * 0.5
        w_c = (w[i, j, k] + w1) * 0.5
        
    src_x = x - u_c * dt
    src_y = y - v_c * dt
    src_z = z - w_c * dt
    
    grid_src_x = (src_x / DH) - 0.5
    grid_src_y = (src_y / DH) - 0.5
    grid_src_z = (src_z / DH) - 0.5
    
    rho_new[i, j, k] = sample_field_3d(rho_old, grid_src_x, grid_src_y, grid_src_z)

@wp.kernel
def divergence_3d(u: wp.array3d(dtype=float), v: wp.array3d(dtype=float), w: wp.array3d(dtype=float), div: wp.array3d(dtype=float), bc_type: int):
    i, j, k = wp.tid()
    
    u_r = float(0.0)
    u_l = float(0.0)
    v_t = float(0.0)
    v_b = float(0.0)
    w_f = float(0.0) # front (z+)
    w_k = float(0.0) # back (z)
    
    if bc_type == BC_PERIODIC:
        u_r = u[cyclic_index(i+1), j, k]
        u_l = u[i, j, k]
        v_t = v[i, cyclic_index(j+1), k]
        v_b = v[i, j, k]
        w_f = w[i, j, cyclic_index(k+1)]
        w_k = w[i, j, k]
    else:
        u_l = u[i, j, k]
        
        if i == N_GRID-1:
            u_r = 0.0 
        else:
            u_r = u[i+1, j, k]
            
        v_b = v[i, j, k]
        
        if j == N_GRID-1:
            v_t = 0.0 
        else:
            v_t = v[i, j+1, k]
            
        w_k = w[i, j, k]
        
        if k == N_GRID-1:
            w_f = 0.0 
        else:
            w_f = w[i, j, k+1]
        
    div[i, j, k] = (u_r - u_l + v_t - v_b + w_f - w_k) / DH

@wp.kernel
def jacobi_iter_3d(div: wp.array3d(dtype=float), p0: wp.array3d(dtype=float), p1: wp.array3d(dtype=float), bc_type: int):
    i, j, k = wp.tid()
    
    val_l = float(0.0)
    val_r = float(0.0)
    val_d = float(0.0)
    val_u = float(0.0)
    val_b = float(0.0)
    val_f = float(0.0)
    
    if bc_type == BC_PERIODIC:
        val_l = p0[cyclic_index(i-1), j, k]
        val_r = p0[cyclic_index(i+1), j, k]
        val_d = p0[i, cyclic_index(j-1), k]
        val_u = p0[i, cyclic_index(j+1), k]
        val_b = p0[i, j, cyclic_index(k-1)]
        val_f = p0[i, j, cyclic_index(k+1)]
    else:
        # Left
        if i == 0:
            val_l = p0[i, j, k] 
        else:
            val_l = p0[i-1, j, k]
            
        # Right
        if i == N_GRID-1:
            val_r = p0[i, j, k] 
        else:
            val_r = p0[i+1, j, k]
            
        # Down
        if j == 0:
            val_d = p0[i, j, k] 
        else:
            val_d = p0[i, j-1, k]
            
        # Up
        if j == N_GRID-1:
            val_u = p0[i, j, k] 
        else:
            val_u = p0[i, j+1, k]
            
        # Back
        if k == 0:
            val_b = p0[i, j, k] 
        else:
            val_b = p0[i, j, k-1]
            
        # Front
        if k == N_GRID-1:
            val_f = p0[i, j, k] 
        else:
            val_f = p0[i, j, k+1]
        
    sum_neighbors = val_l + val_r + val_d + val_u + val_b + val_f
    p1[i, j, k] = (1.0/6.0) * (sum_neighbors - div[i, j, k] * DH * DH)

@wp.kernel
def update_velocities_3d(
    p: wp.array3d(dtype=float),
    u_in: wp.array3d(dtype=float),
    v_in: wp.array3d(dtype=float),
    w_in: wp.array3d(dtype=float),
    u_out: wp.array3d(dtype=float),
    v_out: wp.array3d(dtype=float),
    w_out: wp.array3d(dtype=float),
    bc_type: int
):
    i, j, k = wp.tid()
    
    if bc_type == BC_PERIODIC:
        grad_p_x = (p[i, j, k] - p[cyclic_index(i-1), j, k]) / DH
        u_out[i, j, k] = u_in[i, j, k] - grad_p_x
        
        grad_p_y = (p[i, j, k] - p[i, cyclic_index(j-1), k]) / DH
        v_out[i, j, k] = v_in[i, j, k] - grad_p_y
        
        grad_p_z = (p[i, j, k] - p[i, j, cyclic_index(k-1)]) / DH
        w_out[i, j, k] = w_in[i, j, k] - grad_p_z
    else:
        if i == 0:
            u_out[i, j, k] = 0.0
        else:
            grad_p_x = (p[i, j, k] - p[i-1, j, k]) / DH
            u_out[i, j, k] = u_in[i, j, k] - grad_p_x
            
        if j == 0:
            v_out[i, j, k] = 0.0
        else:
            grad_p_y = (p[i, j, k] - p[i, j-1, k]) / DH
            v_out[i, j, k] = v_in[i, j, k] - grad_p_y
            
        if k == 0:
            w_out[i, j, k] = 0.0
        else:
            grad_p_z = (p[i, j, k] - p[i, j, k-1]) / DH
            w_out[i, j, k] = w_in[i, j, k] - grad_p_z

@wp.kernel
def compute_velocity_loss_3d(
    vx: wp.array3d(dtype=float),
    vy: wp.array3d(dtype=float),
    vz: wp.array3d(dtype=float),
    target_vx: wp.array3d(dtype=float),
    target_vy: wp.array3d(dtype=float),
    target_vz: wp.array3d(dtype=float),
    loss: wp.array(dtype=float)
):
    i, j, k = wp.tid()
    
    diff_x = vx[i, j, k] - target_vx[i, j, k]
    diff_y = vy[i, j, k] - target_vy[i, j, k]
    diff_z = vz[i, j, k] - target_vz[i, j, k]
    
    val = (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) / float(N_GRID * N_GRID * N_GRID)
    wp.atomic_add(loss, 0, val)

@wp.kernel
def enforce_noslip_velocity_3d(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float)
):
    i, j, k = wp.tid()
    
    # Left/Right (i=0, i=N-1)
    if i == 0: 
        u[i, j, k] = 0.0 # No pen
        v[i, j, k] = 0.0 # No slip
        w[i, j, k] = 0.0 # No slip
    elif i == N_GRID - 1:
        v[i, j, k] = 0.0 # No slip
        w[i, j, k] = 0.0 # No slip
        
    # Bottom/Top (j=0, j=N-1)
    if j == 0:
        v[i, j, k] = 0.0 # No pen
        u[i, j, k] = 0.0 # No slip
        w[i, j, k] = 0.0 # No slip
    elif j == N_GRID - 1:
        u[i, j, k] = 0.0 # No slip
        w[i, j, k] = 0.0 # No slip
        
    # Back/Front (k=0, k=N-1)
    if k == 0:
        w[i, j, k] = 0.0 # No pen
        u[i, j, k] = 0.0 # No slip
        v[i, j, k] = 0.0 # No slip
    elif k == N_GRID - 1:
        u[i, j, k] = 0.0 # No slip
        v[i, j, k] = 0.0 # No slip


class FluidOptimizer:
    def __init__(self, num_basis_fields=5, sim_steps=50, pressure_iterations=100, device=None, bc_type=BoundaryCondition.PERIODIC, dim=2):
        self.device = device if device else wp.get_device()
        self.bc_type = bc_type
        self.num_basis_fields = num_basis_fields
        self.sim_steps = sim_steps
        self.pressure_iterations = pressure_iterations
        self.dt = DT
        self.dim = dim
        
        # Grid shape
        if self.dim == 2:
            self.shape = (N_GRID, N_GRID)
        else:
            self.shape = (N_GRID, N_GRID, N_GRID)
            

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
        
        # We'll use 4D array [num_basis, N, N, N] for 3D basis fields or [num_basis, N, N] for 2D
        
        if self.dim == 2:
             basis_shape_np = (num_basis_fields, N_GRID, N_GRID)
        else:
             basis_shape_np = (num_basis_fields, N_GRID, N_GRID, N_GRID)

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
                for y in range(N_GRID):
                    for x in range(N_GRID):
                        yf = y * DH
                        xf = x * DH
                        self.basis_fx_np[k, y, x] = np.sin(xf * freq_x + phase_x) * np.cos(yf * freq_y)
                        self.basis_fy_np[k, y, x] = np.cos(xf * freq_x) * np.sin(yf * freq_y + phase)
            else:
                # 3D generation
                phase_x = rng.uniform(0, 2*np.pi)
                for z in range(N_GRID):
                    for y in range(N_GRID):
                        for x in range(N_GRID):
                            zf = z * DH
                            yf = y * DH
                            xf = x * DH
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

    def run_step(self, t):
        """Execute one simulation step t -> t+1."""
        if self.dim == 2:
            # 1. Advect Velocities (Self-Advection)
            wp.launch(advect_mac_u, self.shape, inputs=[self.dt, self.vx_arrays[t], self.vy_arrays[t], self.wx_arrays[t]])
            wp.launch(advect_mac_v, self.shape, inputs=[self.dt, self.vx_arrays[t], self.vy_arrays[t], self.wy_arrays[t]])
            
            if self.bc_type == BoundaryCondition.WALL:
                wp.launch(enforce_noslip_velocity, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t]])
                
            # 2. Apply Forces
            wp.launch(apply_forces, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.basis_fx, self.basis_fy, self.weights, self.dt])
            
            # 3. Pressure
            wp.launch(divergence, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.div_arrays[t], self.bc_type.value])
            
            self.pressure_arrays[t][0].zero_()
            for k in range(self.pressure_iterations):
                wp.launch(jacobi_iter, self.shape, inputs=[self.div_arrays[t], self.pressure_arrays[t][k], self.pressure_arrays[t][k+1], self.bc_type.value])
                
            final_p_idx = self.pressure_iterations
            wp.launch(update_velocities, self.shape, inputs=[self.pressure_arrays[t][final_p_idx], self.wx_arrays[t], self.wy_arrays[t], self.vx_arrays[t+1], self.vy_arrays[t+1], self.bc_type.value])
            
            # 4. Advect Density
            wp.launch(advect_density, self.shape, inputs=[self.dt, self.vx_arrays[t+1], self.vy_arrays[t+1], self.density_arrays[t], self.density_arrays[t+1], self.bc_type.value])
            
        else:
            # 3D
            inputs_uvw = [self.dt, self.vx_arrays[t], self.vy_arrays[t], self.vz_arrays[t]]
            wp.launch(advect_mac_u_3d, self.shape, inputs=inputs_uvw + [self.wx_arrays[t]])
            wp.launch(advect_mac_v_3d, self.shape, inputs=inputs_uvw + [self.wy_arrays[t]])
            wp.launch(advect_mac_w_3d, self.shape, inputs=inputs_uvw + [self.wz_arrays[t]])
            
            if self.bc_type == BoundaryCondition.WALL:
                wp.launch(enforce_noslip_velocity_3d, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t]])
                
            wp.launch(apply_forces_3d, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.basis_fx, self.basis_fy, self.basis_fz, self.weights, self.dt])
            wp.launch(divergence_3d, self.shape, inputs=[self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.div_arrays[t], self.bc_type.value])
            
            self.pressure_arrays[t][0].zero_()
            for k in range(self.pressure_iterations):
                wp.launch(jacobi_iter_3d, self.shape, inputs=[self.div_arrays[t], self.pressure_arrays[t][k], self.pressure_arrays[t][k+1], self.bc_type.value])
                
            final_p_idx = self.pressure_iterations
            wp.launch(update_velocities_3d, self.shape, inputs=[self.pressure_arrays[t][final_p_idx], self.wx_arrays[t], self.wy_arrays[t], self.wz_arrays[t], self.vx_arrays[t+1], self.vy_arrays[t+1], self.vz_arrays[t+1], self.bc_type.value])
            wp.launch(advect_density_3d, self.shape, inputs=[self.dt, self.vx_arrays[t+1], self.vy_arrays[t+1], self.vz_arrays[t+1], self.density_arrays[t], self.density_arrays[t+1], self.bc_type.value])

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
                        self.loss
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
                        self.loss
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
             basis_shape = (N_GRID, N_GRID)
        else:
             basis_shape = (N_GRID, N_GRID, N_GRID)

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
