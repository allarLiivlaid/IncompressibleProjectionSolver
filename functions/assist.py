from matplotlib.path import Path
from numba import njit, prange
import numpy as np
import pyvista as pv

#Assisting functions for geometry indentification
def point_in_poly(x, y, poly):
    return Path(poly).contains_point((x, y))

def points_in_poly(points, poly):
    return Path(poly).contains_points(points)

def point_in_triangle(p, a, b, c):
    return Path([a, b, c, a]).contains_point(p)

def load_su2_vtu(vtu_file):
    mesh =pv.read(vtu_file)
    points= mesh.points[:, :2]
    vel =mesh.point_data["Velocity"][:, :2]
    pressure= mesh.point_data["Pressure"]

    return points, vel, pressure

@njit(parallel=True, cache=True)
def _build_cost(x_opt, allowed_coor, type2mask, inflowmask,
                dt, type2_pen, inflow_pen, inflow):
    n_par = len(x_opt)
    n_allowed = len(allowed_coor)
    cost = np.empty((n_par, n_allowed),dtype=np.float64)

    for pi in prange(n_par):
        ox = x_opt[pi, 0]
        oy = x_opt[pi, 1]

        for ai in range(n_allowed):
            dx = allowed_coor[ai, 0]-ox
            dy = allowed_coor[ai, 1]-oy
            c = (dx/dt)*(dx/dt)+(dy/dt)*(dy/dt)
            if type2mask[ai]:
                c += type2_pen
            if (dx*dx + dy*dy) ** 0.5 > 0.2: #0.2 a changeable constant, computational time
                c += type2_pen* 0.5
            if inflow and inflowmask[ai]:
                c += inflow_pen
            cost[pi,ai] = c
    return cost


@njit(cache=True)
def _points_in_poly(points, poly):
    inside = np.zeros(points.shape[0], dtype=np.bool_)
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i, 0], poly[i, 1]
        x1, y1 = poly[(i+1) % n, 0], poly[(i+1) % n, 1]
        for k in range(points.shape[0]):
            px, py = points[k, 0], points[k, 1]
            if ((y0 > py) != (y1 > py)):
                if px < (x1 - x0) * (py - y0) / (y1 - y0 + 1e-15) + x0:
                    inside[k] = ~inside[k]
    return inside


@njit(cache=True)
def _build_backward_cost(fluid_coor, fluid_vel, vacant_coor, dt):
    n_flu  = len(fluid_coor)
    n_vac = len(vacant_coor)
    cost = np.empty((n_flu, n_vac), dtype=np.float64)
    for n in range(n_flu):
        xchange = fluid_coor[n, 0]-fluid_vel[n, 0]*dt
        ychange = fluid_coor[n, 1]-fluid_vel[n, 1]*dt
        for m in range(n_vac):
            xdif = xchange- vacant_coor[m, 0]
            ydif = ychange - vacant_coor[m, 1]
            cost[n, m] = xdif*xdif+ydif*ydif
    return cost

@njit(cache=True, fastmath=True)
def _apply_assignments_with_memory(
    occ_new, v_new, v_free,mov_id, allowed_id,
    x_start, lattice_coor,row_ind, col_ind,dt, dx,
):
    n_ass = len(row_ind)
    n_mov = len(mov_id)
    proj_dif_sum = np.float64(0.0)
    proj_v = np.zeros_like(v_new)

    for n in range(n_ass):
        ri = row_ind[n]
        ci = col_ind[n]
        y_dest, x_dest = allowed_id[ci, 0], allowed_id[ci, 1]
        dest_pos = lattice_coor[y_dest, x_dest]
        st_pos  = x_start[ri]
        opt_x = st_pos[0]+v_free[ri, 0]*dt
        opt_y = st_pos[1]+v_free[ri, 1]*dt
        dif_x = dest_pos[0]-opt_x
        dif_y = dest_pos[1]-opt_y

        if abs(dif_x) < dx*0.5 and abs(dif_y) < dx*0.5:
            v_new[y_dest, x_dest, 0] = v_free[ri, 0]
            v_new[y_dest, x_dest, 1] = v_free[ri, 1]
        else:
            v_new[y_dest, x_dest, 0] = (dest_pos[0]-st_pos[0])/dt
            v_new[y_dest, x_dest, 1] = (dest_pos[1]-st_pos[1])/dt

        occ_new[y_dest, x_dest] = 1
        delta_x = v_new[y_dest, x_dest, 0] - v_free[ri, 0]
        delta_y = v_new[y_dest, x_dest, 1] - v_free[ri, 1]
        pro_y = int(np.round((st_pos[1] - lattice_coor[0, 0, 1]) / dx))
        pro_x = int(np.round((st_pos[0] - lattice_coor[0, 0, 0]) / dx))
        ny, nx = proj_v.shape[0], proj_v.shape[1]
        if pro_y < 0:
            pro_y = 0
        elif pro_y >= ny:
            pro_y = ny-1
        if pro_x < 0:
            pro_x = 0
        elif pro_x >= nx:
            pro_x = nx-1
        proj_v[pro_y, pro_x, 0] =delta_x/dt
        proj_v[pro_y, pro_x, 1] = delta_y/dt
        proj_dif_sum +=(delta_x**2+delta_y**2)**0.5
    avg_proj_dif = proj_dif_sum / n_mov
    return occ_new, v_new, proj_v, avg_proj_dif

def save_state(occ, v, filename="state.npz"):
    np.savez_compressed(filename,occ=occ,v=v)

def load_state(filename="state.npz"):
    data = np.load(filename)
    occ = data["occ"]
    v = data["v"]
    return occ, v


