from . import assist
import numpy as np
from scipy.spatial import cKDTree
import meshio
import random

#Creation of appropriate size grids
def initial(lattice_spacing=0.01, bounds=(0.0, 1.0, 0.0, 1.0)):

    x0, x1, y0, y1 = bounds
    nrx = int((x1 - x0) / lattice_spacing)
    nry = int((y1 - y0) / lattice_spacing)

    xs, ys = np.meshgrid(
        np.linspace(x0 + lattice_spacing / 2, x1 - lattice_spacing / 2, nrx),
        np.linspace(y0 + lattice_spacing / 2, y1 - lattice_spacing / 2, nry),
        indexing="xy"
    )

    occ = np.zeros((nry, nrx), dtype=np.int8)
    v = np.zeros((nry, nrx, 2))
    lattice_coords = np.stack([xs, ys], axis=-1)

    return occ, v, lattice_coords

#Boundary particle creation
def boundary_quad(
    occ,
    lattice_coords,
    corners,
    particle_type=2
):

    corners = np.asarray(corners, dtype=float)
    ny, nx = occ.shape

    for j in range(ny):
        for i in range(nx):
            x, y = lattice_coords[j, i]
            if assist.point_in_poly(x, y, corners):
                occ[j, i] = particle_type

    return occ

def boundary_quad_parabolic(
    occ,
    v,
    lattice_coords,
    ends,
    flow_direction,
    thickness_center,
    particle_type=2
):
    ends = np.asarray(ends, dtype=float)
    p0, p1 = ends[0], ends[1]

    seg = p1 - p0
    half_width = np.linalg.norm(seg) / 2.0
    seg_dir = seg / (2.0 * half_width)
    midpoint = 0.5 * (p0 + p1)
    flow_direction = np.asarray(flow_direction, dtype=float)
    flow_direction /= np.linalg.norm(flow_direction)
    dx = np.linalg.norm(lattice_coords[0, 1] - lattice_coords[0, 0])
    ny, nx = occ.shape
    n_created = 0

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] != 0:
                continue

            pt = lattice_coords[j, i]
            r = np.dot(pt - midpoint, seg_dir)
            if abs(r) >half_width:
                continue
            r_norm = abs(r) /half_width
            thickness = max(1, int(round(1.0 + (thickness_center - 1.0) * (1.0 - r_norm ** 2))))
            d_flow = np.dot(pt - midpoint, flow_direction)
            layer = int(np.floor(d_flow / dx))
            if not (0 <= layer < thickness):
                continue
            occ[j, i] = particle_type
            v[j, i] = 0.0
            n_created += 1
    return occ, v, n_created

def boundary_triangle(
    occ,
    lattice_coords,
    corners,
    particle_type=2
):

    corners = np.asarray(corners, dtype=float)
    ny, nx = occ.shape
    a, b, c = corners

    for j in range(ny):
        for i in range(nx):
            p = lattice_coords[j, i]
            if assist.point_in_triangle(p, a, b, c):
                occ[j, i] = particle_type

    return occ


#Deletion for geometry creation or outflow
def delete_quad(
    occ,
    v,
    lattice_coords,
    corners
):
    corners = np.asarray(corners, dtype=float)
    ny, nx = occ.shape
    n_deleted = 0

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] == 0:
                continue
            x, y = lattice_coords[j, i]
            if assist.point_in_poly(x, y, corners):
                occ[j, i] = 0
                v[j, i, :] = 0.0
                n_deleted += 1

    return occ, v,n_deleted

def delete_outflow_parabolic(
    occ, v, lattice_coords, ends, v_center, flow_direction=None
):
    ends = np.asarray(ends, dtype=float)
    p0, p1 = ends[0], ends[1]

    seg = p1 - p0
    half_width = np.linalg.norm(seg) / 2.0
    seg_dir = seg / (2.0 * half_width)
    midpoint = 0.5 * (p0 + p1)

    if flow_direction is None:
        flow_direction = np.array([-seg_dir[1], seg_dir[0]])
    flow_direction = np.asarray(flow_direction, dtype=float)
    flow_direction = flow_direction / np.linalg.norm(flow_direction)

    dx = np.linalg.norm(lattice_coords[0, 1] - lattice_coords[0, 0])
    ny, nx = occ.shape
    n_deleted = 0

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] == 0:
                continue
            x, y = lattice_coords[j, i]
            pt = np.array([x, y])
            r = np.dot(pt - midpoint, seg_dir)
            if abs(r) > half_width:
                continue

            r_norm = abs(r) / half_width
            speed = 1.0 + (v_center - 1.0) * (1.0 - r_norm ** 2)
            depth = max(1, int(round(speed))) * dx

            d_flow = np.dot(pt - midpoint, flow_direction)
            if not (-depth <= d_flow <= 0.0):
                continue

            occ[j, i] = 0
            v[j, i, :] = 0.0
            n_deleted += 1

    return occ, v, n_deleted

def delete_quad_save(
    occ,
    v,
    lattice_coords,
    corners
):
    corners = np.asarray(corners, dtype=float)
    ny, nx = occ.shape
    deleted_velocities = []
    deleted_types = []
    n_deleted = 0

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] == 0:
                continue

            x, y = lattice_coords[j, i]
            if assist.point_in_poly(x, y, corners):
                deleted_velocities.append(v[j, i].copy())
                deleted_types.append(occ[j, i])
                occ[j, i] = 0
                v[j, i, :] = 0.0
                n_deleted += 1

    deleted_velocities = np.asarray(deleted_velocities)
    deleted_types = np.asarray(deleted_types)

    return occ, v, deleted_velocities, deleted_types, n_deleted

#Fill functions with simple velocity input or FEM results
def fill_empty(
    occ,
    v,
    velocity=(1.0, 0.0),
    empty_value=0,
    particle_type=1
):

    velocity = np.asarray(velocity, dtype=float)
    empty_mask = occ == empty_value
    occ[empty_mask] = particle_type
    v[empty_mask] = velocity

    return occ, v

def fill_empty_noise(
    occ,
    v,
    x_bounds=(5.0, 5.0),
    y_bounds=(-2.5, 2.5),
    empty_value=0,
    particle_type=1
):

    empty_mask = occ == empty_value
    occ[empty_mask] = particle_type
    n_empty = np.count_nonzero(empty_mask)
    vx = np.random.uniform(x_bounds[0], x_bounds[1], size=n_empty)
    vy = np.random.uniform(y_bounds[0], y_bounds[1], size=n_empty)
    v[empty_mask] = np.column_stack((vx, vy))
    return occ, v

def fill_parabolic(
    occ,
    v,
    lattice_coords,
    ends,
    v_center,
    flow_direction=None,
    empty_value=0,
    particle_type=1
):
    ends = np.asarray(ends, dtype=float)
    p0, p1 = ends[0], ends[1]

    seg = p1 - p0
    half_width = np.linalg.norm(seg) / 2.0
    seg_dir = seg / (2.0 * half_width)
    midpoint = 0.5 * (p0 + p1)

    if flow_direction is None:
        flow_direction = np.array([-seg_dir[1], seg_dir[0]])
    flow_direction = np.asarray(flow_direction, dtype=float)
    flow_direction /= np.linalg.norm(flow_direction)
    ny, nx = occ.shape

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] != empty_value:
                continue
            pt = lattice_coords[j, i]
            r = np.dot(pt - midpoint, seg_dir)
            r_norm = abs(r) / half_width
            speed = 1.0 + (v_center - 1.0) * (1.0 - r_norm ** 2)
            occ[j, i] = particle_type
            v[j, i] = speed * flow_direction

    return occ, v

def fill_empty_with_velocity(
    occ,
    v,
    lattice_coords,
    velocity_file="VelfieldInput/flow.vtu",
    empty_value=0,
    particle_type=1,
    v_in=1.0,
):

    if velocity_file.endswith('.vtu'):
        positions, velocities, _ = assist.load_su2_vtu(velocity_file)
        velocities *= v_in
    else:
        data       = np.loadtxt(velocity_file, comments='%')
        positions  = data[:, :2]
        velocities = data[:, 2:4]

    tree = cKDTree(positions)
    empty_mask    = occ == empty_value
    empty_indices = np.argwhere(empty_mask)
    if len(empty_indices) == 0:
        return occ, v
    coords   = lattice_coords[empty_indices[:, 0], empty_indices[:, 1]]
    _, nearest = tree.query(coords, workers=-1)

    for k, idx in enumerate(empty_indices):
        j, i = idx
        v[j, i] = velocities[nearest[k]]

    occ[empty_mask] = particle_type

    return occ, v

def inflow_segment(
    occ,
    v,
    lattice_coords,
    corners,
    inflow_velocity,
    particle_type=1
):



    corners = np.asarray(corners, dtype=float)
    inflow_velocity = np.asarray(inflow_velocity, dtype=float)
    ny, nx = occ.shape
    n_created = 0

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] != 0:
                continue
            x, y = lattice_coords[j, i]
            if assist.point_in_poly(x, y, corners):
                occ[j, i] = particle_type
                v[j, i] = inflow_velocity
                n_created += 1

    return occ, v, n_created


def inflow_segment_noise(
        occ,
        v,
        lattice_coords,
        corners,
        inflow_velocity,
        particle_type=1,
        x_bounds=(5.0, 5.0),
        y_bounds=(-2.5, 2.5),
):
    empty_mask = occ == 0
    occ[empty_mask] = particle_type
    n_empty = np.count_nonzero(empty_mask)
    vx = np.random.uniform(x_bounds[0], x_bounds[1], size=n_empty)
    vy = np.random.uniform(y_bounds[0], y_bounds[1], size=n_empty)
    v[empty_mask] = np.column_stack((vx, vy))

    corners = np.asarray(corners, dtype=float)
    inflow_velocity = np.asarray(inflow_velocity, dtype=float)
    ny, nx = occ.shape
    n_created = 0

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] != 0:
                continue
            x, y = lattice_coords[j, i]
            if assist.point_in_poly(x, y, corners):
                occ[j, i] = particle_type
                v[j, i] = inflow_velocity
                n_created += 1

    return occ, v, n_created

def inflow_segment_save(
    occ,
    v,
    lattice_coords,
    corners,
    deleted_velocities,
    deleted_types
):

    corners = np.asarray(corners, dtype=float)
    ny, nx = occ.shape
    n_created = 0
    k = 0

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] != 0:
                continue
            x, y = lattice_coords[j, i]
            if assist.point_in_poly(x, y, corners):
                if k >= len(deleted_velocities):
                    break
                occ[j, i] = deleted_types[k]
                v[j, i] = deleted_velocities[k]
                k += 1
                n_created += 1

    return occ, v, n_created

def inflow_segment_parabolic(
    occ,
    v,
    lattice_coords,
    ends,
    v_center,
    flow_direction=None,
    particle_type=1
):
    ends = np.asarray(ends, dtype=float)
    p0, p1 = ends[0], ends[1]

    seg = p1 - p0
    seg_len = np.linalg.norm(seg)
    half_width = seg_len / 2.0
    seg_dir = seg / seg_len
    midpoint = 0.5 * (p0 + p1)

    if flow_direction is None:
        flow_direction = np.array([-seg_dir[1], seg_dir[0]])
    flow_direction = np.asarray(flow_direction, dtype=float)
    flow_direction /= np.linalg.norm(flow_direction)


    dx = np.linalg.norm(lattice_coords[0, 1] - lattice_coords[0, 0])
    ny, nx = occ.shape
    n_created = 0

    for j in range(ny):
        for i in range(nx):
            if occ[j, i] != 0:
                continue
            pt = lattice_coords[j, i]
            r = np.dot(pt - midpoint, seg_dir)
            if abs(r) > half_width:
                continue
            r_norm = abs(r) / half_width
            speed = 1.0 + (v_center- 1.0) * (1.0 -r_norm**2)
            d_flow = np.dot(pt - midpoint, flow_direction)
            layer = int(np.floor(d_flow / dx))
            thickness = max(1, int(round(speed)))
            if not (0 <= layer < thickness):
                continue
            occ[j, i] = particle_type
            v[j, i] = speed * flow_direction
            n_created += 1

    n_samples = 50
    t = np.linspace(-1.0, 1.0, n_samples)
    parabola_pts = np.array([
        midpoint
        + r * half_width * seg_dir
        + (1.0 + (v_center - 1.0) * (1.0 - r**2)) * flow_direction *dx
        for r in t
    ])
    segment_pts = np.array([
        midpoint + r * half_width * seg_dir
        for r in t[::-1]
    ])
    corners = np.vstack([parabola_pts, segment_pts])

    return occ, v, corners, n_created

def add_particle_at_position(
    occ,
    v,
    lattice_coords,
    x,
    y,
    vx,
    vy,
    particle_type=1,
    radius=0.15,
):
    dx = lattice_coords[..., 0] - x
    dy = lattice_coords[..., 1] - y
    dist2 = dx**2 + dy**2
    mask = dist2 <= radius**2
    occ[mask] = particle_type
    v[mask, 0] = vx
    v[mask, 1] = vy
    n_inserted = np.count_nonzero(mask)

    return occ, v, n_inserted