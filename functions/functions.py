import os
from . import assist
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import heapq
import time
import imageio.v2 as imageio


def print_function(list_t, list_avg_proj, list_n_created, list_n_deleted, list_curl, save_dir="."):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(dpi=300)
    plt.plot(list_t, list_avg_proj, marker='o', color='blue')
    plt.xlabel("$t$")
    plt.ylabel("Average projection difference")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/avg_proj_diff_timeseries.png", bbox_inches="tight")
    plt.close()

    plt.figure(dpi=300)
    plt.plot(list_t, list_curl, marker='o', color='blue')
    plt.xlabel("$t$")
    plt.ylabel("Curl value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/curl_list.png", bbox_inches="tight")
    plt.close()

    plt.figure(dpi=300)
    plt.plot(list_t, list_n_created, marker='o', color='green', label="$n_{created}$")
    plt.plot(list_t, list_n_deleted, marker='x', color='red', label="$n_{deleted}$")
    plt.xlabel("Time")
    plt.ylabel("Number of particles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/n_created_n_deleted_timeseries.png", bbox_inches="tight")
    plt.close()

def make_video(frame_dir, out_mp4,fps):
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not files:
        return
    writer = imageio.get_writer(out_mp4,fps=fps,codec="libx264",quality=8,pixelformat="yuv420p")

    for f in files:
        frame = imageio.imread(os.path.join(frame_dir, f))
        writer.append_data(frame)
    writer.close()
    print(f"[viz] Saved {out_mp4} ({len(files)} frames)")

def generate_viz_videos(
    speed_dir="speed_viz",
    pressure_dir="pressure_viz",
    proj_dir="proj_viz",
    out_speed_mp4="Animation/speed.mp4",
    out_pressure_mp4="Animation/pressure.mp4",
    out_proj_mp4="Animation/proj.mp4",
    fps=4,
):

    if os.path.isdir(speed_dir):
        make_video(speed_dir, out_speed_mp4,fps)
    if os.path.isdir(pressure_dir):
        make_video(pressure_dir, out_pressure_mp4,fps)
    if os.path.isdir(proj_dir):
        make_video(proj_dir, out_proj_mp4, fps)


def plot_avg_speed_error(nu,occ,speed_c,v_c,p_c,x0, x1, y0, y1,size,lat_coor,
    velocity_file="VelfieldInput/flow.vtu",
    out_file="Plots/avg_speed_lattice.png",
    out_error_file="Plots/speed_error_lattice.png",
    out_pressure_file="Plots/avg_pressure_lattice.png",
    out_pressure_error_file="Plots/pressure_error_lattice.png",
    vmin=None,vmax=None,figsize=(6, 6),dpi=300,
    cmap="coolwarm",error_cmap="magma",v_in=1, error=False,
):
    ny = len(speed_c)
    nx = len(speed_c[0])

    l2_num = 0.0
    l2_den = 0.0

    avg_spe = np.zeros((ny, nx))
    avg_v = np.zeros((ny, nx,2))
    for i in range(ny):
        for j in range(nx):
            if speed_c[i][j]:
                avg_spe[i, j] = np.mean(speed_c[i][j])
            if v_c[i][j]:
                avg_v[i, j] = np.mean(v_c[i][j], axis=0)

    assist.save_state(occ, avg_spe, filename=f"FlowStates/avg_speed_mag.npz")
    assist.save_state(occ, avg_v, filename=f"FlowStates/avg_speed_vec.npz")
    assist.save_state(occ, p_c, filename="FlowStates/avg_pressure.npz")

    ref_pos, ref_v, ref_p = assist.load_su2_vtu(velocity_file)
    ref_v*=v_in
    ref_speed = np.linalg.norm(ref_v, axis=1)
    ref_p = np.array(ref_p, dtype=float)*v_in**2
    tree = cKDTree(ref_pos)
    error_grid = np.zeros((ny, nx))
    p_error_grid = np.zeros((ny, nx))
    sum_err = 0.0
    n_used = 0
    max_dist = size
    en_lat = np.zeros((ny, nx))
    en_ref = np.zeros((ny, nx))
    tot_en_err = 0.0
    tot_en_dif = 0.0
    tot_en_ref = 0.0

    p_l2_num = 0.0
    p_l2_den = 0.0


    for i in range(ny):
        for j in range(nx):
            x_cell, y_cell = lat_coor[i, j]
            if avg_spe[i, j] == 0.0 or x_cell > 1.0: #0.7 limit region
                error_grid[i, j] = 0
                continue

            dist, nearest = tree.query([x_cell, y_cell])
            if dist > max_dist:
                continue

            ref_s = ref_speed[nearest]
            diff = avg_spe[i, j] - ref_s

            err = abs(diff)
            error_grid[i, j] = err
            sum_err += err
            n_used += 1
            l2_num += diff ** 2
            l2_den += ref_s ** 2
            e_lattice = 0.5 * avg_spe[i, j]**2
            e_ref = 0.5 * ref_s**2
            en_lat[i, j] = e_lattice
            en_ref[i, j] = e_ref
            tot_en_ref += e_ref
            tot_en_err += abs(e_lattice - e_ref)
            tot_en_dif += e_lattice - e_ref

            if p_c[i, j] != 0.0:  # ← direct 2D index, no list check
                ref_p_val = ref_p[nearest]
                p_diff = p_c[i, j] - ref_p_val
                p_error_grid[i, j] = abs(p_diff)
                p_l2_num += p_diff ** 2
                p_l2_den += ref_p_val ** 2

    tot_en_err/= n_used
    tot_en_dif/= n_used
    rel_energy_error = tot_en_err/tot_en_ref
    rel_energy_diff = tot_en_dif/tot_en_ref

    rel_l2_error = np.sqrt(l2_num / l2_den)
    rel_p_l2_error = np.sqrt(p_l2_num / max(p_l2_den, 1e-30))

    p_sim_vals = []
    p_ref_vals = []

    for i in range(ny):
        for j in range(nx):
            x_cell, y_cell = lat_coor[i, j]
            if p_c[i, j] == 0.0 or x_cell > 0.7:
                continue
            dist, nearest = tree.query([x_cell, y_cell])
            if dist > max_dist:
                continue
            if ref_p[nearest] != 0.0:
                p_sim_vals.append(p_c[i, j])
                p_ref_vals.append(ref_p[nearest])

    p_sim_vals = np.array(p_sim_vals)
    p_ref_vals = np.array(p_ref_vals)

    ratio = p_sim_vals / p_ref_vals
    print(f"[diag] p_sim  range: {p_sim_vals.min():.4f}  {p_sim_vals.max():.4f}")
    print(f"[diag] p_ref  range: {p_ref_vals.min():.4f}  {p_ref_vals.max():.4f}")
    print(f"[diag] ratio  mean:  {ratio.mean():.4f}  std: {ratio.std():.4f}")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    extent = (x0, x1, y0, y1)
    im = ax.imshow(avg_spe,origin="lower",cmap=cmap,extent=extent,vmin=vmin,vmax=vmax,aspect="equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.tight_layout()
    plt.colorbar(im, ax=ax,label=r"$|\mathbf{u}|$",fraction=0.035, pad=0.02)
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(p_c, origin="lower", cmap="RdBu_r", extent=extent, aspect="equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.tight_layout()
    plt.colorbar(im, ax=ax, label=r"$p$", fraction=0.035, pad=0.02)
    fig.savefig(out_pressure_file, bbox_inches="tight")
    plt.close(fig)

    if error:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(error_grid,origin="lower",cmap=error_cmap,extent=extent,aspect="equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.xlim(-0.3, 1.3)
        fig.tight_layout()
        cbar=plt.colorbar(im, ax=ax, label=r"$|v_{sim} - v_{ref}|$",fraction=0.035, pad=0.02)
        cbar.set_label(r"$Error$")
        fig.savefig(out_error_file, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(p_error_grid, origin="lower", cmap=error_cmap,
                       extent=extent, aspect="equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.xlim(-0.3, 1.3)
        fig.tight_layout()
        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(r"$|p_{sim} - p_{ref}|$")
        fig.savefig(out_pressure_error_file, bbox_inches="tight")
        plt.close(fig)

        print(f"[info] rel_l2_error: {rel_l2_error:.4f}")
        print(f"[info] Relative energy error:           {rel_energy_error:.8f}")
        print(f"[info] Relative energy signed difference:{rel_energy_diff:.8f}")

        print(f"[info] rel_p_l2_error: {rel_p_l2_error:.4f}")


    return sum_err



def step_fixed_dt(occ,v,lat_coor,dt,inflow_cor=None,speed_fr=None,v_fr=None,proj_v_avg=None,g=9.81,inflow_penalty=1e10,type2_penalty=1e10,nu=0.0,dx_space=0.02):

    ny,nx = occ.shape
    mov_mask = occ == 1

    if nu > 0:
        fluid_mask = (occ == 1).astype(float)
        wall_mask = (occ == 2).astype(float)
        solidfluid = fluid_mask + wall_mask
        dt_stab = 0.125 * dx_space ** 2 /nu
        n_sub = max(1, int(np.ceil(dt / dt_stab)))
        dt_sub=dt/n_sub
        v_free_f = v.copy()

        weights = [
            (0, 1, 13 / 15), (0, -1, 13 / 15),(1, 0, 13 / 15), (-1, 0, 13 / 15),
            (1, 1, 4 / 15), (1, -1, 4 / 15),(-1, 1, 4 / 15), (-1, -1, 4 / 15),
            (0, 2, -1 / 60), (0, -2, -1 / 60),(2, 0, -1 / 60), (-2, 0, -1 / 60),
            (2, 1, -1 / 30), (2, -1, -1 / 30),(-2, 1, -1 / 30), (-2, -1, -1 / 30),
            (1, 2, -1 / 30), (1, -2, -1 / 30),(-1, 2, -1 / 30), (-1, -2, -1 / 30),
            (2, 2, 0), (2, -2, 0),(-2, 2, 0), (-2, -2, 0),(0, 0, -21 / 5),
        ]

        for i in range(2):
            field = v_free_f[:, :, i].copy()
            for _ in range(n_sub):
                lap = np.zeros_like(field)
                for dy, dx, w in weights:
                    shif_field = np.zeros_like(field)
                    shif_valid = np.zeros_like(solidfluid)
                    ys = slice(max(0, dy), ny + min(0, dy))
                    yt = slice(max(0, -dy), ny - max(0, dy))
                    xs = slice(max(0, dx), nx + min(0, dx))
                    xt = slice(max(0, -dx), nx - max(0, dx))
                    shif_field[yt, xt] = field[ys, xs]
                    shif_valid[yt, xt] = solidfluid[ys, xs]
                    lap += w * shif_valid * (shif_field-field)

                lap /= dx_space ** 2
                field += dt_sub * nu * lap
            v_free_f[:, :, i] = np.where(fluid_mask, field, v_free_f[:, :, i])
    else:
        v_free_f = v.copy()

    t0 = time.perf_counter()
    v_free_f[..., 1] -= g*dt
    x_st_all = lat_coor[mov_mask]
    v_free = v_free_f[mov_mask].copy()
    allw_mask = occ<=2
    allw_coor = lat_coor[allw_mask]
    allw_ind = np.argwhere(allw_mask)
    type2_mask = occ[allw_mask] == 2

    x_opt = x_st_all+v_free*dt
    x_opt_f = np.zeros((ny, nx, 2))
    x_opt_f[mov_mask] = x_opt

    if inflow_cor is not None:
        inflow_mask = np.zeros(len(allw_coor), dtype=np.bool_)
        for poly in inflow_cor:
            poly_arr = np.asarray(poly, dtype=float)
            inflow_mask |=assist.points_in_poly(allw_coor, poly_arr)
        has_inflow = True
    else:
        inflow_mask = np.zeros(len(allw_coor), dtype=np.bool_)
        has_inflow = False

    cost = assist._build_cost(x_opt, allw_coor,type2_mask, inflow_mask,dt, type2_penalty, inflow_penalty, has_inflow)

    print(f"Prep build: {time.perf_counter() - t0:.3f}s")
    print(f"cost_sq shape: {cost.shape}, memory: {cost.nbytes / 1e6:.1f} MB")
    t0 = time.perf_counter()
    row_ind, col_ind = linear_sum_assignment(cost)
    print(f"Calc block: {time.perf_counter() - t0:.3f}s")

    occ_new = occ.copy()
    occ_new[mov_mask] = 0
    v_new = np.zeros_like(v)
    moving_indices_arr = np.argwhere(mov_mask).astype(np.int64)
    allowed_indices_arr = allw_ind.astype(np.int64)

    occ_new, v_new, proj_v, avg_proj_dif = assist._apply_assignments_with_memory(
        occ_new, v_new, v_free,moving_indices_arr, allowed_indices_arr,
        x_st_all, lat_coor,row_ind.astype(np.int64), col_ind.astype(np.int64),dt, dx_space)

    if speed_fr is not None:
        for k in range(len(row_ind)):
            ci = col_ind[k]
            y, x = allw_ind[ci]
            speed_fr[y][x].append(np.linalg.norm(v_new[y, x]))
            v_fr[y][x].append(v_new[y, x].copy())

    occ[:] = occ_new
    v[:] = v_new

    avg_proj_v = np.zeros((ny, nx, 2))
    for y in range(ny):
        for x in range(nx):
            if occ[y, x] == 1:
                proj_v_avg[y][x].append(proj_v[y, x].copy())
            if proj_v_avg[y][x]:
                avg_proj_v[y, x] = np.mean(proj_v_avg[y][x], axis=0)

    return occ, v, x_opt_f, avg_proj_dif, proj_v, avg_proj_v, v_free_f

def utility(proj_v,occ, v, latt_coor, occ_or, timestep,outflow_cor=None,pressure_calc=True,
    viz=True, size=0.02, vel_in=0, xmax=0,inflowend = None, density=1,p_line=None):

    v_origin_field = v.copy()
    collected_occ = [occ.copy()]
    occ_fin = collected_occ[-1].copy()
    v_fin = v_origin_field.copy()
    v_fin[occ_fin != 1] = 0.0
    outflow_points = p_line

    if pressure_calc:
        #p = reconstruct_pressure(proj_v, occ_fin, latt_coor,outflow_points,dx=size,x_inflow_max = inflowend)
        p = reconstruct_pressure(proj_v, occ_fin, latt_coor, outflow_points, dx=size)
        #p =p*density*vel_in**2

    if viz:
        os.makedirs("speed_viz", exist_ok=True)
        os.makedirs("pressure_viz", exist_ok=True)
        os.makedirs("proj_viz", exist_ok=True)
        occu = np.argwhere(occ_fin == 1)
        if len(occu) > 0:
            pos_all = latt_coor[occu[:, 0], occu[:, 1]]
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = outflow_cor
            x_min_out = min(x0, x1, x2, x3)
            x_max_out = max(x0, x1, x2, x3)
            y_min_out = min(y0, y1, y2, y3)
            y_max_out = max(y0, y1, y2, y3)

            new_fil = (occ_fin == 1) & (occ_or == 0)
            in_outflow = ((pos_all[:, 0] >= x_min_out) & (pos_all[:, 0] <= x_max_out)&(pos_all[:, 1] >= y_min_out) & (pos_all[:, 1] <= y_max_out))
            inv = in_outflow|new_fil[occu[:, 0], occu[:, 1]]
            occu = occu[~inv]
        if len(occu) > 0:
            pos = latt_coor[occu[:, 0], occu[:, 1]]
            vel = v_fin[occu[:, 0], occu[:, 1]]
            speed = np.linalg.norm(vel, axis=1)

            xmin= latt_coor[..., 0].min()
            ymin, ymax = latt_coor[..., 1].min(), latt_coor[..., 1].max()

            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(pos[:, 0], pos[:, 1],c=speed, cmap="jet", s=size*300, vmin=0.0,linewidths=0, vmax=2*vel_in)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            cbar =fig.colorbar(sc, ax=ax,fraction=0.035, pad=0.02)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            cbar.set_label(r"$|\mathbf{u}|$")
            ax.set_title(f"T = {timestep:.4f}")
            fig.tight_layout()
            fig.savefig(f"speed_viz/speed_t_{timestep:.4f}.png", dpi=500)
            plt.close(fig)

        if pressure_calc:
            p_vals = p[occu[:, 0], occu[:, 1]]
            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(pos[:, 0], pos[:, 1],c=p_vals, cmap="seismic",s=size*300)
            ax.set_xlim(inflowend, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_aspect("equal")
            cbar=fig.colorbar(sc, ax=ax,fraction=0.035, pad=0.02)
            cbar.set_label(r"$p$")
            ax.set_title(f"T = {timestep:.4f}")
            fig.tight_layout()
            fig.savefig(f"pressure_viz/pressure_t_{timestep:.4f}.png", dpi=500)
            plt.close(fig)

        if len(occu) > 0:
            pos = latt_coor[occu[:, 0], occu[:, 1]]
            proj_v_mat = proj_v[occu[:, 0], occu[:, 1]]
            proj = np.linalg.norm(proj_v_mat, axis=1)

            xmin = latt_coor[..., 0].min()
            ymin, ymax = latt_coor[..., 1].min(), latt_coor[..., 1].max()

            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=proj, cmap="jet", s=size * 300, linewidths=0)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            cbar.set_label(r"$proj$")
            ax.set_title(f"T = {timestep:.4f}")
            fig.tight_layout()
            fig.savefig(f"proj_viz/proj_t_{timestep:.4f}.png", dpi=500)
            plt.close(fig)

    occ[:] = occ_fin
    v[:] = v_fin

    return occ, v, p

def save_flow_state(occ, v, folder="FlowStates"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f"occ_v_file.npz")
    np.savez_compressed(filename, occ=occ, v=v)
    print(f"Saved flow state to {filename}")

def reconstruct_pressure1(proj_v, occ_fin,latt_coor, zero_points, dx, x_inflow_max):
    xn, yn = occ_fin.shape
    stencil = [
        (-1, 0, 1.0),(1, 0, 1.0),(0, -1, 1.0),(0, 1, 1.0),
        (-1, -1, np.sqrt(2)),(-1, 1, np.sqrt(2)),(1, -1, np.sqrt(2)),(1, 1, np.sqrt(2))]
    val = np.zeros((xn, yn), dtype=bool)
    for r in range(xn):
        for c in range(yn):
            if occ_fin[r, c] == 1:
                x = latt_coor[r, c, 0]
                if x >= x_inflow_max:
                    val[r, c] = True

    def near_cell(xycoor):
        px, py = xycoor
        best = None
        best_dist = np.inf
        for xi in range(xn):
            for yi in range(yn):
                if not val[xi, yi]:
                    continue
                cx, cy = latt_coor[xi, yi]
                dist = (cx-px)**2 + (cy-py)**2
                if dist<best_dist:
                    best_dist=dist
                    best=(xi,yi)
        return best

    def dijkstra(src_rc):
        dist = np.full((xn, yn), np.inf)
        pred = np.full((xn, yn, 2), -1, dtype=int)
        sr, sc = src_rc
        dist[sr, sc] = 0.0
        heap = [(0.0, sr, sc)]
        while heap:
            cost, r, c = heapq.heappop(heap)
            if cost > dist[r, c]:
                continue
            for dr, dc, w in stencil:
                nr, nc = r + dr, c + dc
                if 0 <= nr < xn and 0 <= nc < yn and val[nr, nc]:
                    edge_cost = w * dx
                    new_cost = cost + edge_cost
                    if new_cost < dist[nr, nc]:
                        dist[nr, nc] = new_cost
                        pred[nr, nc] = [r, c]
                        heapq.heappush(heap, (new_cost, nr, nc))
        return dist, pred

    def pathint(x, y, pred):
        path = []
        cur_x, cur_y = x, y
        while True:
            px, py = pred[cur_x, cur_y]
            if px == -1:
                break
            path.append((cur_x, cur_y))
            cur_x, cur_y = px, py
        path.append((cur_x, cur_y))
        path = path[::-1]
        p_acc = 0.0
        for i in range(len(path) - 1):
            from_x, from_y = path[i]
            to_x, to_y = path[i + 1]
            from_pos = latt_coor[from_x, from_y]
            to_pos = latt_coor[to_x, to_y]
            step_vec = to_pos - from_pos
            step_unit = step_vec / np.linalg.norm(step_vec)
            pv = proj_v[to_x, to_y]
            p_acc -= float(np.dot(pv, step_unit)) * dx
        return p_acc

    cell  = [near_cell(zp) for zp in zero_points]
    dijkstra_res = [dijkstra(zp) for zp in cell]
    p = np.zeros((xn, yn))
    for i in range(xn):
        for j in range(yn):
            if not val[i, j]:
                continue
            p_sum = 0.0
            n_paths = 0
            for dist, pred in dijkstra_res:
                p_sum += pathint(i, j, pred)
                n_paths += 1
            if n_paths > 0:
                p[i, j] = p_sum/n_paths
    return p

def reconstruct_pressure(
    proj_v,        # (ny, nx, 2)  [0]=x-component, [1]=y-component
    occ_fin,       # (ny, nx)     1=fluid
    latt_coor,     # (ny, nx, 2)  [0]=x, [1]=y
    zero_points,   # ((x0,y0),(x1,y1)) physical coords of zero-pressure line
    dx,
    n_iter=300,
    omega=0.6,
):
    ny, nx = occ_fin.shape
    fluid = occ_fin == 1

    # ── Zero-line mask ───────────────────────────────────────────────────
    p0 = np.array(zero_points[0], dtype=float)
    p1 = np.array(zero_points[1], dtype=float)
    seg = p1 - p0
    seg_len = np.linalg.norm(seg)

    zero_line_mask = np.zeros((ny, nx), dtype=bool)
    for r in range(ny):
        for c in range(nx):
            if not fluid[r, c]:
                continue
            pt = latt_coor[r, c] - p0
            if seg_len < 1e-12:
                dist = np.linalg.norm(pt)
            else:
                t = np.clip(np.dot(pt, seg) / seg_len**2, 0.0, 1.0)
                dist = np.linalg.norm(pt - t * seg)
            if dist <= dx * 0.5:
                zero_line_mask[r, c] = True

    # ── Stencil: uv is in physical (x,y) space ──────────────────────────
    # Array layout: [row, col] = [y, x]
    # So dr=Δrow=Δy, dc=Δcol=Δx → uv = (dc, dr) in (x,y) coords
    INV_SQRT2 = 1.0 / np.sqrt(2.0)
    stencil_raw = [
        (-1,  0,  1/3,  np.array([ 0.0, -1.0])),
        ( 1,  0,  1/3,  np.array([ 0.0,  1.0])),
        ( 0, -1,  1/3,  np.array([-1.0,  0.0])),
        ( 0,  1,  1/3,  np.array([ 1.0,  0.0])),
        (-1, -1,  1/12, np.array([-INV_SQRT2, -INV_SQRT2])),
        (-1,  1,  1/12, np.array([ INV_SQRT2, -INV_SQRT2])),
        ( 1, -1,  1/12, np.array([-INV_SQRT2,  INV_SQRT2])),
        ( 1,  1,  1/12, np.array([ INV_SQRT2,  INV_SQRT2])),
    ]
    total_w = sum(s[2] for s in stencil_raw)
    stencil = [(dr, dc, w / total_w, uv) for dr, dc, w, uv in stencil_raw]

    # ── Iterative solve ──────────────────────────────────────────────────
    p = np.zeros((ny, nx), dtype=float)

    for _ in range(n_iter):
        p_old = p.copy()  # neighbours always read from the frozen snapshot
        p_old[zero_line_mask] = 0.0  # guarantee boundary is clean in the read copy

        for r in range(ny):
            for c in range(nx):
                if not fluid[r, c] or zero_line_mask[r, c]:
                    continue

                p_num = 0.0
                w_total = 0.0

                for dr, dc, w, uv in stencil:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < ny and 0 <= nc < nx):
                        continue
                    if not fluid[nr, nc]:
                        continue
                    grad_contrib = float(np.dot(proj_v[nr, nc], uv)) * dx
                    p_num += w * (p_old[nr, nc] + grad_contrib)  # read from p_old
                    w_total += w

                if w_total > 0:
                    p_new = p_num / w_total
                    p[r, c] += omega * (p_new - p[r, c])

        # Hard reset — redundant now but kept as a guarantee
        p[zero_line_mask] = 0.0

    return p

def curl_proj_v(proj_v, occ, dx_space=0.02):
    vx = proj_v[..., 0]
    vy = proj_v[..., 1]
    dvx_dy = np.gradient(vx, dx_space, axis=0)
    dvy_dx = np.gradient(vy, dx_space, axis=1)
    curl = dvy_dx - dvx_dy
    fluid = (occ == 1)

    interior = (
        fluid &
        (np.roll(occ,  1, axis=0) == 1) &
        (np.roll(occ, -1, axis=0) == 1) &
        (np.roll(occ,  1, axis=1) == 1) &
        (np.roll(occ, -1, axis=1) == 1)
    )

    curl = np.where(interior, curl, 0.0)
    total_curl = np.sum(curl[interior]) * dx_space**2

    return curl, total_curl

def curl_proj_v_viscous(proj_v, v_free_f, occ, nu, dt, dx_space=0.02):
    """
    curl(-grad(p)/rho + nu*lap(v))
    = nu * lap(omega)

    All terms in acceleration units [m/s²].
    """
    ny, nx = occ.shape
    fluid = (occ == 1)

    # nu * lap(v)  [m/s²]
    lap_v = np.zeros_like(v_free_f)
    for i in range(2):
        field = v_free_f[:, :, i]
        lap_v[:, :, i] = (
            np.roll(field,  1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field,  1, axis=1) + np.roll(field, -1, axis=1) -
            4.0 * field
        ) / dx_space**2
    lap_v = np.where(fluid[..., None], lap_v, 0.0)

    # proj_v/dt → acceleration [m/s²]
    proj_a = proj_v / dt

    total = proj_a + nu * lap_v   # [m/s²]

    vx = total[..., 0]
    vy = total[..., 1]
    dvx_dy = np.gradient(vx, dx_space, axis=0)
    dvy_dx = np.gradient(vy, dx_space, axis=1)
    curl = dvy_dx - dvx_dy        # [m/s² / m] = [1/s²]

    interior = (
        fluid &
        (np.roll(occ,  1, axis=0) == 1) &
        (np.roll(occ, -1, axis=0) == 1) &
        (np.roll(occ,  1, axis=1) == 1) &
        (np.roll(occ, -1, axis=1) == 1)
    )
    curl = np.where(interior, curl, 0.0)

    n = np.count_nonzero(interior)
    total_curl = np.sum(curl[interior]) * dx_space**2

    return curl, total_curl