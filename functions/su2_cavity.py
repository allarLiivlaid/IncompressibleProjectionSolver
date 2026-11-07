"""
SU2 lid-driven-cavity reference utilities.

This module provides two things for the Hungarian/lattice cavity calculation:

1. Interpolate a converged SU2 velocity field onto the lattice and use it as
   the initial condition on fluid cells only.
2. Compare the final Hungarian field against the same SU2 reference over the
   complete 2-D cavity and along the two standard centre lines.

The wall velocities already present in ``v`` are deliberately left untouched.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from scipy.interpolate import (
    LinearNDInterpolator,
    RegularGridInterpolator,
    PchipInterpolator,
)
from scipy.spatial import cKDTree


def _normalise_key(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _extract_velocity_from_vtk(mesh):
    """Return points[:, :2] and velocity[:, :2] from an SU2 ParaView file."""

    # Prefer point data. If SU2/VTK happened to store the solution as cell data,
    # convert it to point data first so that interpolation is unambiguous.
    if len(mesh.point_data) == 0 and len(mesh.cell_data) > 0:
        mesh = mesh.cell_data_to_point_data(pass_cell_data=False)

    pdata = mesh.point_data

    # Most SU2 ParaView output stores velocity as one vector called Velocity.
    for key, values in pdata.items():
        nk = _normalise_key(key)
        arr = np.asarray(values)
        if nk in {"velocity", "vel"} and arr.ndim == 2 and arr.shape[1] >= 2:
            return np.asarray(mesh.points[:, :2], float), np.asarray(arr[:, :2], float)

    # Also accept separate component fields.
    x_aliases = {
        "velocityx", "xvelocity", "velx", "velocity0", "velocityxcomponent",
    }
    y_aliases = {
        "velocityy", "yvelocity", "vely", "velocity1", "velocityycomponent",
    }

    ux = None
    uy = None
    for key, values in pdata.items():
        nk = _normalise_key(key)
        arr = np.asarray(values).reshape(-1)
        if nk in x_aliases:
            ux = arr
        elif nk in y_aliases:
            uy = arr

    if ux is None or uy is None:
        raise KeyError(
            "Could not identify velocity in the SU2 VTU file. "
            f"Available point-data arrays: {list(pdata.keys())}"
        )

    return np.asarray(mesh.points[:, :2], float), np.column_stack((ux, uy)).astype(float)


class _VelocityInterpolator:
    """Fast interpolation with regular-grid detection and nearest fallback."""

    def __init__(self, points, velocity):
        points = np.asarray(points, dtype=float)
        velocity = np.asarray(velocity, dtype=float)

        self.points = points
        self.velocity = velocity
        self.tree = cKDTree(points)

        # The supplied cavity mesh is Cartesian/transfinite. Detect that case
        # and use RegularGridInterpolator, which is faster and smoother than a
        # general Delaunay interpolation for a 256x256 reference mesh.
        xr = np.round(points[:, 0], 12)
        yr = np.round(points[:, 1], 12)
        xu = np.unique(xr)
        yu = np.unique(yr)

        self.structured = (len(xu) * len(yu) == len(points))

        if self.structured:
            ix = np.searchsorted(xu, xr)
            iy = np.searchsorted(yu, yr)

            ug = np.full((len(yu), len(xu)), np.nan, dtype=float)
            vg = np.full_like(ug, np.nan)
            ug[iy, ix] = velocity[:, 0]
            vg[iy, ix] = velocity[:, 1]

            if np.any(~np.isfinite(ug)) or np.any(~np.isfinite(vg)):
                self.structured = False
            else:
                self._u = RegularGridInterpolator(
                    (yu, xu), ug,
                    method="linear", bounds_error=False, fill_value=np.nan,
                )
                self._v = RegularGridInterpolator(
                    (yu, xu), vg,
                    method="linear", bounds_error=False, fill_value=np.nan,
                )

        if not self.structured:
            self._u = LinearNDInterpolator(points, velocity[:, 0], fill_value=np.nan)
            self._v = LinearNDInterpolator(points, velocity[:, 1], fill_value=np.nan)

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        shape = np.broadcast_shapes(x.shape, y.shape)
        x = np.broadcast_to(x, shape)
        y = np.broadcast_to(y, shape)

        q = np.column_stack((y.ravel(), x.ravel())) if self.structured else np.column_stack((x.ravel(), y.ravel()))

        if self.structured:
            u = np.asarray(self._u(q), dtype=float).reshape(shape)
            vv = np.asarray(self._v(q), dtype=float).reshape(shape)
        else:
            u = np.asarray(self._u(x.ravel(), y.ravel()), dtype=float).reshape(shape)
            vv = np.asarray(self._v(x.ravel(), y.ravel()), dtype=float).reshape(shape)

        bad = ~np.isfinite(u) | ~np.isfinite(vv)
        if np.any(bad):
            q_near = np.column_stack((x[bad], y[bad]))
            _, idx = self.tree.query(q_near)
            u[bad] = self.velocity[idx, 0]
            vv[bad] = self.velocity[idx, 1]

        return np.stack((u, vv), axis=-1)


def _centreline_from_lattice(occ, velocity, lattice_coords, orientation, value):
    x = lattice_coords[..., 0]
    y = lattice_coords[..., 1]
    fluid = occ == 1

    if orientation == "vertical":
        perpendicular = x
        along = y
        component = velocity[..., 0]
    elif orientation == "horizontal":
        perpendicular = y
        along = x
        component = velocity[..., 1]
    else:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    if not np.any(fluid):
        return np.array([]), np.array([])

    dmin = np.min(np.abs(perpendicular[fluid] - value))
    band = fluid & np.isclose(
        np.abs(perpendicular - value), dmin,
        atol=max(1e-12, 1e-8 * max(dmin, 1.0)), rtol=0.0,
    )

    c = along[band]
    f = component[band]
    if len(c) == 0:
        return np.array([]), np.array([])

    rounded = np.round(c, 12)
    unique = np.unique(rounded)
    c_out = np.empty(len(unique))
    f_out = np.empty(len(unique))
    for k, val in enumerate(unique):
        m = rounded == val
        c_out[k] = np.mean(c[m])
        f_out[k] = np.mean(f[m])

    order = np.argsort(c_out)
    return c_out[order], f_out[order]


def initialise_lid_driven_cavity(
    occ,
    v,
    lattice_coords,
    *,
    cavity_bounds=(0.0, 1.0, 0.0, 1.0),
    lid_velocity=(1.0, 0.0),
    fluid_type=1,
    wall_type=2,
):
    """
    Fallback cavity occupancy/boundary initialiser.

    Cell centres inside the physical square are fluid. Cell centres outside the
    square (the supplied ghost layers) are wall cells. Ghost cells immediately
    above the physical cavity and horizontally within its width carry the moving
    lid velocity; all other wall/ghost cells are stationary.

    If the user's newer ``initialisation.lid_driven_cavity`` exists, prefer that
    in order to preserve the project's exact wall convention.
    """
    x0, x1, y0, y1 = map(float, cavity_bounds)
    x = lattice_coords[..., 0]
    y = lattice_coords[..., 1]

    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    wall = ~inside

    occ[...] = 0
    v[...] = 0.0
    occ[inside] = fluid_type
    occ[wall] = wall_type

    top = wall & (y > y1) & (x >= x0) & (x <= x1)
    v[top] = np.asarray(lid_velocity, dtype=float)

    return occ, v


class SU2CavityReference:
    """Load one SU2 cavity solution once and reuse its interpolator."""

    def __init__(self, vtu_path):
        self.vtu_path = Path(vtu_path)
        if not self.vtu_path.is_file():
            raise FileNotFoundError(
                f"SU2 reference VTU not found: {self.vtu_path}\n"
                "Run cavity_reference_pycharm.py first."
            )

        mesh = pv.read(str(self.vtu_path))
        self.points, self.velocity = _extract_velocity_from_vtk(mesh)
        self.interpolator = _VelocityInterpolator(self.points, self.velocity)

    def velocity_at(self, x, y):
        return self.interpolator(x, y)

    def check_metadata(
        self,
        metadata_path,
        *,
        lid_speed,
        reynolds,
        cavity_length,
        density=None,
        rtol=1e-10,
        require_metadata=False,
    ):
        """Check that the VTU was generated for the same physical case."""
        metadata_path = Path(metadata_path)
        if not metadata_path.is_file():
            msg = (
                f"SU2 metadata file not found: {metadata_path}. "
                "The VTU will be used, but U/Re/L compatibility cannot be verified."
            )
            if require_metadata:
                raise FileNotFoundError(msg)
            print("WARNING:", msg)
            return None

        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "lid_speed": float(lid_speed),
            "reynolds": float(reynolds),
            "cavity_length": float(cavity_length),
        }
        if density is not None:
            expected["density"] = float(density)

        mismatches = []
        for key, val in expected.items():
            if key not in meta:
                mismatches.append(f"{key}: missing in metadata")
                continue
            ref = float(meta[key])
            if not np.isclose(ref, val, rtol=rtol, atol=1e-12):
                mismatches.append(f"{key}: SU2={ref:g}, Hungarian={val:g}")

        if mismatches:
            raise ValueError(
                "The SU2 reference does not match the Hungarian cavity case:\n  - "
                + "\n  - ".join(mismatches)
                + "\nRegenerate the SU2 reference with matching settings."
            )

        print(
            "SU2 reference metadata matches: "
            f"U_lid={lid_speed:g}, Re={reynolds:g}, L={cavity_length:g}"
        )
        return meta

    def apply_initial_condition(
        self,
        occ,
        v,
        lattice_coords,
        *,
        cavity_bounds=(0.0, 1.0, 0.0, 1.0),
        fluid_type=1,
    ):
        """
        Put the SU2 velocity field on all physical fluid lattice cells.

        Wall/ghost cells are not changed. This should therefore be called after
        ``lid_driven_cavity`` has created the occupancy and moving-wall values.
        """
        x0, x1, y0, y1 = cavity_bounds
        x = lattice_coords[..., 0]
        y = lattice_coords[..., 1]

        mask = (
            (occ == fluid_type)
            & (x >= x0) & (x <= x1)
            & (y >= y0) & (y <= y1)
        )

        if not np.any(mask):
            raise ValueError("No fluid cells were found inside cavity_bounds.")

        ref_v = self.velocity_at(x[mask], y[mask])
        v[mask] = ref_v

        speed = np.linalg.norm(ref_v, axis=1)
        info = {
            "n_cells": int(np.count_nonzero(mask)),
            "mean_speed": float(np.mean(speed)),
            "max_speed": float(np.max(speed)),
        }

        print(
            "Applied SU2 initial condition to "
            f"{info['n_cells']} fluid cells "
            f"(mean |u|={info['mean_speed']:.6g}, max |u|={info['max_speed']:.6g})."
        )
        return v, info

    def compare(
        self,
        occ,
        v,
        lattice_coords,
        *,
        lid_speed,
        cavity_bounds=(0.0, 1.0, 0.0, 1.0),
        out_dir="Plots",
        prefix="lid_cavity_su2",
        verbose=True,
    ):
        """Full-field and centreline comparison against the SU2 reference."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        x0, x1, y0, y1 = cavity_bounds
        lx = x1 - x0
        ly = y1 - y0

        x = lattice_coords[..., 0]
        y = lattice_coords[..., 1]
        fluid = (
            (occ == 1)
            & (x >= x0) & (x <= x1)
            & (y >= y0) & (y <= y1)
        )

        ref = self.velocity_at(x[fluid], y[fluid])
        num = np.asarray(v[fluid], dtype=float)
        diff = num - ref

        speed_num = np.linalg.norm(num, axis=1)
        speed_ref = np.linalg.norm(ref, axis=1)

        vector_rmse_nd = np.sqrt(np.mean(np.sum(diff**2, axis=1))) / lid_speed
        vector_rel_l2 = np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-14)
        speed_rmse_nd = np.sqrt(np.mean((speed_num - speed_ref)**2)) / lid_speed
        speed_rel_l2 = np.linalg.norm(speed_num - speed_ref) / max(np.linalg.norm(speed_ref), 1e-14)

        active = (speed_num > 1e-4 * lid_speed) & (speed_ref > 1e-4 * lid_speed)
        if np.any(active):
            cosine = np.sum(num[active] * ref[active], axis=1) / (
                speed_num[active] * speed_ref[active]
            )
            direction_cosine = float(np.mean(np.clip(cosine, -1.0, 1.0)))
        else:
            direction_cosine = np.nan

        u_ref_grid = np.full(occ.shape, np.nan)
        v_ref_grid = np.full(occ.shape, np.nan)
        speed_ref_grid = np.full(occ.shape, np.nan)
        speed_num_grid = np.full(occ.shape, np.nan)
        speed_err_grid = np.full(occ.shape, np.nan)
        vector_err_grid = np.full(occ.shape, np.nan)

        u_ref_grid[fluid] = ref[:, 0]
        v_ref_grid[fluid] = ref[:, 1]
        speed_ref_grid[fluid] = speed_ref
        speed_num_grid[fluid] = speed_num
        speed_err_grid[fluid] = speed_num - speed_ref
        vector_err_grid[fluid] = np.linalg.norm(diff, axis=1)

        np.savez_compressed(
            out_dir / f"{prefix}_reference_on_lattice.npz",
            u=u_ref_grid,
            v=v_ref_grid,
            speed=speed_ref_grid,
            numerical_speed=speed_num_grid,
            speed_error=speed_err_grid,
            vector_error=vector_err_grid,
            fluid_mask=fluid,
            lattice_coords=lattice_coords,
        )

        def save_field(data, title, cbar_label, filename):
            fig, ax = plt.subplots(figsize=(7.2, 6.0))
            im = ax.pcolormesh(
                (x - x0) / lx,
                (y - y0) / ly,
                np.ma.masked_invalid(data),
                shading="nearest",
            )
            fig.colorbar(im, ax=ax, label=cbar_label)
            ax.set_xlabel("x/L")
            ax.set_ylabel("y/L")
            ax.set_aspect("equal")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(out_dir / filename, dpi=200)
            plt.close(fig)

        save_field(
            speed_ref_grid / lid_speed,
            "SU2 reference speed",
            r"$|u|/U_{lid}$",
            f"{prefix}_su2_speed.png",
        )
        save_field(
            speed_err_grid / lid_speed,
            "Hungarian - SU2 speed-magnitude error",
            r"$(|u|_{H}-|u|_{SU2})/U_{lid}$",
            f"{prefix}_speed_error.png",
        )
        save_field(
            vector_err_grid / lid_speed,
            "Hungarian - SU2 vector-error magnitude",
            r"$|\mathbf{u}_{H}-\mathbf{u}_{SU2}|/U_{lid}$",
            f"{prefix}_vector_error.png",
        )

        # Vertical centreline: u/U_lid vs y/L.
        y_num, u_num = _centreline_from_lattice(
            occ, v, lattice_coords, "vertical", 0.5 * (x0 + x1)
        )
        yy = np.linspace(y0, y1, 501)
        vv_ref = self.velocity_at(np.full_like(yy, 0.5 * (x0 + x1)), yy)

        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        ax.plot(vv_ref[:, 0] / lid_speed, (yy - y0) / ly, lw=2.2, label="SU2")
        ax.scatter(u_num / lid_speed, (y_num - y0) / ly, s=13, alpha=0.55, label="Hungarian lattice")
        if len(y_num) >= 4:
            yc = (y_num - y0) / ly
            p = PchipInterpolator(yc, u_num / lid_speed)
            ys = np.linspace(yc.min(), yc.max(), 500)
            ax.plot(p(ys), ys, lw=1.6, label="Hungarian (PCHIP plot only)")
        ax.set_xlabel(r"$u/U_{lid}$")
        ax.set_ylabel("y/L")
        ax.set_title("Lid-driven cavity: vertical centreline")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_u_centerline.png", dpi=200)
        plt.close(fig)

        # Horizontal centreline: v/U_lid vs x/L.
        x_num, v_num = _centreline_from_lattice(
            occ, v, lattice_coords, "horizontal", 0.5 * (y0 + y1)
        )
        xx = np.linspace(x0, x1, 501)
        vv_ref = self.velocity_at(xx, np.full_like(xx, 0.5 * (y0 + y1)))

        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        ax.plot(vv_ref[:, 1] / lid_speed, (xx - x0) / lx, lw=2.2, label="SU2")
        ax.scatter(v_num / lid_speed, (x_num - x0) / lx, s=13, alpha=0.55, label="Hungarian lattice")
        if len(x_num) >= 4:
            xc = (x_num - x0) / lx
            p = PchipInterpolator(xc, v_num / lid_speed)
            xs = np.linspace(xc.min(), xc.max(), 500)
            ax.plot(p(xs), xs, lw=1.6, label="Hungarian (PCHIP plot only)")
        ax.set_xlabel(r"$v/U_{lid}$")
        ax.set_ylabel("x/L")
        ax.set_title("Lid-driven cavity: horizontal centreline")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_v_centerline.png", dpi=200)
        plt.close(fig)

        report = {
            "vector_rmse_normalized": float(vector_rmse_nd),
            "vector_relative_l2": float(vector_rel_l2),
            "speed_rmse_normalized": float(speed_rmse_nd),
            "speed_relative_l2": float(speed_rel_l2),
            "direction_cosine_mean": float(direction_cosine),
        }

        if verbose:
            print("=" * 68)
            print("LID-DRIVEN CAVITY: HUNGARIAN vs SU2")
            print("=" * 68)
            print(f"  vector RMSE / U_lid             : {vector_rmse_nd:.6f}")
            print(f"  vector relative L2              : {vector_rel_l2:.6f}")
            print(f"  speed-magnitude RMSE / U_lid    : {speed_rmse_nd:.6f}")
            print(f"  speed-magnitude relative L2     : {speed_rel_l2:.6f}")
            print(f"  mean directional cosine         : {direction_cosine:.6f}")
            print("=" * 68)

        return report
