import numpy as np
import function
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import shutil
for folder in ["vector_viz", "speed_viz"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

center = (0.5, 0.5)
radius = 0.1
line_y = 5
dt_start = 0.0001
dt_min = 0.0001
gval=40
size=0.02

x0,x1,y0,y1=-0.0,1.0,0.0,1.0
limits = (x0,x1,y0,y1)

v_in=4.0
endt=4/v_in
inflow_speed = (v_in, 0.0)

inflow_area = [(-0.3,0), (-0.3+size*v_in,0), (-0.3+size*v_in,1), (-0.3,1)]
outflow_area =([0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0])
bound =([0.0,0.0],[1.0,0.0],[1.0,0.1],[0.0,0.1])

triangle = [
    (0.05, 1.0),
    (0.80, 0.60),
    (0.80, 1.0),
]

p2 = (1.0, 0.0)
p3 = (1.0, 0.6)

occ, v, p, lattice_coords = function.initial(lattice_spacing=size, bounds=limits)
#occ = function.circular_drop(occ,center, radius, lattice_coords)
occ, v, inserted = function.add_particle_at_position(occ,v,lattice_coords,x=0.5,y=0.5,vx=0.0,vy=-5.0)
occ = function.boundary_quad(occ,lattice_coords,bound,particle_type=2)

fig, ax = plt.subplots(figsize=(6,6))
cmap = ListedColormap(["white", "blue", "green"])
norm = BoundaryNorm([0, 1,2,3],cmap.N)
mask = occ > 0
sc = ax.scatter(lattice_coords[mask][:, 0],lattice_coords[mask][:, 1],c=occ[mask],s=10,cmap=cmap,norm=norm)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect('equal')

quiver = None
viz=False
stepviz=True
list_t = []
list_avg_proj = []
list_n_created = []
list_n_deleted = []
speed_cell = [[[] for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
t = 0.0
bestdt=size
n_out=function.count_inflow_sites(lattice_coords,inflow_area)
#n_out=100
print("N out", n_out)
def update(frame):
    global occ,v, t,quiver
    global ts, ts_avg_proj_diff, ts_n_created, ts_n_deleted
    if t >= endt:
        function.print_function(list_t, list_avg_proj, list_n_created, list_n_deleted)
        function.generate_viz_videos(vector_dir="vector_viz",speed_dir="speed_viz",out_vector_mp4="vectors.mp4",out_speed_mp4="speed.mp4",fps=5)
        error = function.plot_avg_speed_error(speed_cell, x0, x1, y0, y1, size, lattice_coords,out_file="avg_speed_lattice.png",vmin=0.8,vmax=12.0)
        raise StopIteration

    if viz:
        function.test_visualisation(occ,v, p, lattice_coords,bestdt,g=gval)

    occ, v, avg_proj_diff, proj_v = function.step_fixed_dt_inflow2(
        occ,
        v,
        lattice_coords,
        bestdt,
        viz=stepviz,
        t=t,
        speed_frames=speed_cell,
        g=gval
    )

    list_t.append(t)
    list_avg_proj.append(avg_proj_diff)
    list_n_created.append(0)
    list_n_deleted.append(0)

    if quiver is not None:
        quiver.remove()
    mask=occ>0
    sc.set_offsets(lattice_coords[mask])
    sc.set_array(occ[mask].astype(float))
    quiver = ax.quiver(lattice_coords[...,0][mask],lattice_coords[...,1][mask],
        proj_v[...,0][mask],proj_v[...,1][mask],color="red",scale=1/size,scale_units="xy",angles="xy")
    sc.set_offsets(lattice_coords[occ > 0])
    ax.set_title(f"t ={t:.5f}, dt ={bestdt:.5f}")

    t+=bestdt
    return sc,quiver

ani=FuncAnimation(fig,update,frames=1, interval=500, blit=False)
plt.show()
