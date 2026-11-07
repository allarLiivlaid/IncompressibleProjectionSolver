import functions.functions as functions
import functions.initialisation as initialisation
import functions.assist as assist
import os
import time
from collections import deque

os.makedirs("Animation", exist_ok=True)
os.makedirs("Plots", exist_ok=True)
for folder in [ "speed_viz", "pressure_viz", "proj_viz"]:
    if os.path.exists(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            os.unlink(item_path)

#End time, inflow condition, gravity, viscosity and mesh size assigning
endt=2
v_in=1.8; inflow_speed = (v_in, 0.0)
gval=10
nu=0.002
size=0.01
density=1

#General size of simulation box
x0,x1,y0,y1=-1.2,1.2,0.0,1.1 ;limits = (x0,x1,y0,y1)

#Definition of inflow/outflow areas
outflow_start=1.2
test_area=((-1.2,0),(1.2,0.0),(1.2,1.1),(-1.2, 1.1))
inflow_area = ((1.2,1.1), (1.2,1.1), (1.2,1.1), (1.2,1.1))
outflow_area=((-1.2,1.1),(-1.2,1.1),(-1.2,1.1),(-1.2,1.1))

inflow_end=-1.2
outflow_point=([1.1,0.05],[1.1,1.1])
pressure_line=([-1.1,0.05],[1.1,0.05])


#Rigid boundary definition
bound1 =([-1.2,0.0],[1.2,0.0],[1.2,0.05],[-1.2,0.05])
bound2 =([-1.2,0.0],[-1.1,0.0],[-1.1,1.1],[-1.2,1.1])
bound3 =([1.2,0.0],[1.1,0.0],[1.1,1.1],[1.2,1.1])
bound4 =([-1.1,0.05],[1.1,0.05],[1.1,0.25],[-1.1,0.25])

#Creation of occupation, velocity and coordinate matrices
occ, v, lattice_coords = initialisation.initial(lattice_spacing=size, bounds=limits)
occ, v, inserted = initialisation.add_particle_at_position(occ,v,lattice_coords,x=0.0,y=0.8,vx=0.0,vy=-0.0,radius=0.25,particle_type=1)

#Define inflow, boundaries, fill condition
occ = initialisation.boundary_quad(occ,lattice_coords,bound1,particle_type=2)
occ = initialisation.boundary_quad(occ,lattice_coords,bound2,particle_type=2)
occ = initialisation.boundary_quad(occ,lattice_coords,bound3,particle_type=2)
occ = initialisation.boundary_quad(occ,lattice_coords,bound4,particle_type=1)

#Bookkeeping
list_t = [];list_avg_proj = [];list_n_created = [];list_n_deleted = [];list_curl = []
v_cell     = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
speed_cell = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
proj_v_cell = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
bestdt=2*size
t = 0.0+bestdt
endt-=bestdt
Nsteps=round(endt/bestdt)

#Main timestepping loop
counter = 0
tSt = time.perf_counter()
while counter <= Nsteps:
    print(f"[t={t:.5f}] timestep")
    occ_pre = occ.copy()

    #Main timestepping function
    occ, v, x_opt, avg_proj_diff, proj_v, average_proj_v, v_free_f = functions.step_fixed_dt(
        occ,
        v,
        lattice_coords,
        bestdt,
        inflow_cor=[inflow_area],
        speed_fr=speed_cell,
        v_fr=v_cell,
        g=gval,
        nu=nu, dx_space=size,
        proj_v_avg=proj_v_cell
    )

    #Addition of inflow
    n_created = 0
    print("Created", n_created)

    #Suppression of voids and visualisation
    occ, v, p = functions.utility(
        average_proj_v, occ, v, lattice_coords, bestdt,
        timestep=t,
        outflow_cor=outflow_area,
        size=size,
        vel_in=v_in,
        xmax=outflow_start,
        inflowend=inflow_end,
        density=density,
        p_line=outflow_point,
    )

    average_proj_v = average_proj_v * size
    curl, curlRMS = functions.curl_proj_v(average_proj_v, occ, dx_space=size)
    list_curl.append(curlRMS)
    print("Curl:", curlRMS)

    #Deletion of particles from the outflow region
    n_deleted = 0
    print("Deleted",n_deleted)

    #Bookkeeping
    list_t.append(t)
    list_avg_proj.append(avg_proj_diff)
    list_n_created.append(n_created)
    list_n_deleted.append(n_deleted)

    #Final plotting and video creation
    if counter == Nsteps:
        print(f"Calc block: {time.perf_counter() - tSt:.3f}s")
        assist.save_state(occ, v, filename=f"FlowStates/end_v.npz")
        functions.print_function(list_t, list_avg_proj, list_n_created, list_n_deleted, list_curl)
        functions.generate_viz_videos(fps=5)
        error1 = functions.plot_avg_speed_error(nu, occ, speed_cell, v_cell, p, x0, x1, y0, y1, size, lattice_coords,
                                                velocity_file="VelfieldInput/flow.vtu",
                                                out_error_file="Plots/speed_error_lattice_big.png",
                                                out_file="Plots/avg_speed_lattice_big.png", vmin=0.0, vmax=6,
                                                v_in=v_in, error=True)
    counter += 1
    t += bestdt