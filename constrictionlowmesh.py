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
endt=0.45 #seconds
v_in=4.0; inflow_speed = (v_in, 0.0) #m/s
gval=0.0
size=0.02

density=1 #kg/m^3
nu=0.0 #m^2/s

#General size of simulation box
x0,x1,y0,y1=-0.8,1.8,0.0,1.0 ;limits = (x0,x1,y0,y1) #meters

#Definition of inflow/outflow areas
outflow_start=x1-size*v_in*2
inflow_end=x0+size*v_in
inflow_area = ((x0,0), (inflow_end,0), (inflow_end,1), (x0,1))
outflow_area =((outflow_start,0.0),(x1,0.0),(x1,0.5),(outflow_start,0.5))
triangle = ((0.05, 1.0),(0.80, 0.50),(0.80, 1.0))
outflow_points=((1.0, 0.05),(1.0, 0.25),(1.0, 0.45))
outflow_point=((1.0, 0.05),(1.0, 0.5))


#Rigid boundary definition
bound3 =([0.80,0.5],[x1,0.5],[x1,1.0],[0.80,1.0])

#Creation of occupation, velocity and coordinate matrices
occ, v, lattice_coords = initialisation.initial(lattice_spacing=size, bounds=limits)

#Define inflow, boundaries, fill condition
occ, v,n_created = initialisation.inflow_segment(occ,v,lattice_coords,inflow_area,inflow_velocity=inflow_speed)
occ = initialisation.boundary_triangle(occ,lattice_coords,triangle,particle_type=2)
occ = initialisation.boundary_quad(occ,lattice_coords,bound3,particle_type=2)
occ, v = initialisation.fill_empty_with_velocity(occ,v,lattice_coords, v_in=v_in)
occ, v,n_deleted = initialisation.delete_quad(occ,v, lattice_coords,outflow_area)

#Bookkeeping
list_t = [];list_avg_proj = [];list_n_created = [];list_n_deleted = [];list_curl = []
v_cell     = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
speed_cell = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
proj_v_cell = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
bestdt=size
t = 0.0+bestdt
endt-=bestdt
Nsteps=round(endt/bestdt)

#Main timestepping loop
counter = 0

tSt = time.perf_counter()
while counter <= Nsteps:
    print(f"[t={t:.5f}] timestep")
    occ_pre = occ.copy()

    v_prev=v.copy()

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
        nu=nu,dx_space=size,
        proj_v_avg=proj_v_cell
    )

    #Addition of inflow
    occ, v, n_created = initialisation.inflow_segment(occ, v, lattice_coords, inflow_area, inflow_velocity=inflow_speed)
    print("Created", n_created)

    #Suppression of voids and visualisation
    occ, v,p = functions.utility(
        average_proj_v,occ, v, lattice_coords, bestdt,
        timestep=t,
        outflow_cor=outflow_area,
        size=size,
        vel_in=v_in,
        xmax=outflow_start,
        inflowend=inflow_end,
        density=density,
        p_line=outflow_point,
    )
    #Deletion of particles from the outflow region
    occ, v, n_deleted = initialisation.delete_quad(occ, v, lattice_coords, outflow_area)
    print("Deleted",n_deleted)

    average_proj_v = average_proj_v*size
    curl, curlRMS = functions.curl_proj_v(average_proj_v, occ, dx_space=size)
    list_curl.append(curlRMS)
    print("Curl:", curlRMS)

    #Bookkeeping
    list_t.append(t)
    list_avg_proj.append(avg_proj_diff)
    list_n_created.append(n_created)
    list_n_deleted.append(n_deleted)

    #Final plotting and video creation
    if counter == Nsteps:
        print(f"Calc block: {time.perf_counter() - tSt:.3f}s")
        assist.save_state(occ, v, filename=f"FlowStates/end_v.npz")
        functions.print_function(list_t, list_avg_proj, list_n_created, list_n_deleted,list_curl)
        functions.generate_viz_videos(fps=5)
        error1 = functions.plot_avg_speed_error(nu,occ,speed_cell,v_cell,p, x0, x1, y0, y1, size, lattice_coords,velocity_file="VelfieldInput/flow.vtu",out_error_file="Plots/speed_error_lattice_big.png",
                                               out_file="Plots/avg_speed_lattice_big.png", vmin=0.0, vmax=2.2 * v_in,v_in=v_in,error=True)
    counter += 1
    t += bestdt
