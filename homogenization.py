import functions.functions as functions
import functions.initialisation as initialisation
import os
from collections import deque


os.makedirs("Animation", exist_ok=True)
os.makedirs("Plots", exist_ok=True)
for folder in ["vector_viz", "speed_viz", "proj_viz", "pressure_viz"]:
    if os.path.exists(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            os.unlink(item_path)

#End time, inflow condition, gravity, viscosity and mesh size assigning
endt=0.3
v_in=4.0; inflow_speed = (v_in, 0.0)
gval=0.0
size=0.015

nu=0.0
density=1 #kg/m^3

#General size of simulation box
x0,x1,y0,y1=-0.0,1.8,0.0,2.0 ;limits = (x0,x1,y0,y1)

#Definition of inflow/outflow areas
outflow_start=x1-size*v_in
inflow_end=x0+size*v_in

inflow_area = [(x0,y0+0.05), (inflow_end,y0+0.05), (inflow_end,y1-0.05), (x0,y1-0.05)]
test_area =([inflow_end,y0+0.05],[x1-0.3,y0+0.05],[x1-0.3,y1-0.05],[inflow_end,y1-0.05])
outflow_area =([outflow_start,y0+0.05],[x1,y0+0.05],[x1,y1-0.05],[outflow_start,y1-0.05])
outflow_points=((outflow_start, y0+0.05),(outflow_start, (y0+y1)/2),(outflow_start, y1-0.05))
outflow_point=((outflow_start, y0+0.05),(outflow_start, y1-0.05))

#Rigid boundary definition
bound1 =([x0,y0],[x1,y0],[x1,y0+0.05],[x0,y0+0.05])
bound2 =([x0,y1-0.05],[x1,y1-0.05],[x1,y1],[x0,y1])

#Creation of occupation, velocity and coordinate matrices
occ, v, lattice_coords = initialisation.initial(lattice_spacing=size, bounds=limits)

#Define inflow, boundaries, fill condition
occ, v,n_created = initialisation.inflow_segment(occ,v,lattice_coords,inflow_area,inflow_velocity=inflow_speed)
occ, v, inserted = initialisation.add_particle_at_position(occ,v,lattice_coords,x=0.4,y=0.2,vx=0.0,vy=-0.0,radius=0.1,particle_type=2)
occ, v, inserted = initialisation.add_particle_at_position(occ,v,lattice_coords,x=0.4,y=0.6,vx=0.0,vy=-0.0,radius=0.1,particle_type=2)
occ, v, inserted = initialisation.add_particle_at_position(occ,v,lattice_coords,x=0.4,y=1.0,vx=0.0,vy=-0.0,radius=0.1,particle_type=2)
occ, v, inserted = initialisation.add_particle_at_position(occ,v,lattice_coords,x=0.4,y=1.4,vx=0.0,vy=-0.0,radius=0.1,particle_type=2)
occ, v, inserted = initialisation.add_particle_at_position(occ,v,lattice_coords,x=0.4,y=1.8,vx=0.0,vy=-0.0,radius=0.1,particle_type=2)
occ = initialisation.boundary_quad(occ,lattice_coords,bound1,particle_type=2)
occ = initialisation.boundary_quad(occ,lattice_coords,bound2,particle_type=2)
#occ, v = initialisation.fill_empty_with_velocity(occ,v,lattice_coords, v_in=v_in,velocity_file="VelfieldInput/vortex.txt")
occ, v = initialisation.fill_empty(occ,v,velocity=(v_in, 0.0))
occ, v,n_deleted = initialisation.delete_quad(occ,v, lattice_coords,outflow_area)

#Bookkeeping
list_t = [];list_avg_proj = [];list_n_created = [];list_n_deleted = [];list_curl = []
speed_cell = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
v_cell     = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
proj_v_cell = [[deque(maxlen=25) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
bestdt=size
t = 0.0+bestdt
endt-=bestdt
Nsteps=round(endt/bestdt)

#Main timestepping loop
counter = 0
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
        nu=nu,dx_space=size,
        proj_v_avg=proj_v_cell,
    )

    #Addition of inflow
    occ, v, n_created = initialisation.inflow_segment(occ, v, lattice_coords, inflow_area, inflow_velocity=inflow_speed)
    print("Created", n_created)

    #Visualisation
    occ, v, p = functions.utility(
        proj_v,occ, v, lattice_coords,
        timestep=t,
        occ_or=occ_pre,
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

    average_proj_v = average_proj_v
    #curl, curlRMS = functions.curl_proj_v(average_proj_v, occ, dx_space=size)
    curl, curlRMS = functions.curl_proj_v_viscous(proj_v, v_free_f, occ, nu, bestdt, dx_space=size)
    list_curl.append(curlRMS)
    print("Curl:", curlRMS)

    #Bookkeeping
    list_t.append(t)
    list_avg_proj.append(avg_proj_diff)
    list_n_created.append(n_created)
    list_n_deleted.append(n_deleted)

    #Final plotting and video creation
    if counter == Nsteps:
        functions.print_function(list_t, list_avg_proj, list_n_created, list_n_deleted, list_curl)
        functions.generate_viz_videos(fps=5)
        # error = functions.plot_avg_speed_error(speed_cell, x0, x1, y0, y1, size, lattice_coords,out_file="Plots/avg_speed_lattice.png", vmin=0.0, vmax=2*v_in)
        error1 = functions.plot_avg_speed_error(nu, occ, speed_cell, v_cell,p, x0, x1, y0, y1, size, lattice_coords,
                                                out_error_file="Plots/speed_error_lattice_big.png",
                                                out_file="Plots/avg_speed_lattice_big.png", vmin=0.0, vmax=2.2 * v_in,
                                                v_in=v_in)
    counter += 1
    t += bestdt