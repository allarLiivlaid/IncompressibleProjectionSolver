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
endt=0.75
v_in=10.0; inflow_speed = (v_in, 0.0)
gval=0.0
nu=0.0
size=0.011

#General size of simulation box
x0,x1,y0,y1=-0.3,1.3,0.0,1.0 ;limits = (x0,x1,y0,y1)

#Definition of inflow/outflow areas
inflow_area = ((-0.3,0), (-0.3+size*v_in,0), (-0.3+size*v_in,1), (-0.3,1))
outflow_area =((1.0,0.0),(1.3,0.0,),(1.3,0.6),(1.0,0.6))
triangle = ((0.05, 1.0),(0.80, 0.60),(0.80, 1.0))
outflow_start=1.0
test_area=((-0.3+size*v_in,0),(1.0,0.0),(1.0,0.6),(0.05, 1.0),(-0.3+size*v_in,1))


#Rigid boundary definition
bound1 =([-0.3,0.0],[1.6,0.0],[1.6,0.05],[-0.3,0.05])
bound2 =([-0.3,1.45],[1.6,1.45],[1.6,1.5],[-0.3,1.5])
bound3 =([0.80,0.6],[1.3,0.6],[1.3,1.0],[0.80,1.0])

#Creation of occupation, velocity and coordinate matrices
occ, v, lattice_coords = initialisation.initial(lattice_spacing=size, bounds=limits)

#Define inflow, boundaries, fill condition
occ, v,n_created = initialisation.inflow_segment(occ,v,lattice_coords,inflow_area,inflow_velocity=inflow_speed)
occ = initialisation.boundary_triangle(occ,lattice_coords,triangle,particle_type=2)
occ = initialisation.boundary_quad(occ,lattice_coords,bound3,particle_type=2)
occ, v = initialisation.fill_empty(occ,v, velocity=(v_in, 0.0))
occ, v,n_deleted = initialisation.delete_quad(occ,v, lattice_coords,outflow_area)

#Bookkeeping
list_t = [];list_avg_proj = [];list_n_created = [];list_n_deleted = []
speed_cell = [[deque(maxlen=20) for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
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
    occ, v, x_opt, avg_proj_diff, proj_v = functions.step_fixed_dt(
        occ,
        v,
        lattice_coords,
        bestdt,
        inflow_corners=inflow_area,
        speed_frames=speed_cell,
        g=gval,
        nu=nu
    )

    #Addition of inflow
    occ, v, n_created = initialisation.inflow_segment(occ, v, lattice_coords, inflow_area, inflow_velocity=inflow_speed)
    print("Created", n_created)

    #Suppression of voids and visualisation
    occ, v, n_back_iters = functions.fill_vacancies_backward(
        occ, v, lattice_coords, bestdt,
        timestep=t,
        occ_origin=occ_pre,
        outflow_corners=outflow_area,
        test_corners=test_area,
        size=size,
        x_opt=x_opt,
        vel_in=v_in,
        xmax=outflow_start
    )
    #Deletion of particles from the outflow region
    occ, v, n_deleted = initialisation.delete_quad(occ, v, lattice_coords, outflow_area)
    print("Deleted",n_deleted)

    #Bookkeeping
    list_t.append(t)
    list_avg_proj.append(avg_proj_diff)
    list_n_created.append(n_created)
    list_n_deleted.append(n_deleted)

    #Final plotting and video creation
    if counter == Nsteps:
        functions.print_function(list_t, list_avg_proj, list_n_created, list_n_deleted)
        functions.generate_viz_videos(fps=5)
        error = functions.plot_avg_speed_error(speed_cell, x0, x1, y0, y1, size, lattice_coords,velocity_file="VelfieldInput/flowsmall.vtu",out_error_file="Plots/speed_error_lattice_small.png",
                                               out_file="Plots/avg_speed_lattice_small.png", vmin=0.0, vmax=2.5*v_in,v_in=v_in)
        error1 = functions.plot_avg_speed_error(speed_cell, x0, x1, y0, y1, size, lattice_coords,velocity_file="VelfieldInput/flowbig.vtu",out_error_file="Plots/speed_error_lattice_big.png",
                                               out_file="Plots/avg_speed_lattice_big.png", vmin=0.0, vmax=2.5 * v_in,v_in=v_in)
    counter += 1
    t += bestdt