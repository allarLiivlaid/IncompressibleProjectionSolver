import functions.functions as functions
import functions.initialisation as initialisation
import os
functions.generate_viz_videos(fps=5)

os.makedirs("Animation", exist_ok=True)
os.makedirs("Plots", exist_ok=True)
for folder in ["speed_viz", "pressure_viz"]:
    if os.path.exists(folder):
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            os.unlink(item_path)

#End time, inflow condition, gravity, viscosity and mesh size assigning
endt=0.6
v_in=10.0; inflow_speed = (v_in, 0.0); inflow_speed1=(-v_in, 0.0),
gval=0.0
nu=0.0
size=0.01

#General size of simulation box
x0,x1,y0,y1=-0.3,3.5,0.0,0.8 ;limits = (x0,x1,y0,y1)

#Definition of inflow/outflow areas
inflow_area1 = [(x0,y0), (x0+size*v_in,y0), (x0+size*v_in,y1/2), (x0,y1/2)]
inflow_area2 = [(x1-size*v_in,y1/2), (x1,y1/2), (x1,y1), (x1-size*v_in,y1)]

test_area =([-0,0],[0,0],[0,0],[0,0])
outflow_start=x1-0.3

outflow_area1 =([x1-size*v_in,y0],[x1,y0],[x1,y1/2],[x1-size*v_in,y1/2])
outflow_area2 =([x0,y1/2],[x0+size*v_in,y1/2],[x0+size*v_in,y1],[x0,y1])

right_area = [(x0+size*v_in,y0), (x1-size*v_in,y0), (x1-size*v_in,y1/2), (x0+size*v_in,y1/2)]
left_area = [(x0+size*v_in,y1/2), (x1-size*v_in,y1/2), (x1-size*v_in,y1), (x0+size*v_in,y1)]

single_area = [(1.55,y1/2), (1.56,y1/2), (1.56,y1/2-0.01), (1.55,y1/2-0.01)]

#Creation of occupation, velocity and coordinate matrices
occ, v, lattice_coords = initialisation.initial(lattice_spacing=size, bounds=limits)

#Define inflow, boundaries, fill condition
occ, v,n_created = initialisation.inflow_segment(occ,v,lattice_coords,inflow_area1,inflow_velocity=inflow_speed)
occ, v,n_created = initialisation.inflow_segment(occ,v,lattice_coords,inflow_area2,inflow_velocity=inflow_speed1)

occ, v,n_created = initialisation.inflow_segment(occ,v,lattice_coords,right_area,inflow_velocity=inflow_speed)
occ, v,n_created = initialisation.inflow_segment(occ,v,lattice_coords,left_area,inflow_velocity=inflow_speed1)

occ, v,n_deleted = initialisation.delete_quad(occ,v, lattice_coords,single_area)
occ, v,n_created = initialisation.inflow_segment(occ,v,lattice_coords,single_area,inflow_velocity=(0.0,0.0))

occ, v,n_deleted = initialisation.delete_quad(occ,v, lattice_coords,outflow_area1)
occ, v,n_deleted = initialisation.delete_quad(occ,v, lattice_coords,outflow_area2)

#Bookkeeping
list_t = [];list_avg_proj = [];list_n_created = [];list_n_deleted = []
speed_cell = [[[] for _ in range(int((x1-x0)/size))] for _ in range(int((y1-y0)/size))]
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
        inflow_corners=[inflow_area1, inflow_area2],
        speed_frames=speed_cell,
        g=gval,
        nu=nu
    )

    #Addition of inflow
    occ, v, n_created = initialisation.inflow_segment(occ, v, lattice_coords, inflow_area1, inflow_velocity=inflow_speed)
    occ, v, n_created = initialisation.inflow_segment(occ, v, lattice_coords, inflow_area2,
                                                      inflow_velocity=inflow_speed1)
    print("Created", n_created)

    #Suppression of voids and visualisation
    occ, v = functions.utility(
        proj_v,occ, v, lattice_coords,
        timestep=t,
        occ_origin=occ_pre,
        outflow_corners=outflow_area1,
        size=size,
        vel_in=v_in,
        xmax=outflow_start,
    )

    #Deletion of particles from the outflow region
    occ, v, n_deleted = initialisation.delete_quad(occ, v, lattice_coords, outflow_area1)
    occ, v, n_deleted = initialisation.delete_quad(occ, v, lattice_coords, outflow_area2)
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
        error = functions.plot_avg_speed_error(speed_cell, x0, x1, y0, y1, size, lattice_coords,out_file="Plots/avg_speed_lattice.png", vmin=0.0, vmax=2*v_in)

    counter += 1
    t += bestdt