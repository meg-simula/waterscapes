import math
import os
from dolfin import *

directories = ["/nu_0.4999_theta_0.5_dt_0.0125_formulationtype_standard_solvertype_direct/",\
               "/nu_0.4999_theta_0.5_dt_0.0125_formulationtype_total_pressure_solvertype_direct/"]

for d in directories:

    prefix = "results_brain_transfer_1e-06/" + d
    A = 4
    mesh = Mesh()
    D = mesh.topology().dim()
    file = HDF5File(MPI.comm_world, "colin27_coarse_boundaries.h5", "r")
    file.read(mesh, "/mesh", False)

    filep = [HDF5File(MPI.comm_world, prefix + "/p%d.h5" % (i+1), "r")
             for i in range(A)]

    # for i in range(A):
    #     os.remove(prefix + "/v_" + str(i+1) +".xdmf")
    #     os.remove(prefix + "/v_" + str(i+1) +".h5")

    filev = [XDMFFile(MPI.comm_world, prefix + "/v_"+ str(i+1) +".xdmf")
             for i in range(A)]
    
    Q = FunctionSpace(mesh, "CG", 1)
    p = Function(Q)
    DG = VectorFunctionSpace(mesh, "DG", 0)

    kappa = (1.4e-14, 1.e-10, 1.e-10, 1.e-10) # Vardakis et al, 2016, Oedema
    eta = (8.9e-4, 2.67e-3, 2.67e-3, 2.67e-3)
    scaling = 1.e6 # Scaling from Vardakis values to mm, g, s

    K = [kappa[i]/eta[i]*scaling for i in range(4)]


    M = 240

    p0 = Point(89.9, 108.9, 82.3)  # "Origin"
    p1 = Point(102.2, 139.3, 82.3) # Point in the center z-plane, near "butterfly"
    p2 = Point(110.7, 108.9, 98.5) # Point in the center y-plane, near "eye"
               
    points = [p0, p1, p2]

    v_values = [None for i in range(A)]
    for i in range(A):
        v_values[i] = [[], [], []]

    dt = 0.0125
    times = []
    for i in range(M):

        t = i*dt
        times += [t]
        print("t = ", t)

        # Evaluate clearance from pressures
        for a in range(A):
            attribute_name = "/p_%d/vector_%d" % (a, i)
            filep[a].read(p, attribute_name)
            v = project(-K[a]*grad(p), DG)
            filev[a].write_checkpoint(v, "v_%d" %a, t)
            for (k, x) in enumerate(points):
                v_x = v(x)
                v_mag = math.sqrt(sum(vv**2 for vv in v_x))
                print("v_%d_mag(x) = " % a, v_mag)
                v_values[a][k] += [v_mag]

                
    import pylab
    import matplotlib
    matplotlib.rcParams["lines.linewidth"] = 3
    matplotlib.rcParams["axes.linewidth"] = 3
    matplotlib.rcParams["axes.labelsize"] = "xx-large"
    matplotlib.rcParams["grid.linewidth"] = 1
    matplotlib.rcParams["xtick.labelsize"] = "xx-large"
    matplotlib.rcParams["ytick.labelsize"] = "xx-large"
    matplotlib.rcParams["legend.fontsize"] = "xx-large"
    
    for a in range(A):
        print (v_values[a][0])
        pylab.figure(figsize=(11, 8))
        pylab.plot(times, v_values[a][0], label="x_0")
        pylab.plot(times, v_values[a][1], label="x_1")
        pylab.plot(times, v_values[a][2], label="x_2")
        pylab.legend()
        pylab.grid(True)
        pylab.xlabel("t (s)", fontsize=20)
        pylab.ylabel("(mm/s)", fontsize=20)
        # pylab.xticks(fontsize = 20)
        #pylab.savefig("p%d.png" % a)
        pylab.savefig(prefix+"/v%d.png" % (a+1))
        pylab.close()
