import math
import os
from dolfin import *

directories = ["/nu_0.4999_theta_0.5_dt_0.0125_formulationtype_standard_solvertype_direct/",\
               "/nu_0.4999_theta_0.5_dt_0.0125_formulationtype_total_pressure_solvertype_direct/"]

for d in directories:
    prefix = "results_brain_transfer_1e-06/" + d
    print("prefix = ", prefix)

    mesh = Mesh()
    D = mesh.topology().dim()
    file = HDF5File(MPI.comm_world, "colin27_coarse_boundaries.h5", "r")
    file.read(mesh, "/mesh", False)

    filep_e = HDF5File(MPI.comm_world, prefix + "/p1.h5", "r")
    os.remove(prefix + "/v_e.xdmf")
    os.remove(prefix + "/v_e.h5")
    filev_e = XDMFFile(MPI.comm_world, prefix + "/v_e.xdmf")
    
    Q = FunctionSpace(mesh, "CG", 1)
    DG = VectorFunctionSpace(mesh, "DG", 0)

    K_e = 1.4e-14/8.9e-4*1.0e6
    p_e = Function(Q)
    M = 240

    p0 = Point(89.9, 108.9, 82.3)  # "Origin"
    p1 = Point(102.2, 139.3, 82.3) # Point in the center z-plane, near "butterfly"
    p2 = Point(110.7, 108.9, 98.5) # Point in the center y-plane, near "eye"
               
    points = [p0, p1, p2]

    v_values = [[], [], []]
    times = []

    for i in range(M):

        attribute_name = "/p_0/vector_%d" % i
        filep_e.read(p_e, attribute_name)
        v_e = project(-K_e*grad(p_e), DG)
        t = filep_e.attributes(attribute_name)["timestamp"]
        times += [t]
        filev_e.write_checkpoint(v_e, "v_e", i)

        for (k, x) in enumerate(points):
            v_x = v_e(x)
            v_mag = math.sqrt(sum(v**2 for v in v_x))
            print("v_mag = ", v_mag)
            v_values[k] += [v_mag]

    print("v_values", v_values)            
    import pylab
    import matplotlib
    matplotlib.rcParams["lines.linewidth"] = 3
    matplotlib.rcParams["axes.linewidth"] = 3
    matplotlib.rcParams["axes.labelsize"] = "xx-large"
    matplotlib.rcParams["grid.linewidth"] = 1
    matplotlib.rcParams["xtick.labelsize"] = "xx-large"
    matplotlib.rcParams["ytick.labelsize"] = "xx-large"
    matplotlib.rcParams["legend.fontsize"] = "xx-large"

    pylab.figure(figsize=(9, 8))
    pylab.plot(times, v_values[0], label="x_0")
    pylab.plot(times, v_values[1], label="x_1")
    pylab.plot(times, v_values[2], label="x_2")
    pylab.legend()
    pylab.grid(True)
    pylab.xlabel("t (s)", fontsize=20)
    pylab.ylabel("(mm)", fontsize=20)
    pylab.savefig(prefix+"/ve_mag.png")

