import math

from dolfin import *

prefix = "results_brain/nu_0.4999_formulationtype_standard_solvertype_direct"
A = 4

mesh = Mesh()
D = mesh.topology().dim()
file = HDF5File(MPI.comm_world, "colin27_coarse_boundaries.h5", "r")
file.read(mesh, "/mesh", False)

fileu = HDF5File(MPI.comm_world, prefix + "/u.h5", "r")
filep = [HDF5File(MPI.comm_world, prefix + "/p%d.h5" % (i+1), "r")
         for i in range(A)]

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

u = Function(V)
p = Function(Q)

M = 100

p0 = Point(89.9, 108.9, 82.3)  # "Origin"
p1 = Point(102.2, 139.3, 82.3) # Point in the center z-plane, near "butterfly"
p2 = Point(110.7, 108.9, 98.5) # Point in the center y-plane, near "eye"
           
points = [p0, p1, p2]

u_values = [[], [], []]
times = []

p_values = [None for i in range(A)]
for i in range(A):
    p_values[i] = [[], [], []]

for i in range(M):

    attribute_name = "/u/vector_%d" % i
    fileu.read(u, attribute_name)

    t = fileu.attributes(attribute_name)["timestamp"]
    times += [t]
    print("t = ", t)

    # Evaluate norm of displacement u
    for (k, x) in enumerate(points):
        u_x = u(x)
        u_mag = math.sqrt(sum(v**2 for v in u_x))
        print("u_mag = ", u_mag)
        u_values[k] += [u_mag]

    # Evaluate pressures p
    for a in range(A):
        attribute_name = "/p_%d/vector_%d" % (a, i)
        filep[a].read(p, attribute_name)

        for (k, x) in enumerate(points):
            p_x = p(x)
            print("p_%d(x) = " % a, p_x)
            p_values[a][k] += [p_x]

            
import pylab
import matplotlib
matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["axes.labelsize"] = "x-large"
matplotlib.rcParams["grid.linewidth"] = 1
matplotlib.rcParams["xtick.labelsize"] = "x-large"
matplotlib.rcParams["ytick.labelsize"] = "x-large"
matplotlib.rcParams["legend.fontsize"] = "x-large"

pylab.figure(figsize=(8, 8))
pylab.plot(times, u_values[0], label="x_0")
pylab.plot(times, u_values[1], label="x_1")
pylab.plot(times, u_values[2], label="x_2")
pylab.legend()
pylab.grid(True)
pylab.xlabel("t (s)")
pylab.ylabel("(mm)")
#pylab.savefig("u_mag.png")
pylab.savefig(prefix+"/u_mag.png")

mmHg2Pa = 133.32  # Conversion factor from mmHg to Pascal
Pa2mmHg = 1.0/mmHg2Pa

for a in range(A):
    pylab.figure(figsize=(8,8))
    pylab.plot(times, [i*Pa2mmHg for i in p_values[a][0]], label="x_0")
    pylab.plot(times, [i*Pa2mmHg for i in p_values[a][1]], label="x_1")
    pylab.plot(times, [i*Pa2mmHg for i in p_values[a][2]], label="x_2")
    pylab.legend()
    pylab.grid(True)
    pylab.xlabel("t (s)")
    pylab.ylabel("(mmHg)")
    #pylab.savefig("p%d.png" % a)
    pylab.savefig(prefix+"/p%d.png" % a)
x
