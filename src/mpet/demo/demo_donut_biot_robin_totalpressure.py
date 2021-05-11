__author__ = "Eleonora Piersanti (eleonora@simula.no) and Marie E. Rognes (meg@simula.no), 2017"

import math
import pytest
from mpet import *
from matplotlib import pylab
from datetime import datetime
import os
import csv
# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

class My_Expression(Expression):
    def __init__(self, mmHg2Pa, time, degree=None):
        self.time = time
        self.mmHg2Pa = mmHg2Pa
    def eval(self, value, x):
        if (x[0]*x[0] + x[1]*x[1])<50.0*50.0:
            value[0] = self.mmHg2Pa*0.15*sin(2*pi*self.time)
        else:
            value[0] = 0.0

def solid_pressure(u, E, nu):
    "Define the standard linear elastic constitutive equation."
    d = u.geometric_dimension()
    I = Identity(d)
    mu = E/(2.0*((1.0 + nu)))
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    s = 2*mu*sym(grad(u)) + lmbda*div(u)*I
    ps = -1.0/d*sum([s[(i,i)] for i in range(d)])
    return ps

def biot_donut():

    "N is t_he mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    theta = 1.0
    dt = 0.01
    T = 1.0
    # Define material parameters in MPET equations
    A = 2
    c = 1.0e-4
    alpha = (None,1.0)
    K = (None, 1.0e-5,)
    S = ((None, None),(None, 0.0))
    E = 5000 # Pa
    nu = 0.479
    params = dict(A=A, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Setting up MPET problem")

    # Read mesh
    mesh = Mesh()
    file = HDF5File(mpi_comm_world(), "donut2D.h5", "r")
    file.read(mesh, "/mesh", False)
    file.close()
    #mesh = refine(mesh)

    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)
    inner_boundary = CompiledSubDomain("on_boundary && ((x[0]*x[0] + x[1]*x[1])<50.0*50.0)")
    outer_boundary = CompiledSubDomain("on_boundary && ((x[0]*x[0] + x[1]*x[1])>50.0*50.0)")

    mmHg2Pa = Constant(133.322)
    p_val = 0.15*mmHg2Pa

    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    problem.s = conditional((x[0]*x[0] + x[1]*x[1] < 50.0*50.0), -p_val*sin(2*pi*time), 0.0)*n

    problem.displacement_nullspace = False
    OB_M = 1
    IB_M = 1
    
    problem.u_bar = Expression(("0.1*sin(2*pi*t)*x[0]/sqrt((x[0]*x[0] + x[1]*x[1]))",\
                                "0.1*sin(2*pi*t)*x[1]/sqrt((x[0]*x[0] + x[1]*x[1]))"),\
                                 t=time, degree=3, domain=problem.mesh)
    inner_boundary.mark(problem.momentum_boundary_markers, IB_M)
    outer_boundary.mark(problem.momentum_boundary_markers, OB_M)

    IB_C = 0
    OB_C = 2

    #Dirichlet
    # problem.p_bar = [Expression("mmHg2Pa*0.15*sin(2*pi*t)",\
    #                              t=time, mmHg2Pa=mmHg2Pa, degree=3, domain=problem.mesh)]

    problem.p_bar = [Constant(0.0), My_Expression(mmHg2Pa, time, degree=3)]

    for i in range(1, A):
        inner_boundary.mark(problem.continuity_boundary_markers[i], IB_C)
        outer_boundary.mark(problem.continuity_boundary_markers[i], OB_C)
    #Neumann

    #Robin
    problem.beta = [Constant(0.0), conditional((x[0]*x[0] + x[1]*x[1] < 50.0*50.0), 1.0e-12, 1.0e-5)]
    problem.p_robin = [Constant(0.0), conditional((x[0]*x[0] + x[1]*x[1] < 50.0*50.0), p_val*sin(2*pi*time),0.0)]

    # Set-up solver
    params = dict(dt=dt, theta=theta, T=T)
    solver = MPETTotalPressureSolver(problem, params)

    # Using zero initial conditions by default
    
    # Solve
    points = [Point(30.0,0.0), Point(50.0,0.0), Point(70.0,0.0), Point(100.0,0.0)]
    solutions = solver.solve()
    up = solver.up_
    u0_values = [[up(point)[0] for point in points],]
    p_values = [[up(point)[2] for point in points],]

    PS = FunctionSpace(mesh, "CG", 1)
    ps_0 = solid_pressure(up.sub(0), E, nu)
    ps_0 = project(ps_0, PS)

    today = datetime.now()
    foldername = "results/"+today.strftime('%Y%m%d_%H%M%S')
    os.makedirs(foldername)
    Fileu = File(foldername + "/u_robin.pvd")
    Filep = File(foldername + "/p_robin.pvd")

    params_file = csv.writer(open(foldername+"/params.csv", "w"))
    for key, val in problem.params.items():
        params_file.writerow([key, val])

    params_file.writerow(["IB_M", IB_M])
    params_file.writerow(["OB_M", OB_M])
    params_file.writerow(["IB_C", IB_C])
    params_file.writerow(["OB_C", OB_C])    
    ps_values = [[ps_0(point) for point in points],]
    times = [0.0,]
    for (up, t) in solutions:
        info("t = %g" % t)
        Fileu << up.sub(0)
        Filep << up.sub(1)         
        plot(up.sub(0), key="u")
        plot(up.sub(1), key="p0")
        times += [t]
        u0_values.append([up(point)[0] for point in points])
        p_values.append([up(point)[2] for point in points])

        ps = solid_pressure(up.sub(0), E, nu)
        ps = project(ps, PS)
        ps_values.append([ps(point) for point in points])
        plot(ps, key="ps")

    interactive()

    a = zip(*u0_values)
    b = zip(*p_values)
    d = zip(*ps_values)

    pylab.figure()
    for (k, i) in enumerate(a):
        pylab.plot(times, i, "-*", label ="x_%d" %k)
    pylab.grid(True)
    pylab.xlabel("time")
    pylab.ylabel("u0")
    pylab.legend()
    pylab.savefig(foldername + "/u.png")

    pylab.figure()
    for (k, i) in enumerate(b):
        pylab.plot(times, i, "-*", label ="x_%d" %k)
    pylab.grid(True)
    pylab.xlabel("time")
    pylab.ylabel("p")
    pylab.legend()
    pylab.savefig(foldername + "/p.png")

    pylab.figure()
    for (k, i) in enumerate(d):
        pylab.plot(times, i, "-*", label ="x_%d" %k)
    pylab.grid(True)
    pylab.xlabel("time")
    pylab.ylabel("ps")
    pylab.legend()
    pylab.savefig(foldername + "/p_solid.png")
    pylab.show()


if __name__ == "__main__":

    biot_donut()
