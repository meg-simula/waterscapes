__author__ = "Eleonora Piersanti (eleonora@simula.no) and Marie E. Rognes (meg@simula.no), 2017"

import math
import pytest
from mpet import *
from matplotlib import pylab

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

def solid_pressure(u, E, nu):
    "Define the standard linear elastic constitutive equation."
    d = u.geometric_dimension()
    I = Identity(d)
    mu = E/(2.0*((1.0 + nu)))
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    s = 2*mu*sym(grad(u)) + lmbda*div(u)*I
    ps = -1.0/d*sum([s[(i,i)] for i in range(d)])
    return ps

def demo_donut(n=8, M=8, theta=1.0):

    "N is t_he mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    dt = 0.05
    T = 1.0
    # Define material parameters in MPET equations
    A = 1
    c = 1.0e-2
    alpha = (1.0,)
    K = (1.0e-5,)
    S = ((0.0,),)
    E = 500 # Pa
    nu = 0.49
    params = dict(A=A, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Setting up MPET problem")

    # Read mesh
    mesh = Mesh()
    file = HDF5File(mpi_comm_world(), "donut2D.h5", "r")
    file.read(mesh, "/mesh", False)
    file.close()

    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)
    inner_boundary = CompiledSubDomain("on_boundary && ((x[0]*x[0] + x[1]*x[1])<50.0*50.0)")
    outer_boundary = CompiledSubDomain("on_boundary && ((x[0]*x[0] + x[1]*x[1])>50.0*50.0)")

    mmHg2Pa = Constant(133.322)
    n = FacetNormal(mesh)
#    problem.s = Expression("x[0]*x[0] + x[1]*x[1] > 40.0*40.0 ? mmHg2Pa*15.0*sin(2*pi*t) : 0.0", mmHg2Pa=mmHg2Pa, t=time, degree=3)*n
    x = SpatialCoordinate(mesh)
    problem.s = conditional(x[0]*x[0] + x[1]*x[1] > 40.0*40.0, mmHg2Pa*0.15*sin(2*pi*time), 0.0)*n

    problem.displacement_nullspace = True
    outer_boundary.mark(problem.momentum_boundary_markers, 1)
    
    #problem.u_bar = Expression(("0.1*sin(2*pi*t)*x[0]/sqrt((x[0]*x[0] + x[1]*x[1]))","0.1*sin(2*pi*t)*x[1]/sqrt((x[0]*x[0] + x[1]*x[1]))"),\
    #                             t=time, degree=3, domain=problem.mesh)
    inner_boundary.mark(problem.momentum_boundary_markers, 1)

    for i in range(A):
        inner_boundary.mark(problem.continuity_boundary_markers[i], 1)
        outer_boundary.mark(problem.continuity_boundary_markers[i], 1)

    problem.pressure_nullspace = [False]    

    # Set-up solver
    params = dict(dt=dt, theta=theta, T=T)
    solver = MPETSolver(problem, params)

    # Using zero initial conditions by default
    
    # Solve
    points = [Point(30.0,0.0), Point(50.0,0.0), Point(70.0,0.0), Point(100.0,0.0)]
    Fileu = File("u.pvd")
    Filep = File("p.pvd")
    solutions = solver.solve()
    up = solver.up_
    u0_values = [[up(point)[0] for point in points],]
    p_values = [[up(point)[2] for point in points],]

    PS = FunctionSpace(mesh, "CG", 1)
    ps_0 = solid_pressure(up.sub(0), E, nu)
    ps_0 = project(ps_0, PS)

    ps_values = [[ps_0(point) for point in points],]
    times = [0.0,]
    for (up, t) in solutions:
        info("t = %g" % t)
        #Fileu << up.sub(0)
        #Filep << up.sub(1)         
        # plot(up.sub(0), key="u")
        # plot(up.sub(1), key="p0")
        times += [t]
        # u0_values.append([up(point)[0] for point in points])
        p_values.append([up(point)[2] for point in points])

        ps = solid_pressure(up.sub(0), E, nu)
        ps = project(ps, PS)
        ps_values.append([ps(point) for point in points])
        # plot(ps, key="ps")

    # interactive()

    a = zip(*u0_values)
    b = zip(*p_values)
    c = zip(*ps_values)

    # pylab.figure()
    # for (k, i) in enumerate(a):
    #     pylab.plot(times, i, "-*", label ="x_%d" %k)
    # pylab.grid(True)
    # pylab.xlabel("time")
    # pylab.ylabel("u0")
    # pylab.legend()

    # pylab.figure()
    # for (k, i) in enumerate(b):
    #     pylab.plot(times, i, "-*", label ="x_%d" %k)
    # pylab.grid(True)
    # pylab.xlabel("time")
    # pylab.ylabel("p")
    # pylab.legend()

    # pylab.figure()
    # for (k, i) in enumerate(c):
    #     pylab.plot(times, i, "-*", label ="x_%d" %k)
    # pylab.grid(True)
    # pylab.xlabel("time")
    # pylab.ylabel("ps")
    # pylab.legend()
    # pylab.show()


if __name__ == "__main__":

    demo_donut()
