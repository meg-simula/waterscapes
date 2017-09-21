__author__ = "Eleonora Piersanti (eleonora@simula.no) and Marie E. Rognes (meg@simula.no), 2017"

import math
import pytest
from mpet import *

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

def constant_on_the_donut(n=8, M=8, theta=1.0):

    "N is t_he mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    dt = 0.1
    T = 2*dt

    # Define material parameters in MPET equations
    A = 1
    c = 0.0
    alpha = (1.0, 1.0)
    K = (1.e-2,)
    S = ((0.0,),)
    E = 500 # Pa
    nu = 0.35
    params = dict(A=A, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Setting up MPET problem")

    # Read mesh
    mesh = Mesh()
    file = HDF5File(mpi_comm_world(), "donut2D.h5", "r")
    file.read(mesh, "/mesh", False)
    file.close()

    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)

    n = FacetNormal(mesh)
    problem.s = Expression("t", t=time, degree=0)*n
    problem.displacement_nullspace = True
    
    on_boundary = CompiledSubDomain("on_boundary")
    on_boundary.mark(problem.momentum_boundary_markers, 1)

    problem.p_bar = [Expression("-t", t=time, degree=0) for i in range(A)]
    for i in range(A):
        on_boundary.mark(problem.continuity_boundary_markers[i], 0)

    # Set-up solver
    params = dict(dt=dt, theta=theta, T=T)
    solver = MPETSolver(problem, params)

    # Using zero initial conditions by default
    
    # Solve
    solutions = solver.solve()
    for (up, t) in solutions:
        info("t = %g" % t)

    (u, p, r) = up.split(deepcopy=True)
    print norm(u, "L2")
    volume = math.sqrt(assemble(1*dx(domain=mesh)))
    p_x = p((0.0, 50.0))
    assert(abs(p_x + 0.2) < 1.e-8), "Point value of p not matching reference"
    assert(abs(norm(p, "L2")/volume - 0.2) < 1.e-10), "Point value of p not matching reference"

def constant_on_the_donut_nullspaces(n=8, M=8, theta=1.0):
        
    "N is t_he mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    dt = 0.1
    T = 2*dt

    # Define material parameters in MPET equations
    A = 2
    c = 0.0
    alpha = (1.0, 1.0)
    K = (1.e-2, 1.e-1)
    S = ((0.0, 0.0), (0.0, 0.0))
    E = 500 # Pa
    nu = 0.35
    params = dict(A=A, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Setting up MPET problem")

    # Read mesh
    mesh = Mesh()
    file = HDF5File(mpi_comm_world(), "donut2D.h5", "r")
    file.read(mesh, "/mesh", False)
    file.close()

    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)

    # Mark the entire boundary as Neumann boundary for the momentum equation
    n = FacetNormal(mesh)
    problem.s = Expression("t", t=time, degree=0)*n
    problem.displacement_nullspace = True
    on_boundary = CompiledSubDomain("on_boundary")
    on_boundary.mark(problem.momentum_boundary_markers, 1)

    # Mark the entire boundary as Neumann boundary for the continuity
    # equation(s)
    problem.pressure_nullspace = (True, True)
    on_boundary.mark(problem.continuity_boundary_markers[0], 1)

    # Set-up solver
    params = dict(dt=dt, theta=theta, T=T)
    solver = MPETSolver(problem, params)

    # Using zero initial conditions by default
    
    # Solve
    solutions = solver.solve()
    for (up, t) in solutions:
        info("t = %g" % t)

def test_donut():
    constant_on_the_donut()
    constant_on_the_donut_nullspaces()
    
if __name__ == "__main__":

    test_donut()
