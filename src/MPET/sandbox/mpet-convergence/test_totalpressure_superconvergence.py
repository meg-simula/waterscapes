__author__ = "Eleonora Piersanti (eleonora@simula.no), 2016-2017"

# Modified by Marie E. Rognes

import pytest
from mpet import *

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

from test_mpetsolver_semidiscrete import exact_solutions, convergence_rates

def elliptic_interpolant(K_j, p_j, mesh, l_j):

    V = FunctionSpace(mesh, "CG", l_j)
    p = TrialFunction(V)
    q = TestFunction(V)

    W = FunctionSpace(mesh, "CG", 4)
    p_j = interpolate(p_j, W)
    
    a = inner(K_j*grad(p), grad(q))*dx()
    L = inner(K_j*grad(p_j), grad(q))*dx()
    bc = DirichletBC(V, 0.0, "on_boundary") # Applies only in this case

    p = Function(V)
    solve(a == L, p, bc)
    return p
    
def single_run(n=8, M=8, theta=1.0):

    "N is t_he mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    T = 0.5
    dt = float(T/M)

    # Define material parameters in MPET equations
    A = 2
    c = (1.0, 1.0)
    alpha = (1.0, 1.0)
    K = (1.0, 1.0)
    S = ((0.0, 0.0), (0.0, 0.0))
    E = 1.0
    nu = 0.49999
    params = dict(A=A, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Deriving exact solutions")
    u_e, p_e, f, g, p0_e = exact_solutions(params)

    info("Setting up MPET problem")
    mesh = UnitSquareMesh(n, n)
    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)
    problem.f = Expression(f, t=time, degree=4)
    problem.g = [Expression(g[a], t=time, degree=4) for a in range(A)]
    problem.u_bar = Expression(u_e, t=time, degree=4)
    problem.p_bar = [Expression(p_e[a], t=time, degree=4) for a in range(A)]
    
    # Apply Dirichlet conditions everywhere (indicated by the zero marker)
    on_boundary = CompiledSubDomain("on_boundary")
    on_boundary.mark(problem.momentum_boundary_markers, 0)
    for a in range(A):
        on_boundary.mark(problem.continuity_boundary_markers[a], 0)

    # Set-up solver
    params = dict(dt=dt, theta=theta, T=T)
    solver = MPETTotalPressureSolver(problem, params)

    # All solutions are zero at t=0, no need to set initial conditions

    # Solve
    solutions = solver.solve()
    for (up, t) in solutions:
        info("t = %g" % t)
    (u, p0, p1, p2) = up.split()

    # Compute errors
    problem.p_bar = [Expression(p0_e, t=time, degree=4),] + \
                    [Expression(p_e[a], t=time, degree=4) for a in range(A)]
    p = (p0, p1, p2)
    u_err_L2 = errornorm(problem.u_bar, u, "L2")
    u_err_H1 = errornorm(problem.u_bar, u, "H1")
    p_err_L2 = [errornorm(problem.p_bar[a], p[a], "L2") for a in range(A+1)]
    p_err_H1 = [errornorm(problem.p_bar[a], p[a], "H1") for a in range(A+1)]
    h = mesh.hmin()
    
    # Compute auxiliary interpolation of exact solutions
    Pi_p_j = []
    for j in range(A):
        Pi_p_j += [elliptic_interpolant(K[j], problem.p_bar[j+1], mesh, 1)]

    # Compute discretization errors (difference between interpolant
    # and approximation)
    uh_err_L2 = errornorm(problem.u_bar, u, "L2")
    uh_err_H1 = errornorm(problem.u_bar, u, "H1")
    ph_err_L2 = [errornorm(Pi_p_j[a], p[a+1], "L2") for a in range(A)]
    ph_err_H1 = [errornorm(Pi_p_j[a], p[a+1], "H1") for a in range(A)]

        
    return {"e": (u_err_L2, u_err_H1, p_err_L2, p_err_H1, h),
            "e_h": (uh_err_L2, uh_err_H1, ph_err_L2, ph_err_H1, h)}
    
def convergence_exp(theta):
    import time
    
    # Remove all output from FEniCS (except errors)
    set_log_level(LogLevel.ERROR)

    # Make containers for errors
    u_errorsL2 = []
    u_errorsH1 = []
    p_errorsL2 = [[] for i in range(3)]
    p_errorsH1 = [[] for i in range(3)]

    ph_errorsL2 = [[] for i in range(2)]
    ph_errorsH1 = [[] for i in range(2)]

    hs = []

    # Iterate over mesh sizes/time steps and compute errors
    start = time.time()
    ns = [4, 8, 16, 32, 64]
    ms = [4, ]*len(ns)

    for (n, m) in zip(ns, ms):
        print("(n, m) = ", (n, m))
        errors = single_run(n, m, theta)
        (erruL2, erruH1, errpL2, errpH1, h) = errors["e"]
        (erruhL2, erruhH1, errphL2, errphH1, h) = errors["e_h"]

        hs += [h]
        u_errorsL2 += [erruL2]
        u_errorsH1 += [erruH1]
        # Collect total errors
        for (i, errpi) in enumerate(errpL2):
            p_errorsL2[i] += [errpi]
        for (i, errpi) in enumerate(errpH1):
            p_errorsH1[i] += [errpi]

        # Collect approximation errors
        for (i, errpi) in enumerate(errphL2):
            ph_errorsL2[i] += [errpi]
        for (i, errpi) in enumerate(errphH1):
            ph_errorsH1[i] += [errpi]

            
    print("u_errorsL2 = ", ["%0.2e" % i for i in u_errorsL2])
    print("u_errorsH1 = ", ["%0.2e" % i for i in u_errorsH1])
    for a in range(3):
        print("p[%d]_errorsL2 = " % (a), ["%0.2e" % i for i in p_errorsL2[a]])
        print("p[%d]_errorsH1 = " % (a), ["%0.2e" % i for i in p_errorsH1[a]])

    print("")

    for a in range(3):
        print("p[%d]_errorsL2 = " % (a), ["%0.2e" % i for i in p_errorsL2[a]])
        print("p[%d]_errorsH1 = " % (a), ["%0.2e" % i for i in p_errorsH1[a]])

    print("")

    for a in range(2):
        print("ph[%d]_errorsL2 = " % (a+1), ["%0.2e" % i for i in ph_errorsL2[a]])
        print("ph[%d]_errorsH1 = " % (a+1), ["%0.2e" % i for i in ph_errorsH1[a]])

    print("")
    
    # Compute convergence rates:
    u_ratesL2 = convergence_rates(u_errorsL2, hs)
    u_ratesH1 = convergence_rates(u_errorsH1, hs)
    p0_ratesL2 = convergence_rates(p_errorsL2[0], hs)
    p1_ratesL2 = convergence_rates(p_errorsL2[1], hs)
    p2_ratesL2 = convergence_rates(p_errorsL2[2], hs)
    p0_ratesH1 = convergence_rates(p_errorsH1[0], hs)
    p1_ratesH1 = convergence_rates(p_errorsH1[1], hs)
    p2_ratesH1 = convergence_rates(p_errorsH1[2], hs)

    ph1_ratesL2 = convergence_rates(ph_errorsL2[0], hs)
    ph2_ratesL2 = convergence_rates(ph_errorsL2[1], hs)
    ph1_ratesH1 = convergence_rates(ph_errorsH1[0], hs)
    ph2_ratesH1 = convergence_rates(ph_errorsH1[1], hs)

    print("u_ratesL2 = ", ["%0.2f" % i for i in u_ratesL2])
    print("u_ratesH1 = ", ["%0.2f" % i for i in u_ratesH1])
    print("p0_ratesL2 = ", ["%0.2f" % i for i in p0_ratesL2])
    print("p1_ratesL2 = ", ["%0.2f" % i for i in p1_ratesL2])
    print("p2_ratesL2 = ", ["%0.2f" % i for i in p2_ratesL2])
    print("p1_ratesH1 = ", ["%0.2f" % i for i in p1_ratesH1])
    print("p2_ratesH1 = ", ["%0.2f" % i for i in p2_ratesH1])
    print("")
    print("ph1_ratesL2 = ", ["%0.2f" % i for i in ph1_ratesL2])
    print("ph2_ratesL2 = ", ["%0.2f" % i for i in ph2_ratesL2])
    print("ph1_ratesH1 = ", ["%0.2f" % i for i in ph1_ratesH1])
    print("ph2_ratesH1 = ", ["%0.2f" % i for i in ph2_ratesH1])
    
    end = time.time()
    print("Time_elapsed = ", end - start)

def test_convergence():
    convergence_exp(0.5)
    #convergence_exp(1.0)

if __name__ == "__main__":

    test_convergence()
