from __future__ import print_function

__author__ = "Eleonora Piersanti (eleonora@simula.no), 2016-2017"
__all__ = []

import pytest
from mpet import *

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

def convergence_rates(errors, hs):
    import math
    rates = [(math.log(errors[i+1]/errors[i]))/(math.log(hs[i+1]/hs[i]))
             for i in range(len(hs)-1)]
    return rates

def exact_solutions(params):
    import math
    import sympy

    J = params["J"]
    nu = params["nu"]
    E = params["E"]
    alpha = params["alpha"]
    c = params["c"]
    K = params["K"]
    S = params["S"]
    
    # Convert (nu, E) to (mu, labda)
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    mu = E/(2.0*(1.0+nu))
    
    # Sympy utensils
    pi = math.pi
    omega = 2*pi
    sin = sympy.sin
    diff = sympy.diff
    
    x = sympy.symbols(("x[0]", "x[1]"))
    t = sympy.symbols("t")

    # Define exact solutions u and p
    u = [sin(2*pi*x[0] + pi/2.0)*sin(2*pi*x[1] + pi/2.0)*sin(omega*t + t),
         sin(2*pi*x[0] + pi/2.0)*sin(2*pi*x[1] + pi/2.0)*sin(omega*t + t)]

    p = []
    p += [0]
    for i in range(1, J+1):
        p += [-(i)*sin(2*pi*x[0] + pi/2.0)*sin(2*pi*x[1] + pi/2.0)*sin(omega*t + t)]

    d = len(u)
    div_u = sum([diff(u[i], x[i]) for i in range(d)])
    p[0] = lmbda*div_u - sum([alpha[i]*p[i+1] for i in range(J)])

    # Simplify symbolics 
    u = [sympy.simplify(u[i]) for i in range(d)]
    p = [sympy.simplify(p[i]) for i in range(J+1)]

    # Compute sigma_ast
    grad_u = [[diff(u[i], x[j]) for j in range(d)] for i in range(d)]

    eps_u = [[0.5*(grad_u[i][j] + grad_u[j][i]) for j in range(d)]
             for i in range(d)]

    grad_p = [[diff(p[i], x[j]) for j in range(d)] for i in range(J+1)]

    sigma_ast = [[2*mu*eps_u[i][j] for j in range(d)] for i in range(d)]

    div_sigma_ast = [sum([diff(sigma_ast[i][j], x[j]) for j in range(d)])
                     for i in range(d)]

    # for i in range(d):
    #     sigma_ast[i][i] += lmbda*div_u

    # Compute f
    div_sigma_ast = [sum([diff(sigma_ast[i][j], x[j]) for j in range(d)])
                     for i in range(d)]
    f = [-(div_sigma_ast[j] + diff(p[0],x[j])) for j in range(d)]
    f = [sympy.simplify(fi) for fi in f]

    # Compute g
    g = [0 for i in range(J)]
    for i in range(J):
        g[i] = - c[i]*diff(p[i+1], t) \
               - alpha[i]/lmbda *diff((p[0] + sum([alpha[j]*p[j+1] for j in range(J)])), t) \
               + sum([diff(K[i]*grad_p[i+1][j], x[j]) for j in range(d)]) \
               - sum(S[i][j]*(p[i+1] - p[j+1]) for j in range(J))

    g = [sympy.simplify(gi) for gi in g]

    # Print sympy expressions as c++ code
    u_str = [sympy.printing.ccode(u[i]) for i in range(d)]
    p_str = [sympy.printing.ccode(p[i]) for i in range(J+1)]
    f_str = [sympy.printing.ccode(f[i]) for i in range(d)]
    g_str = [sympy.printing.ccode(g[i]) for i in range(J)]
    
    return (u_str, p_str, f_str, g_str)
    
def single_run(n=8, M=8, theta=1.0, direct_solver=True):

    "N is the mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    T = 1.0
    dt = float(T/M)

    # Define material parameters in MPET equations
    J = 2
    c = (1.0, 1.0)
    alpha = (1.0, 1.0)
    K = (1.0, 1.0)
    S = ((1.0, 1.0), (1.0, 1.0))
    E = 1.0
    nu = 0.35
    params = dict(J=J, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Deriving exact solutions")
    u_e, p_e, f, g = exact_solutions(params)

    info("Setting up MPET problem")
    mesh = UnitSquareMesh(n, n)
    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)
    problem.f = Expression(f, t=time, degree=3)
    problem.g = [Expression(g[i], t=time, degree=3) for i in range(J)]
    problem.u_bar = Expression(u_e, t=time, degree=3)

    problem.displacement_nullspace=False
    p_ex = [Expression(p_e[i], t=time, degree=3) for i in range(J+1)]
    problem.p_bar = [Expression(p_e[i], t=time, degree=3) for i in range(1,J+1)]

    # Jpply Dirichlet conditions everywhere (indicated by the zero marker)
    on_boundary = CompiledSubDomain("on_boundary")
    on_boundary.mark(problem.momentum_boundary_markers, 0)
    for i in range(J):
        on_boundary.mark(problem.continuity_boundary_markers[i], 0)

    # Set-up solver
    params = dict(dt=dt, theta=theta, T=T, testing=False, direct_solver=direct_solver)
    solver = MPETTotalPressureSolver(problem, params)

    # Set initial conditions
    # Initial conditions are needed for the total pressure too
    VP = solver.up_.function_space()
    V = VP.sub(0).collapse()
    assign(solver.up_.sub(0), interpolate(problem.u_bar, V))
    for i in range(J+1):
        Q = VP.sub(i+1).collapse()
        assign(solver.up_.sub(i+1), interpolate(p_ex[i], Q))
    
    # Solve
    solutions = solver.solve()
    for (up, t) in solutions:
        info("t = %g" % t)

    (u, p0, p1, p2) = up.split()
    p = (p1, p2)
    u_err_L2 = errornorm(problem.u_bar, u, "L2", degree_rise=5)
    u_err_H1 = errornorm(problem.u_bar, u, "H1", degree_rise=5)
    p_err_L2 = [errornorm(problem.p_bar[i], p[i], "L2", degree_rise=5) for i in range(J)]
    p_err_H1 = [errornorm(problem.p_bar[i], p[i], "H1", degree_rise=5) for i in range(J)]
    h = mesh.hmin()
    return (u_err_L2, u_err_H1, p_err_L2, p_err_H1, h)
    
def convergence_exp(theta, direct_solver):
    import time
    
    # Remove all output from FEniCS (except errors)
    set_log_level(LogLevel.ERROR)

    # Make containers for errors
    u_errorsL2 = []
    u_errorsH1 = []
    p_errorsL2 = [[] for i in range(2)]
    p_errorsH1 = [[] for i in range(2)]
    hs = []

    # Iterate over mesh sizes/time steps and compute errors
    start = time.time()
    if theta == 0.5:
        ns = [8, 16, 32]
        ms = [4, 8, 16]
    else:
        ns = [8, 16, 32]
        ms = [8, 8*4, 8*4**2]

    for (n, m) in zip(ns, ms):
        print("(n, m) = ", (n, m))
        (erruL2, erruH1, errpL2, errpH1, h) = single_run(n, m, theta, direct_solver)
        hs += [h]
        u_errorsL2 += [erruL2]
        u_errorsH1 += [erruH1]
        for (i, errpi) in enumerate(errpL2):
            p_errorsL2[i] += [errpi]
        for (i, errpi) in enumerate(errpH1):
            p_errorsH1[i] += [errpi]

    # Compute convergence rates:
    u_ratesL2 = convergence_rates(u_errorsL2, hs)
    u_ratesH1 = convergence_rates(u_errorsH1, hs)
    p0_ratesL2 = convergence_rates(p_errorsL2[0], hs)
    p1_ratesL2 = convergence_rates(p_errorsL2[1], hs)
    p0_ratesH1 = convergence_rates(p_errorsH1[0], hs)
    p1_ratesH1 = convergence_rates(p_errorsH1[1], hs)

    print("u_ratesL2 = ", u_ratesL2)
    print("u_ratesH1 = ", u_ratesH1)
    print("p0_ratesL2 = ", p0_ratesL2)
    print("p1_ratesL2 = ", p1_ratesL2)
    print("p0_ratesH1 = ", p0_ratesH1)
    print("p1_ratesH1 = ", p1_ratesH1)

    end = time.time()
    print("Time_elapsed = ", end - start)

    # Test that convergence rates are in agreement with theoretical
    # expectation asymptotically
    assert (u_ratesL2[-1] > 1.70), "L2 convergence in u failed"
    assert (u_ratesH1[-1] > 1.70), "H1 convergence in u failed"
    assert (p0_ratesL2[-1] > 1.70), "L2 convergence in p0 failed"
    assert (p1_ratesL2[-1] > 1.70), "L2 convergence in p1 failed"
    assert (p0_ratesH1[-1] > 0.95), "H1 convergence in p0 failed"
    assert (p1_ratesH1[-1] > 0.95), "H1 convergence in p1 failed"

def test_convergence():
    #test for direct_solver=True
    convergence_exp(0.5, True)
    convergence_exp(1.0, True)
    #test for direct_solver=False
    convergence_exp(0.5, False)
    convergence_exp(1.0, False)

if __name__ == "__main__":

    test_convergence()
