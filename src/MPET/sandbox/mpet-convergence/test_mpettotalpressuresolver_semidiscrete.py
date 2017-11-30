__author__ = "Eleonora Piersanti (eleonora@simula.no), 2016-2017"

# Modified by Marie E. Rognes

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

    A = params["A"]
    nu = params["nu"]
    E = params["E"]
    alpha = params["alpha"]
    c = params["c"]
    K = params["K"]
    S = params["S"]
    
    # I is short-hand for range(d):
    gdim = 2
    I = range(gdim)
    As = range(A)
    
    # Convert (nu, E) to (mu, labda)
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    mu = E/(2.0*(1.0+nu))
    print "(mu, lmbda) = ", (mu, lmbda)

    # Sympy utensils
    pi = math.pi
    omega = 2*pi
    sin = sympy.sin
    cos = sympy.cos
    diff = sympy.diff
    
    x = sympy.symbols(("x[0]", "x[1]"))
    t = sympy.symbols("t")

    # Define exact solutions u and p
    u = [(sin(2*pi*x[1])*(-1.0 + cos(2*pi*x[0])) + 1.0/(mu + lmbda)*sin(pi*x[0])*sin(pi*x[1]))*t,
         (sin(2*pi*x[0])*(1.0 - cos(2*pi*x[1])) + 1.0/(mu + lmbda)*sin(pi*x[0])*sin(pi*x[1]))*t]
    p = [-(a+1)*sin(pi*x[0])*sin(pi*x[1])*t for a in As]
        
    div_u = sum([diff(u[i], x[i]) for i in I])

    # Define total pressure p0
    p0 = lmbda*div_u - sum([alpha[a]*p[a] for a in As])

    # Simplify symbolics 
    u = [sympy.simplify(u_i) for u_i in u]
    p = [sympy.simplify(p_i) for p_i in p]
    p0 = sympy.simplify(p0)
    
    # Compute sigma_ast
    grad_u = [[diff(u[i], x[j]) for j in I] for i in I]
    eps_u = [[0.5*(grad_u[i][j] + grad_u[j][i]) for j in I] for i in I]
    grad_p = [[diff(p[a], x[j]) for j in I] for a in As]
    sigma_ast = [[2*mu*eps_u[i][j] for j in I] for i in I]

    # Compute f (vector of length gdim)
    div_sigma_ast = [sum([diff(sigma_ast[i][j], x[j]) for j in I]) for i in I]
    f = [- (div_sigma_ast[j] + diff(p0, x[j])) for j in I]
    f = [sympy.simplify(f_i) for f_i in f]

    # Compute g (vector of length A)
    g = [None for i in range(A)]

    divu_t = diff((p[0] + sum([alpha[b]*p[b] for b in As])), t)/lmbda
    for a in As:
        g[a] = - c[a]*diff(p[a], t) \
               - alpha[a]*divu_t \
               + sum([diff(K[a]*grad_p[a][j], x[j]) for j in I]) \
               - sum(S[a][b]*(p[a] - p[b]) for b in range(A))
    g = [sympy.simplify(gi) for gi in g]
    
    # Print sympy expressions as c++ code
    u_str = [sympy.printing.ccode(u[i]) for i in I]
    p_str = [sympy.printing.ccode(p[a]) for a in As]
    p0_str = sympy.printing.ccode(p0)
    f_str = [sympy.printing.ccode(f[i]) for i in I]
    g_str = [sympy.printing.ccode(g[a]) for a in As]

    print "Exact u = ", u_str
    print "Exact p_0 = ", p0_str
    for a in As:
        print "Exact p_%d = %s" % (a+1, p_str[a])

    return (u_str, p_str, p0_str, f_str, g_str)
    
def single_run(n=8, M=8, theta=1.0):

    "N is t_he mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    T = 1.0
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
    u_e, p_e, p0_e, f, g = exact_solutions(params)

    info("Setting up MPET problem")
    mesh = UnitSquareMesh(n, n)
    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)
    problem.f = Expression(f, t=time, degree=3)
    problem.g = [Expression(g[a], t=time, degree=3) for a in range(A)]
    problem.u_bar = Expression(u_e, t=time, degree=3)
    problem.p_bar = [Expression(p_e[a], t=time, degree=3) for a in range(A)]
    
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
    p = (p1, p2)
    u_err_L2 = errornorm(problem.u_bar, u, "L2")
    u_err_H1 = errornorm(problem.u_bar, u, "H1")
    p_err_L2 = [errornorm(problem.p_bar[a], p[a], "L2") for a in range(A)]
    p_err_H1 = [errornorm(problem.p_bar[a], p[a], "H1") for a in range(A)]
    h = mesh.hmin()
    return (u_err_L2, u_err_H1, p_err_L2, p_err_H1, h)
    
def convergence_exp(theta):
    import time
    
    # Remove all output from FEniCS (except errors)
    set_log_level(ERROR)

    # Make containers for errors
    u_errorsL2 = []
    u_errorsH1 = []
    p_errorsL2 = [[] for i in range(2)]
    p_errorsH1 = [[] for i in range(2)]
    hs = []

    # Iterate over mesh sizes/time steps and compute errors
    start = time.time()
    ns = [4, 8, 16, 32, 64]
    ms = [4, ]*len(ns)

    for (n, m) in zip(ns, ms):
        print "(n, m) = ", (n, m)
        (erruL2, erruH1, errpL2, errpH1, h) = single_run(n, m, theta)
        hs += [h]
        u_errorsL2 += [erruL2]
        u_errorsH1 += [erruH1]
        for (i, errpi) in enumerate(errpL2):
            p_errorsL2[i] += [errpi]
        for (i, errpi) in enumerate(errpH1):
            p_errorsH1[i] += [errpi]

    print "u_errorsL2 = ", ["%0.2e" % i for i in u_errorsL2]
    print "u_errorsH1 = ", ["%0.2e" % i for i in u_errorsH1]
    for a in range(2):
        print "p[%d]_errorsL2 = " % (a+1), ["%0.2e" % i for i in p_errorsL2[a]]
        print "p[%d]_errorsH1 = " % (a+1), ["%0.2e" % i for i in p_errorsH1[a]]

    print
    
    # Compute convergence rates:
    u_ratesL2 = convergence_rates(u_errorsL2, hs)
    u_ratesH1 = convergence_rates(u_errorsH1, hs)
    p0_ratesL2 = convergence_rates(p_errorsL2[0], hs)
    p1_ratesL2 = convergence_rates(p_errorsL2[1], hs)
    p0_ratesH1 = convergence_rates(p_errorsH1[0], hs)
    p1_ratesH1 = convergence_rates(p_errorsH1[1], hs)

    print "u_ratesL2 = ", ["%0.2f" % i for i in u_ratesL2]
    print "u_ratesH1 = ", ["%0.2f" % i for i in u_ratesH1]
    print "p0_ratesL2 = ", ["%0.2f" % i for i in p0_ratesL2]
    print "p1_ratesL2 = ", ["%0.2f" % i for i in p1_ratesL2]
    print "p0_ratesH1 = ", ["%0.2f" % i for i in p0_ratesH1]
    print "p1_ratesH1 = ", ["%0.2f" % i for i in p1_ratesH1]

    end = time.time()
    print "Time_elapsed = ", end - start

def test_convergence():
    convergence_exp(0.5)
    #convergence_exp(1.0)

if __name__ == "__main__":

    test_convergence()
