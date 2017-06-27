__author__ = "Eleonora Piersanti (eleonora@simula.no), 2016-2017"
__all__ = []

import pytest
from mpet import *

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

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
    u = [sin(2*pi*x[0])*sin(2*pi*x[1])*sin(omega*t + 1.0),
         sin(2*pi*x[0])*sin(2*pi*x[1])*sin(omega*t + 1.0)]
    p = []
    for i in range(A):
        p += [-(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*sin(omega*t + 1.0)]

    # Simplify symbolics 
    d = len(u)
    u = [sympy.simplify(u[i]) for i in range(d)]
    p = [sympy.simplify(p[i]) for i in range(A)]

    # Compute sigma_ast
    grad_u = [[diff(u[i], x[j]) for j in range(d)] for i in range(d)]
    eps_u = [[0.5*(grad_u[i][j] + grad_u[j][i]) for j in range(d)]
             for i in range(d)]
    div_u = sum([diff(u[i], x[i]) for i in range(d)])
    grad_p = [[diff(p[i], x[j]) for j in range(d)] for i in range(A)]
    sigma_ast = [[2*mu*eps_u[i][j] for j in range(d)] for i in range(d)]
    for i in range(d):
        sigma_ast[i][i] += lmbda*div_u

    # Compute f
    div_sigma_ast = [sum([diff(sigma_ast[i][j], x[j]) for j in range(d)])
                     for i in range(d)]
    f = [- (div_sigma_ast[j] - sum(alpha[i]*grad_p[i][j] for i in range(A)))
         for j in range(d)]
    f = [sympy.simplify(fi) for fi in f]
    
    # Compute g
    g = [None for i in range(A)]
    for i in range(A):
        g[i] = - c*diff(p[i], t) - alpha[i]*diff(div_u, t) \
               + sum([diff(K[i]*grad_p[i][j], x[j]) for j in range(d)]) \
               - sum(S[i][j]*(p[i] - p[j]) for j in range(A))
    g = [sympy.simplify(gi) for gi in g]
        
    # Print sympy expressions as c++ code
    u_str = [sympy.printing.ccode(u[i]) for i in range(d)]
    p_str = [sympy.printing.ccode(p[i]) for i in range(A)]
    f_str = [sympy.printing.ccode(f[i]) for i in range(d)]
    g_str = [sympy.printing.ccode(g[i]) for i in range(A)]
    
    return (u_str, p_str, f_str, g_str)
    
def test_single_run(n=8, M=8):

    "N is t_he mesh size, M the number of time steps."
    
    # Define end time T and timestep dt
    T = 1.0
    dt = float(T/M)

    # Define material parameters in MPET equations
    A = 2
    c = 1.0
    alpha = (1.0, 1.0)
    K = (1.0, 1.0)
    S = ((1.0, 1.0), (1.0, 1.0))
    E = 1.0
    nu = 0.35
    params = dict(A=A, alpha=alpha, K=K, S=S, c=c, nu=nu, E=E)

    info("Deriving exact solutions")
    u_e, p_e, f, g = exact_solutions(params)

    info("Setting up MPET problem")
    mesh = UnitSquareMesh(n, n)
    time = Constant(0.0)
    problem = MPETProblem(mesh, time, params=params)
    problem.f = Expression(f, t=time, degree=3)
    problem.g = [Expression(g[i], t=time, degree=3) for i in range(A)]
    problem.u_bar = Expression(u_e, t=time, degree=3)
    problem.p_bar = [Expression(p_e[i], t=time, degree=3) for i in range(A)]

    on_boundary = CompiledSubDomain("on_boundary")
    on_boundary.mark(problem.momentum_boundary_markers, 0)
    for i in range(A):
        on_boundary.mark(problem.continuity_boundary_markers[i], 0)

    # Set-up solver
    params = dict(dt=dt, theta=1.0)
    solver = MPETSolver(problem, params)

    # Set initial conditions
    VP = solver.up_.function_space()
    V = VP.sub(0).collapse()
    assign(solver.up_.sub(0), interpolate(problem.u_bar, V))
    for i in range(A):
        Q = VP.sub(i+1).collapse()
        assign(solver.up_.sub(i+1), interpolate(problem.p_bar[i], Q))
    
    # Solve
    solutions = solver.solve()
    for (up, t) in solutions:
        print "t =", t
        plot(problem.g[0], mesh=mesh, key="g0")
        plot(problem.g[1], mesh=mesh, key="g1")
        plot(problem.f, mesh=mesh, key="f")

        plot(up.sub(0), key="u")
        plot(up.sub(1), key="p0")
        plot(up.sub(2), key="p1")
        pass

    (u, p0, p1) = up.split()
    print "\| u - u_h \|_0 = ", errornorm(problem.u_bar, u, "L2")
    print "\| p0 - p0_h \|_0 = ", errornorm(problem.p_bar[0], p0, "L2")
    print "\| p1 - p1_h \|_0 = ", errornorm(problem.p_bar[1], p1, "L2")
    
    interactive()
    up_vec_l2_norm = 12.2519728885
    assert(abs(up.vector().norm("l2") - up_vec_l2_norm) < 1.e-10), \
        "l2-norm of solution (%g) not matching reference (%g)." \
        % (up.vector().norm("l2"), up_vec_l2_norm)

if __name__ == "__main__":

    # Just test a single run
    test_single_run(n=16, M=16)

