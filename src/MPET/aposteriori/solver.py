import numpy

from dolfin import *
from mpet import *

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

# Remove logging (lower value, less logging removed).
set_log_level(60)

def str2exp(u):
    """ Converts strings to FEniCS string expresions """
    from sympy2fenics import str2sympy, sympy2exp
    u = str2sympy(u)
    return sympy2exp(u)

def mpet_source_terms(u, p, params):
    "Compute source terms f, g in MPET equations symbolically."
    
    from sympy2fenics import str2sympy, sympy2exp, grad, div, sym, eps
    from sympy import eye, symbols, diff, simplify

    t = symbols("t")

    E, nu = params["E"], params["nu"] 
    mu = E/(2.0*(1.0 + nu))
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))

    alpha = params["alpha"]
    A = params["A"]
    J = range(A)
    c = params["c"]
    K = params["K"]
    S = params["S"]
    
    # Convert from string to sympy representation
    u = str2sympy(u)
    p = str2sympy(p)

    sigma = lambda u: 2.0*mu*eps(u) + lmbda*div(u)*eye(len(u))
    
    # Compute right-hand side for momentum equation (split sum intentional)
    f = -div(sigma(u))
    for j in J:
        f += alpha[j]*grad(p[j]).T
    
    # Compute right-hand sides for mass equations
    g = [c[i]*diff(p[i], t) + alpha[i]*div(diff(u, t)) - K[i]*div(grad(p[i])) for i in J]
    for i in J:
        for j in J:
            g[i] += S[j][i]*(p[i] - p[j]) 

    # NB: MPET module assumes -g! Consider fixing this.
    g = [-g_i for g_i in g] 

    # Convert from sympy to FEniCS Expression C++ strings
    f = sympy2exp(simplify(f))
    g = [sympy2exp(simplify(g_i)) for g_i in g]
    
    return (f, g)
    
def exact_solutions(params):
    """Return exact solutions u and p (list) and right hand sides f and g
    (list) as FEniCS Expression strings.
    """
    
    u = "(cos(pi*x)*sin(pi*y)*sin(pi*t), sin(pi*x)*cos(pi*y)*sin(pi*t))"
    p = "(sin(pi*x)*cos(pi*y)*sin(2*pi*t), cos(pi*x)*sin(pi*y)*sin(2*pi*t))" 	

    # Derive source terms f and g (list) as FEniCS Expression strings
    f, g = mpet_source_terms(u, p, params)

    # Convert strings to FEniCS Expression strings
    u = str2exp(u)
    p = str2exp(p)

    return (u, p, f, g)

def solve_example1(n, T, dt, params):
    
    # Mesh and time
    mesh = UnitSquareMesh(n, n)
    time = Constant(0.0)
    J = range(params["A"])
    
    # Get exact solutions and corresponding right-hand sides as
    # Expression strings
    (u_str, p_str, f_str, g_str) = exact_solutions(params)

    # Convert to FEniCS expressions and attach the time
    u_e = Expression(u_str, degree=3, t=time)
    p_e = [Expression(p_str[j], degree=3, t=time) for j in J]
    f = Expression(f_str, degree=3, t=time)
    g = [Expression(g_str[i], degree=3, t=time) for i in J]

    # Create MPETProblem object and attach sources and Dirichlet boundary values 
    problem = MPETProblem(mesh, time, params=params)
    problem.f = f
    problem.g = g
    problem.u_bar = u_e
    problem.p_bar = p_e
    
    # Apply Dirichlet conditions everywhere for all fields (indicated by the zero marker)
    on_boundary = CompiledSubDomain("on_boundary")
    on_boundary.mark(problem.momentum_boundary_markers, 0)
    for j in J:
        on_boundary.mark(problem.continuity_boundary_markers[j], 0)

    # Set-up solver
    theta = 1.0
    params = dict(dt=dt, theta=theta, T=T, direct_solver=True)
    solver = MPETSolver(problem, params)

    # No need to set initial conditions since all zero at start in
    # this example

    # Solve
    solutions = solver.solve()
    for (up, t) in solutions:
        pass

    oops = up.split()
    u = oops[0]
    p = oops[1:]

    # Compute errors
    u_err_L2 = errornorm(problem.u_bar, u, "L2")
    u_err_H1 = errornorm(problem.u_bar, u, "H1")
    p_err_L2 = [errornorm(problem.p_bar[i], p[i], "L2") for i in J]
    p_err_H1 = [errornorm(problem.p_bar[i], p[i], "H1") for i in J]

    return (mesh.hmax(), u_err_L2, u_err_H1, p_err_L2, p_err_H1)

def rate(E, h):
    rates = [numpy.log(E[i+1]/E[i])/numpy.log(h[i+1]/h[i]) for i in range(len(E)-1)]
    return rates
    
def uniform_convergence_2networks():

    T = 1.0 
    
    # Convert mu, lbmda to E, nu
    mu = 1.
    lmbda = 10.0
    E = mu*(3*lmbda+2*mu)/(lmbda + mu)
    nu = lmbda/(2*(lmbda + mu))
    params = dict(E=E, nu=nu, alpha=(0.5, 0.5), A=2, c=(1.0, 1.0), K=(1.0, 1.0),
                  S=((0.0, 1.0), (1.0, 0.0)))
    print("params = ", params)

    errors = dict(u_L2=[], u_H1=[], p1_L2=[], p1_H1=[], p2_L2=[], p2_H1=[])
    hs = []
    dts = [1./2**2, 1./4**2, 1./8**2, 1./16**2]
    for (i, n) in enumerate([4, 8, 16, 32]):
        # Just initialize table rows
        for key in errors.keys():
            errors[key] += [[]]

        for dt in dts:
            # Solve system and compute errors
            (h, u_err_L2, u_err_H1, p_err_L2, p_err_H1) = solve_example1(n, T, dt, params)

            # Group computed errors in table
            errors["u_L2"][i] += [u_err_L2]
            errors["u_H1"][i] += [u_err_H1]
            errors["p1_L2"][i] += [p_err_L2[0]]
            errors["p1_H1"][i] += [p_err_H1[0]]
            errors["p2_L2"][i] += [p_err_L2[1]]
            errors["p2_H1"][i] += [p_err_H1[1]]

        hs.append(h)
        
    numpy.set_printoptions(threshold=20, edgeitems=10, linewidth=140,
                           formatter=dict(float=lambda x: "%.3e" % x))
    for key in errors.keys():
        print(key)
        print(numpy.array(errors[key]))
        rates = rate([errors[key][i][i] for i in range(len(dts))], hs)
        print("Diagonal rate = ", rates)
        print()
    print("Refinement in time (horizontal, columns), in space (vertical, rows)")
    print("Delta x")
    print(hs)
    print("Delta t")
    print(dts)
    
if __name__ == "__main__":
    uniform_convergence_2networks()
