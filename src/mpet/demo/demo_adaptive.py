from mpet import *

def rate(E, h):
    rates = [numpy.log(E[i+1]/E[i])/numpy.log(h[i+1]/h[i]) for i in range(len(E)-1)]
    return numpy.array(rates)

def str2exp(u):
    """Converts strings to FEniCS string expresions."""
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
    Js = range(params["J"])
    c = params["c"]
    K = params["K"]
    S = params["S"]
    
    # Convert from string to sympy representation
    u = str2sympy(u)
    p = str2sympy(p)

    sigma = lambda u: 2.0*mu*eps(u) + lmbda*div(u)*eye(len(u))
    
    # Compute right-hand side for momentum equation (split sum intentional)
    f = -div(sigma(u))
    for j in Js:
        f += alpha[j]*grad(p[j]).T
    
    # Compute right-hand sides for mass equations (split sum intentional)
    g = [c[i]*diff(p[i], t) + alpha[i]*div(diff(u, t)) - K[i]*div(grad(p[i])) for i in Js]
    for i in Js:
        for j in Js:
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
    
    u = "(0.1*cos(pi*x)*sin(pi*y)*sin(pi*t), 0.1*sin(pi*x)*cos(pi*y)*sin(pi*t))"
    p = "(sin(pi*x)*cos(pi*y)*sin(2*pi*t), cos(pi*x)*sin(pi*y)*sin(2*pi*t))" 	

    # Derive source terms f and g (list) as FEniCS Expression strings
    f, g = mpet_source_terms(u, p, params)

    # Convert strings to FEniCS Expression strings
    u = str2exp(u)
    p = str2exp(p)

    return (u, p, f, g)

def main():

    # -----------------------------------------------------------------------------
    # Define the problem
    # -----------------------------------------------------------------------------
    
    # Define mesh
    n = 2
    mesh = UnitSquareMesh(n, n)

    # Define time (and Constant time_ for previous time)
    time = Constant(0.0) 

    # Define initial time step
    dt0 = 0.2
    dt = Constant(dt0)

    # Set end-time
    T = 0.5

    # Define material parameters
    mu = 1.0
    lmbda = 10.0
    E, nu = convert_to_E_nu(mu, lmbda)
    J = 2
    material = dict(E=E, nu=nu, alpha=(0.5, 0.5), J=J, c=(1.0, 1.0), K=(1.0, 1.0),
                    S=((0.0, 1.0), (1.0, 0.0)))

    # Get exact solutions and corresponding right-hand sides as
    # Expression strings
    (u_str, p_str, f_str, g_str) = exact_solutions(material)

    # Convert to FEniCS expressions and attach the time
    Js = range(J)
    u_e = Expression(u_str, degree=3, t=time)
    p_e = [Expression(p_str[j], degree=3, t=time) for j in Js]
    f = Expression(f_str, degree=3, t=time)
    #f_ = Expression(f_str, degree=3, t=time_)
    g = [Expression(g_str[j], degree=3, t=time) for j in Js]

    # Create MPETProblem object and attach sources and Dirichlet boundary values 
    problem = MPETProblem(mesh, time, params=material)
    problem.f = f
    problem.g = g
    problem.u_bar = u_e
    problem.p_bar = p_e
    
    # Apply Dirichlet conditions everywhere for all fields (indicated by the zero marker)
    on_boundary = CompiledSubDomain("on_boundary")
    on_boundary.mark(problem.momentum_boundary_markers, 0)
    for j in Js:
        on_boundary.mark(problem.continuity_boundary_markers[j], 0)

    # -----------------------------------------------------------------------------
    # Set-up solver
    # -----------------------------------------------------------------------------

    theta = 1.0
    params = AdaptiveMPETSolver.default_params()
    params["MPETSolver"]["dt"] = dt
    params["MPETSolver"]["theta"] = theta
    params["MPETSolver"]["T"] = T

    solver = AdaptiveMPETSolver(problem, params)
    info(solver.params, True)
    
    # Containers for comparing exact solution with error estimator
    # (this test case specific)
    u_H1_errors = []
    p_H1_errors = [[] for j in range(J)]
    p_L2_errors = [[] for j in range(J)]
    p_L2H1 = 0.0

    # -----------------------------------------------------------------------------
    # Adaptive loop
    # -----------------------------------------------------------------------------
    done = False
    while(not done):

        # Lists of error estimate terms, one for each time
        eta_1s = []
        eta_2s = []
        eta_3s = []
        eta_4s = []

        # For each time step, compute solutions at this time step
        solutions = solver.inner_solver.solve()
        for (up, t) in solutions:
            # Solve system at this time step

            # Compute element-wise error indicators
            eta_u_K_m = assemble(solver.zeta_u_K)
            eta_p_K_m = assemble(solver.zeta_p_K)
            eta_u_dt_K_m = assemble(solver.zeta_u_dt_K)
            
            # Compute error estimators
            eta_u_m = eta_u_K_m.sum()
            eta_p_m = eta_p_K_m.sum()
            eta_u_dt_m = eta_u_dt_K_m.sum()
            eta_dt_p_m = assemble(solver.R4p)

            # Add time-wise entries to estimator lists
            tau_m = float(solver.inner_solver.params["dt"])
            eta_1s += [tau_m*eta_p_m]
            eta_2s += [eta_u_m]
            eta_3s += [tau_m*numpy.sqrt(eta_u_dt_m)] 
            eta_4s += [tau_m*eta_dt_p_m]

            # Compute actual errors for comparison and for computation of efficiency index
            oops = up.split()
            u = oops[0]
            p = oops[1:]
    
            # Compute H^1_0 error of u at t, and of p_j
            u_H1_errors += [errornorm(problem.u_bar, u, "H10")]
            for j in Js:
                p_H1_errors[j] += [errornorm(problem.p_bar[j], p[j], "H10")]
                p_L2_errors[j] += [errornorm(problem.p_bar[j], p[j], "L2")]

            # Compute the error involving the piecewise constant interpolant.
            ys = [[] for j in Js]
            m = 5
            for j in Js:
                # Store current time for resetting later
                t_n = float(problem.p_bar[j].t)
                
                subtimes = [t_n + float(dt)*float(q)/(m-1) for q in range(m)]
                for t_m in subtimes:
                    problem.p_bar[j].t.assign(t_m)
                    ys[j] += [errornorm(problem.p_bar[j], p[j], "H10")]

                # Reset time
                problem.p_bar[j].t.assign(t_n)

            p_L2H1 += scipy.integrate.trapz(sum(numpy.square(ys)), dx=float(dt)/(m-1))
        
            # Assign "previous time" to time_ for use in the error estimates
            solver.time_.assign(t)

        # Compute error estimators
        eta_1 = numpy.sqrt(numpy.sum(eta_1s))
        eta_2 = numpy.sqrt(numpy.max(eta_2s))
        eta_3 = numpy.sum(eta_3s)
        eta_4 = numpy.sqrt(numpy.sum(eta_4s))

        print("eta_1 = %.3e, eta_2 = %.3e, eta_3 = %.3e, eta_4 = %.3e"
              % (eta_1, eta_2, eta_3, eta_4))
        eta = eta_1 + eta_2 + eta_3 + eta_4
        print("eta = ", eta)
    
        # Compute total error
        u_errors = numpy.array(u_H1_errors)
        p_H1_errors = [numpy.array(p_H1_errors[j]) for j in Js]
        p_L2_errors = [numpy.array(p_L2_errors[j]) for j in Js]
        
        # max_t || u - u_{h, tau} ||_L^{\infty}(0, T; H^1_0):
        u_Linfty = max(u_errors)
        
        # max_t || p - p_{h, tau} ||_L^{\infty}(0, T; L_2):
        p_Linfty = max(numpy.sqrt(sum([numpy.square(p_L2_errors[j]) for j in Js])))
        
        # || p - p_{h, tau} ||_L^2(0, T; H^1_0):
        p_L2 = numpy.sqrt(scipy.integrate.trapz(sum([numpy.square(p_H1_errors[j])
                                                     for j in Js]), dx=float(dt)))
        
        # || p - \pi^0 p_{h, tau} ||_L^2(0, T; H^1_0):
        p_L2H1 = numpy.sqrt(p_L2H1)
        
        error_infty = u_Linfty + p_Linfty
        error_L2 = p_L2 + p_L2H1
        total_error = error_infty + error_L2
        
        print("e1 = %0.3e, e2 = %0.3e, e3 = %0.3e, e4 = %0.3e"
              % (u_Linfty, p_Linfty, p_L2, p_L2H1))
        
        print("error_infty = %0.3e, error_L2 = %0.3e, total_error = %0.3e"
              % (error_infty, error_L2, total_error))
    
        # Compute efficiency indices
        I_eff_dag = eta/error_L2
        I_eff_star = eta/error_infty
        I_eff = eta/total_error
        
        print("I_eff_dag", I_eff_dag)
        print("I_eff_star", I_eff_star)
        print("I_eff", I_eff)
        print("-"*80)
        
        errors = (u_Linfty, p_Linfty, p_L2, p_L2H1)
        estimates = (eta_1, eta_2, eta_3, eta_4)

        ## Estimate contribution to error
        
        ## (Adapt timestep)
        
        # Evaluate total error
        if True:
            done = True
        
        # Mark mesh
        
        # Refine mesh

    exit()


if __name__ == "__main__":

    # Turn on FEniCS optimizations
    parameters["form_compiler"]["cpp_optimize"] = True
    flags = ["-O3", "-ffast-math", "-march=native"]
    parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
    
    # Remove logging (lower value, less logging removed).
    #set_log_level(60)
    
    (h, errors, estimates, I_eff) = main()

    print(h)
    print(errors)
    print(estimates)
    print(I_eff)

