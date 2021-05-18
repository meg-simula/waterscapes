from dolfin import *

import numpy
import scipy.integrate
import itertools

from mpet.mpetproblem import *
from mpet.mpetsolver import *

class AdaptiveMPETSolver():
    def __init__(self, problem, f_, params=None):

        # Set problem and update parameters if given
        self.problem = problem
        self.f_ = f_
        
        self.params = self.default_params()
        if params is not None:
            self.params.update(params)

        self.inner_solver = MPETSolver(problem,
                                       params=self.params["MPETSolver"])
        
        self.__init_error_estimators(self.problem, self.inner_solver)

    def __init_error_estimators(self, problem, solver):
        "Initialize forms for error estimation."

        # Extract current and previous solutions from MPETSolver
        up = split(solver.up)
        u = up[0]
        p = up[1:]
        up_ = split(solver.up_)
        u_ = up_[0]
        p_ = up_[1:]

        # Define geometric objects needed for error estimation
        mesh = solver.up.function_space().mesh()
        h = CellDiameter(mesh)
        n = FacetNormal(mesh)
        DG0 = FunctionSpace(mesh, "DG", 0) 
        w = TestFunction(DG0) 	

        material = problem.params
        nu = material["nu"]
        E = material["E"]
        mu, lmbda = convert_to_mu_lmbda(E, nu)
        alpha = material["alpha"]
        c = material["c"]
        K = material["K"]
        S = material["S"]
        J = range(material["J"])
        
        sigma = lambda u: 2.0*mu*sym(grad(u)) + lmbda*div(u)*Identity(len(u))

        f = problem.f  
        g = problem.g  
        f_ = self.f_
        
        dt = Constant(solver.params["dt"])
        
        # Define cell and edge residuals of momentum equation
        RK_u = f + div(sigma(u)) - sum([grad(alpha[i]*p[i]) for i in J])
        RE_u = sum([jump(alpha[i]*p[i], n) for i in J]) - jump(sigma(u), n)

        # Define cell and edge residuals of momentum equation at previous time
        RK_u_ = f_ + div(sigma(u_)) - sum([grad(alpha[i]*p_[i]) for i in J])
        RE_u_ = sum([jump(alpha[i]*p_[i], n) for i in J]) - jump(sigma(u_), n) 

        # Define cell and edge residuals of mass equations
        # NB: - g here because of MPET Solver - g convention!
        RK_p = [- g[j] - c[j]*(p[j]-p_[j])/dt - alpha[j]*div(u - u_)/dt
                + K[j]*div(grad(p[j])) - sum([S[i][j]*(p[j] - p[i]) for i in J])
                for j in J]
        RE_p = [-K[j]*jump(grad(p[j]), n) for j in J]
    
        # Define discrete time derivative of cell and edge residuals of
        # momentum equation
        RK_u_dt = (RK_u - RK_u_)/dt
        RE_u_dt = (RE_u - RE_u_)/dt

        # Define symbolic error indicators and estimators
        self.zeta_u_K = w*(h**2)*RK_u**2*dx + avg(h)*avg(w)*RE_u**2*dS
        self.zeta_p_K = sum([w*(h**2)*(RK_p[i]**2)*dx + avg(h)*avg(w)*(RE_p[i]**2)*dS for i in J])
        self.zeta_u_dt_K = w*(h**2)*RK_u_dt**2*dx + avg(h)*avg(w)*RE_u_dt**2*dS

        d_j = lambda p, q: sum([S[i][j]*(p[j] - p[i])*q[i] for i in J])*dx

        R4p = [None for j in J]
        for j in J:
            R4p[j] = inner(K[j]*grad(p[j] - p_[j]), grad(p[j] - p_[j]))*dx
            for i in J:
                if S[i][j] > DOLFIN_EPS:
                    R4p[j] += (S[i][j]*((p[j] - p_[j]) - (p[i] - p_[i]))*(p[j] - p_[j]))*dx

        self.R4p = sum(R4p)

        # Lists of error estimate terms, one for each time
        self.eta_1s = []
        self.eta_2s = []
        self.eta_3s = []
        self.eta_4s = []

    def estimate_error_at_t(self):
        # Compute element-wise error indicators
        eta_u_K_m = assemble(self.zeta_u_K)
        eta_p_K_m = assemble(self.zeta_p_K)
        eta_u_dt_K_m = assemble(self.zeta_u_dt_K)

        # Compute error estimators
        eta_u_m = eta_u_K_m.sum()
        eta_p_m = eta_p_K_m.sum()
        eta_u_dt_m = eta_u_dt_K_m.sum()
        eta_dt_p_m = assemble(self.R4p)

        # Add time-wise entries to estimator lists
        tau_m = float(self.inner_solver.params["dt"])
        self.eta_1s += [tau_m*eta_p_m]
        self.eta_2s += [eta_u_m]
        self.eta_3s += [tau_m*numpy.sqrt(eta_u_dt_m)] 
        self.eta_4s += [tau_m*eta_dt_p_m]

    def compute_error_estimate(self):
        # Sum lists of error estimators over time, to compute final error estimate
        eta_1 = numpy.sqrt(numpy.sum(self.eta_1s))
        eta_2 = numpy.sqrt(numpy.max(self.eta_2s))
        eta_3 = numpy.sum(self.eta_3s)
        eta_4 = numpy.sqrt(numpy.sum(self.eta_4s))
        etas = (eta_1, eta_2, eta_3, eta_4)
        eta = sum(etas)
        return (eta, etas)
        
    @staticmethod
    def default_params():
        "Define default solver parameters."
        params = Parameters("AdaptiveMPETSolver")
        params.add(MPETSolver.default_params())

        #dt0 = params["MPETSolver"]["dt"]
        #params.add("dt0", dt0)

        return params

# def adaptive_solve(adaptive, material, mesh):

#     # Define tolerances for spatial and temporal error 
#     tol = adaptive["tol"]
#     tol_eta_space = 0.5*tol
#     tol_eta_time = 0.5*tol

#     # Set max iterations etc.
#     max_iterations = adaptive["max_iterations"]
#     num_iterations = 0

#     eta = 2*tol
#     while(eta >= tol and num_iterations < max_iterations):
#         print("num_iterations = ", num_iterations)
        
#         # Solve adaptively in time on a given mesh
#         adaptive_solve_in_time(adaptive, material, mesh)
        
#         # Compute error estimates
#         eta_time = eta
#         eta_space = eta
#         eta = eta_time + eta_space
        
#         # If spatial error is too large, mark for refinement
#         h = mesh.hmin()
#         print("\th = ", h, " (h_min = ", adaptive["hmin"], ")")
#         if (eta_space >= tol_eta_space and h >= adaptive["hmin"]):

#             # Mark mesh based on mesh markers and strategy
#             markers = MeshFunction("bool", mesh, 0)
#             markers.set_all(True) # FIXME
            
#             # Refine mesh in space based on markers
#             print("Refining mesh")
#             mesh = refine(mesh, markers)
#             print("\tnum_vertices = ", mesh.num_vertices(), "h = ", mesh.hmin())
            
#             num_iterations += 1
#         print("")
            
# def adaptive_solve_in_time(adaptive, material, mesh):
#     # adaptive: Adaptive parameters
#     # material: Material parameters
#     # mesh: Current mesh

#     J = range(material["J"])
#     alpha = material["alpha"]
#     c = material["c"]
#     S = material["S"]
#     K = material["K"]

#     E, nu = material["E"], material["nu"] 
#     mu = E/(2.0*(1.0 + nu))
#     lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    
#     # Extract initial values of mesh size, time step 
#     dt = adaptive["dt"]
#     T = adaptive["T"] 
#     tol = adaptive["tol"]
    
#     # Mesh and time
#     dt = Constant(dt)

#     # Solve through in time on given mesh
#     time = Constant(0.0)
#     time_ = Constant(0.0)

#     # Get exact solutions and corresponding right-hand sides as
#     # Expression strings
#     (u_str, p_str, f_str, g_str) = exact_solutions(material)
    
#     # Convert to FEniCS expressions and attach the time
#     u_e = Expression(u_str, degree=3, t=time)
#     p_e = [Expression(p_str[j], degree=3, t=time) for j in J]
#     f = Expression(f_str, degree=3, t=time)
#     f_ = Expression(f_str, degree=3, t=time_)
#     g = [Expression(g_str[i], degree=3, t=time) for i in J]
    
#     # Create MPETProblem object and attach sources and Dirichlet boundary values 
#     problem = MPETProblem(mesh, time, params=material)
#     problem.f = f
#     problem.g = g
#     problem.u_bar = u_e
#     problem.p_bar = p_e

#     # Apply Dirichlet conditions everywhere for all fields (indicated by the zero marker)
#     on_boundary = CompiledSubDomain("on_boundary")
#     on_boundary.mark(problem.momentum_boundary_markers, 0)
#     for j in J:
#         on_boundary.mark(problem.continuity_boundary_markers[j], 0)

#     # Set-up solver
#     theta = 1.0
#     params = dict(dt=dt, theta=theta, T=T, direct_solver=True)
#     solver = MPETSolver(problem, params)

#     # Extract current and previous solutions from MPETSolver
#     up = split(solver.up)
#     u = up[0]
#     p = up[1:]
#     up_ = split(solver.up_)
#     u_ = up_[0]
#     p_ = up_[1:]

#     # Define geometric objects needed for error estimation
#     h = CellDiameter(mesh)
#     n = FacetNormal(mesh)
#     DG0 = FunctionSpace(mesh, "DG", 0) 
#     w = TestFunction(DG0) 	

#     # NB: Be a bit careful with spatially-constant alphas here.
#     sigma = lambda u: 2.0*mu*sym(grad(u)) + lmbda*div(u)*Identity(len(u))

#     # Define cell and edge residuals of momentum equation
#     RK_u = f + div(sigma(u)) - sum([grad(alpha[i]*p[i]) for i in J])
#     RE_u = sum([jump(alpha[i]*p[i], n) for i in J]) - jump(sigma(u), n)

#     # Define cell and edge residuals of momentum equation at previous time
#     RK_u_ = f_ + div(sigma(u_)) - sum([grad(alpha[i]*p_[i]) for i in J])
#     RE_u_ = sum([jump(alpha[i]*p_[i], n) for i in J]) - jump(sigma(u_), n) 

#     # Define cell and edge residuals of mass equations
#     # NB: - g here because of MPET Solver - g convention!
#     RK_p = [- g[j] - c[j]*(p[j]-p_[j])/dt - alpha[j]*div(u - u_)/dt
#             + K[j]*div(grad(p[j])) - sum([S[i][j]*(p[j] - p[i]) for i in J])
#             for j in J]
#     RE_p = [-K[j]*jump(grad(p[j]), n) for j in J]
    
#     # Define discrete time derivative of cell and edge residuals of
#     # momentum equation
#     RK_u_dt = (RK_u - RK_u_)/dt
#     RE_u_dt = (RE_u - RE_u_)/dt

#     # Define symbolic error indicators and estimators
#     zeta_u_K = w*(h**2)*RK_u**2*dx + avg(h)*avg(w)*RE_u**2*dS
#     zeta_p_K = sum([w*(h**2)*(RK_p[i]**2)*dx + avg(h)*avg(w)*(RE_p[i]**2)*dS for i in J])
#     zeta_u_dt_K = w*(h**2)*RK_u_dt**2*dx + avg(h)*avg(w)*RE_u_dt**2*dS

#     d_j = lambda p, q: sum([S[i][j]*(p[j] - p[i])*q[i] for i in J])*dx

#     R4p = [None for j in J]
#     for j in J:
#         R4p[j] = inner(K[j]*grad(p[j] - p_[j]), grad(p[j] - p_[j]))*dx
#         for i in J:
#             if S[i][j] > DOLFIN_EPS:
#                 R4p[j] += (S[i][j]*((p[j] - p_[j]) - (p[i] - p_[i]))*(p[j] - p_[j]))*dx

#     R4p = sum(R4p)
    
#     # Lists of error estimate terms, one for each time
#     eta_1s = []
#     eta_2s = []
#     eta_3s = []
#     eta_4s = []

#     u_H1_errors = []
#     p_H1_errors = [[] for j in J]
#     p_L2_errors = [[] for j in J]

#     p_L2H1 = 0.0

#     # Solve
#     solutions = solver.solve()

#     for (up, t) in solutions:

#         # Compute element-wise error indicators
#         eta_u_K_m = assemble(zeta_u_K)
#         eta_p_K_m = assemble(zeta_p_K)
#         eta_u_dt_K_m = assemble(zeta_u_dt_K)

#         # Compute error estimators
#         eta_u_m = eta_u_K_m.sum()
#         eta_p_m = eta_p_K_m.sum()
#         eta_u_dt_m = eta_u_dt_K_m.sum()
#         eta_dt_p_m = assemble(R4p)
        
#         # Add time-wise entries to estimator lists
#         tau_m = float(dt)
#         eta_1s += [tau_m*eta_p_m]
#         eta_2s += [eta_u_m]
#         eta_3s += [tau_m*numpy.sqrt(eta_u_dt_m)] 
#         eta_4s += [tau_m*eta_dt_p_m]

#         # Compute actual errors for comparison and for computation of efficiency index
#         oops = up.split()
#         u = oops[0]
#         p = oops[1:]
    
#         # Compute H^1_0 error of u at t, and of p_j
#         u_H1_errors += [errornorm(problem.u_bar, u, "H10")]
#         for j in J:
#             p_H1_errors[j] += [errornorm(problem.p_bar[j], p[j], "H10")]
#             p_L2_errors[j] += [errornorm(problem.p_bar[j], p[j], "L2")]

#         # Compute the error involving the piecewise constant interpolant.
#         ys = [[] for j in J]
#         m = 5
#         for j in J:
#             # Store current time for resetting later
#             t_n = float(problem.p_bar[j].t)

#             subtimes = [t_n + float(dt)*float(q)/(m-1) for q in range(m)]
#             for t_m in subtimes:
#                 problem.p_bar[j].t.assign(t_m)
#                 ys[j] += [errornorm(problem.p_bar[j], p[j], "H10")]

#             # Reset time
#             problem.p_bar[j].t.assign(t_n)

#         p_L2H1 += scipy.integrate.trapz(sum(numpy.square(ys)), dx=float(dt)/(m-1))
        
#         # Assign "previous time" to time_ for use in the error estimates
#         time_.assign(t)

#     # Compute error estimators
#     eta_1 = numpy.sqrt(numpy.sum(eta_1s))
#     eta_2 = numpy.sqrt(numpy.max(eta_2s))
#     eta_3 = numpy.sum(eta_3s)
#     eta_4 = numpy.sqrt(numpy.sum(eta_4s))

#     print("eta_1 = %.3e, eta_2 = %.3e, eta_3 = %.3e, eta_4 = %.3e"
#           % (eta_1, eta_2, eta_3, eta_4))
#     eta = eta_1 + eta_2 + eta_3 + eta_4
#     print("eta = ", eta)
    
#     # Compute total error
#     u_errors = numpy.array(u_H1_errors)
#     p_H1_errors = [numpy.array(p_H1_errors[j]) for j in J]
#     p_L2_errors = [numpy.array(p_L2_errors[j]) for j in J]

#     # max_t || u - u_{h, tau} ||_L^{\infty}(0, T; H^1_0):
#     u_Linfty = max(u_errors)

#     # max_t || p - p_{h, tau} ||_L^{\infty}(0, T; L_2):
#     p_Linfty = max(numpy.sqrt(sum([numpy.square(p_L2_errors[j]) for j in J])))

#     # || p - p_{h, tau} ||_L^2(0, T; H^1_0):
#     p_L2 = numpy.sqrt(scipy.integrate.trapz(sum([numpy.square(p_H1_errors[j]) for j in J]), dx=float(dt)))
    
#     # || p - \pi^0 p_{h, tau} ||_L^2(0, T; H^1_0):
#     p_L2H1 = numpy.sqrt(p_L2H1)
    
#     error_infty = u_Linfty + p_Linfty
#     error_L2 = p_L2 + p_L2H1
#     total_error = error_infty + error_L2

#     print("e1 = %0.3e, e2 = %0.3e, e3 = %0.3e, e4 = %0.3e"
#           % (u_Linfty, p_Linfty, p_L2, p_L2H1))

#     print("error_infty = %0.3e, error_L2 = %0.3e, total_error = %0.3e"
#           % (error_infty, error_L2, total_error))
    
#     # Compute efficiency indices
#     I_eff_dag = eta/error_L2
#     I_eff_star = eta/error_infty
#     I_eff = eta/total_error
    
#     print("I_eff_dag", I_eff_dag)
#     print("I_eff_star", I_eff_star)
#     print("I_eff", I_eff)
#     print("-"*80)

#     errors = (u_Linfty, p_Linfty, p_L2, p_L2H1)
#     estimates = (eta_1, eta_2, eta_3, eta_4)

#     return (mesh.hmax(), errors, estimates, I_eff)

