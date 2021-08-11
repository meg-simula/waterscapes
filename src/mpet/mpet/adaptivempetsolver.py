from dolfin import *

import numpy
import scipy.integrate
import itertools

from mpet.mpetproblem import *
from mpet.mpetsolver import *


def dorfler_mark(indicators, fraction):
    """Compute Boolean markers given the error indicators field and the
    Dorfler fraction parameter. 

    Note that fraction = 1.0 marks all cells, fraction=0.0 marks no
    cells.
    """
    mesh = indicators.function_space().mesh()
    markers = MeshFunction("bool", mesh, mesh.topology().dim(), False)

    # Sort indicators (higher to lower, note negative sort trick)
    etas = indicators.vector().get_local()
    indices = numpy.argsort(-etas)
    
    # Compute sum of indicators and decide upon error limit given fraction
    S = numpy.sum(etas)
    stop = fraction*S

    # Iterate over the indices, mark the largest first, and stop when
    # the tolerance is reached.
    eta = 0.0
    for i in indices:
        if eta >= stop:
            break
        else:
            markers[i] = True
            eta += etas[i]

    return markers

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
        mu = Constant(mu)
        lbmda = Constant(lmbda)
        alpha = Constant(material["alpha"])
        c = [Constant(ci) for ci in material["c"]]
        K = [Constant(Ki) for Ki in material["K"]]
        S = [[Constant(Sij) for Sij in Sj] for Sj in material["S"]]
        J = range(material["J"])
        
        sigma = lambda u: 2.0*mu*sym(grad(u)) + lmbda*div(u)*Identity(len(u))

        f = problem.f  
        g = problem.g  
        f_ = self.f_
        
        dt = solver.dt
        
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
                if float(S[i][j]) > DOLFIN_EPS:
                    R4p[j] += (S[i][j]*((p[j] - p_[j]) - (p[i] - p_[i]))*(p[j] - p_[j]))*dx

        self.R4p = sum(R4p)

        # Lists of error estimate terms, one for each time
        self.eta_1s = []
        self.eta_2s = []
        self.eta_3s = []
        self.eta_4s = []

        # Error indicator functions
        self.eta_1Ks = Function(w.function_space())
        self.eta_2Ks = Function(w.function_space())
        self.eta_3Ks = Function(w.function_space())
        
    def estimate_error_at_t(self):
        # Compute element-wise error indicators
        eta_p_K_m = assemble(self.zeta_p_K)
        eta_u_K_m = assemble(self.zeta_u_K)
        eta_u_dt_K_m = assemble(self.zeta_u_dt_K)

        # Compute error estimators
        eta_p_m = eta_p_K_m.sum()
        eta_u_m = eta_u_K_m.sum()
        eta_u_dt_m = eta_u_dt_K_m.sum()
        eta_dt_p_m = assemble(self.R4p)

        tau_m = float(self.inner_solver.dt)

        # Add time-wise entries to estimator lists
        self.eta_1s += [tau_m*eta_p_m]
        self.eta_2s += [eta_u_m]
        self.eta_3s += [tau_m*numpy.sqrt(eta_u_dt_m)] 
        self.eta_4s += [tau_m*eta_dt_p_m]

        return (eta_p_K_m, eta_u_K_m, eta_u_dt_K_m, tau_m)
        
    def add_to_indicators(self, eta_Ks):
        print("Adding to error indicator fields")
        (eta_p_K_m, eta_u_K_m, eta_u_dt_K_m, tau_m) = eta_Ks

        self.eta_1Ks.vector().axpy(tau_m, eta_p_K_m)
        self.eta_3Ks.vector()[:] = tau_m*numpy.sqrt(eta_u_dt_K_m.get_local())

        e0 = self.eta_2Ks.vector().get_local()
        e1 = eta_u_K_m.get_local()
        self.eta_2Ks.vector()[:] = numpy.maximum(e0, e1)

    def pop_error_estimators(self):
        "Remove last items from error estimators lists."
        self.eta_1s.pop()
        self.eta_2s.pop()
        self.eta_3s.pop()
        self.eta_4s.pop()

    def compute_error_indicators(self):
        # Create the field  eta_Ks combining the right contributions
        # from eta_1Ks (squared, so taking square root), eta_2Ks, and
        # eta_3Ks.
        eta_Ks = self.eta_2Ks.copy(deepcopy=True)
        eta_Ks.vector().axpy(1.0, self.eta_3Ks.vector())
        eta_Ks.vector()[:] += numpy.sqrt(self.eta_1Ks.vector().get_local())
        
        return eta_Ks
        
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
        params.add("alpha", 0.1)
        params.add("beta", 2)
        params.add("tau_max", 0.2)
        params.add("tau_min", 0.0)
        
        return params


