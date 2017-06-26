__author__ = "Eleonora Piersanti (eleonora@simula.no), 2016-2017"
__all__ = []

import pytest

from mpet import *
import math
from cbcpost import *
import time

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)

class FirstTest(MPET):
    """2 networks:
    -1st: extracellular CSF
    -2nd: capillary
    Parameters taken from Tully and Ventikos 2010"""

    def __init__(self, params=None):

        MPET.__init__(self, params)

        # Create mesh
        x0 = Point(0.0, 0.0)
        x1 = Point(self.params.L, self.params.L)
        n = self.params.N
        # self.mesh = RectangleMesh(x0, x1, n, n, "crossed")
        self.mesh = UnitSquareMesh(n, n)
        # plot(self.mesh)
        self.period = 2*math.pi
        # Create boundary markers and initialize all facets to +
        self.facet_domains = FacetFunction("size_t", self.mesh)
        self.facet_domains.set_all(0)

        self.bc_type = "Neumann"
        # Mark all exterior boundary facets as 1
        # Note: The CompiledSubDomain class gives faster code than
        # Python subclassing
        #self.allboundary = AllBoundary()
        self.allboundary = CompiledSubDomain("on_boundary")
        self.left = CompiledSubDomain("on_boundary && (x[0] < DOLFIN_EPS)")
        self.right = CompiledSubDomain("on_boundary && (x[0] > 1.0 - DOLFIN_EPS)")
        self.top = CompiledSubDomain("on_boundary && (x[1] > 1.0 - DOLFIN_EPS)")
        self.bottom = CompiledSubDomain("on_boundary && (x[1] < DOLFIN_EPS)")

        self.allboundary.mark(self.facet_domains, 1)
        # self.left.mark(self.facet_domains, 2) #!!!
        # self.right.mark(self.facet_domains, 3) #!!!
        # self.top.mark(self.facet_domains, 4) #!!!
        # self.bottom.mark(self.facet_domains, 5) #!!!
        #plot(self.facet_domains)
        #interactive()

    def exact_solutions(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        
        x = SpatialCoordinate(self.mesh)

        u = as_vector( ( sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0),\
                         sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0) ))
        
        p = [-(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0) for i in range(AA)]
        
        # p = [p0,]
        return u, p

    def exact_solutions_expression(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        
        u = Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0)",\
                        "sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0)"),
                        L=L, n=n, t=t, domain=self.mesh, degree=5)

        
        p = [Expression("-sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", t=t, n=n, L=L, domain=self.mesh, degree=4),
             Expression("-2.0*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", t=t, n=n, L=L, domain=self.mesh, degree=4)]
        
        # p = [p0,]
        return u, p
        
    def initial_conditions(self, t):

        AA = self.params.AA
        L = self.params.L
        n = self.period

        uex, pex = self.exact_solutions(t)
        
        return uex, pex

    def boundary_conditions_u(self, t0, t1, theta):

        E = self.params.E
        nu = self.params.nu
        alphas = self.params.alphas
        mesh = self.mesh
        L = self.params.L
        n = self.period
        AA = self.params.AA
        
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0+nu))
        d = 2
        I = Identity(d)

        uex1, pex1 = self.exact_solutions(t1)

        sigma1 = lmbda * div(uex1) * I + 2.0 * mu * sym(grad(uex1))
        n1 = -sigma1 + sum(alphas[i]*pex1[i]for i in range(AA))*I

        if self.bc_type == "Dirichlet":
            bcu = {1: {"Dirichlet": uex1}}
        if self.bc_type == "Neumann":
            bcu = {1: {"Neumann": n1}}

        return bcu

    def boundary_conditions_p(self, t0, t1, theta):

        uex0, pex0 = self.exact_solutions(t0)
        uex1, pex1 = self.exact_solutions(t1)

        Ks = self.params.Ks
        
        if self.bc_type == "Dirichlet":
            bcp = [{1: {"Dirichlet": pex1[0]}},
                   {1: {"Dirichlet": pex1[1]}}]
        if self.bc_type == "Neumann":
            bcp = [{1: {"Neumann": Ks[0]*(grad(theta*pex1[0] + (1.0-theta)*pex0[0]))}},
                   {1: {"Neumann": Ks[1]*(grad(theta*pex1[1] + (1.0-theta)*pex0[1]))}}]

        return bcp



    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

  
        p0 = []
        u0 = []
        return u0, p0

    def f(self, t):

        E = self.params.E
        nu = self.params.nu
        alphas = self.params.alphas
        mesh = self.mesh
        L = self.params.L
        n = self.period
        AA = self.params.AA
        uex, pex = self.exact_solutions(t)
        
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0+nu))
        d = 2
        I = Identity(d)
        sigma = lmbda * div(uex) * I + 2.0 * mu * sym(grad(uex))
        ff = -div(sigma) + sum(alphas[i]*grad(pex[i]) for i in range(AA))
 
        return ff

    def g(self, t):

        Q = self.params.Q
        alphas = self.params.alphas
        Ks = self.params.Ks
        mesh = self.mesh
        AA = self.params.AA
        G = self.params.G
        L = self.params.L
        n = self.period

        uex, pex = self.exact_solutions(t)

        gg = [0]*AA

        gg = [- 1.0/Q*diff(pex[i], t) - alphas[i]*div(diff(uex, t)) \
              + div(Ks[i]*grad(pex[i])) - sum(G[j] * (pex[i] - pex[j]) for j in range(AA)) for i in range(AA)]

        return gg


    def nullspace(self):
      
        # No null space
        if self.bc_type == "Dirichlet":
            null = False
        if self.bc_type == "Neumann":        
            null = True
        return null

def single_run(N=64, M=64):
    "N is the mesh size, M the number of time steps."

    # Define end time T and timestep dt
    T = 1.0
    dt = float(T/M)

    # Define material parameters in MPET equations
    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    Ks = (1.0, 1.0)
    G = (1.0, 1.0)

    E = 1.0
    nu = 0.35
    Incompressible = False

    # Create problem set-up
    print "N = ", N 
    params = dict(N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                  Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu)
    problem = FirstTest(params)

    # Create solver
    solver_params = {"dt": dt, "T": T, "theta": 0.5, "direct_solver": True, "testing": False, "fieldsplit": False,\
                     "krylov_solver": {"monitor_convergence":False, "nonzero_initial_guess":True,\
                                       "relative_tolerance": 1.e-6, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}}
    solver = SimpleSolver(problem, solver_params)

    # Solve
    solutions = solver.solve_symmetric()
    for (U, t) in solutions:
        pass

    u = U.split(deepcopy=True)[0]
    p = U.split(deepcopy=True)[1:AA+1]

    # Compute the errors between approximation and exact solutions
    uex, pex = problem.exact_solutions_expression(T)
    erru, errpL2, errpH1 = compute_error(u, p, uex, pex, problem.mesh)

    # Return errors and meshsize
    h = problem.mesh.hmax()

    return (erru, errpL2, errpH1, h)

def convergence_rates(errors, hs):
    rates = [(math.log(errors[i+1]/errors[i]))/(math.log(hs[i+1]/hs[i]))
             for i in range(len(hs)-1)]
    return rates

def run_quick_convergence_test():

    # Remove all output from FEniCS (except errors)
    set_log_level(WARNING)

    # Make containers for errors
    u_errors = []
    p_errorsL2 = [[] for i in range(2)]
    p_errorsH1 = [[] for i in range(2)]
    hs = []

    # Iterate over mesh sizes/time steps and compute errors
    start = time.time()
    print "Start"
    for j in [8, 16, 32]:
        print "i = ", j
        (erru, errpL2, errpH1, h) = single_run(N=j, M=j)
        hs += [h]
        u_errors += [erru]
        print "\| u(T)  - u_h(T) \|_1 = %r" % erru
        for (i, errpi) in enumerate(errpL2):
            print "\| p_%d(T)  - p_h_%d(T) \|_0 = %r" % (i, i, errpi)
            p_errorsL2[i] += [errpi]
        for (i, errpi) in enumerate(errpH1):
            print "\| p_%d(T)  - p_h_%d(T) \|_1 = %r" % (i, i, errpi)
            p_errorsH1[i] += [errpi]

    # Compute convergence rates:
    # print "hs = ", hs
    u_rates = convergence_rates(u_errors, hs)
    p0_ratesL2 = convergence_rates(p_errorsL2[0], hs)
    p1_ratesL2 = convergence_rates(p_errorsL2[1], hs)
    p0_ratesH1 = convergence_rates(p_errorsH1[0], hs)
    p1_ratesH1 = convergence_rates(p_errorsH1[1], hs)

    print "u_rates = ", u_rates
    print "p0_ratesL2 = ", p0_ratesL2
    print "p1_ratesL2 = ", p1_ratesL2
    print "p0_ratesH1 = ", p0_ratesH1
    print "p1_ratesH1 = ", p1_ratesH1

    end = time.time()
    print "Time_elapsed = ", end - start

    # Test that convergence rates are in agreement with theoretical
    # expectation asymptotically
    assert (u_rates[-1] > 1.95), "H1 convergence in u failed"
    assert (p0_ratesL2[-1] > 1.95), "L2 convergence in p0 failed"
    assert (p1_ratesL2[-1] > 1.95), "L2 convergence in p1 failed"
    assert (p0_ratesH1[-1] > 0.95), "H1 convergence in p0 failed"
    assert (p1_ratesH1[-1] > 0.95), "H1 convergence in p1 failed"

def test_convergence():
    run_quick_convergence_test()


if __name__ == "__main__":

    # Just test a single run
    single_run(N=8, M=8)

    # Run quick convergence test:
    #run_quick_convergence_test()

    # Store all errors
    #main()
