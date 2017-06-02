
from mpet import *
from numpy import zeros
import sys
import os
import math
from cbcpost import *
import time

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
#parameters["form_compiler"]["representation"] = "uflacs"

            
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

        # Mark all exterior boundary facets as 1
        # Note: The CompiledSubDomain class gives faster code than
        # Python subclassing
        #self.allboundary = AllBoundary()
        self.allboundary = CompiledSubDomain("on_boundary")
        self.allboundary.mark(self.facet_domains, 1) #!!!

        gdim = self.mesh.geometry().dim()
        self.cell_domains = MeshFunction("size_t", self.mesh, gdim,
                                         self.mesh.domains())


    def exact_solutions(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        u = Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", "sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)"),
                        n=n, L=L, t=t, domain=self.mesh, degree=5)
        p0 = Expression("-sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)",
                        n=n, L=L, t=t, domain=self.mesh, degree=5)
        p1 = Expression("-2.0*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)",
                        n=n, L=L, t=t, domain=self.mesh, degree=5)
        p = [p0, p1]
        return u, p
    
    
    def initial_conditions(self, t):

        AA = self.params.AA
        L = self.params.L
        n = self.period
        nu = self.params.nu
        E = self.params.E
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0 + nu))
        
        alphas = self.params.alphas
        
        uex, pex = self.exact_solutions(t)
        
        x = SpatialCoordinate(self.mesh)
        
        # u0 = Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", "sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)"),
        #                 n=n, L=L, t=t, domain=self.mesh, degree=5)
        ux = sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        uy = sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        u = as_vector((ux, uy))
        
        p0 = -sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        p1 = -2.0 * sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        divu = ux.dx(0) + uy.dx(1)
        pt = lmbda * divu - alphas[0] * p0 - alphas[1] * p1
        
        p = [pt, pex[0], pex[1]]

        return uex, p

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        AA = self.params.AA
        L = self.params.L
        n = self.period

        uex, pex = self.exact_solutions(t)
        

        u0 = [(uex, self.allboundary)]

        p0 = [((2), pex[0], self.allboundary), ((3), pex[1], self.allboundary)]

        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        u0 = []
        p0 = []
        return u0, p0

    def f(self, t):
    
        E = self.params.E
        nu = self.params.nu
        print "nu = ", nu
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0 + nu))
    
        alphas = self.params.alphas
        mesh = self.mesh
        L = self.params.L
        n = self.period
        x = SpatialCoordinate(self.mesh)

        ux = sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        uy = sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        u = as_vector((ux, uy))
        
        p0 = -sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        p1 = -2.0 * sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        
        I = Identity(2) 
        ff = -div(2*mu*sym(grad(u)) + lmbda*div(u)*I - alphas[0]*p0*I - alphas[1]*p1*I)
        
        return ff
    
    def g(self, t):

        E = self.params.E
        nu = self.params.nu
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0 + nu))
    
        Q = self.params.Q
        alphas = self.params.alphas
        Ks = self.params.Ks
        mesh = self.mesh
        AA = self.params.AA
        G = self.params.G
        L = self.params.L
        n = self.period
    
        gg = [0]*AA
        
        x = SpatialCoordinate(self.mesh)
        ux = sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        uy = sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        u = as_vector((ux, uy))
        
        dotux = sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*n*cos(n*t + 1.0)
        dotuy = sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*n*cos(n*t + 1.0)
        
        p0 = -sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        p1 = -2.0 * sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)
        
        dotp0 = -sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*n*cos(n*t + 1.0)
        dotp1 = -2.0 * sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*n*cos(n*t + 1.0)
        
        gg[0] = -1./Q*dotp0 -alphas[0] * (dotux.dx(0) + dotuy.dx(1)) + div(Ks[0]*grad(p0)) - G[0] * (p0-p1)
        gg[1] = -1./Q*dotp1 -alphas[1] * (dotux.dx(0) + dotuy.dx(1)) + div(Ks[1]*grad(p1)) - G[1] * (p1-p0)
        
        return gg
    

    
    def nullspace(self):
      
        # No null space
        null = []
        return null


def single_run(N=64, M=64):
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    T = 1.0
    dt = float(T/M)
    # T = 1.0/float(M)
    # dt = T

    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    Ks = (1.0, 1.0)
    G = (1.0, 1.0)

    E = 1.0
    nu = 0.49999
    print "lmbda = ", nu*E/((1.0-2.0*nu)*(1.0+nu))
    Incompressible = False

    # Create problem set-up
    print "N = ", N 
    problem = FirstTest(dict(N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))

    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 0.5, "testing":True, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":False, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-6, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})

    # Solve
    solutions = solver.solve_totalpressure()
    for (U, t) in solutions:
        pass

    u = U.split(deepcopy=True)[0]
    p = U.split(deepcopy=True)[2:]

    # plot(u, mesh=mesh)
    # plot(p[0], mesh=mesh)
    # interactive()
    uex, pex = problem.exact_solutions(T)
    # plot(uex, mesh=mesh)
    # interactive()

    erru, errpL2, errpH1 = compute_error(u, p, uex, pex, problem.mesh)

    # Return errors and meshsize
    h = problem.mesh.hmax()
    #h = L/N
    #print h, erru, errpL2, errpH1

    return (erru, errpL2, errpH1, h)


def convergence_rates(errors, hs):
    rates = [(math.log(errors[i+1]/errors[i]))/(math.log(hs[i+1]/hs[i])) for i in range(len(hs)-1)]

    return rates

def run_quick_convergence_test():

    # Remove all output from FEniCS (except errors)
    set_log_level(60)

    # Make containers for errors
    u_errors = []
    p_errorsL2 = [[] for i in range(2)]
    p_errorsH1 = [[] for i in range(2)]
    hs = []

    # Iterate over mesh sizes/time steps and compute errors
    start = time.time()
    print "Start"
    for j in [8, 16, 32, 64]:
        print "i = ", j
        (erru, errpL2, errpH1, h) = single_run(N=j, M=j)
        hs += [h]
        u_errors += [erru]
        print "\| u(T)  - u_h(T) \|_1 = %r" % erru
        for (i, errpi) in enumerate(errpL2):
            print "\| p_%d(T)  - p_h_%d(T) \|_0 = %r" % (i, i, errpi)
            p_errorsL2[i] += [errpi]
        for (i, errpi) in enumerate(errpH1):
            print "\| p_%d(T)  - p_h_%d(T) \|_0 = %r" % (i, i, errpi)
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

    # Add test:
    for i in u_rates:
        assert (i > 1.9), "H1 convergence in u failed"
    for i in p0_ratesL2:
        assert (i > 1.9), "L2 convergence in p0 failed"
    for i in p1_ratesL2:
        assert (i > 1.9), "L2 convergence in p1 failed"
    for i in p0_ratesH1:
        assert (i > 0.9), "H1 convergence in p0 failed"
    for i in p1_ratesH1:
        assert (i > 0.9), "H1 convergence in p1 failed"

if __name__ == "__main__":

    # Just test a single run
    # single_run(N=64, M=10)

    # Run quick convergence test:
    run_quick_convergence_test()

    # Store all errors
    #main()
