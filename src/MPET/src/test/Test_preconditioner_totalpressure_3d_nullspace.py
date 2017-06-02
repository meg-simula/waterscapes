
from mpet import *
from numpy import zeros
import sys
import os
import math
from cbcpost import *
import time
import pylab

# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
set_log_level(30)
#parameters["form_compiler"]["representation"] = "uflacs"
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# pylab.rc('text', usetex=True)
# pylab.rc('font', family='serif')
            
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
        self.mesh = UnitCubeMesh(n, n, n)
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
        u = Expression(("pi*x[0]*cos(pi*x[0]*x[1])*t", "-pi*x[1]*cos(pi*x[0]*x[1])*t", "pi*x[2]*cos(pi*x[0]*x[1])*t"),
                        t=t, domain=self.mesh, degree=3)
        p0 = Expression("sin(2*pi*x[0])*sin(2*pi*x[1]) * t",
                        n=n, L=L, t=t, domain=self.mesh, degree=3)
        p1 = Expression("2.0*sin(2*pi*x[0])*sin(2*pi*x[1]) * t",
                        n=n, L=L, t=t, domain=self.mesh, degree=3)
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
        ux =  pi*x[0]*cos(pi*x[0]*x[1]) * t
        uy = -pi*x[1]*cos(pi*x[0]*x[1]) * t
        uz =  pi*x[2]*cos(pi*x[0]*x[1]) * t
        u = as_vector((ux, uy))
        
        p0 = sin(2*pi*x[0])*sin(2*pi*x[1]) * t
        p1 = 2.0*sin(2*pi*x[0])*sin(2*pi*x[1]) * t
        divu = ux.dx(0) + uy.dx(1) + uz.dx(2)
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

        u0 = []
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        u0 = []
        p0 = []
        return u0, p0

    def f(self, t):
    
        E = self.params.E
        nu = self.params.nu
        lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
        mu = E/(2.0*(1.0 + nu))
    
        alphas = self.params.alphas
        mesh = self.mesh
        L = self.params.L
        n = self.period
        x = SpatialCoordinate(self.mesh)
        
        ux = pi*x[0]*cos(pi*x[0]*x[1]) * t
        uy = -pi*x[1]*cos(pi*x[0]*x[1]) * t
        uz = pi*x[2]*cos(pi*x[0]*x[1]) * t

        u = as_vector((ux, uy, uz))
        
        p0 = sin(2*pi*x[0])*sin(2*pi*x[1]) * t
        p1 = 2.0*sin(2*pi*x[0])*sin(2*pi*x[1]) * t
        
        I = Identity(3) 
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
        
        ux = pi*x[0]*cos(pi*x[0]*x[1]) * t
        uy = -pi*x[1]*cos(pi*x[0]*x[1]) * t
        uz = pi*x[2]*cos(pi*x[0]*x[1]) * t
        u = as_vector((ux, uy, uz))
        
        p0 = sin(2*pi*x[0])*sin(2*pi*x[1]) * t
        p1 = 2.0*sin(2*pi*x[0])*sin(2*pi*x[1]) * t
        
        dotux = pi*x[0]*cos(pi*x[0]*x[1])
        dotuy = -pi*x[1]*cos(pi*x[0]*x[1])
        dotuz = pi*x[2]*cos(pi*x[0]*x[1])
        dotp0 = sin(2*pi*x[0])*sin(2*pi*x[1])
        dotp1 = 2.0*sin(2*pi*x[0])*sin(2*pi*x[1])
        
        gg[0] = -1./Q*dotp0 -alphas[0] * (dotux.dx(0) + dotuy.dx(1) + dotuz.dx(2)) + div(Ks[0]*grad(p0)) - G[0] * (p0-p1)
        gg[1] = -1./Q*dotp1 -alphas[1] * (dotux.dx(0) + dotuy.dx(1) + dotuz.dx(2)) + div(Ks[1]*grad(p1)) - G[1] * (p1-p0)
        
        return gg
    

    
    def nullspace(self):
      
        # No null space
        null = True
        return null


def single_run(N=64, M=64):
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    T = 1.0
    dt = float(T/M)
    T = dt
    # dt = T

    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    Ks = (1.0, 1.0)
    G = (1.0,1.0)

    E = 1.0
    # nu = 0.49
    #lmbda = 1000
    # nu = 0.4998333703662547
    nus = [0.3660254037844386, 0.4998333703662547, 0.499998333337037]
    #nus = [0.4998333703662547]
    KK = [(1.0, 1.0), (1.0e-3, 1.0e-3), (1.0e-5, 1.0e-5), (1.0e-8, 1.0e-8)]
    #KK = [(1.0e-8, 1.0e-8)]
    Incompressible = False

    print "*"*200
    print "N = ", N
    for nu in nus:

        print "lmbda = ", nu*E/((1.0-2.0*nu)*(1.0+nu))
        print "mu = ", E/(2.0*(1+0+nu))
        
        for Ks in KK:

            print "Ks = ", Ks
            # Create problem set-up 
            problem = FirstTest(dict(N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                                     Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))
        
            # Create solver
            solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 1.0, "direct_solver":False, "testing":True, "fieldsplit": False,\
                                            "krylov_solver": {"monitor_convergence":False, "nonzero_initial_guess":True,\
                                                "relative_tolerance": 1.e-6, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})
        
    
            # Solve
            solutions = solver.solve_totalpressure()
            for (U, t) in solutions:
                pass

    print "*"*200

if __name__ == "__main__":

    # Just test a single run
    for N in [8, 16, 32, 64]:
    # for N in [64]:
        single_run(N=N, M=64)

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
