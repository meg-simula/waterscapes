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
        # x0 = Point(0.0, 0.0)
        # x1 = Point(self.params.L, self.params.L)
        # n = self.params.N
        
        # Create mesh
        m = self.params.mesh_file
        if isinstance(m, Mesh):
            self.mesh = m
        else:
            self.mesh = Mesh(m)

        # self.mesh = RectangleMesh(x0, x1, n, n, "crossed")
        # self.mesh = UnitSquareMesh(n, n)
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
        self.left = CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS")
        self.right = CompiledSubDomain("on_boundary && x[0] > 1.0 - DOLFIN_EPS")
        self.top = CompiledSubDomain("on_boundary && x[1] < 1.0 + DOLFIN_EPS")
        self.bottom = CompiledSubDomain("on_boundary && x[1] < DOLFIN_EPS")

        self.left = CompiledSubDomain("on_boundary && x[0] < DOLFIN_EPS")        
        
        self.allbutleft = CompiledSubDomain("on_boundary && x[0] > DOLFIN_EPS")
        self.allboundary.mark(self.facet_domains, 6)       
        # 
        # self.allbutleft.mark(self.facet_domains, 1) #!!!
        # self.left.mark(self.facet_domains, 2) #!!!
        # self.right.mark(self.facet_domains, 3) #!!!
        # self.top.mark(self.facet_domains, 4) #!!!
        # self.bottom.mark(self.facet_domains, 5) #!!!

        gdim = self.mesh.geometry().dim()
        self.cell_domains = MeshFunction("size_t", self.mesh, gdim,
                                         self.mesh.domains())

        self.ds = Measure("ds")(domain=self.mesh, subdomain_data=self.facet_domains)
        
    def exact_solutions(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        
        x = SpatialCoordinate(self.mesh)

        u = as_vector( ( sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0), sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0) ))
        p = [-(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0) for i in range(AA)]
        
        # p = [p0,]
        return u, p
    
    def exact_solutions_expression(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        u0 = Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)","sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)"), domain=self.mesh, degree=5, L=L, t=t, n=n)
        p0 = [Expression("-(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", domain=self.mesh, degree=5, L=L, t=t, i=i, n=n) for i in range(AA)]
        return u0, p0
        
    def initial_conditions(self, t):

        AA = self.params.AA
        L = self.params.L
        n = self.period

        uex, pex = self.exact_solutions(t)
        
        return uex, pex

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        AA = self.params.AA
        L = self.params.L
        n = self.period

        uex, pex = self.exact_solutions_expression(t)
        
        # u0 = [(uex, self.allboundary),]

        p0 = [((i+1), pex[i], self.allboundary)
              for i in range(AA)]

        u0 = []
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):

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
        uex0, pex0 = self.exact_solutions(t0)

        sigma1 = lmbda * div(uex1) * I + 2.0 * mu * sym(grad(uex1))
        sigma0 = lmbda * div(uex0) * I + 2.0 * mu * sym(grad(uex0))
        n1 = -sigma1 + sum(alphas[i]*pex1[i]for i in range(AA))*I
        n0 = -sigma0 + sum(alphas[i]*pex0[i]for i in range(AA))*I
        
        # u0 = [(n1, 2), (n1, 3), ]
        # This does not work, why???
        u0 = [(n1, 6),]        
        # u0 = []
        p0 = []
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

        gg = [-1.0/Q*diff(pex[i], t) -alphas[i]*div(diff(uex, t)) + div(Ks[i]*grad(pex[i])) - sum(G[j] * (pex[i] - pex[j]) for j in range(AA)) for i in range(AA)]

        return gg


    def nullspace(self):
      
        # No null space
        null = False
        # null = True
        return null

def single_run(N=64, M=64):
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    T = 1.0
    dt = float(T/M)
    # T = 4.0*dt
    # dt = T

    L = 70.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    Ks = (1.0, 1.0)
    G = (1.0, 1.0)

    E = 1.0
    nu = 0.35
    # nu = 0.49999
    Incompressible = False

    # Create problem set-up
    mesh=Mesh()
    filename = "../../meshes/2D_brain_refined7.xdmf"
    f = XDMFFile(mpi_comm_world(), filename)
    f.read(mesh, True)
    
    problem = FirstTest(dict(mesh_file=mesh, N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))

    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 0.5, "direct_solver":True, "testing":False, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":False, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-6, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})

    # Solve
    solutions = solver.solve_symmetric()
    for (U, t) in solutions:
        pass

    u = U.split(deepcopy=True)[0]
    p = U.split(deepcopy=True)[1:AA+1]
    print len(p)
    uex, pex = problem.exact_solutions_expression(T)
    erru, errpL2, errpH1 = compute_error(u, p, uex, pex, problem.mesh)

    # Return errors and meshsize
    h = problem.mesh.hmax()
    #h = L/N
    print h, erru, errpL2, errpH1
    plot(uex, mesh = mesh, title="uex")
    plot(pex[0], mesh = mesh, title="p0")
    plot(pex[1], mesh = mesh, title="p1")
    plot(u, mesh = mesh, title="u")
    plot(p[0], mesh = mesh, title="p0")
    plot(p[1], mesh = mesh, title="p1")
    interactive()
    return (erru, errpL2, errpH1, h)


def convergence_rates(errors, hs):
    rates = [(math.log(errors[i+1]/errors[i]))/(math.log(hs[i+1]/hs[i])) for i in range(len(hs)-1)]

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
        assert (i > 1.88), "H1 convergence in u failed"
    for i in p0_ratesL2:
        assert (i > 1.89), "L2 convergence in p0 failed"
    for i in p1_ratesL2:
        assert (i > 1.89), "L2 convergence in p1 failed"
    for i in p0_ratesH1:
        assert (i > 0.9), "H1 convergence in p0 failed"
    for i in p1_ratesH1:
        assert (i > 0.9), "H1 convergence in p1 failed"

if __name__ == "__main__":

    # Just test a single run
    single_run(N=64, M=60)

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
