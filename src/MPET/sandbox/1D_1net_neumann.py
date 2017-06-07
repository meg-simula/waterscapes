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
#parameters["form_compiler"]["representation"] = "uflacs"
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
# pylab.rc('text', usetex=True)
# pylab.rc('font', family='serif')
            
class FirstTest(MPET):

    """
    First setup considering a 1D mesh and 1 metwork.
    Only neumann conditions are imposed on the displacement.
    Rigid motions are taken into account using lagrange multipliers.
    Parameters taken from Tully and Ventikos 2010"""

    def __init__(self, params=None):

        MPET.__init__(self, params)

        # Create mesh
        m = self.params.mesh_file
        if isinstance(m, Mesh):
            self.mesh = m
        else:
            self.mesh = Mesh(m)

        self.period = 2*math.pi
        # Create boundary markers and initialize all facets to +
        self.facet_domains = FacetFunction("size_t", self.mesh)
        self.facet_domains.set_all(0)

        # Mark all exterior boundary facets as 1
        # Note: The CompiledSubDomain class gives faster code than
        # Python subclassing
        #self.allboundary = AllBoundary()
        self.allboundary = CompiledSubDomain("on_boundary")
        self.ventricles = CompiledSubDomain("near(x[0], 0.0) && on_boundary")
        self.skull = CompiledSubDomain("near (x[0], L) && on_boundary", L = self.params.L)
        
        self.allboundary.mark(self.facet_domains, 1) #!!!
        self.ventricles.mark(self.facet_domains, 2) #!!!
        self.skull.mark(self.facet_domains, 3) #!!!

        gdim = self.mesh.geometry().dim()
        self.cell_domains = MeshFunction("size_t", self.mesh, gdim,
                                         self.mesh.domains())
        



    def exact_solutions(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        
        class MyExpression(Expression):
            def eval(self, value, x):
                value[0] = 0.0
            def value_shape(self):
                return (1,)

        u = MyExpression()
        # p0 = Expression("A0*sin(2.0*pi*t)+C0",
        #                 A0 = 666.611825, C0 = 1999.835475, t=t,domain=self.mesh, degree=3)
        # p0 = Expression("A0 * sin(2.0 * pi * t) + (133.0 - 133/(L) * x[0])",
        #                 A0 = 666.611825, L = L, t = t, domain=self.mesh, degree=3)
        # p0 = Expression("(A0 - 13.0/(L) * x[0]) * sin(2*pi*t)",
        #                 A0 = 666.611825, L = L, t = t, domain=self.mesh, degree=3)
                        # A0 = 0.0, L = L, t = t, domain=self.mesh, degree=3)
        # p0 = Expression("A0*sin(2.0*pi*t)",
                        # A0 = 666.611825, t=t,domain=self.mesh, degree=3)
        p0 = Expression("A*(1.0 - x[0]/L) + B", degree=1, A=13.0, B=0.0, L=L)

        p = [p0,]
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
        
        # p0 = 1999.835475
        p0 = pex[0]
        divu = 0.0
        pt = lmbda * divu - alphas[0] * p0 
        
        p = [pex[0],]

        return uex, p

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        AA = self.params.AA
        L = self.params.L
        n = self.period

        uex, pex = self.exact_solutions(t)
        
        # u0 = []
        u0 = [(uex, self.allboundary)]
        
        p0 = [((1), pex[0], self.allboundary)]
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        
        uex0, pex0 = self.exact_solutions(t0)
        uex1, pex1 = self.exact_solutions(t1)
        # u0 = [(theta * pex1[0] + (1.0 - theta) * pex0[0], 2), (theta * pex1[0] + (1.0 - theta) * pex0[0], 3)]
        p0 = []
        u0 = []

        return u0, p0

    def f(self, t):
        
        class MyExpression(Expression):
            def eval(self, value, x):
                value[0] = 0.0
            def value_shape(self):
                return (1,)
    
        ff = MyExpression()
        
        return ff
    
    def g(self, t):

        AA = self.params.AA

        gg = [0]*AA
        
        gg[0] = Expression("0.0", domain=self.mesh)
        
        return gg
    

    
    def nullspace(self):
      
        # No null space
        null = False
        return null


def single_run():
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    N = 1000
    M = 10.0
    dt = float(1.0/M)
    T = 100.0
    # T = 4*dt
    L = 70.0 
    Q = 1.0/(4.5e-10)
    AA = 1
    alphas = (1.0,)
    # Ks = (1.573e-5,)
    Ks = (1.57e-5,)
    G = (1.0e-13,)

    E = 584.0
    nu = 0.35

    Incompressible = False

    mesh = IntervalMesh(N, 0.0, L)
    Vf = VectorFunctionSpace(mesh, "CG", 1)
    vfinterp = Function(Vf)
    print "lmbda = ", nu*E/((1.0-2.0*nu)*(1.0+nu))
    print "mu = ", E/(2.0*(1.0+nu))

    print "Ks = ", Ks
    # Create problem set-up 
    problem = FirstTest(dict(mesh_file=mesh, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))

    print problem.params
    
    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T, "theta": 1.0, "direct_solver":True, "testing":False, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":True, "nonzero_initial_guess":True,\
                                        "relative_tolerance": 1.e-10, "absolute_tolerance": 1.e-10, "divergence_limit": 1.e10}})


    # Solve
    filenameu = "results/1D_1Net_Neumann4/u_1D.xdmf"
    filenamep0 = "results/1D_1Net_Neumann4/p0_1D.xdmf"
    filenamep1 = "results/1D_1Net_Neumann4/1p1_1D.xdmf"
    filenamevf = "results/1D_1Net_Neumann4/vf_1D.pvd"
    
    fileu = XDMFFile(mpi_comm_world(), filenameu)
    filep0 = XDMFFile(mpi_comm_world(), filenamep0)
    filep1 = XDMFFile(mpi_comm_world(), filenamep1)
    filevf = XDMFFile(mpi_comm_world(), filenamevf)

    # solutions = solver.solve_totalpressure()
    solutions = solver.solve_symmetric()
    
    for (U, t) in solutions:
        u = U.split(deepcopy=True)[0]
        p = U.split(deepcopy=True)[1:AA+2]
        vf = -Ks[0] * grad(p[0])
        vf = project(vf, Vf)
        # The adjoint should be in the solver?
        
        # p_CSF = Function(Ks[0])
        # p_CSF.assign(p[0])
        # J = Functional(inner(grad(p[0]), grad(p[0]))*dx*dt[FINISH_TIME])
        
        

        fileu.write(u)
        filep0.write(p[0])
        filevf.write(vf)
        plot(u, key="u", mesh=mesh)
        plot(p[0], key="p", mesh=mesh)
        plot(vf, key="vf", mesh=mesh)


    # plot(u, key="u", mesh=mesh)
    # plot(p[0], key="p", mesh=mesh)
    # plot(vf, key="vf", mesh=mesh)        
    # interactive()
    print "*"*200

if __name__ == "__main__":

    single_run()

    # Run quick convergence test:
    # run_quick_convergence_test()

    # Store all errors
    #main()
