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

    def __init__(self, params=None, mesh=None):

        MPET.__init__(self, params)

        # Create mesh
        x0 = Point(0.0, 0.0)
        x1 = Point(self.params.L, self.params.L)
        n = self.params.N
        # self.mesh = RectangleMesh(x0, x1, n, n, "crossed")
        if not mesh:
            self.mesh = UnitSquareMesh(n, n)
        else:
            self.mesh=mesh
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

    def initial_conditions(self, t):

        AA = self.params.AA
        L = self.params.L
        n = self.period

        u0 = Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0)","sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0)"), domain=self.mesh, degree=5, t=t, L=L, n=n)
        p0 = [Expression("(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", domain=self.mesh, degree=5, t=t, L=L, i=i, n=n) for i in range(AA)]

        return u0, p0

    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        AA = self.params.AA
        L = self.params.L
        n = self.period

        u0 = [(Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)","sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)"),
                          domain=self.mesh, degree=5, t=t, L=L, n=n), self.allboundary)]

        p0 = [((i+1), Expression("(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", domain=self.mesh, degree=5, t=t, L=L, i=i, n=n), self.allboundary)
              for i in range(AA)]

        return u0, p0

    def neumann_conditions(self, t0, t1, theta):
        u0 = []
        p0 = []
        return u0, p0

    def f(self, t):

        E = self.params.E
        nu = self.params.nu
        alphas = self.params.alphas
        mesh = self.mesh
        L = self.params.L
        n = self.period

        ff = Expression(("- E*nu*(-4*(pi*pi)*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + 4*(pi*pi)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/(L*L))/((-2*nu + 1)*(nu + 1)) \
                          - E*(-2.0*(pi*pi)*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + 2.0*(pi*pi)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/(L*L))/(nu + 1) \
                          + 4.0*(pi*pi)*E*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/((L*L)*(nu + 1)) + 2*pi*a0*sin(n*t + 1.0)*sin(2*pi*x[1]/L)*cos(2*pi*x[0]/L)/L \
                          + 4*pi*a1*sin(n*t + 1.0)*sin(2*pi*x[1]/L)*cos(2*pi*x[0]/L)/L",\
                         "-E*nu*(-4*(pi*pi)*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + 4*(pi*pi)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/(L*L))/((-2*nu + 1)*(nu + 1)) \
                         - E*(-2.0*(pi*pi)*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + 2.0*(pi*pi)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/(L*L))/(nu + 1) \
                         + 4.0*(pi*pi)*E*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/((L*L)*(nu + 1)) + 2*pi*a0*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/L \
                         + 4*pi*a1*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/L"), \
                        domain=mesh, degree=5, t=t, nu=nu, E=E, a0=alphas[0], a1=alphas[1], L=L, n=n)
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

        gg = [0]*AA

        if self.params.Incompressible == False:
            gg[0] = Expression(' 8*(pi*pi)*Ks*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + a0*(2*pi*n*sin(2*pi*x[0]/L)*cos(n*t + 1.0)*cos(2*pi*x[1]/L)/L \
                               + 2*pi*n*sin(2*pi*x[1]/L)*cos(n*t + 1.0)*cos(2*pi*x[0]/L)/L) - G01*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L) \
                               + n*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*cos(n*t + 1.0)/Q',
                             domain=mesh, degree=5, t=t, Q=Q, L=L, a0=alphas[0], Ks=Ks[0], G01=G[1], n=n)

            gg[1] = Expression(' 16*(pi*pi)*Ks*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + a1*(2*pi*n*sin(2*pi*x[0]/L)*cos(n*t + 1.0)*cos(2*pi*x[1]/L)/L \
                               + 2*pi*n*sin(2*pi*x[1]/L)*cos(n*t + 1.0)*cos(2*pi*x[0]/L)/L) + G10*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L) \
                               + 2*n*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*cos(n*t + 1.0)/Q',
                             domain=mesh, degree=5, t=t, Q=Q, Ks=Ks[1], a1=alphas[1], G10 = G[0], L=L, n=n)

        else:
            gg[0] = Expression(' 8*(pi*pi)*Ks*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + a0*(2*pi*n*sin(2*pi*x[0]/L)*cos(n*t + 1.0)*cos(2*pi*x[1]/L)/L \
                               + 2*pi*n*sin(2*pi*x[1]/L)*cos(n*t + 1.0)*cos(2*pi*x[0]/L)/L) - G01*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)',
                             domain=mesh, degree=5, t=t, Q=Q, L=L, a0=alphas[0], Ks=Ks[0], G01=G[1], n=n)

            gg[1] = Expression(' 16*(pi*pi)*Ks*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + a1*(2*pi*n*sin(2*pi*x[0]/L)*cos(n*t + 1.0)*cos(2*pi*x[1]/L)/L \
                               + 2*pi*n*sin(2*pi*x[1]/L)*cos(n*t + 1.0)*cos(2*pi*x[0]/L)/L) + G10*sin(n*t + 1.0)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)',
                             domain=mesh, degree=5, t=t, Q=Q, Ks=Ks[1], a1=alphas[1], G10 = G[0], L=L, n=n)
        return gg

    def exact_solutions(self, t):
        AA = self.params.AA
        L = self.params.L
        n = self.period
        u0 = Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)","sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)"), domain=self.mesh, degree=5, L=L, t=t, n=n)
        p0 = [Expression("(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", domain=self.mesh, degree=5, L=L, t=t, i=i, n=n) for i in range(AA)]
        return u0, p0
    
    def nullspace(self):
      
        # No null space
        null = []
        return null

def single_run(N=64, M=8):
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    T = 0.1
    k = 0.1
    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    mesh = UnitSquareMesh(N,N)
    R = FunctionSpace(mesh, "DG", 0)
    K0 = Function(R)
    K0.vector()[:] = 1.0
    K0 = Constant(1.0)
    Ks = (K0, Constant(1.0))
    G = (1.0, 1.0)

    E = 1.0
    nu = 0.35
    Incompressible = False

    # Create problem set-up
    problem = FirstTest(dict(N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu), mesh=mesh)

    # Create solver
    solver = SimpleSolver(problem, {"dt": k, "T": T, "direct_solver":True, "fieldsplit": False,\
                                    "krylov_solver": {"monitor_convergence":False, "nonzero_initial_guess": False,\
                                    "relative_tolerance": 1.e-200, "absolute_tolerance": 1.e-16}})

    # Solve
    solutions = solver.solve()
    for (U, t) in solutions:
        pass

    # adj_html("forward.html", "forward")
    # adj_html("adjoint.html", "adjoint")
    
    # parameters["adjoint"]["stop_annotating"] = True
    
    # print "Replaying"
    #replay_dolfin(forget=False, tol=0.0, stop=True)
    
    # Extract views to components of U
    (u, p0, p1) = split(U)
    
    # j = inner(grad(p0), grad(p0))*dx
    # J = assemble(j)
    # print "J = ", J

    J = Functional(inner(grad(p0), grad(p0))*dx*dt[FINISH_TIME])
    
    m = Control(Ks[0])
    Jr = ReducedFunctional(J, m)
    
    # Ks[0].vector()[:] += 0.001
    
    # print "Functional value", Jr(Ks[0])
        
    
    
    
    
    
    
    
    #print assemble(inner(grad(p0), grad(p0))*dx)
    #Ks[0].vector()[:] += 0.001
    
    PETScOptions().set("ksp_monitor_true_residual")
    PETScOptions().set("ksp_converged_reason")
        
    Jr.taylor_test(Ks[0], seed=0.0001)
    dJdm = compute_gradient(J, m, project=True)
    # print " dJdm = ", float(dJdm)
    #plot(dJdm, mesh)
    #plot(p0, mesh)
    #interactive()
    # print " dJdm = ", float(dJdm)

    u = U.split(deepcopy=True)[0]
    p = U.split(deepcopy=True)[1:]

    uex, pex = problem.exact_solutions(T)
    erru, errpL2, errpH1 = compute_error(u, p, uex, pex, problem.mesh)

    # Return errors and meshsize
    h = problem.mesh.hmax()
    #h = L/N
    print h, erru, errpL2, errpH1

    # return (erru, errpL2, errpH1, h)


if __name__ == "__main__":

    # Just test a single run
    single_run(N=8, M=8)

