"""run with python prova.py "$(<params.txt)"
use cbcpost and set_parse_command_line_arguments(True)
"""
from mpet import *
# from solver import *
# from error import *
from numpy import zeros
import sys
import os
import math
from cbcpost import *
import time
# set_parse_command_line_arguments(True)


# Turn on FEniCS optimizations
parameters["form_compiler"]["cpp_optimize"] = True
flags = ["-O3", "-ffast-math", "-march=native"]
parameters["form_compiler"]["cpp_optimize_flags"] = " ".join(flags)
#parameters["form_compiler"]["representation"] = "uflacs"

class Left(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return (on_boundary and x[0] < DOLFIN_EPS )

class Right(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return (on_boundary and x[0] > 1 - DOLFIN_EPS)

class Front(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return (on_boundary and x[1] < 1 + DOLFIN_EPS)

class Back(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return (on_boundary and x[1] < DOLFIN_EPS)

class Top(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return (on_boundary and x[1] > 1 - DOLFIN_EPS)

class Bottom(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)
    def inside(self, x, on_boundary):
        return (on_boundary and x[1] < DOLFIN_EPS )

class Permeability(Expression):
    def eval(self, v, x):
        if x[1] >= 1.0/4.0 and x[1] <= 3.0/4.0 : v[0] = 1.0
        else: v[0] = 1.0
        
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
        self.period = 5./2*math.pi
        # Create boundary markers and initialize all facets to +
        self.facet_domains = FacetFunction("size_t", self.mesh)
        self.facet_domains.set_all(0)

        # Mark all exterior boundary facets as 1
        # Note: The CompiledSubDomain class gives faster code than
        # Python subclassing
        #self.allboundary = AllBoundary()
        self.allboundary = CompiledSubDomain("on_boundary")
        self.allboundary.mark(self.facet_domains, 1) #!!!
        # 
        self.uboundary = CompiledSubDomain("on_boundary & (near(x[0], 0.0) || near(x[1], 0.0) || near(x[1], 1.0))")
        self.uboundary.mark(self.facet_domains, 2)
        
        self.pboundary = CompiledSubDomain("on_boundary & (near(x[0], 1.0) || near(x[1], 0.0) || near(x[1], 1.0))")
        self.pboundary.mark(self.facet_domains, 3)

        self.stress = CompiledSubDomain("on_boundary & near(x[0], 1.0)")
        self.stress.mark(self.facet_domains, 4)
        
        self.flux = CompiledSubDomain("on_boundary & near(x[0], 0.0)")
        self.flux.mark(self.facet_domains, 5)
        
        # plot(self.facet_domains)
        # self.left = Left()
        # self.left.mark(self.facet_domains, 2)
        # 
        # self.right = Right()
        # self.right.mark(self.facet_domains, 3)
        # 
        # # self.front = Front()
        # # self.front.mark(self.facet_domains, 4)
        # # 
        # # self.back = Back()
        # # self.back.mark(self.facet_domains, 5)
        # 
        # self.top = Top()
        # self.top.mark(self.facet_domains, 6)
        # 
        # self.bottom = Bottom()
        # self.bottom.mark(self.facet_domains, 7)
        
        gdim = self.mesh.geometry().dim()
        self.cell_domains = MeshFunction("size_t", self.mesh, gdim,
                                         self.mesh.domains())

        # Attach domains to measures for convenience
        self.ds = Measure("ds")(domain=self.mesh, subdomain_data=self.facet_domains)

        #self.dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.facet_domains)
        #self.dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=self.cell_domains)

        #It is needed to update the quantities that depend on the num. of networks: G, alphas, rhos...

        #length expressed in mm!!

        # self.params.update(G = zeros((self.params.AA, self.params.AA)))
        # self.params.G[0][1] = 1.0e-17
        # self.params.G[1][0] = 1.0e-17
        # self.params.update(alphas = [1.0, 1.0])
        # self.params.update(Ks = [0.015730337078651686, 37.453183520599254])
        self.params.update(exact_u = "sin(2*pi*x[0])*sin(2*pi*x[1])*sin(n*t + 1.0), sin(2*pi*x[0])*sin(2*pi*x[1])*sin(n*t+1.0)")
        self.params.update(exact_p = "(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*sin(n*t + 1.0) for i in range(AA)")
        # self.params.Incompressible = str(self.params.Incompressible)

    def initial_conditions(self, t):

        AA = self.params.AA
        L = self.params.L
        n = self.period
        # u0 = Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])*t","sin(2*pi*x[0])*sin(2*pi*x[1])*t"), domain = self.mesh, degree = 5, t = t)
        # p0 = [Expression("(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*t", domain = self.mesh, degree = 5, t = t, i=i) for i in range(AA)]

        # u0 = Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])*sin(t)","sin(2*pi*x[0])*sin(2*pi*x[1])*sin(t)"), domain = self.mesh, degree = 5, t = t)
        # p0 = [Expression("(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*sin(t)", domain = self.mesh, degree = 5, t = t, i=i) for i in range(AA)]

        u0 = Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0)","sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n * t + 1.0)"), domain=self.mesh, degree=5, t=t, L=L, n=n)
        p0 = [Expression("(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", domain=self.mesh, degree=5, t=t, L=L, i=i, n=n) for i in range(AA)]

        # u0 = Constant((0.0,0.0, 0.0))
        # p0 = [(Constant(0.0))]

        # u0 = Constant((0.0,0.0))
        # p0 = [(Constant(0.0))]
        return u0, p0
    


    def boundary_conditions(self, t):
        "Specify essential boundary conditions at time t as a given tuple."

        AA = self.params.AA
        L = self.params.L
        n = self.period
        # u0 = [(Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])*t","sin(2*pi*x[0])*sin(2*pi*x[1])*t"), domain = self.mesh, degree = 5, t = t), self.allboundary)]
        # #first entry is the space, second entry is the expression, 3rd entry is the boundary
        # p0 = [ ( (i+1), Expression("(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*t", domain = self.mesh, degree = 5, t = t, i=i), self.allboundary) for i in range(AA)]

        # u0 = [(Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])*t","sin(2*pi*x[0])*sin(2*pi*x[1])*t"), domain = self.mesh, degree = 5, t = t), 1)]
        # #first entry is the space, second entry is the expression, 3rd entry is the boundary
        # p0 = [ ( (i+1), Expression("(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*t", domain = self.mesh, degree = 5, t = t, i=i), 1) for i in range(AA)]

        # u0 = [(Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])*sin(t)","sin(2*pi*x[0])*sin(2*pi*x[1])*sin(t)"), domain = self.mesh, degree = 5, t = t), self.allboundary)]
        # #first entry is the space, second entry is the expression, 3rd entry is the boundary
        # p0 = [ ( (i+1), Expression("(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*sin(t)", domain = self.mesh, degree = 5, t = t, i=i), self.allboundary) for i in range(AA)]

        # Value, boundary subdomain
        # u0 = [(0, Expression("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t)",
        #                   domain=self.mesh, degree=5, t=t, L=L, n=n), self.allboundary),
        #       (1, Expression("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t)",
        #                   domain=self.mesh, degree=5, t=t, L=L, n=n), self.allboundary)]

        u0 = [(Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)","sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)"),\
                          domain=self.mesh, degree=5, t=t, L=L, n=n), self.allboundary)]
        
        p0 = [((i+1), Expression("(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", domain=self.mesh, degree=5, t=t, L=L, i=i, n=n), self.allboundary) for i in range(AA)]

        #BH Test case
        # u0 = [(0, Constant(0.0), self.left),
        #       (0, Constant(0.0), self.right),
        #       (1, Constant(0.0), self.front),
        #       (1, Constant(0.0), self.back),
        #       (2, Constant(0.0), self.bottom)]
        # 
        # u0 = [(0, Constant(0.0), self.left),
        #       (0, Constant(0.0), self.right),
        #       # (1, Constant(0.0), self.front),
        #       # (1, Constant(0.0), self.back),
        #       (1, Constant(0.0), self.bottom)]
        # 
        # p0 = [(1, Constant(0.0), self.top)]
        # u0 = []
        return u0, p0

    def neumann_conditions(self, t0, t1, theta):

        E = self.params.E
        nu = self.params.nu
        L = self.params.L
        n = self.period
        alphas = self.params.alphas
        # u0 = [ ( Expression( (" - 1.0*E*nu*(2*pi*sin(2*pi*x[0]/L)*sin(n*t + 1.0)*cos(2*pi*x[1]/L)/L + 2*pi*sin(2*pi*x[1]/L)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)/L)/((-2*nu + 1)*(nu + 1)) \
        #                         - 2.0*pi*E*sin(2*pi*x[1]/L)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)/(L*(nu + 1)) +\
        #                         a0 * sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0) + \
        #                         a1 * 2 * sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)",
        #                       " - 1.0*E*(1.0*pi*sin(2*pi*x[0]/L)*sin(n*t + 1.0)*cos(2*pi*x[1]/L)/L + 1.0*pi*sin(2*pi*x[1]/L)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)/L)/(nu + 1)"), \
        #                           t = t1, nu=nu, E=E, a0 = alphas[0], a1 = alphas[1], L=L, n=n), 4) ]
        # 
        # p0 = [(Expression("theta * (2.0*pi*sin(2*pi*x[1]/L)*sin(n*t_dt + 1.0)*cos(2*pi*x[0]/L)/L) + (1.0-theta)*(2.0*pi*sin(2*pi*x[1]/L)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)/L)", \
        #                     t=t0, t_dt=t1, theta=theta, nu=nu, E=E, L=L, n=n), 0, 5),
        #       (Expression("theta * (4.0*pi*sin(2*pi*x[1]/L)*sin(n*t_dt + 1.0)*cos(2*pi*x[0]/L)/L) + (1.0-theta)*(4.0*pi*sin(2*pi*x[1]/L)*sin(n*t + 1.0)*cos(2*pi*x[0]/L)/L)", \
        #                     t=t0, t_dt=t1, theta=theta, nu=nu, E=E, L=L, n=n), 1, 5)]

        # #fixit  
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

        # ff = Expression(("- E*nu*(-4*(pi*pi)*sin(n*t)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + 4*(pi*pi)*sin(n*t)*cos(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/(L*L))/((-2*nu + 1)*(nu + 1)) \
        #                   - E*(-2.0*(pi*pi)*sin(n*t)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + 2.0*(pi*pi)*sin(n*t)*cos(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/(L*L))/(nu + 1) \
        #                   + 4.0*(pi*pi)*E*sin(n*t)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/((L*L)*(nu + 1)) + 2*pi*a0*sin(n*t)*sin(2*pi*x[1]/L)*cos(2*pi*x[0]/L)/L",\
        #                  "-E*nu*(-4*(pi*pi)*sin(n*t)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + 4*(pi*pi)*sin(n*t)*cos(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/(L*L))/((-2*nu + 1)*(nu + 1)) \
        #                  - E*(-2.0*(pi*pi)*sin(n*t)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + 2.0*(pi*pi)*sin(n*t)*cos(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/(L*L))/(nu + 1) \
        #                  + 4.0*(pi*pi)*E*sin(n*t)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/((L*L)*(nu + 1)) + 2*pi*a0*sin(n*t)*sin(2*pi*x[0]/L)*cos(2*pi*x[1]/L)/L"), \
        #                 domain=mesh, degree=5, t=t, nu=nu, E=E, a0=alphas[0], L=L, n=n)

        # ff = Expression(("0.0","0.0"), degree=3)
        # ff = Expression(("0.0","0.0","0.0"), degree=3)
  
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
        #1 network
        # gg = [Expression('8*pi*pi*Ks*t*sin(2*pi*x[0])*sin(2*pi*x[1]) + alphas*(2*pi*sin(2*pi*x[0])*cos(2*pi*x[1]) + 2*pi*sin(2*pi*x[1])*cos(2*pi*x[0])) + sin(2*pi*x[0])*sin(2*pi*x[1])/Q',
        #                 domain=mesh, degree=5, t=t, Q=Q, Ks=Ks, alphas=alphas[0]) for i in range(AA)] #attention to alphaaaaaaa as well

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

        # gg[0] = Expression("0.0", degree=3)

        # gg[0] = Expression(' 8*(pi*pi)*Ks*sin(n*t)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)/(L*L) + a0*(2*pi*n*sin(2*pi*x[0]/L)*cos(n*t)*cos(2*pi*x[1]/L)/L \
        #                    + 2*pi*n*sin(2*pi*x[1]/L)*cos(n*t)*cos(2*pi*x[0]/L)/L) + n*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*cos(n*t)/Q',
        #                  domain=mesh, degree=5, t=t, Q=Q, L=L, a0=alphas[0], Ks=Ks[0], n=n)
        
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

def single_run(N=64, M=64):
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    T = 1.0
    dt = float(T)/M
    # dt = 1.0
    L = 1.0
    Q = 1.0
    AA = 1
    alphas = (1.0, )
    # Ks = (Permeability(), )
    Ks = (1.0, )
    G = (1.0, )
    
    E = 1.0
    nu = 0.35
    
    lmbda = 1.0
    mu = 1.0
    
    Incompressible = False

    # Create problem set-up
    problem = FirstTest(dict(N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu, lmbda=lmbda, mu=mu))


    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T})

    # Solve
    solutions = solver.solve()
    for (U, t) in solutions:
        pass

    u = U.split(deepcopy=True)[0]
    p = U.split(deepcopy=True)[1:]

    # interactive()
    # plot(1.0/3.0 * tr( sigma(U0.split()[0]) ) )
    # interactive()

    # pfile = File("p.pvd")  
    # ufile = File("u.pvd")
    # pfile << p[0]
    # ufile << u
    # Compute error at end time
    uex, pex = problem.exact_solutions(T)
    erru, errpL2, errpH1 = compute_error(u, p, uex, pex, problem.mesh)
    #FIXME!!
    # Return errors and meshsize
    h = problem.mesh.hmax
    print erru, errpL2, errpH1
    
    return (erru, errpL2, errpH1, h)

    # return 1

def main():

    # Create error directory for storing errors if not existing
    if not os.path.isdir("Errors"):
        os.mkdir("Errors")
 
    casedir = "Errors_%s" % time.strftime("%Y_%m_%d__H%H_M%M_S%S")
    print "Storing errors to %s" % casedir
    os.mkdir(casedir)

    NN = [8, 16, 32, 64]#, 128]
    T = 1.0
    dts = [float(T)/i for i in NN]
    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    E = 1.0
    Incompressible = False    
    # 
    # kappa1_list = range(-1,2)
    # kappa2_list = range(-1,2)
    # lambda_list = range(-1,2)
    
    kappa1 = Constant(1.0)
    kappa2 = Constant(1.0)
    gamma = Constant(1.0)   
    nu = Constant(0.35)

    G = (gamma, gamma)

    param_str = "_k1=" + str(float(kappa1)) + "_k2=" + str(float(kappa2)) + "_nu=" + str(float(nu)) + "_E=" + str(float(E)) + "_gamma=" + str(float(gamma))  
    Error = SaveError(NN, dts, 2, param_str, casedir)

    cc = 1
    for N in NN:
        print " N = ", N
        problem = FirstTest({"N":N, "L":L, "Ks": (kappa1, kappa2), "alphas": alphas, "G": G,
                             "AA":AA, "E":E, "nu":nu, "Incompressible": Incompressible})
 

        rr = 1
        for dt in dts:

            solver = SimpleSolver(problem, {"dt": dt, "T": T})
            solutions = solver.solve()

            for (U0, t) in solutions:
                pass
            
            u0 = U0.split(deepcopy=True)[0]
            p0 = U0.split(deepcopy=True)[1:]

            uex, pex = problem.exact_solutions(solver.params.T)

            erru, errpL2, errpH1 = compute_error(u0, p0, uex, pex, problem.mesh)
            Error.store_error(erru, errpL2, errpH1, cc, rr)
            rr += 1

        cc += 1

    #FIXME
    Error.save_file(2)

    paramfile = open('Errors/parameters_' + param_str, 'w')
    paramfile.write(str(problem.params))
    paramsolverfile = open('Errors/parameters_solver_' + param_str, 'w')
    paramsolverfile.write(str(solver.params))


def convergence_rates(errors, hs):
    rates = [(math.log(errors[i+1]/errors[i]))/(math.log(hs[i+1]/hs[i]))
             for i in range(len(hs)-1)]

    return rates

def run_quick_convergence_test():

    # Remove all output from FEniCS (except errors)
    set_log_level(ERROR)

    # Make containers for errors
    u_errors = []
    p_errors = [[] for i in range(2)]
    hs = []

    # Iterate over mesh sizes/time steps and compute errors
    start = time.time()
    print "Start"
    for j in [8, 16, 32]:#, 64, 128]:#, 8, 16]:
        print "i = ", j
        (erru, errpL2, errpH1, h) = single_run(N=j, M=j)
        hs += [h]
        u_errors += [erru]
        print "\| u(T)  - u_h(T) \|_1 = %r" % erru
        for (i, errpi) in enumerate(errpL2):
            print "\| p_%d(T)  - p_h_%d(T) \|_0 = %r" % (i, i, errpi)
            p_errors[i] += [errpi]
        print

    # Compute convergence rates:
    u_rates = convergence_rates(u_errors, hs)
    p0_rates = convergence_rates(p_errors[0], hs)
    p1_rates = convergence_rates(p_errors[1], hs)

    # Print convergence rates
    print "u_rates = ", u_rates
    print "p0_rates = ", p0_rates
    print "p1_rates = ", p1_rates
    end = time.time()
    print "Time_elapsed = ", end - start


if __name__ == "__main__":

    # Run quick convergence test:
    # run_quick_convergence_test()
    main()
    # single_run()