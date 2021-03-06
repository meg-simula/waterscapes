"""run with python prova.py "$(<params.txt)"
use cbcpost and set_parse_command_line_arguments(True)
"""
from mpetproblem import *
from solver import *
from error import *
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

        # Attach domains to measures for convenience
        self.ds = Measure("ds")(domain=self.mesh, subdomain_data=self.facet_domains)
        self.params.update(exact_u = "sin(2*pi*x[0])*sin(2*pi*x[1])*sin(n*t + 1.0), sin(2*pi*x[0])*sin(2*pi*x[1])*sin(n*t+1.0)")
        self.params.update(exact_p = "(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*sin(n*t + 1.0) for i in range(AA)")

    def initial_conditions(self, t):

        AA = self.params.AA
        L = self.params.L
        n = self.period

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

        u0 = [(Expression(("sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)","sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)"),\
                          domain=self.mesh, degree=5, t=t, L=L, n=n), self.allboundary)]
        
        p0 = [((i+1), Expression("(i+1)*sin(2*pi*x[0]/L)*sin(2*pi*x[1]/L)*sin(n*t + 1.0)", domain=self.mesh, degree=5, t=t, L=L, i=i, n=n), self.allboundary) for i in range(AA)]

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

def single_run(N=64, M=64):
    "N is the mesh size, M the number of time steps."

    # Specify discretization parameters
    T = 1.0
    dt = float(T)/M
    # dt = 1.0
    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    # Ks = (Permeability(), )
    Ks = (1.0, 1.0)
    G = (1.0e-6, 1.0e-6)
    
    E = 1.0
    nu = 0.35
    # 
    # lmbda = 1.0
    # mu = 1.0
    # 
    Incompressible = False

    # Create problem set-up
    problem = FirstTest(dict(N=N, L=L, Q=Q, AA=AA, alphas=alphas,
                             Ks=Ks, G=G, Incompressible=Incompressible, E=E, nu=nu))


    # Create solver
    solver = SimpleSolver(problem, {"dt": dt, "T": T})

    # Solve
    solutions = solver.solve()
    for (U, t) in solutions:
        pass

    u = U.split(deepcopy=True)[0]
    p = U.split(deepcopy=True)[1:]

    uex, pex = problem.exact_solutions(T)
    erru, errpL2, errpH1 = compute_error(u, p, uex, pex, problem.mesh)
    #FIXME!!
    # Return errors and meshsize
    # h = problem.mesh.hmax
    h = L/N
    # print erru, errpL2, errpH1
    
    return (erru, errpL2, errpH1, h)

    # return 1

def main():

    # Create error directory for storing errors if not existing
    if not os.path.isdir("Errors"):
        os.mkdir("Errors")
 

    NN = [8, 16, 32, 64]#, 128]
    T = 1.0
    dts = [float(T)/i for i in NN]
    L = 1.0
    Q = 1.0
    AA = 2
    alphas = (1.0, 1.0)
    E = 1.0
    Incompressible = False    
      
    kappa1 = Constant(1.0)
    kappa2 = Constant(1.0)
    gamma = Constant(1.0e-6)   
    nu = Constant(0.35)

    G = (gamma, gamma)

    param_str = "_k1=" + str(float(kappa1)) + "_k2=" + str(float(kappa2)) + "_nu=" + str(float(nu)) + "_E=" + str(float(E)) + "_gamma=" + str(float(gamma))
    folder = "Error_MMS1"
    Error = SaveError(NN, dts, 2, param_str, folder)

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


def convergence_rates(errors, hs):
    rates = [(math.log(errors[i+1]/errors[i]))/(math.log(hs[i+1]/hs[i])) for i in range(len(hs)-1)]

    return rates

def run_quick_convergence_test():

    # Remove all output from FEniCS (except errors)
    set_log_level(ERROR)

    # Make containers for errors
    u_errors = []
    p_errorsL2 = [[] for i in range(2)]
    p_errorsH1 = [[] for i in range(2)]
    hs = []

    # Iterate over mesh sizes/time steps and compute errors
    start = time.time()
    print "Start"
    for j in [8, 16, 32, 64]:#, 128]:#, 8, 16]:
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


if __name__ == "__main__":

    # Run quick convergence test:
    run_quick_convergence_test()
    # main()
    # single_run()