from dolfinimport import *
import numpy as np
#from block import *
#from block.algebraic.petsc import *
#from block.iterative import *
from rm_basis_L2 import rigid_motions 
import resource

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def elastic_stress(u, E, nu):
    d = u.geometric_dimension()
    I = Identity(d)
    s = E/(1.0 + nu)*sym(grad(u)) + nu*E/((1.0-2.0*nu)*(1.0+nu))*div(u)*I
    return s

class MPETSolver(object):
    """This solver solves the multiple-network poroelasticity equations
    (MPET): find a vector field (the displacement) u and the network
    pressures p_a for a set of networks a = 1, ..., A such that:

        - div ( sigma(u) - sum_{a} alpha_a p_a ) = f     (1)
        c_a p_t + alpha_a div(u_t) - div K_a p_a = g_a   (2)

    where 
    
      sigma(u) = 2*mu*eps(u) + lmbda div(u) I 

    and eps(u) = sym(grad(u)), and mu and lmbda are the standard Lame
    parameters. For each network a, c_a is the saturation coefficient,
    alpha_a is the Biot-Willis coefficient, and K_a is the hydraulic
    conductivity.

    f is a given body force and g_a source(s) in network a.

    See e.g. Tully and Ventikos, 2011 for further details on the
    multiple-network poroelasticity equations.

    Boundary conditions:

    We assume that there is a facet function marking the different
    subdomains of the boundary. 

    For the momentum equation (1):
    
    We assume that each part of the boundary of the domain is one of
    the following types:

    Dirichlet: 

      u(., t) = \bar u(t) 

    Neumann:

      (sigma(u) - sum_{a} alpha_a p_a I) * n = traction

    For the continuity equations (2):

    Dirichet:

      p_a(., t) = \bar p_a(t) 
      
    Neumann

      K grad p_a(., t) * n = flux_a(t)

    Robin

     ...

    Initial conditions:

      u(x, t_0) = u_0(x)

      p_a(x, t_0) = p0_a(x) if c_a > 0

    FIXME: Document which solver formulation this solver uses.
    """

    def __init__(self, problem, params=None):
        "Create solver with given MPET problem and parameters."
        self.problem = problem

        # Update parameters if given
        self.params = self.default_params()
        if params is not None:
            self.params.update(params)

        # Initialize objects
        self.F = self.initialize()

        # Exact left and right hand side

        
    def solve(self):
        "."
        
        dt = self.params["dt"]
        T = self.params["T"]
        time = self.problem.time

        (a, L) = system(self.F)

        # Assemble left-hand side matrix
        A = assemble(a)

        solver = LUSolver(A)

        self.up.assign(self.up_)
        while float(time) < T:
            
            # Assemble vector
            b = assemble(L)
            
            # Apply boundary conditions

            # Solve
            solver.solve(self.up.vector(), b)

            yield self.up, float(time)

            # Update previous solution
            assign(self.up_, self.up)

            # Update time
            time.assign(float(time) + dt)
            
        
    @staticmethod
    def default_params():
        "Define default solver parameters."
        params = Parameters("SimpleSolver")
        params.add("dt", 0.05)
        params.add("t", 0.0)
        params.add("T", 1.0)
        params.add("theta", 0.5)
        params.add("u_degree", 2)
        params.add("p_degree", 1)
        params.add("direct_solver", True)
        params.add(KrylovSolver.default_parameters())
        params.add(LUSolver.default_parameters())
        params.add("testing", False)
        params.add("fieldsplit", False)
        params.add("symmetric", False)
        return params

    def initialize(self):

        # Extract mesh from problem
        mesh = self.problem.mesh

        # Redefine measures based on subdomain information provided by
        # the problem
        #self.ds = Measure("ds", domain=self.problem.mesh,
        #                  subdomain_data=self.problem.facet_domains)

        # Extract time and time step
        dt = Constant(self.params.dt)
        #t = self.problem.time
        #t_ = Constant(self.params.t)
        #t = Constant(self.params.t + float(dt))
        
        # Extract the number of networks
        # FIXME: Problem networks should be called just A
        A = self.problem.params["A"]

        # Create function spaces 
        V = VectorElement("CG", mesh.ufl_cell(), self.params.u_degree)
        W = FiniteElement("CG", mesh.ufl_cell(), self.params.p_degree)
        M = MixedElement([V] + [W for i in range(A)])
        VW = FunctionSpace(mesh, M)

        # Create previous solution field(s) and extract previous
        # displacement solution u_ and pressures p_ = (p_1, ..., p_A)
        up_ = Function(VW)
        u_ = split(up_)[0]
        p_ = split(up_)[1:]

        # Create trial functions and extract displacement u and pressure
        # trial functions p = (p_1, ..., p_A)
        up = TrialFunctions(VW)
        u = up[0]
        p = up[1:]

        # Create test functions and extract displacement u and pressure
        # test functions p = (p_1, ..., p_A)
        vw = TestFunctions(VW)
        v = vw[0]
        w = vw[1:]

        # um and pm represent the solutions at time t + dt*theta
        theta = self.params.theta
        um = theta*u + (1.0 - theta)*u_
        pm = [(theta*p[i] + (1.0-theta)*p_[i]) for i in range(A)]
        
        # Define geometry related objects

        # Extract material parameters from problem
        E = self.problem.params["E"]           # Young's modulus
        nu = self.problem.params["nu"]         # Poisson ratio
        alpha = self.problem.params["alpha"]
        K = self.problem.params["K"]
        S = self.problem.params["S"]
        c = self.problem.params["c"]

        # Define the extra/elastic stress
        sigma = lambda u: elastic_stress(u, E, nu)

        # First equation (elliptic)
        F00 = inner(sigma(u), sym(grad(v)))*dx()
        F01 = sum([-alpha[i]*p[i]*div(v) for i in range(A)])*dx()
        f = self.problem.f
        F02 = - dot(f, v)*dx()
        F0 = F00 + F01 + F02
        
        # Second equation (parabolic)
        F10 = sum([-alpha[i]*div(u-u_)*w[i] for i in range(A)])*dx()
        F11 = sum([-dt*K[i]*inner(grad(pm[i]), grad(w[i]))
                   for i in range(A)])*dx()
        F12 = sum([sum([-dt*S[i][j]*(pm[i] - pm[j])*w[i] for j in range(A)])
                   for i in range(A)])*dx()
        g = self.problem.g
        F13 = - sum([dt*g[i]*w[i] for i in range(A)])*dx()

        # Saturation coefficient 
        F14 = sum([c*(p[i] - p_[i])*w[i] for i in range(A)])*dx()
        F1 = F10 + F11 + F12 + F13 + F14

        # Combined variational form (F = 0 to be solved)
        F = F0 + F1

        self.up = Function(VW)
        self.up_ = up_
        
        return F

        
        
