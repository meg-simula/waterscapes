# from dolfin import *
from dolfinimport import *
import ufl
from cbcpost import ParamDict, Parameterized

class MPET(Parameterized):

    def __init__(self, params=None):
        """Initialize problem instance.

        Params will be taken from default_params and overridden
        by the values given to this constructor.
        """
        Parameterized.__init__(self, params)


    @classmethod
    def default_params(cls):

        params = ParamDict(
            # Physical parameters:
            L = 1.0,
            Q = 1.0, # FIXME: Remove, use c instead
            AA = 1,
            alphas = [1.0],
            rhos = [1.0],
            nu = 0.35, #must be less than 0.5
            rho = 1.0,
            E = 584e-3,
            mu=1.0,
            Ks = 1.0,
            G = 1.0,
            mesh_file = None,
            N = 32,
            Incompressible = False, # FIXME: Remove when you have removed Q
            Acceleration = False,
            c = 1.0 # c = 1/Q, Saturation coefficient 
        )
        return params

    def initial_conditions(self, t):
        """Return initial conditions.
        """
        raise NotImplementedError("initial conditions must be overridden in subclass")


    def boundary_conditions(self, t):
        """
        Return initial conditions:
        for displacement:
        1st entry is the expression, 2nd entry is the boundary

        for pressures:
        first entry is the subspace, 2nd entry expression, 3rd entry boundary

        Here it is an example:

            AA = self.params.AA
            u0 = [(Expression(("sin(2*pi*x[0])*sin(2*pi*x[1])*t","sin(2*pi*x[0])*sin(2*pi*x[1])*t"), domain = self.mesh, degree = 5, t = t), self.allboundary)]
            p0 = [ ( (i+1), Expression("(i+1)*sin(2*pi*x[0])*sin(2*pi*x[1])*t", domain = self.mesh, degree = 5, t = t, i=i), self.allboundary) for i in range(AA)]

            return u0, p0
        """
        raise NotImplementedError("boundary conditions must be overridden in subclass")

    def boundary_conditions_u(self, t_dt, t, theta):
        
        raise NotImplementedError("boundary conditions must be overridden in subclass")


    def boundary_conditions_p(self, t_dt, t, theta):
        
        raise NotImplementedError("boundary conditions must be overridden in subclass")



    def neumann_conditions(self, t_dt, t, theta):

        """Return neumann conditions.
        """
        raise NotImplementedError("neumann conditions must be overridden in subclass")

    def robin_conditions(self, t_dt, t, theta):

        """Return robin conditions.
        """
        raise NotImplementedError("robin conditions must be overridden in subclass")

    def exact_solutions(self, t):

        """
        exact solution
        """
        raise NotImplementedError("exact displacement must be overridden in subclass")


    def f(self, t):

        raise NotImplementedError("exact right hand side for the momentum equation must be overridden in subclass")


    def g(self, t):

        raise NotImplementedError("right hand side for the continuity equation be overridden in subclass")
    
    def nullspace(self):

        raise NotImplementedError("null space to be overridden in subclass")
    
class MPETProblem(object):

    def __init__(self, mesh, time, params=None):
        """Initialize problem instance.

        Params will be taken from default_params and overridden
        by the values given to this constructor.
        """

        self.mesh = mesh
        self.time = time

        # Update problem parameters
        self.params = self.default_parameters()
        if params is not None:
            self.params.update(params)

        As = range(self.params["A"])

        # Default values for forces and sources
        self.f = Constant((0.0, 0.0))
        self.s = Constant((0.0, 0.0))
        self.g = [Constant(0.0) for i in As]
        self.I = [Constant(0.0) for i in As]

        # Default values for Dirichlet boundary conditions
        self.u_bar = Constant((0.0, 0.0))
        self.p_bar = [Constant(0.0) for i in As]

        # Default markers
        INVALID = 7101982
        markers = FacetFunction("size_t", mesh)
        markers.set_all(INVALID)
        self.momentum_boundary_markers = markers

        self.continuity_boundary_markers = []
        for i in As:
            markers = FacetFunction("size_t", mesh)
            markers.set_all(INVALID)
            self.continuity_boundary_markers += [markers]
        
    @classmethod
    def default_parameters(cls):
        "Define the set of parameters to define the problem."
        ps = {"A": 1.0,
              "alpha": (1.0, ),
              "rho": 1.0,
              "nu": 0.479,
              "E": 584e-3,
              "K": (1.0, ),
              "G": ((1.0,),),
              "c": 0.0,
              }
        return ps
        
        