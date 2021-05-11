__author__ = "Eleonora Piersanti <eleonora@simula.no>"

# Modified by Marie E. Rognes <meg@simula.no>, 2017

from dolfin import *

class MPETProblem(object):
    """An MPETProblem represents an instance of the multiple-network
    poroelasticity (MPET) equations: find the displacement vector
    field u and the network pressures p_j for a set of networks j = 1,
    ..., J such that:

    (1)    - div ( sigma(u) - sum_{j=1}^J alpha_j p_j I) = f          
    (2)    c_j (p_j)_t + alpha_j div(u_t) - div K_j grad p_j + sum_{i} S_{ji} (p_j - p_i) = g_j   

    holds for t in (0, T) and for all x in the domain Omega, where
    
      sigma(u) = 2*mu*eps(u) + lmbda div(u) I 

    and eps(u) = sym(grad(u)), and mu and lmbda are the standard Lame
    parameters. For each network j, c_j is the specific storage
    coefficient, alpha_j is the Biot-Willis coefficient, and K_j is
    the hydraulic conductivity.

    f is a given body force and g_j is a source in network j.

    See e.g. Tully and Ventikos, Cerebral water transport using
    multiple-network poroelastic theory: application to normal
    pressure hydrocephalus, 2011 for an introduction to the
    multiple-network poroelasticity equations.

    Boundary conditions:

    We assume that there are (possibly up to J+1) facet functions
    marking the different subdomains of the boundary. We assume that

    * Dirichlet conditions are marked by 0, 
    * Neumann conditions are marked by 1 and 
    * Robin conditions are marked by 2.

    For the momentum equation (1):
    
    We assume that each part of the boundary of the domain is one of
    the following two types:

    *Dirichlet*: 

      u(., t) = \bar u(t) 

    *Neumann*:

      (sigma(u) - sum_{j} alpha_j p_j I) * n = s

    For the continuity equations (2):

    We assume that each part of the boundary of the domain is one of
    the following three types:

    *Dirichet*

      p_j(., t) = \bar p_j(t) 
      
    *Neumann*

      K grad p_j(., t) * n = I_j(t)

    *Robin* 
       
      K grad p_j(., t) * n = \beta_j (p_j - p_j_robin)
    
    Initial conditions:

      u(x, t_0) = u_0(x)

      p_j(x, t_0) = p0_j(x) if c_j > 0
    """

    def __init__(self, mesh, time, params=None):
        """Initialize problem instance.

        Params will be taken from default_params and overridden
        by the values given to this constructor.

        This problem class assumes that the following attributes are
        set:

        self.f : body force for the momentum equation (vector field)
        self.g : list of body sources for the continuity equations

        self.s : boundary force for the momentum equation (vector field)
        self.I : list of boundary fluxes for the continuity equations

        self.u_bar : essential boundary condition for u
        self.p_bar : list of essential boundary conditions for p = (p_1, ..., p_A)

        self.displacement_nullspace : boolean (True/False) if rigid motions should be eliminated
        self.pressure_nullspace : list of boolean (True/False) if constants should be eliminated
        """
        self.mesh = mesh
        # NB: Assumption that everything that depends on time does so
        # through this Constant time
        self.time = time

        # Update problem parameters
        self.params = self.default_parameters()
        if params is not None:
            self.params.update(params)

        As = range(self.params["A"])

        gdim = self.mesh.geometry().dim()
        
        # Default values for forces and sources
        self.f = Constant((0.0,)*gdim) 
        self.s = Constant((0.0,)*gdim)
        self.g = [Constant(0.0) for i in As]
        self.I = [Constant(0.0) for i in As]
        self.beta = [Constant(0.0) for i in As]
        self.p_robin = [Constant(0.0) for i in As]
        # Default values for Dirichlet boundary conditions
        self.u_bar = Constant((0.0,)*gdim)
        self.p_bar = [Constant(0.0) for i in As]

        # Default markers, one for the momentum equation and one for
        # each network continuity equation

        # Assumption: Neumann condition is marked by 1
        # Assumption: Robin conditions is maked by 2

        INVALID = 7101982
        tdim = mesh.topology().dim() 
        markers = MeshFunction("size_t", mesh, tdim-1)
        markers.set_all(INVALID)
        self.momentum_boundary_markers = markers

        self.continuity_boundary_markers = []
        for i in As:
            markers = MeshFunction("size_t", mesh, tdim-1)
            markers.set_all(INVALID)
            self.continuity_boundary_markers += [markers]

        self.displacement_nullspace = False
        self.pressure_nullspace = list(False for i in As)
            
    @classmethod
    def default_parameters(cls):
        "Define the set of parameters to define the problem."
        ps = {"A": 1.0,
              "alpha": (1.0,),
              "rho": 1.0,
              "nu": 0.479,
              "E": 584e-3,
              "K": (1.0, ),
              "G": ((1.0,),),
              "c": 0.0,
              }
        return ps
        
        
