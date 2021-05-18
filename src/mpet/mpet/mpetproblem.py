__author__ = "Eleonora Piersanti <eleonora@simula.no>"

# Modified by Marie E. Rognes <meg@simula.no>, 2017

import sys
from dolfin import *

def convert_to_E_nu(mu, lmbda):
    E = mu*(3*lmbda+2*mu)/(lmbda + mu)
    nu = lmbda/(2*(lmbda + mu))
    return (E, nu)

def convert_to_mu_lmbda(E, nu):
    mu = E/(2.0*((1.0 + nu)))
    lmbda = nu*E/((1.0-2.0*nu)*(1.0+nu))
    return (mu, lmbda)
    
def elastic_stress(u, E, nu):
    "Define the standard linear elastic constitutive equation."
    d = u.geometric_dimension()
    I = Identity(d)
    (mu, lmbda) = convert_to_mu_lmbda(E, nu)
    s = 2*mu*sym(grad(u)) + lmbda*div(u)*I
    return s

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
    parameters. 

    # FIXME: sigma expressed in terms of E, nu in code, update
    # description here.

    For each network j, c_j is the specific storage
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

        mesh should be the FEniCS Mesh, time should be a Constant
        representing time. Assumed that all functions depending on
        time do so via this Constant.

        This problem class assumes that the following attributes are
        set:

        self.f: body force for the momentum equation (vector field)
        self.g: list of body sources for the continuity equations

        self.s: boundary force for the momentum equation (vector field)
        self.I: list of boundary fluxes for the continuity equations

        self.u_bar: essential boundary condition for u
        self.p_bar: list of essential boundary conditions for p = (p_1, ..., p_A)

        self.displacement_nullspace: boolean (True/False) if rigid motions should be eliminated
        self.pressure_nullspace: list of boolean (True/False) if constants should be eliminated

        """
        # Set mesh and time
        self.mesh = mesh
        self.time = time

        # Update problem parameters
        self.params = self.default_parameters()
        if params is not None:
            self.params.update(params)

        Js = range(self.params["J"])
        gdim = self.mesh.geometry().dim()
        
        # Set default values for forces and sources
        self.f = Constant((0.0,)*gdim) 
        self.s = Constant((0.0,)*gdim)
        self.g = [Constant(0.0) for i in Js]
        self.I = [Constant(0.0) for i in Js]
        self.beta = [Constant(0.0) for i in Js]
        self.p_robin = [Constant(0.0) for i in Js]
        self.u_bar = Constant((0.0,)*gdim)
        self.p_bar = [Constant(0.0) for i in Js]

        # Default markers, one for the momentum equation and one for
        # each network continuity equation
        INVALID = sys.maxsize
        tdim = mesh.topology().dim() 
        markers = MeshFunction("size_t", mesh, tdim-1)
        markers.set_all(INVALID)
        self.momentum_boundary_markers = markers

        self.continuity_boundary_markers = []
        for i in Js:
            markers = MeshFunction("size_t", mesh, tdim-1)
            markers.set_all(INVALID)
            self.continuity_boundary_markers += [markers]

        self.u_has_nullspace = False
        self.p_has_nullspace = list(False for i in Js)
            
    @classmethod
    def default_parameters(cls):
        "Define the set of parameters to define the problem."
        ps = {"J": 1.0,
              "rho": Constant(1.0),
              "nu": Constant(0.479),
              "E": Constant(1500),
              "alpha": (Constant(1.0),),
              "K": (Constant(1.0), ),
              "X": ((Constant(1.0),),),
              "c": (Constant(1.0),)
        }
        return ps
        
        
