__author__ = "Eleonora Piersanti <eleonora@simula.no>"

# Modified by Marie E. Rognes <meg@simula.no>, 2017

from dolfin import *

class MPETProblem(object):

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
        
        
