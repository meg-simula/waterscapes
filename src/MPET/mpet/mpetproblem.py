__author__ = "Eleonora Piersanti <eleonora@simula.no>"

# Modified by Marie E. Rognes <meg@simula.no>, 2017

from dolfin import *

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
        
        
