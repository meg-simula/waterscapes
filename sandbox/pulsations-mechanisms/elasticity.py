from dolfin import *
from mshr import *

class ElasticityModel(object):
    def __init__(self, mesh, time, f, g, params=None):
        if params:
            self.params = params
        else:
            self.params = self.default_parameters()
            
        self.mesh = mesh
        self.time = time
        self.f = f or Constant((0.0,)*self.mesh.topology().dim()) # Body force
        self.g = g or Constant((0.0,)*self.mesh.topology().dim()) # Traction
        
    def default_parameters(self):

        params = Parameters("elasticity_model")
        params.add("mu", 100.0)
        params.add("lmbda", 10000.0)
        
        return params
        
class DynamicElasticitySolver(object):
    """
    u_tt - div(sigma(u)) = f

    Weak formulation:

    <u_tt, v> + <sigma(u), eps(v)> - <sigma(u) * n, v>_{dO} = <f, v>

    Temporal discretization

    1/(dt^2) <u^+ - 2 u + u^-, v> + <sigma(u), eps(v)> - sigma(u)*n, v>_{dO} 
    = <f, v>

    """

    def __init__(self, model, params=None):
        self.model = model

        mesh = self.model.mesh
        V = VectorFunctionSpace(mesh, "CG", 1)
        self.solution = Function(V) # Current solution
        self.u_ = Function(V)       # Previous solution
        self.u__ = Function(V)      # Previous previous solution
        
    def sigma(self, u):

        mu = self.model.params["mu"]
        lmbda = self.model.params["lmbda"]

        I = Identity(self.model.mesh.topology().dim())
        return 2*mu*sym(grad(u)) + lmbda*div(u)*I
            
    def solve(self, T, dt):

        if not isinstance(dt, Constant):
            dt = Constant(dt)

        mesh = self.model.mesh
        V = self.solution.function_space()

        u = TrialFunction(V)
        v = TestFunction(V)
        u_ = self.u_   
        u__ = self.u__ 

        f = self.model.f
        g = self.model.g
        
        F = inner(1/(dt**2)*(u - 2*u_ + u__), v)*dx() + inner(self.sigma(u_), sym(grad(v)))*dx() - inner(g, v)*ds() - inner(f, v)*dx()
        (a, L) = system(F)
        
        t = self.model.time 

        bdry = CompiledSubDomain("(sqrt(x[0]*x[0] + x[1]*x[1]) < 50) && on_boundary")
        bc = DirichletBC(V, (0.0, 0.0), bdry)
        while (float(t) <= (T + DOLFIN_EPS)):

            # Update time
            t.assign(float(t) + float(dt))

            # Solve system
            solve(a == L, self.solution, bc)

            yield self.solution

            # Update
            self.u__.assign(self.u_)
            self.u_.assign(self.solution)

    
def create_annulus():

    # N gives the number of points on the outer boundary
    N = 50
    
    outer = Circle(Point(0.0, 0.0), 100., N)
    inner = Circle(Point(0.0, 0.0), 30., N)
    annulus = outer - inner

    info("Generating mesh")
    M = 20
    mesh = generate_mesh(annulus, M)

    plot(mesh, interactive=True)

    info("Storing mesh to 'annulus.h5'")
    hdf = HDF5File(mpi_comm_world(), "annulus.h5", "w")
    hdf.write(mesh, "/mesh")
    hdf.close()

    return mesh
    
def solve_dynamic_elasticity():

    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), "annulus.h5", "r")
    hdf.read(mesh, "/mesh", False)

    mesh = refine(mesh)
    
    time = Constant(0.0)

    n = FacetNormal(mesh)
    x = SpatialCoordinate(mesh)
    p0 = conditional(((x[0]*x[0] + x[1]*x[1]) > 50*50), - 0.1*sin(2*DOLFIN_PI*time), 0.0)
    g = p0*n
    model = ElasticityModel(mesh, time, None, g)

    solver = DynamicElasticitySolver(model)

    file = File("output/u.pvd")
    solutions = solver.solve(3.0, 0.005)

    points = [(0.0, 50.), (0.0, 90.0), (0.0, 70.0)]
    
    values = []
    for u in solutions:
        values += [[u(p) for p in points]]
        #file << u
        plot(u, key="u", mode="displacement")
        
    import pylab
    for j in (0, 1, 2):
        pylab.plot([values[i][j][1] for i in range(len(values))])
    pylab.grid(True)
    pylab.show()
    
    interactive()
        
if __name__ == "__main__":

    #create_annulus()
    solve_dynamic_elasticity()
    
    
    
    
