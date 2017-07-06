"Generate mesh of an idealized 2D brain aka a donut."

# Eleonora Piersanti and Marie E. Rognes, June 26 2017

from dolfin import *
from mshr import *

def generate_brain_mesh():

    # Get as much output as possible by setting debug type log level.
    set_log_level(DEBUG)

    origin = Point(0.0, 0.0)
    r1 = 100.0 # Outer radius (mm)
    r2 = 30.0 # Inner radius  (mm)
    
    parenchyma = Circle(origin, r1)
    ventricles = Circle(origin, r2)
    
    geometry = parenchyma - ventricles
    dolfin.info(geometry, True)
    
    # Create mesh, N controls the resolution (N higher -> more cells)
    N = 20
    mesh = generate_mesh(geometry, N)
    
    plot(mesh, title="mesh")
    
    # Store mesh to h5 (good for further FEniCS input/output)
    file = HDF5File(mpi_comm_world(), "donut2D.h5", "w")
    file.write(mesh, "/mesh")
    file.close()

    # Store mesh to pvd/vtu (good for Paraview)
    file = File("donut2D.pvd")
    file << mesh

    return mesh
    
def solve_poisson(mesh):
    "Solve basic Poisson problem on given mesh"
    
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    f = Expression("pow(x[0] - 0.5, 2)", degree=2)
    L = f*v*dx()

    bc = DirichletBC(V, 0.0, "on_boundary")

    u = Function(V)
    solve(a == L, u, bc)
    return u
    

if __name__ == "__main__":

    mesh = generate_brain_mesh()

    # Test that mesh can be read back in and solved on
    mesh = Mesh()
    file = HDF5File(mpi_comm_world(), "donut2D.h5", "r")
    file.read(mesh, "/mesh", False)
    file.close()
    
    u = solve_poisson(mesh)
    file = File("u.pvd")
    file << u
    plot(u, title="solution")

    interactive()
