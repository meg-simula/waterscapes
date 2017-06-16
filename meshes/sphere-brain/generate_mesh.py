"""Generate mesh of an idealized brain with a spherical outer surface,
a spherical ventricular surface and a channel from the inner surface
to the outer surface representing the aqueduct.

"""

# Marie E. Rognes, June 17 2017

from dolfin import *
from mshr import *

def generate_brain_mesh():

    # Get as much output as possible by setting debug type log level.
    set_log_level(DEBUG)

    origin = Point(0.0, 0.0, 0.0)
    r1 = 10.0 # Outer radius (cm)
    r2 = 3.0 # Inner radius  (cm)
    r3 = 0.5 # Aqueduct radius
    cistern = Point(origin.x(), origin.y(), origin.z() - r1)
    
    parenchyma = Sphere(origin, r1)
    ventricles = Sphere(origin, r2)
    aqueduct = Cylinder(origin, cistern, r3, r3)
    
    geometry = parenchyma - ventricles - aqueduct
    dolfin.info(geometry, True)
    
    # Create mesh, N controls the resolution (N higher -> more cells)
    N = 20
    mesh = generate_mesh(geometry, N)
    
    plot(mesh, title="mesh")
    
    # Store mesh to h5 (good for further FEniCS input/output)
    file = HDF5File(mpi_comm_world(), "sphere_brain.h5", "w")
    file.write(mesh, "/mesh")
    file.close()

    # Store mesh to pvd/vtu (good for Paraview)
    file = File("mesh.pvd")
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
    file = HDF5File(mpi_comm_world(), "sphere_brain.h5", "r")
    file.read(mesh, "/mesh", False)
    file.close()
    
    u = solve_poisson(mesh)
    file = File("u.pvd")
    file << u
    plot(u, title="solution")

    interactive()
