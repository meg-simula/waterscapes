"""Generate a 2D mesh of an idealized brain with a circular outer surface,
a circular ventricular surface and a channel from the inner surface
to the outer surface representing the aqueduct. This mesh is generated within
a larger square mesh in such a way that the two meshes are nested. This
is useful for Multilevel Monte Carlo.

This is the 2D version of the sphere brain mesh generated in generate_mesh.py

"""

# Marie E. Rognes, June 17 2017
# Matteo Croci,    June 19 2017

from dolfin import *
from mshr import *

def generate_brain_mesh():

    # Get as much output as possible by setting debug type log level.
    set_log_level(DEBUG)

    origin = Point(0.0, 0.0)
    r1 = 10.0 # Outer radius (cm)
    r2 = 3.0 # Inner radius  (cm)
    r3 = 0.5 # Aqueduct radius
    cistern1 = Point(origin.x() + r3, origin.y() - r1)
    cistern2 = Point(origin.x() - r3, origin.y())

    
    parenchyma = Circle(origin, r1)
    ventricles = Circle(origin, r2)
    aqueduct   = Rectangle(cistern1, cistern2)
    
    geometry  = parenchyma - ventricles - aqueduct
    dolfin.info(geometry, True)

    lmbda = 0.3
    outer_coor1 = -r1*(1.+lmbda)
    outer_coor2 =  r1*(1.+lmbda)
    p1 = Point(outer_coor1, outer_coor1)
    p2 = Point(outer_coor2, outer_coor2)
    rectangle = Rectangle(p1,p2) 
    dolfin.info(rectangle, True)

    rectangle.set_subdomain(1, geometry)
    
    # Create mesh, N controls the resolution (N higher -> more cells)
    N = 25
    outer_mesh = generate_mesh(rectangle, N)
    brain_geom_markers = MeshFunction("size_t", outer_mesh, 2, outer_mesh.domains())
    inner_mesh = SubMesh(outer_mesh, brain_geom_markers, 1)
    
    plot(outer_mesh, title="outer mesh")
    plot(inner_mesh, title="inner mesh")
    
    # Store mesh to h5 (good for further FEniCS input/output)
    file = HDF5File(mpi_comm_world(), "2D_circle_brain.h5", "w")
    file.write(inner_mesh, "/inner_mesh")
    file.write(outer_mesh, "/outer_mesh")
    file.close()

    # Store mesh to pvd/vtu (good for Paraview)
    file = File("inner_mesh.pvd")
    file << inner_mesh

    file = File("outer_mesh.pvd")
    file << outer_mesh

    return (inner_mesh, outer_mesh)
    
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
    inner_mesh = Mesh()
    outer_mesh = Mesh()
    file = HDF5File(mpi_comm_world(), "2D_circle_brain.h5", "r")
    file.read(inner_mesh, "/inner_mesh", False)
    file.read(outer_mesh, "/outer_mesh", False)
    file.close()
    
    inner_u = solve_poisson(inner_mesh)
    file = File("inner_u.pvd")
    file << inner_u
    plot(inner_u, title="solution")

    outer_u = solve_poisson(outer_mesh)
    file = File("outer_u.pvd")
    file << outer_u
    plot(outer_u, title="solution")

    interactive()
