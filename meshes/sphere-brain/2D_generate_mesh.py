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

    r1 = 10.0 # Outer radius (cm)
    r2 = 3.0 # Inner radius  (cm)
    r3 = 0.5 # Aqueduct radius
    r4 = (r1 - r2)*0.8 + r2 # white matter radius
    r5 = (r1 - r2)*0.2 # distance between aqueduct and grey matter

    origin = Point(0.0, 0.0)
    cistern1 = Point(origin.x() + r3, origin.y() - r1)
    cistern2 = Point(origin.x() - r3, origin.y())

    parenchyma   = Circle(origin, r1)
    ventricles   = Circle(origin, r2)
    aqueduct     = Rectangle(cistern1, cistern2)

    # Define white matter regions
    white = Circle(origin, r4)
    cistern3 = Point(origin.x() + r3 + r5, origin.y() - r1)
    cistern4 = Point(origin.x() - r3 - r5, origin.y())
    aqueduct_white = Rectangle(cistern3, cistern4) # white matter around the aqueduct

    # white and grey matter geometries
    geometry_grey = parenchyma - white - aqueduct_white
    dolfin.info(geometry_grey, True)
    geometry_white = parenchyma - geometry_grey - ventricles - aqueduct
    dolfin.info(geometry_white, True)

    # outer box mesh incapsulating the brain mesh
    lmbda = 0.3
    outer_coor1 = -r1*(1.+lmbda)
    outer_coor2 =  r1*(1.+lmbda)
    p1 = Point(outer_coor1, outer_coor1)
    p2 = Point(outer_coor2, outer_coor2)
    rectangle = Rectangle(p1,p2) 
    dolfin.info(rectangle, True)

    # set subdomains for mshr so that the meshes are nested
    rectangle.set_subdomain(1, geometry_grey)
    rectangle.set_subdomain(2, geometry_white)

    # Create mesh, N controls the resolution (N higher -> more cells)
    N = 25
    outer_mesh = generate_mesh(rectangle, N)

    # extract the outer mesh subdomains so that we can extract the inner brain mesh
    # and we can define the domain marker IDs for white and grey matter
    brain_geom_markers = MeshFunction("size_t", outer_mesh, 2, outer_mesh.domains())
    markers = brain_geom_markers.array().copy()
    markers[markers == 2] = 0 # 0 - white matter, 1 - grey matter

    # set markers for inner brain mesh extraction
    brain_geom_markers.array()[brain_geom_markers.array() == 2] = 1
    # extract inner brain mesh
    inner_mesh = SubMesh(outer_mesh, brain_geom_markers, 1)

    # Create inner_mesh markers from outer_mesh
    whitegrey_markers = MeshFunction("size_t", inner_mesh, 2)
    cell_map = inner_mesh.data().array("parent_cell_indices", 2)
    for cell in cells(inner_mesh):
        i = cell.index()
        j = int(cell_map[i])
        whitegrey_markers.array()[i] = markers[j]

    # save markers to pvd and xml
    File("markers.pvd") << whitegrey_markers
    File("markers.xml") << whitegrey_markers

    # create boundary IDs
    boundary_markers = FacetFunction("size_t", inner_mesh, value = 0)
    eps = 1.0e-3 #NOTE: this tolerance is really high, but otherwise it cannot find the boundaries, odd.
    # ventricle boundary
    ventricle_bnd = CompiledSubDomain("(sqrt(pow(x[0],2) + pow(x[1],2)) < r + eps) && (sqrt(pow(x[0],2) + pow(x[1],2)) > r - eps) && on_boundary", r = r2, eps = eps)
    # subaracnoidal space boundary
    SAS_bnd       = CompiledSubDomain("(sqrt(pow(x[0],2) + pow(x[1],2)) < r + eps) && (sqrt(pow(x[0],2) + pow(x[1],2)) > r - eps) && on_boundary", r = r1, eps = eps)
    # aqueduct boundary
    aqueduct_bnd  = CompiledSubDomain("((x[0] <= xleft + eps) && (x[0] >= xleft - eps)) || ((x[0] <= xright + eps) && (x[0] >= xright - eps)) && (x[1] <= 0.0) && (x[1] >= ybottom) && on_boundary", xleft = -r3, xright = r3, ybottom = -r1, eps = eps)

    ventricle_bnd.mark(boundary_markers, 1)
    SAS_bnd.mark(boundary_markers, 2)
    aqueduct_bnd.mark(boundary_markers, 3)

    # save boundary IDs to pvd and xml
    File("boundary_markers.pvd") << boundary_markers
    File("boundary_markers.xml") << boundary_markers

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
    
def solve_poisson(mesh, markers, boundary_markers):
    """Solve basic Poisson problem on inner brain mesh with piecewise
       constant diffusion coefficient (1 on the white matter and 2 on
       the grey matter
    """
    
    DG = FunctionSpace(mesh, "DG", 0)
    K = Function(DG)
    K.vector()[:] = markers.array()
    File("markers_fun.pvd") << K

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    

    a = inner((Constant(1.0) + K)*grad(u), grad(v))*dx
    f = Expression("pow(x[0] - 0.5, 2)", degree=2)
    L = f*v*dx()

    bc = [DirichletBC(V, 0.0, boundary_markers, 1), DirichletBC(V, Constant(200.), boundary_markers, 2)]

    u = Function(V)
    solve(a == L, u, bc)
    return u
    

if __name__ == "__main__":

    inner_mesh, outer_mesh = generate_brain_mesh()

    # Test that mesh can be read back in and solved on
    inner_mesh = Mesh()
    outer_mesh = Mesh()
    file = HDF5File(mpi_comm_world(), "2D_circle_brain.h5", "r")
    file.read(inner_mesh, "/inner_mesh", False)
    file.read(outer_mesh, "/outer_mesh", False)
    file.close()

    # test only on the inner brain mesh
    markers = MeshFunction("size_t", inner_mesh, "markers.xml")
    boundary_markers = MeshFunction("size_t",  inner_mesh, "boundary_markers.xml")
    u = solve_poisson(inner_mesh, markers, boundary_markers)
    file = File("u.pvd")
    file << u
    plot(u, title="solution")

    interactive()
