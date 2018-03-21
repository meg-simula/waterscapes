# Script to extract boundary markers from the colin27 white-gray
# mesh. Boundary facets bordering on gray matter marked cells is
# defined as "Skull", boundary facets bordering on white matter is
# defined as "Ventricles".

# NB: Runs with Python 2 and FEniCS 2017.X, not with FEniCS 2018.1

from dolfin import *

file = HDF5File(mpi_comm_world(), "colin27_whitegray.h5", "r")
mesh = Mesh()
file.read(mesh, "/mesh", True)

mesh.init()

D = mesh.topology().dim()
markers = MeshFunction("size_t", mesh, D)
file.read(markers, "/markers")

#file = File("markers.pvd")
#file << markers

bdry = BoundaryMesh(mesh, "exterior")
a = bdry.entity_map(bdry.topology().dim())
print a

gray_facets = []
white_facets = []

boundaries = MeshFunction("size_t", mesh, D-1)
boundaries.set_all(0)

# skull == 1
# ventricles == 2
SKULL = 1
VENTRICLES = 2

for facet in cells(bdry):
    parent_facet_index = a[facet]

    # Get corresponding facet in parent mesh
    parent_facet = Facet(mesh, parent_facet_index)

    # Get neighbouring cell index
    cell_indices = parent_facet.entities(D)
    assert len(cell_indices) == 1, "Expect only 1 neighbour since boundary"
    i = cell_indices[0]

    # Extract marker for this cell
    marker = markers.array()[i]

    # marker == 3 means gray matter
    # marker == 4 means white matter
    if marker == 3:
        boundaries[parent_facet.index()] = SKULL
    elif marker == 4:
        boundaries[parent_facet.index()] = VENTRICLES
    else:
        error("Shoundn't be here!")

file = File("colin27_boundaries.pvd")
file << boundaries

file = HDF5File(mpi_comm_world(), "colin27_whitegray_copy.h5", "w")

file.write(mesh, "/mesh")
file.write(markers, "/markers")
file.write(boundaries, "/boundaries")
file.close()

        
