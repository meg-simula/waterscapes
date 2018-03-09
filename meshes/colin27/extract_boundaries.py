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
        error("Soundn't be here!")

file = File("colin27_boundaries.pvd")
file << boundaries

        
