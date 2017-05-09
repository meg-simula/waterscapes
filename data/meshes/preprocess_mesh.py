# This module provides utilities for converting matlab style meshes
# with subdomain markers to FEniCS format. Customized for the colin27
# and adult_mni_152 brain meshes.

import scipy.io
from dolfin import *

def create_submesh_markers(markers, submesh):

    # creates the submesh markers
    submesh_markers = MeshFunction("size_t", submesh, markers.dim())

    # submesh to mesh mappings
    cell_map = submesh.data().array("parent_cell_indices", 3)
    #vertex_map = submesh.data().mesh_function("parent_vertex_indices")

    # iterate over the cells
    for cell in cells(submesh):
        i = cell.index()
        j = int(cell_map[i])
        submesh_markers.array()[i] = markers[j]

        # extracts the facets and the corresponding one of the full mesh
        #facets_sub = cell.entities(2)
        #facets = Cell(mesh, cell_map.array()[cell.index()]).entities(2)

        #for facet_sub in facets_sub :
        #    # mapped vertices
        #    vert_mapped = pfun.array()[Facet(submesh, facet_sub).entities(0)]
        #    # find the corresponding facet on the full mesh
        #    for facet in facets :
        #        # vertices
        #        vert = Facet(mesh, facet).entities(0)
        #        # intersection
        #        common = set(vert).intersection(set(vert_mapped))
        #        if len(common) == 3 :
        #            markers_submesh.array()[facet_sub] = markers.array()[facet]
        #            break

    return submesh_markers

def split_meshes(prefix):

    #mesh = Mesh()
    #hdf = HDF5File(mpi_comm_world(), "%s/%s.h5" % (prefix, prefix), "r")
    #hdf.read(mesh, "/mesh", False)
    #markers = CellFunction("size_t", mesh)
    #hdf.read(markers, "/markers")
    #hdf.close()
    #plot(markers, title="Cell markers")

    begin("Separating out white and gray matter")
    mesh = Mesh("%s/%s_mesh.xml" % (prefix, prefix))
    markers = MeshFunction("size_t", mesh, "%s/%s_markers.xml" % (prefix, prefix))
    if prefix == "colin27":
        GRAY = 3 
        WHITE = 4
    elif prefix == "adultmni152":
        GRAY = 4
        WHITE = 5
    
    # Create mesh of the white and gray matter
    tissue = CellFunction("size_t", mesh, 0)
    tissue.array()[markers.array()==GRAY] = 1
    tissue.array()[markers.array()==WHITE] = 1
    whitegray = SubMesh(mesh, tissue, 1)

    # Map supermesh markers to submesh
    whitegray_markers = create_submesh_markers(markers, whitegray)

    # Store submesh and markers
    file = HDF5File(mpi_comm_world(), "%s/%s_whitegray.h5" % (prefix, prefix), "w")
    file.write(whitegray, "/mesh")
    file.write(whitegray_markers, "/markers")
    file.close()

    #plot(white, title="White matter")
    #plot(gray, title="Gray matter")
    plot(whitegray, title="Mesh")
    plot(whitegray_markers, title="Gray and white matter")
    end()
    interactive()
    
    
def convert_mat_to_xml(prefix, gdim=3, celltype="tetrahedron"):

    # Load matlab data from Colin27 or Adult MNI mesh
    if prefix == "colin27":
        mat_contents = scipy.io.loadmat('%s/MMC_Collins_Atlas_Mesh_Version_2L.mat' % prefix)
        # Extract into separate arrays
        node = mat_contents["node"]
        elem = mat_contents["elem"]
        face = mat_contents["face"]

    elif prefix == "adultmni152":
        mat_contents = scipy.io.loadmat('%s/HeadVolumeMesh.mat' % prefix)
        node =  mat_contents["HeadVolumeMesh"]["node"][0][0]
        elem =  mat_contents["HeadVolumeMesh"]["elem"][0][0]
        face =  mat_contents["HeadVolumeMesh"]["face"][0][0]
    else:
        error("Unknown mesh data file: %s" % prefix)
        
    num_vertices = node.size
    print("Mesh has %d vertices" % num_vertices)
    num_cells = elem.size
    print("Mesh has %d cells" % num_cells)

    # -- Write mesh

    # NB note the -1 offset here (matlab starts counting at 1, python
    # starts counting at 0)
    cells = "\n".join(["""      <%s index="%d" v0="%d" v1="%d" v2="%d" v3="%d"/>""" % (celltype, i, c[0]-1, c[1]-1, c[2]-1, c[3]-1)
                          for (i, c) in enumerate(elem)])
    vertices = "\n".join(["""      <vertex index="%d" x="%f" y="%f" z="%f"/>""" % (i, v[0], v[1], v[2])
                          for (i, v) in enumerate(node)])

    contents = """
<?xml version="1.0" encoding="UTF-8"?>

<dolfin xmlns:dolfin="http://fenicsproject.org">
  <mesh celltype="%s" dim="%d">
    <vertices size="%d">
    %s
    </vertices>

    <cells size="%d">
    %s
    </cells>
  </mesh>
</dolfin>
    """ % (celltype, gdim, num_vertices, vertices, num_cells, cells)

    # -- Domain markers
    N = gdim+1 # Which row contains the markers?
    markers = "\n".join(["""      <value cell_index="%d" local_entity="0" value="%d" />""" % (i, c[N])
                         for (i, c) in enumerate(elem)])

    markers_code = """
<?xml version="1.0"?>

<dolfin xmlns:dolfin="http://fenicsproject.org">
  <mesh_function>
    <mesh_value_collection name="m" type="uint" dim="%d" size="%d">
    %s
    </mesh_value_collection>
  </mesh_function>
</dolfin>
    """ % (gdim, num_cells, markers)

    filename1 = "%s/%s_mesh.xml"  % (prefix, prefix)
    print "Writing data to %s" % filename1
    file = open(filename1, 'w')
    file.write(contents)
    file.close()

    filename2 = "%s/%s_markers.xml"  % (prefix, prefix)
    print "Writing data to %s" % filename2
    file = open(filename2, 'w')
    file.write(markers_code)
    file.close()

    print "Success!"

    return filename1, filename2

def convert_mesh(prefix):

    print("Converting %s mesh..." % prefix)
    f1, f2 = convert_mat_to_xml(prefix)

    # Read mesh back in
    mesh = Mesh(f1)
    
    # Read markers back in
    markers = MeshFunction("size_t", mesh, f2)
 
    # Store as h5 as well
    print("Storing h5")
    file = HDF5File(mpi_comm_world(), "%s/%s.h5" % (prefix, prefix), 'w')
    file.write(mesh, "/mesh")
    file.write(markers, "/markers")
    file.close()

def generate_colin_mesh(prefix):

    print("Converting %s mesh..." % prefix)
    f1, f2 = convert_mat_to_xml(prefix)

    # Read mesh back in
    mesh = Mesh(f1)
    
    # Read markers back in
    markers = MeshFunction("size_t", mesh, f2)
 
    # Store as h5 as well
    print("Storing h5")
    hdf = HDF5File(mpi_comm_world(), "%s/%s.h5" % (prefix, prefix), 'w')
    hdf.write(mesh, "/mesh")
    hdf.write(markers, "/markers")
    hdf.flush()
    hdf.close()
    
if __name__ == "__main__":
    
    #convert_mesh("colin27")
    #split_meshes("colin27")

    convert_mesh("adultmni152")
    split_meshes("adultmni152")
