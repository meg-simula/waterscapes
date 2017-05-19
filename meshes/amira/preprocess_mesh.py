import scipy.io
from dolfin import *
import numpy as np

def create_submesh_markers(markers, submesh, name):

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
        #        common = set(vert).intersection(snp.savetxt('test.out', x, fmt='%1.4e')et(vert_mapped))
        #        if len(common) == 3 :
        #            markers_submesh.array()[facet_sub] = markers.array()[facet]
        #            break
    np.save(name + "_cellmap", cell_map)


    return submesh_markers

def split_meshes():

    mesh = Mesh("bi18.xml")
    markers = MeshFunction("size_t", mesh, "bi18_sub.xml")
    #plot(markers, title="Cell markers")

    # Create mesh of the white and gray matter
    RGM = 22 
    LGM = 23
    RWM = 9
    LWM = 10 
    tissue = CellFunction("size_t", mesh, 0)
    tissue.array()[markers.array()==RGM] = 1
    tissue.array()[markers.array()==LGM] = 1
    tissue.array()[markers.array()==RWM] = 1
    tissue.array()[markers.array()==LWM] = 1
    whitegray = SubMesh(mesh, tissue, 1)

    whitetissue = CellFunction("size_t", mesh, 0)
    whitetissue.array()[markers.array()==RWM] = 1
    whitetissue.array()[markers.array()==LWM] = 1    

    white = SubMesh(mesh, whitetissue, 1)
  
    graytissue = CellFunction("size_t", mesh, 0)
    graytissue.array()[markers.array()==LGM] = 1
    graytissue.array()[markers.array()==RGM] = 1     

    gray = SubMesh(mesh, graytissue, 1)
    
    #white = SubMesh(mesh, markers, WHITE)
    #gray = SubMesh(mesh, markers, GRAY)

    # Map supermesh markers to submesh
    whitegray_markers = create_submesh_markers(markers, whitegray, "whitegray")
    white_markers = create_submesh_markers(markers, white, "white")
    gray_markers = create_submesh_markers(markers, gray, "gray")

    # Store submesh
    file = File("bi18_whitegray_mesh.xml.gz")
    file << whitegray
    file = File("bi18_white_mesh.xml.gz")
    file << white
    file = File("bi18_gray_mesh.xml.gz")
    file << gray
    #file = File("adult_mni_152_model_whitegray_mesh.xdmf")
    #file << whitegray

    # Store submarkers
    file = File("bi18_whitegray_markers.xml.gz")
    file << whitegray_markers
    file = File("bi18_white_markers.xml.gz")
    file << white_markers
    file = File("bi18_gray_markers.xml.gz")
    file << gray_markers
    #file = File("adult_mni_152_model_whitegray_markers.xdmf")
    #ile << whitegray_markers

    #plot(white, title="White matter")
    #plot(gray, title="Gray matter")
    #plot(whitegray, title="Mesh")
    #plot(whitegray_markers, title="Gray and white matter")
    #interactive()
def write_fibers():
    
    mesh = Mesh("bi18_whitegray_mesh.xml.gz")
    DG = VectorFunctionSpace(mesh, "DG", 0)
    fibers = Function(DG)
    cellmap = np.load("whitegray_cellmap.npy")
    lw_tet = np.loadtxt("bi18_labels7_all_fit_lwm_tetra.info")
    rw_tet = np.loadtxt("bi18_labels7_all_fit_rwm_tetra.info")
    lg_tet = np.loadtxt("bi18_labels7_all_fit_lgm_tetra.info")
    rg_tet = np.loadtxt("bi18_labels7_all_fit_rgm_tetra.info")

    rw_eigfile = open("bi18_rwm_eigvect_230000.asc", "r")
    lw_eigfile = open("bi18_lwm_eigvect_230000.asc", "r")
    lg_eigfile = open("bi18_lgm_eigvect_230000.asc", "r")
    rg_eigfile = open("bi18_rgm_eigvect_230000.asc", "r") 
    
    lw_eigvect = [[line.strip().split()[i] for i in range(3)] for line in lw_eigfile.readlines()[:]]
    rw_eigvect = [[line.strip().split()[i] for i in range(3)] for line in rw_eigfile.readlines()[:]]
    lg_eigvect = [[line.strip().split()[i] for i in range(3)] for line in lg_eigfile.readlines()[:]]
    rg_eigvect = [[line.strip().split()[i] for i in range(3)] for line in rg_eigfile.readlines()[:]]

    #print fibers.vector()[0].x[0]
    # print len(lw_tet)
    for i in range(len(lw_tet)):
            for j in range(3): 
                fibers.vector()[int(np.where(cellmap==(lw_tet[i]-1))[0]*3 + j)] = float(lw_eigvect[i][j])
    for i in range(len(rw_tet)):
            for j in range(3): 
                fibers.vector()[int(np.where(cellmap==(rw_tet[i]-1))[0]*3 + j)] = float(rw_eigvect[i][j])

    for i in range(len(lg_tet)):
            for j in range(3): 
                fibers.vector()[int(np.where(cellmap==(lg_tet[i]-1))[0]*3 + j)] = float(lg_eigvect[i][j])

    for i in range(len(rg_tet)):
            for j in range(3): 
                fibers.vector()[int(np.where(cellmap==(rg_tet[i]-1))[0]*3 + j)] = float(rg_eigvect[i][j])

    File("fibers.xml.gz")<<fibers
    #         pass
    #for cell in cells(mesh)
    

if __name__ == "__main__":

    split_meshes()
    write_fibers()