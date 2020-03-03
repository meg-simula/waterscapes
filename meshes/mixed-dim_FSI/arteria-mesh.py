from pygmsh import generate_mesh
from pygmsh.built_in.geometry import Geometry
import meshio



"""
This code generates a mesh modeling a cylindrical artery and its wall.
The blood and arterial wall volumes/surfaces are marked for FSI simulation.
The mesh is saved in .geo, .msh and .xmdf files.
"""



def create_mesh(res=0.025, inner=0.1, outer=0.105, length=0.5): 
    """
    Arguments
    ---------
    - res: meshing resolution 
    - inner: cylindrical artery inner radius
    - outer: cylindrical artery outer radius
      (arterial wall thickness = outer - outer)
    - length: cylindrical artery length
    """   


    # Define geometry
    # ================


    geo = Geometry()

    # Inner circle
    circle = geo.add_circle((0,0,0),inner)
    # Outer circle
    outer_circle = geo.add_circle((0,0,0), outer, lcar=res, holes=[circle])

    # Inner pipe
    inner_pipe = geo.extrude(circle.plane_surface, translation_axis=[0,0,length], point_on_axis=[0,0,0])

    # Outer pipe
    outer_pipe = geo.extrude(outer_circle.plane_surface, translation_axis=[0,0,length], point_on_axis=[0,0,0])
    

    # Define markers
    # ==============


    # Volume for blood and arteria wall
    geo.add_physical(inner_pipe[1])
    geo.add_physical(outer_pipe[1])

    # Interior interface, common to inner and outer
    geo.add_physical(inner_pipe[2])

    # Blood inlet/outlet
    geo.add_physical(inner_pipe[0])
    geo.add_physical(circle.plane_surface)
    
    # Arteria wall inlet/outlet
    geo.add_physical(outer_pipe[0])
    geo.add_physical(outer_circle.plane_surface)

    # Arteria outer wall
    geo.add_physical(outer_pipe[2][:3])


    # Construct geometry and mesh
    # ===========================


    #mesh = generate_mesh(geo, geo_filename="arteria.geo")
    #meshio.write("arteria.msh", mesh) #not ok with gmsh 4.4.1
    
    mesh = generate_mesh(geo, msh_filename="arteria.msh")


# Main
# =======
create_mesh(0.1E-2, 0.3E-2, 0.315E-2, 5.0E-2)


# Convert to XDMF
input_file = "arteria.msh"
file_name = "arteria"
msh = meshio.read(input_file)

# Write XDMF and physical data
meshio.write(file_name + ".xdmf", meshio.Mesh(points = msh.points, cells = {'tetra': msh.cells_dict['tetra']}))
meshio.write(file_name + "_mfv.xdmf", meshio.Mesh(points=msh.points, cells=[("tetra", msh.cells_dict["tetra"])], cell_data={"name_to_read": [msh.cell_data_dict["gmsh:physical"]["tetra"]]}))
meshio.write(file_name + "_mf.xdmf", meshio.Mesh(points=msh.points, cells=[("triangle", msh.cells_dict["triangle"])], cell_data={"name_to_read": [msh.cell_data_dict["gmsh:physical"]["triangle"]]}))
