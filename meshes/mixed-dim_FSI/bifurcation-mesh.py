from pygmsh import generate_mesh
from pygmsh.opencascade.geometry import Geometry
import meshio
import numpy as np



"""
This code generates a mesh modeling an idealized arterial bifurcation and its wall.
The blood and arterial wall volumes/surfaces are marked for FSI simulation.
The mesh is saved in .geo, .msh and .xmdf files.
"""



def create_mesh(res, R, L0, L1, theta, h):
    """
    Arguments
    ---------
    - res: meshing resolution 
    - R: arteries radius
    - L0: parent branch artery length
    - L1: daugther branches arteries length
      (the two daugther branches are symetric) 
    - theta: angle between the two daugther branches
    - h: arterial wall thickness
    """  


    # Define geometry
    # ================


    # Characteristic_length_min/max define the global min/max for the mesh size
    geo = Geometry(characteristic_length_min=res, characteristic_length_max=2*res)

    # Parent branch inner cylinder
    inner_cylinder0 = geo.add_cylinder(x0=[0,0,0], axis=[0,0,L0], radius=R, char_length=res)
    # First daugther inner cylinder
    inner_cylinder1 = geo.add_cylinder(x0=[0,0,L0], axis=[L1*np.tan(theta),0,L1], radius=R, char_length=res)
    # Second daugther inner cylinder
    inner_cylinder2 = geo.add_cylinder(x0=[0,0,L0], axis=[-L1*np.tan(theta),0,L1], radius=R, char_length=res)
    fluid_domain = geo.boolean_union([inner_cylinder0, inner_cylinder1, inner_cylinder2])

    # Parent branch outer cylinder
    outer_cylinder0 = geo.add_cylinder(x0=[0,0,0], axis=[0,0,L0], radius=R+h, char_length=res)
    # First daugther outer cylinder
    outer_cylinder1 = geo.add_cylinder(x0=[0,0,L0], axis=[L1*np.tan(theta),0,L1], radius=R+h, char_length=res)
    # Second daugther outer cylinder
    outer_cylinder2 = geo.add_cylinder(x0=[0,0,L0], axis=[-L1*np.tan(theta),0,L1], radius=R+h, char_length=res)
    whole_domain = geo.boolean_union([outer_cylinder0, outer_cylinder1, outer_cylinder2])
    
    # Boolean_fragment function automatically matches interface surfaces
    solid_domain = geo.boolean_fragments([whole_domain], [fluid_domain])


    # Define markers
    # ==============


    code = []

    # Volume for blood and arterial wall
    code.append("Physical Volume(1) = {1};")
    code.append("Physical Volume(2) = {2};")

    # Interior interface, common to blood and arterial wall
    code.append("Physical Surface(3) = {3, 6, 7};")

    # Blood inlet/outlet
    code.append("Physical Surface(4) = {10};")
    code.append("Physical Surface(5) = {11};")
    code.append("Physical Surface(6) = {12};")

    # Arterial wall inlet/outlet
    code.append("Physical Surface(7) = {1};")
    code.append("Physical Surface(8) = {8};")
    code.append("Physical Surface(9) = {9};")

    # Arterial outer wall
    code.append("Physical Surface(10) = {2, 4, 5};")

    geo.add_raw_code(code)


    # Construct geometry and mesh
    # ===========================

    #mesh = generate_mesh(geo, geo_filename="bifurcation.geo")
    #meshio.write("bifurcation.msh", mesh)

    mesh = generate_mesh(geo, msh_filename="bifurcation.msh")

# Main
# =======

create_mesh(0.075E-2, 0.3E-2, 5.0E-2, 4.0E-2, np.pi/6, 0.1E-2)

# Convert to XDMF
input_file = "bifurcation.msh"
file_name = "bifurcation"
msh = meshio.read(input_file)

# Write XDMF and physical data
meshio.write(file_name + ".xdmf", meshio.Mesh(points = msh.points, cells = {'tetra': msh.cells_dict['tetra']}))
meshio.write(file_name + "_mfv.xdmf", meshio.Mesh(points=msh.points, cells=[("tetra", msh.cells_dict["tetra"])], cell_data={"name_to_read": [msh.cell_data_dict["gmsh:physical"]["tetra"]]}))
meshio.write(file_name + "_mf.xdmf", meshio.Mesh(points=msh.points, cells=[("triangle", msh.cells_dict["triangle"])], cell_data={"name_to_read": [msh.cell_data_dict["gmsh:physical"]["triangle"]]}))

