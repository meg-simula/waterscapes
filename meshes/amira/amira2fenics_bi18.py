import sys

from dolfin_utils.meshconvert import xml_writer 
import re 


def run(input): 
  """ 
  convert between amira and fenics implemented as state machine 
  0 = look for starters 
  1 = vertex 
  2 = cells 
  3 = labels
  """
  state = 0 
  ifile = open(input[0])
  ofile = open(input[1], "w")
  ofile_sub = open(input[2], "w")
  
  vertices =[] 
  cells = []
  labels = []
  
  index_vertices = 0
  index_cells = 0
  while 1:
    line = ifile.readline()
    if line.startswith("@1"): 
      state = 1
    elif line.startswith("\n"): 
      state = 0  
    elif line.startswith("@2"): 
      state = 2  
    elif line.startswith("@3"):
      state = 3
    elif line.startswith("@4"):
      break
    
    elif state == 1:   
    # example: 
    #-37.168381 50.233715 -12.345407
      #print line.strip().split()
      index_vertices += 1
      x, y, z = line.strip().split()
      vertices.append((index_vertices,x,y,z))

    elif state == 2:   
    # example: 
#    *tetra4(1,1,38366,38367,9253,43441)
      index_cells += 1
      v1, v2, v3, v4= line.strip().split() 
      cells.append((index_cells,v1,v2,v3,v4))

    elif state == 3:   
    # example: 
#    *tetra4(1,1,38366,38367,9253,43441)
      label = line.strip() 
      labels.append(label)

  print "out of loop"
  xml_writer.write_header_mesh(ofile, "tetrahedron", 3) 

  xml_writer.write_header_vertices(ofile, len(vertices))
  counter = 0 
  for v in vertices: 
    if not counter == int(v[0])-1: print "Warning! Numbering assumption invalidated", counter, v[0]  
    xml_writer.write_vertex(ofile, int(v[0])-1, float(v[1]), float(v[2]), float(v[3])) 
    counter += 1 
  xml_writer.write_footer_vertices(ofile)

  xml_writer.write_header_cells(ofile, len(cells))
  counter=0
  for c in cells: 
    if not counter == int(c[0])-1: print "Warning! Numbering assumption invalidated", counter, c[0]  
    xml_writer.write_cell_tetrahedron(ofile, int(c[0])-1, int(c[1])-1, int(c[2])-1, int(c[3])-1, int(c[4])-1) 
    counter += 1 
  xml_writer.write_footer_cells(ofile)


  header = ("""<?xml version="1.0"?>
<dolfin xmlns:dolfin="http://fenicsproject.org">
  <mesh_function>
    <mesh_value_collection name="f" type="uint" dim=\"%d" size=\"%d">\n""") % (3, len(cells))
  
  ofile_sub.write(header)
  
  counter=0
  for c in cells: 
    if not counter == int(c[0])-1: print "Warning! Numbering assumption invalidated", counter, c[0]
    xml_writer.write_entity_meshvaluecollection(ofile_sub, 1, int(c[0])-1, int(labels[counter]), local_entity=0) 
    counter += 1
    
  xml_writer.write_footer_meshvaluecollection(ofile_sub)
  xml_writer.write_footer_meshfunction(ofile_sub)
  xml_writer.write_footer_mesh(ofile)
 
#   not handled: 
#    *component(1,"?",1,1)
#    *tria3(1,1,9253,38366,38367)
#
#    *component(2,"Inside",1,2)
#    *tetra4(1,1,38366,38367,9253,43441)
#      
#    if line.startswith("

if __name__ == "__main__": 

  ifilename=sys.argv[1]
  ofilename=sys.argv[2]
  ofilename_sub=sys.argv[3]

  run((ifilename, ofilename, ofilename_sub))

