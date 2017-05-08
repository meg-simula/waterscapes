==============================================================
=            Collins adult brain atlas FEM mesh              =
=                       Version 2L                           =
=             low-resolution mesh (70226 nodes)              =
=                                                            =
=  Created by Qianqian Fang <fangq at nmr.mgh.harvard.edu>   =
==============================================================

== Introduction ==

Collins adult brain atlas [Collins1998] FEM mesh Version 1 was 
created on 02/05/2011 by Qianqian Fang with iso2mesh [Fang2009a] 
version 1.0.0. The gray/white matter surfaces were created by 
Katherine Perdue with FreeSurfer.

URL: http://mcx.sourceforge.net/cgi-bin/index.cgi?MMC/CollinsAtlasMesh

Please refer to [Fang2010] for details. This data is released in the
public domain.

== Format ==

The mesh data were saved in matlab/octave .mat format. Please 
load the .mat file in matlab/octave and export to your FEM simulator.

	node: node coordinates (in mm)
	face: surface triangles, the last column is the surface ID, 
		1-scalp, 2-CSF, 3-gray matter, 4-white matter
	elem: tetrahedral elements, the last column is the region ID, 
		1-scalp and skull layer, 2-CSF, 3-gray matter, 4-white matter

== Optical properties ==

The optical properties of the brain atlas was described in 
[Fang2010] Table 2 (the same properties were first used in 
[Fang2009b]). The properties for scalp/skull and cerebro-spinal 
fluid (CSF) are based on [Custo2006] and those for gray and 
white matters are based on [Yaroslavsky2002] at 630 nm.

---------------------------------------------------------------
!!Tissue types	!!μa (mm−1)!!μs (mm−1)	!!Anisotropy (g)!!Refract. Index (n)!!
||Scalp & skull	||0.019	||7.8		||0.89		||1.37||
||CSF		||0.004	||0.009		||0.89		||1.37||
||Gray-matter	||0.02	||9.0		||0.89		||1.37||
||White-matter	||0.08	||40.9		||0.84		||1.37||
---------------------------------------------------------------

== Footnote ==

If you use this mesh in your publication, please cite the mesh 
version number to avoid any conflict to further updates of this
mesh.

== Reference ==

[Collins1998] D. L. Collins, A. P. Zijdenbos, V. Kollokian, J. G. Sled, 
N. J. Kabani, C. J. Holmes, and A. C. Evans, “Design and construction 
of a realistic digital brain phantom,” IEEE Trans. Med. Imaging 17(3), 
463–468 (1998). 

[Fang2010] Q. Fang, "Mesh-based Monte Carlo method 
using fast ray-tracing in Plucker coordinates," Biomed. Opt. Express 
1(1), 165-175 (2010)

[Fang2009a] Q. Fang and D. Boas, “Tetrahedral mesh generation from 
volumetric binary and gray-scale images,” Proceedings of IEEE 
International Symposium on Biomedical Imaging 2009, 1142–1145 (2009).

[Fang2009b] Q. Fang and D. A. Boas, “Monte Carlo simulation of photon 
migration in 3D turbid media accelerated by graphics processing units,” 
Opt. Express 17(22), 20178–20190 (2009).

[Custo2006] A. Custo, W. M. Wells 3rd, A. H. Barnett, E. M. Hillman, 
and D. A. Boas, “Effective scattering coefficient of the cerebral 
spinal fluid in adult head models for diffuse optical imaging,” 
Appl. Opt. 45(19), 4747–4755 (2006).

[Yaroslavsky2002] A. N. Yaroslavsky, P. C. Schulze, I. V. Yaroslavsky, 
R. Schober, F. Ulrich, and H.-J. Schwarzmaier, “Optical properties 
of selected native and coagulated human brain tissues in vitro 
in the visible and near infrared spectral range,” Phys. Med. Biol. 
47(12), 2059–2073 (2002).
