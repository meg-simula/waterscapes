====================================================================
mpet: finite element solvers for multiple-network poroelastic theory
====================================================================

mpet is a Python module based on the FEniCS finite element software
for solving the multiple-network poroelasticity equations.


==========
Quickstart
==========

In the top-level mpet directory, do::
  $ fenicsproject run stable
  $ export PYTHONPATH=`pwd`:$PYTHONPATH
  $ python3 test/test_mpetsolver_homogeneousDirichletBC.py

=============
Documentation
=============

See http://waterscapes.readthedocs.org

How to generate the documentation from scratch

1. Make sure you have Sphinx installed:

   http://www.sphinx-doc.org/en/stable/index.html

2. From top level MPET, create directory 'doc:

   mkdir doc

3. From top level MPET directory, run sphinx-quickstart

   sphinx-quickstart

   and answer questions. Use doc as the "root path for the
   documentation" and make sure to press y for autodoc.

   Sphinx will now initialize Makefiles etc for you in the doc
   directory.

4. Add Makefile, and files under doc/source to your version control
   system.

5. Add a generate doc target to the Makefile: for instance:

   generate_api_docs:
	sphinx-apidoc -o source/ -H mpet -A "E. Piersanti" -V 0.1 ../mpet/

   Make sure to have all the necessary modules in your PYTHONPATH
   before running sphinx-apidox!

6. Run make html to generate the actual html.
