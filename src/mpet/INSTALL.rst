============
Dependencies
============

The mpet solvers depend on 

* The FEniCS Project software (version 2019.1 or later). For FEniCS
  installation instructions, see https://fenicsproject.org/download/
  The (stable or dev) Docker images should work just fine.

==========================
Installing the mpet module
==========================

# To install:
python setup.py install --prefix=/your/favorite/folder

# For instance
python setup.py install --user

# If you don't want to install globally, just add mpet to your
# PYTHONPATH via:
export PYTHONPATH=`pwd`:$PYTHONPATH

# Alternatively use pip to do an editable install in this folder
pip install -e .

