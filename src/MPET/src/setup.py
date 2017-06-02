# System imports
from distutils.core import setup
#import platform
import sys
#from os.path import join as pjoin

# Version number
major = 0
minor = 1

setup(name="waterscape",
      version = "{0}.{1}".format(major, minor),
      description = """
      An adjointable multiple-network poroelasticity solver
      """,
      author = "Eleonora Piersanti, Marie E. Rognes",
      author_email = "eleonora@simula.no",
      packages = ["mpet"],
      package_dir = {"mpet": "mpet"},
      )
