from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import os

model_dir = os.path.dirname(__file__) or os.getcwd()
includes = [model_dir, numpy.get_include()]
macros = []
libs = []
c_comp_args = ["-ffast-math","-O3","-fopenmp"]
extra_link_args = ['-fopenmp']
c_comp_args = ["-ffast-math","-O3"]
extra_link_args = []
cpp_comp_args = []



extensions = [
    Extension(
        "model",
        define_macros = macros,
        sources=["model.pyx"],
        language="c++",
        include_dirs = includes,
        extra_compile_args=c_comp_args,
        libraries=libs,
        extra_link_args=extra_link_args,
    ),
#    Extension(
#        "openmp",
#        define_macros = macros,
#        sources=["openmp.pyx"],
#        language="c++",
#        include_dirs = includes,
#        extra_compile_args=c_comp_args,
#        libraries=libs,
#        extra_link_args=extra_link_args,
#    ),
#    Extension(
#        "test",
#        define_macros = macros,
#        sources=["test.pyx"],
#        language="c++",
#        include_dirs = includes,
#        extra_compile_args=c_comp_args,
#        libraries=libs,
#        extra_link_args=extra_link_args,
#    ),


]


setup(
    ext_modules = cythonize(extensions),
)
