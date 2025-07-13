from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

cuda_include = "/opt/cuda/include"
cuda_lib = "/opt/cuda/lib64"

ext_modules = [
    Pybind11Extension(
        "neura",
        ["bindings/bindings.cpp", "src/tensor.cpp"],
        include_dirs=["include", cuda_include],
        library_dirs=[cuda_lib],
        libraries=["cudart"],
        language="c++",
    ),
]

setup(
    name="neura",
    version="0.1",
    packages=["neura"],
    package_dir={"neura": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
