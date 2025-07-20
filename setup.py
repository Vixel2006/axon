from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

cuda_include = "/opt/cuda/include"
cuda_lib = "/opt/cuda/lib64"

ext_modules = [
    Pybind11Extension(
        "nawah",
        [
            "bindings/bindings.cpp",
            "src/tensor.cpp",
            "src/ops/add.cpp",
            "src/ops/sub.cpp",
            "src/ops/mul.cpp",
        ],
        include_dirs=["include", cuda_include],
        library_dirs=[cuda_lib],
        libraries=["cudart"],
        language="c++",
    ),
]

setup(
    name="nawah",
    version="0.1",
    packages=["nawah"],
    package_dir={"nawah": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
