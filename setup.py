import os
import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def find_cuda():
    """Finds the CUDA install path."""
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    if os.path.exists(cuda_home):
        return cuda_home
    try:
        nvcc = subprocess.check_output(["which", "nvcc"]).decode().strip()
        cuda_home = os.path.dirname(os.path.dirname(nvcc))
        return cuda_home
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    raise RuntimeError(
        "Cannot find CUDA installation. Please set the CUDA_HOME environment variable."
    )


CUDA_PATH = find_cuda()


class CudaBuild(build_ext):
    """
    Custom build_ext command to compile CUDA files.
    It inherits from pybind11's build_ext to keep its features.
    """

    def build_extension(self, ext):
        # The CUDA compiler
        nvcc = os.path.join(CUDA_PATH, "bin", "nvcc")

        # Separate the source files into C++ and CUDA files
        cpp_sources = []
        cu_sources = []
        for source in ext.sources:
            if source.endswith(".cu"):
                cu_sources.append(source)
            else:
                cpp_sources.append(source)

        # Compile CUDA files to object files
        cuda_objects = []
        for source in cu_sources:
            # Create object file path
            obj_name = os.path.splitext(os.path.basename(source))[0] + ".o"
            obj_path = os.path.join(self.build_temp, obj_name)

            # Ensure build directory exists
            os.makedirs(os.path.dirname(obj_path), exist_ok=True)

            cuda_objects.append(obj_path)

            command = [
                nvcc,
                "-c",
                source,
                "-o",
                obj_path,
                "--std=c++17",
                "-Xcompiler",
                "-fPIC",
                # Add optimization flags
                "-O3",
                # Add compute capability (adjust as needed for your GPU)
                "-gencode",
                "arch=compute_75,code=sm_75",
                "-gencode",
                "arch=compute_80,code=sm_80",
                "-gencode",
                "arch=compute_86,code=sm_86",
                "-Wno-deprecated-gpu-targets",  # Suppress the warning
            ]

            # Add all include directories for nvcc (including Python headers)
            for include_dir in ext.include_dirs:
                command.extend(["-I", include_dir])

            # Add Python include directories explicitly
            import sysconfig

            python_include = sysconfig.get_path("include")
            if python_include and python_include not in ext.include_dirs:
                command.extend(["-I", python_include])

            # Also add platinclude for platform-specific headers
            python_platinclude = sysconfig.get_path("platinclude")
            if python_platinclude and python_platinclude != python_include:
                command.extend(["-I", python_platinclude])

            print(f"Compiling CUDA source: {' '.join(command)}")
            subprocess.check_call(command)

        # Update extension to use only C++ sources
        ext.sources = cpp_sources

        # Add CUDA objects to extra_objects
        if not hasattr(ext, "extra_objects"):
            ext.extra_objects = []
        ext.extra_objects.extend(cuda_objects)

        # Let the original pybind11 build_ext handle the C++ compilation and linking
        super().build_extension(ext)


ext_modules = [
    Pybind11Extension(
        "nawah",
        [
            "bindings/bindings.cpp",
            "src/allocator/allocatorFactory.cpp",
            "src/tensor.cpp",
            "src/engine/add.cpp",
            "src/engine/sub.cpp",
            "src/engine/mul.cpp",
            "src/engine/matmul.cpp",
            "src/engine/ops/cpu/matmul_cpu.cpp",
            "src/engine/ops/cpu/add_cpu.cpp",
            "src/engine/ops/cpu/sub_cpu.cpp",
            "src/engine/ops/cpu/mul_cpu.cpp",
            "src/engine/ops/cuda/add.cu",
            "src/engine/ops/cuda/sub.cu",
            "src/engine/ops/cuda/mul.cu",
            "src/engine/ops/cuda/matmul.cu",
        ],
        include_dirs=["include", os.path.join(CUDA_PATH, "include")],
        library_dirs=[os.path.join(CUDA_PATH, "lib64")],
        libraries=["cudart"],
        language="c++",
        extra_compile_args=["-std=c++17", "-g", "-O3"],
        extra_link_args=["-lcuda"],
    ),
]

setup(
    name="nawah",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CudaBuild},
    zip_safe=False,
)
