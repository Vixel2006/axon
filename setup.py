import os
import subprocess
import sysconfig # Import sysconfig to find Python paths

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages


def find_cuda():
    """Finds the CUDA install path."""
    # (Same as your version)
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
    """
    def build_extension(self, ext):
        nvcc = os.path.join(CUDA_PATH, "bin", "nvcc")
        cpp_sources = []
        cu_sources = []
        for source in ext.sources:
            if source.endswith(".cu"):
                cu_sources.append(source)
            else:
                cpp_sources.append(source)

        # ==================== ADDITION: Get Python Include Dirs ===================
        python_include_path = sysconfig.get_path("include")
        python_plat_include_path = sysconfig.get_path("platinclude")
        # ========================================================================
        
        cuda_objects = []
        for source in cu_sources:
            obj_name = os.path.splitext(os.path.basename(source))[0] + ".o"
            obj_path = os.path.join(self.build_temp, obj_name)
            os.makedirs(os.path.dirname(obj_path), exist_ok=True)
            cuda_objects.append(obj_path)

            command = [
                nvcc,
                "-c", source, "-o", obj_path,
                "--std=c++17",
                "-Xcompiler", "-fPIC",
                "-O3",
                "-gencode", "arch=compute_75,code=sm_75",
                "-gencode", "arch=compute_86,code=sm_86",
                "-Wno-deprecated-gpu-targets",
            ]
            
            # Add all extension include directories
            for include_dir in ext.include_dirs:
                command.extend(["-I", include_dir])

            # =================== FIX: Explicitly add Python includes ==================
            command.extend(["-I", python_include_path])
            if python_plat_include_path != python_include_path:
                 command.extend(["-I", python_plat_include_path])
            # ========================================================================
            
            print(f"Compiling CUDA source: {' '.join(command)}")
            subprocess.check_call(command)

        ext.sources = cpp_sources
        ext.extra_objects.extend(cuda_objects)
        super().build_extension(ext)

# ... (The rest of your setup.py remains the same)
# (ext_modules definition, setup() call, etc.)
ext_modules = [
    Pybind11Extension(
        "cnawah",
        [
            "bindings/bindings.cpp",
            "src/allocator/allocatorFactory.cpp",
            "src/tensor.cpp",
            "src/registery_backend.cpp",
            "src/engine/ops/cpu/matmul.cpp",
            "src/engine/ops/cpu/add.cpp",
            "src/engine/ops/cpu/sub.cpp",
            "src/engine/ops/cpu/mul.cpp",
            "src/engine/ops/cpu/sum.cpp",
            "src/engine/ops/cpu/mean.cpp",
            "src/engine/ops/cpu/relu.cpp",
            "src/engine/ops/cpu/pow.cpp",
            "src/engine/ops/cpu/log.cpp",
            "src/engine/ops/cpu/div.cpp",
            "src/engine/ops/cpu/exp.cpp",
            "src/engine/ops/cpu/softmax.cpp",
            "src/engine/ops/cuda/add.cu",
            "src/engine/ops/cuda/sub.cu",
            "src/engine/ops/cuda/mul.cu",
            "src/engine/ops/cuda/matmul.cu",
            "src/engine/ops/cuda/sum.cu",
            "src/engine/ops/cuda/mean.cu",
            "src/engine/ops/cuda/relu.cu",
            "src/engine/ops/cuda/exp.cu",
            "src/engine/ops/cuda/log.cu",
            "src/engine/ops/cuda/pow.cu",
            "src/engine/ops/cuda/div.cu",
            "src/engine/ops/cuda/softmax.cu",
            "src/autograd/cpu/badd.cpp",
            "src/autograd/cuda/badd.cu",
            "src/autograd/cpu/bsub.cpp",
            "src/autograd/cuda/bsub.cu",
            "src/autograd/cpu/bmul.cpp",
            "src/autograd/cpu/bdiv.cpp",
            "src/autograd/cpu/bpow.cpp",
            "src/autograd/cpu/blog.cpp",
            "src/autograd/cpu/bmatmul.cpp",
            "src/autograd/cpu/bexp.cpp",
            "src/autograd/cuda/bmul.cu",
            "src/autograd/cuda/bdiv.cu",
            "src/autograd/cuda/bpow.cu",
            "src/autograd/cuda/blog.cu",
            "src/autograd/cuda/bexp.cu",
            "src/autograd/cuda/bmatmul.cu",
            "src/engine/ops/cpu/fill.cpp",
            "src/engine/ops/cuda/fill.cu"
        ],
        include_dirs=["include", os.path.join(CUDA_PATH, "include")],
        library_dirs=[os.path.join(CUDA_PATH, "lib64")],
        libraries=["cudart"],
        language="c++",
        
        extra_compile_args=[
            "-std=c++17", 
            "-g", 
            "-O3", 
            "-mavx2",       
            "-mfma",        
            "-fopenmp"      
        ],
        extra_link_args=[
            "-fopenmp",     
            "-lcuda"
        ]
    ),
]

setup(
    name="nawah",
    version="0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CudaBuild},
    zip_safe=False,
)
