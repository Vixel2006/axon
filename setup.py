import os
import subprocess
import sys
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

def has_cuda():
    try:
        subprocess.check_output(["nvcc", "--version"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

class CMakeBuild(build_ext):
    def run(self):
        build_directory = os.path.abspath(self.build_temp)
        os.makedirs(build_directory, exist_ok=True)

        # --- Build SLEEF ---
        sleef_source_dir = os.path.abspath(os.path.join("vendor", "sleef"))
        sleef_build_dir = os.path.join(build_directory, "sleef_build")
        os.makedirs(sleef_build_dir, exist_ok=True)

        print("Configuring and building SLEEF...")
        sleef_cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSLEEF_STATIC_LIBS=ON",
            "-DSLEEF_TEST=OFF",
            "-DSLEEF_SMP=OFF",
            "-DSLEEF_OPENMP=OFF",
        ]
        subprocess.check_call(["cmake", sleef_source_dir, *sleef_cmake_args], cwd=sleef_build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=sleef_build_dir)
        print("SLEEF built successfully.")

        # Locate static library
        sleef_static_lib_name = "libsleef.a" if sys.platform != "win32" else "sleef.lib"
        potential_paths = [
            os.path.join(sleef_build_dir, sleef_static_lib_name),
            os.path.join(sleef_build_dir, "lib", sleef_static_lib_name),
            os.path.join(sleef_build_dir, "Release", sleef_static_lib_name),
            os.path.join(sleef_build_dir, "src", "lib", sleef_static_lib_name),
        ]
        sleef_library_path = next((p for p in potential_paths if os.path.exists(p)), None)
        if not sleef_library_path:
            raise FileNotFoundError(f"SLEEF static lib not found in expected locations: {potential_paths}")

        sleef_include_dir = os.path.join(sleef_source_dir, "include")

        # --- Build Axon ---
        print("Configuring and building Axon...")
        axon_cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_directory}",
            f"-DSLEEF_INCLUDE_DIR={sleef_include_dir}",
            f"-DSLEEF_LIBRARY={sleef_library_path}",
        ]

        if has_cuda():
            print("CUDA detected. Building with CUDA support.")
            axon_cmake_args.append("-DAXON_BUILD_CUDA=ON")
        else:
            print("CUDA not detected. Building without CUDA support.")

        if sys.platform == "win32":
            axon_cmake_args.append("-GVisual Studio 17 2022")

        subprocess.check_call(["cmake", os.path.abspath("."), *axon_cmake_args], cwd=build_directory)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_directory)
        print("Axon built successfully.")

        # Copy built library into package
        lib_name = {"linux": "libaxon.so", "darwin": "libaxon.dylib", "win32": "axon.dll"}[sys.platform]
        built_lib_path = os.path.join(build_directory, lib_name)
        if not os.path.exists(built_lib_path):
            raise FileNotFoundError(f"Built library not found at {built_lib_path}")

        package_dir = os.path.join(os.path.abspath("."), "axon")
        os.makedirs(package_dir, exist_ok=True)
        shutil.copyfile(built_lib_path, os.path.join(package_dir, lib_name))
        print(f"Copied {lib_name} into axon package.")

def get_platform_package_data():
    return {
        "linux": ["libaxon.so"],
        "darwin": ["libaxon.dylib"],
        "win32": ["axon.dll"],
    }.get(sys.platform, [])

setup(
    name="axon-dl",
    version="0.1.0",
    packages=find_packages(),
    cmdclass={"build_ext": CMakeBuild},
    package_data={"axon": get_platform_package_data()},
    zip_safe=False,
)

