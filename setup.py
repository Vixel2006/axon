import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext

# A custom build extension for CMake
class CMakeBuild(build_ext):
    def run(self):
        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake to configure and build
        cmake_args = [
            "-DCMAKE_BUILD_TYPE=Release",  # Or Debug
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(self.build_temp)}",
        ]
        build_args = ["--config", "Release"]

        subprocess.check_call(["cmake", os.path.abspath("."), *cmake_args], cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", ".", *build_args], cwd=self.build_temp)

        # Determine the correct library name based on the platform
        if sys.platform == "linux":
            lib_name = "libfajr.so"
        elif sys.platform == "darwin":
            lib_name = "libfajr.dylib"
        elif sys.platform == "win32":
            lib_name = "fajr.dll"
        else:
            raise NotImplementedError(f"Unsupported platform: {sys.platform}")

        built_lib_path = os.path.join(self.build_temp, lib_name)
        
        # Copy the built library to the idrak package directory
        # This assumes the idrak package is directly under the project root
        package_dir = os.path.join(os.path.abspath("."), "fajr")
        if not os.path.exists(package_dir):
            os.makedirs(package_dir)
        
        destination_lib_path = os.path.join(package_dir, lib_name)
        self.copy_file(built_lib_path, destination_lib_path)

setup(
    name='fajr',
    version='0.1.0',
    packages=find_packages(),
    cmdclass={
        'build_ext': CMakeBuild,
    },
    package_data={
        'fajr': ['libfajr.so', 'libfajr.dylib', 'fajr.dll'], # Include all for cross-platform
    },
)