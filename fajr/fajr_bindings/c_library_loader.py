import ctypes
import os

import sys
# Determine the correct library name based on the platform
if sys.platform == "linux":
    lib_name = "libfajr.so"
elif sys.platform == "darwin":
    lib_name = "libfajr.dylib"
elif sys.platform == "win32":
    lib_name = "fajr.dll"
else:
    raise NotImplementedError(f"Unsupported platform: {sys.platform}")

# Construct the path to the library within the installed package
# This assumes the library is copied directly into the 'fajr' package directory
# during installation.
library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", lib_name)

tensor_lib = None
try:
    tensor_lib = ctypes.CDLL(library_path)
except OSError as e:
    print(f"Error loading shared library: {e}")
    print(f"Please ensure '{library_path}' exists and is accessible.")
