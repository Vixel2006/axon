import ctypes
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
library_path = os.path.join(project_root, "build", "libelnawah.so")

tensor_lib = None
try:
    tensor_lib = ctypes.CDLL(library_path)
except OSError as e:
    print(f"Error loading shared library: {e}")
    print(f"Please ensure '{library_path}' exists and is accessible.")
