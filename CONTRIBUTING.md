# 🤝 contributing to idrak

welcome, fellow hacker! idrak is not just another deep learning framework; it's a statement. if you resonate with our philosophy of uncompromising hackability, raw performance, and elegant simplicity, then your contributions are invaluable.

we demand code that is **optimized as fuck**, **smaller and compressed**, and **easy to read as fuck**. if your code doesn't meet these standards, it doesn't belong here.

## 🚀 getting started

before you dive in, ensure you have your development environment set up:

1.  **fork the repository.**
2.  **clone your fork:** `git clone https://github.com/your-username/nawah.git`
3.  **navigate to the project root:** `cd nawah`
4.  **install python dependencies:** `pip install -e .`
5.  **build the c backend:** idrak relies on a high-performance c backend. you'll need `cmake` and a c compiler (like `gcc`).
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```
    ensure `libidrak.so` (or `libidrak.dylib` on macos) is generated in the `build/` directory.

## 🧠 codebase overview: the idrak architecture

idrak's power comes from its hybrid python/c architecture, designed for maximum flexibility and performance.

### python frontend (`idrak/`)

this is where the user-facing api and high-level logic reside.

*   **`idrak/core/tensor.py`**: defines the python `tensor` object. this is your primary data structure. it manages shape, device, and `requires_grad`, but crucially, it delegates actual data storage and computation to the c backend via `idrak_bindings`. it also overloads python operators (`+`, `*`, `@`, etc.) to construct the lazy computation graph.
*   **`idrak/core/buffer.py`**: implements `lazybuffer`, the heart of idrak's lazy evaluation and autograd system. it handles topological sorting (`topo_sort`) for efficient execution and `realize()` for triggering computation and `backward()` for gradient propagation.
*   **`idrak/idrak_bindings/`**: the critical bridge between python and c.
    *   **`c_library_loader.py`**: dynamically loads the compiled c shared library (`libidrak.so`).
    *   **`ctypes_definitions.py`**: defines `ctypes` structures that precisely mirror the c `tensor`, `storage`, and `device` structs. this allows python to directly manipulate c data types.
    *   **`c_function_signatures.py`**: declares the argument and return types for all c functions exposed to python, ensuring correct data marshaling.
    *   **`c_wrapper_functions.py`**: provides convenient python functions that wrap the raw `ctypes` calls to the c backend.
*   **`idrak/ops/`**: python-side definitions of all tensor operations.
    *   **`op.py`**: the abstract base class `lazyop`. every operation (unary, binary, movement, reduction) must inherit from this. it defines the interface for calculating output shapes (`calc_out_shape`), creating c-specific context (`create_ctx_struct`), and the python-side `forward` and `backward` calls that delegate to c. the `create_node` method is fundamental for building the lazy computation graph.
    *   **`uop.py`, `bop.py`, `mop.py`, `rop.py`**: concrete implementations for unary, binary, movement, and reduction operations. these classes define the python logic for an operation and then call the appropriate c wrapper function.
*   **`idrak/nn/`**: python-side neural network modules.
    *   **`module.py`**: the abstract base class `module`. all neural network layers inherit from this, defining `forward`, `params`, `buffers`, `freeze`, and `reset_parameters`.
    *   **`pipeline.py`**: the `pipeline` class, a `module` that acts as a sequential container for other `module`s. it enables the `>>` operator for chaining layers and supports list-like access (`model[0]`, `model[1:3]`).
    *   **`conv.py`, `linear.py`, `activations.py`**: implementations of standard neural network layers.
*   **`idrak/functions.py`**: a functional api for common tensor operations (e.g., `zeros`, `add`, `relu`). these functions typically wrap the `lazyop.create_node` calls, providing a more convenient interface.
*   **`idrak/metrics/`**: python-side implementations of loss functions (e.g., `bce`, `mse`).
*   **`idrak/optim/`**: python-side wrappers for optimizers (e.g., `adam`, `sgd`).

### c backend (`src/`)

this is the performance-critical core, where tensor operations and their gradients are implemented in highly optimized c.

*   **`src/tensor.c`**: manages the low-level c `tensor` structure, memory allocation/deallocation (`tmalloc`, `tfree`, `smalloc`, `sfree`), and utility functions like `numel`, `compute_strides`, and `is_contiguous`.
*   **`src/autograd/cpu/`**: contains the c implementations of the **backward passes** (gradient computations) for various operations (e.g., `binary_ops_grad_cpu.c`, `unary_ops_grad_cpu.c`). these are heavily optimized, often utilizing simd instructions (`_mm256_...`) for maximum throughput.
*   **`src/ops/cpu/`**: contains the c implementations of the **forward passes** for various operations (e.g., `binary_ops_cpu.c`, `unary_ops_cpu.c`). like the backward passes, these are designed for raw speed.
*   **`src/optimizers/`**: c implementations of optimization algorithms (e.g., `adam.c`, `sgd.c`, `zero_grad.c`).

## 🎯 coding standards & principles

we are ruthless about these:

1.  **performance is paramount (c code):**
    *   **simd first:** for numerical operations, always prioritize simd intrinsics (avx2, avx512, etc.) where applicable. if you're writing a loop over tensor elements, you should be thinking in terms of `_mm256_loadu_ps`, `_mm256_add_ps`, etc.
    *   **cache efficiency:** design algorithms to be cache-friendly. minimize random memory access.
    *   **minimal overhead:** every clock cycle counts. avoid unnecessary function calls, memory allocations, or data copies.
    *   **contiguous vs. non-contiguous:** handle both cases efficiently. provide fast paths for contiguous tensors and correct (though potentially slower) paths for non-contiguous ones.
2.  **readability is non-negotiable (all code):**
    *   **clear python:** python code should be idiomatic, concise, and self-documenting.
    *   **well-commented c:** c code, especially complex simd or memory management logic, *must* be thoroughly commented. explain *why* you're doing something, not just *what*.
    *   **consistent naming:** follow existing naming conventions (`snake_case` for functions, `camelcase` for classes).
3.  **conciseness & compression:**
    *   **no bloat:** avoid unnecessary abstractions or boilerplate. if it doesn't add value, it's dead weight.
    *   **functional composition, object-oriented state:** while core operations are treated functionally (transforming data without side effects, building pipelines with `>>`), we pragmatically use classes (`module`, `pipeline`, `experiment`) for stateful components (like learnable parameters in layers) and for encapsulating related functionality. this allows for clear, manageable state while maintaining a functional approach to data flow.
4.  **testing:**
    *   **unit tests:** every new feature or bug fix should come with comprehensive unit tests. ensure your changes don't break existing functionality.
    *   **edge cases:** test for edge cases, invalid inputs, and boundary conditions.
5.  **error handling & logging:**
    *   **c logging:** use the `log_info`, `log_warn`, `log_error` macros in the c backend for debugging and critical error reporting.
    *   **python exceptions:** use appropriate python exceptions for invalid api usage.

## 🛠️ how to contribute

### reporting bugs

found a bug? don't just complain; help us fix it.

1.  check the [issue tracker](https://github.com/yushi2006/nawah/issues) to see if it's already reported.
2.  if not, open a new issue.
3.  provide a clear, concise description of the bug.
4.  include steps to reproduce the bug.
5.  specify your environment (os, python version, idrak version).
6.  (optional but highly appreciated) provide a minimal code example that demonstrates the bug.

### suggesting features

have an idea to make idrak even more brutal?

1.  open an issue on the [issue tracker](https://github.com/yushi2006/nawah/issues).
2.  clearly describe the feature and its use case.
3.  explain *why* this feature is important and how it aligns with idrak's philosophy.

### submitting pull requests

ready to contribute code? follow these steps:

1.  **fork** the repository and **clone** your fork.
2.  create a new branch for your feature or bug fix: `git checkout -b feature/my-awesome-feature` or `git checkout -b bugfix/fix-that-nasty-bug`.
3.  make your changes, adhering strictly to the [coding standards & principles](#-coding-standards--principles).
4.  **add/update tests:** ensure your changes are covered by new or updated unit tests.
5.  **build and test locally:** run `make` in your `build/` directory and then run your python tests to ensure everything works.
6.  **commit your changes:** write clear, concise commit messages.
    *   **good:** `feat: implement new conv3d layer with simd optimization`
    *   **bad:** `fix: stuff`
7.  push your branch to your fork: `git push origin feature/my-awesome-feature`
8.  open a pull request against the `main` branch of the `nawah` repository.
9.  provide a detailed description of your changes in the pr.

#### adding new operations (e.g., a new `lazyop`)

1.  **c backend (`src/ops/cpu/` and `src/autograd/cpu/`)**:
    *   implement the `forward` pass for your operation in `src/ops/cpu/your_op_cpu.c`. optimize it with simd.
    *   implement the `backward` pass (gradient) in `src/autograd/cpu/your_op_grad_cpu.c`. this is where the real magic happens.
    *   add function declarations to `include/ops/your_op.h` and `include/autograd/autograd_your_op.h`.
    *   update `cmakelists.txt` to compile your new c files.
2.  **ctypes definitions (`idrak/idrak_bindings/ctypes_definitions.py`)**:
    *   if your operation requires special context for the backward pass (e.g., `conv2dbackwardextras`), define its `ctypes.structure` here.
3.  **c function signatures (`idrak/idrak_bindings/c_function_signatures.py`)**:
    *   declare the `argtypes` and `restype` for your new c `forward` and `backward` functions.
4.  **c wrapper functions (`idrak/idrak_bindings/c_wrapper_functions.py`)**:
    *   create python wrapper functions for your new c functions.
5.  **python `lazyop` (`idrak/ops/your_op.py`)**:
    *   create a new class inheriting from `lazyop` (or `uop`, `bop`, `rop`, `mop` if applicable).
    *   implement `calc_out_shape`, `create_ctx_struct`, `forward`, and `backward` methods, calling your c wrappers.
    *   add your new `lazyop` to `idrak/ops/__init__.py`.
6.  **functional api (`idrak/functions.py`)**:
    *   add a high-level functional wrapper for your operation.
7.  **tests**: write comprehensive tests for your new operation.

#### adding new neural network layers (`idrak/nn/`)

1.  create a new python file (e.g., `idrak/nn/my_layer.py`).
2.  define your layer class, inheriting from `idrak.nn.module`.
3.  implement the `__init__`, `forward`, and `reset_parameters` methods.
4.  use existing `idrak.functions` or `idrak.nn` components.
5.  add your new layer to `idrak/nn/__init__.py`.
6.  write tests.

## 💬 community & support

join the discussion, ask questions, and share your idrak creations! (add links to discord, github discussions, etc., if available).

thank you for contributing to idrak. let's build something truly revolutionary.
