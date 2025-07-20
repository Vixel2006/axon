PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
CUDA = /opt/cuda/bin/nvcc
PYTEST = $(VENV)/bin/pytest
FLAKE8 = $(VENV)/bin/flake8
BLACK = $(VENV)/bin/black
CPPLINT = cpplint
CLANG_FORMAT = clang-format
CMAKE = cmake
BUILD_DIR = build
SRC_DIR = src
TEST_DIR = tests
CMAKE_FLAGS = -D CMAKE_CUDA_COMPILER=$(CUDA)
SOURCES = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
MODULE = nawah.cpython-313-x86_64-linux-gnu.so

all: prepare init build test lint style

prepare:
	rm -rf build

init:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install pytest flake8 black pybind11

build: $(BUILD_DIR)/$(MODULE)

$(BUILD_DIR)/$(MODULE):
	mkdir -p $(BUILD_DIR)
	$(CMAKE) -S . -B $(BUILD_DIR) $(CMAKE_FLAGS) && cd $(BUILD_DIR) && $(MAKE)
	cd ..
	pip install -e . --break-system-packages

test:
	$(PYTEST) $(TEST_DIR) -v

lint:
	$(CPPLINT) $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h 2> cpplint_errors.txt || true
	$(FLAKE8) $(SRC_DIR)/*.py $(TEST_DIR)/*.py

style:
	$(CLANG_FORMAT) -i $(SRC_DIR)/*.cpp $(SRC_DIR)/*.h
	$(BLACK) $(SRC_DIR)/*.py $(TEST_DIR)/*.py

clean:
	rm -rf $(BUILD_DIR) $(VENV) $(MODULE) cpplint_errors.txt

.PHONY: all init build test lint style clean
