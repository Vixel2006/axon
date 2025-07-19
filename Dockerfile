FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

# Use correct ENV syntax
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    cmake \
    g++ \
    clang-format \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /nawah

# Copy project files
COPY . .

# Create and activate virtual environment
RUN python3 -m venv /nawah/venv \
    && . /nawah/venv/bin/activate \
    && pip install --upgrade pip \
# Ensure virtual environment is used in CMD
ENV PATH="/nawah/venv/bin:$PATH"

RUN make init
RUN make build

CMD ["make", "test"]
