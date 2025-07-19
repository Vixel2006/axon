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

RUN make init
RUN make build

CMD ["make", "test"]
