FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

# Use correct ENV syntax
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gpg \
    wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" | tee /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install -y cmake \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
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
