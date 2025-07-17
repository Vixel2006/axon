FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND = noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    cmake \
    g++ \
    clang-format \
    git \
    && rm -rf /var/bin/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pytest flake8 black pybind11 cpplint

WORKDIR /nawah

COPY . .

CMD ["make"]
