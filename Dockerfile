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


WORKDIR /nawah

COPY . .


RUN python3 -m venv venv
ENV PATH="/nawah/venv/bin:$PATH"

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install requirements.txt

CMD ["make", "test"]
