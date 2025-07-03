# Use 10.1-ubuntu16.04 as base
FROM lwpu/lwca:10.0-devel-ubuntu16.04

# Developer
MAINTAINER Chenhan D. Yu <chenhany@lwpu.com>

# Set the working directory to /app
WORKDIR /lwTensor

# Copy the current directory contents into the container at /app
COPY . /lwTensor

# Remove the build directory
RUN rm -rf build

# Install dependencies through apt-get.
RUN apt-get update && apt-get install -y -qq \
    git \
    wget \
    vim \
    build-essential \
    ca-certificates \
    gcovr \
    libnuma-dev \
    doxygen \
    doxygen-latex

# Install CMAKE-3.14
ELW CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v3.14.3/cmake-3.14.3-Linux-x86_64.tar.gz" 
ELW CMAKE_ROOT="/lwTensor/.dependencies/cmake" 
RUN mkdir -p $CMAKE_ROOT 
RUN wget --no-check-certificate --quiet -O - $CMAKE_URL | tar --strip-components=1 -xz -C $CMAKE_ROOT 

# Environment variables
ELW LWDA_ROOT="/usr/local/lwca"
ELW PATH="${CMAKE_ROOT}/bin:${PATH}"

# Build lwTensor (develop build)
RUN mkdir build && \
    cd build && \
    cmake .. && \
    make install -j4
