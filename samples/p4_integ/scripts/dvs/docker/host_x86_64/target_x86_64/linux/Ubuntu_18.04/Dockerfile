
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0  \
    libgl1 \
    libglx0  \
    libegl1 \
    libgles2 && \
    rm -rf /var/lib/apt/lists/*

COPY 10_lwidia.json /usr/share/glvnd/egl_vendor.d/10_lwidia.json

FROM ubuntu:18.04 as vulkan-khronos

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libegl1-mesa-dev \
    libwayland-dev \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libxrandr-dev \
    python3 \
    python3-distutils \
    wget && \
    rm -rf /var/lib/apt/lists/*

ARG VULKAN_VERSION
# Download and compile vulkan components
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    git clone https://github.com/KhronosGroup/Vulkan-ValidationLayers.git /opt/vulkan && \
    cd /opt/vulkan && git checkout "${VULKAN_VERSION}" && \
    mkdir build && cd build && ../scripts/update_deps.py && \
    cmake -C helper.cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . && make install && ldconfig && \
    mkdir -p /usr/local/lib && cp -a Vulkan-Loader/build/install/lib/* /usr/local/lib && \
    mkdir -p /usr/local/include/vulkan && cp -r Vulkan-Headers/build/install/include/vulkan/* /usr/local/include/vulkan && \
    mkdir -p /usr/local/share/vulkan/registry && \
    cp -r Vulkan-Headers/build/install/share/vulkan/registry/* /usr/local/share/vulkan/registry && \
    rm -rf /opt/vulkan

ARG LWDA_VERSION
FROM lwpu/lwdagl:10.2-base-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libwayland-dev \
    libxrandr-dev \
    build-essential \
    freeglut3-dev \
    pkg-config  \
    libxml2-utils \
    libglfw3 \
    libglfw3-dev \
    libegl1-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=vulkan-khronos /usr/local/bin /usr/local/bin
COPY --from=vulkan-khronos /usr/local/lib /usr/local/lib
COPY --from=vulkan-khronos /usr/local/include/vulkan /usr/local/include/vulkan
COPY --from=vulkan-khronos /usr/local/share/vulkan /usr/local/share/vulkan

COPY lwidia_icd.json /etc/vulkan/icd.d/lwidia_icd.json

# remove the lwca 10.2 toolkit from the base image  lwpu/lwdagl:10.2-base-ubuntu18.04
RUN rm -rf /usr/local/lwca-10.2
RUN rm -rf /usr/local/lwca

RUN groupadd -g 1000 lwdauser
RUN useradd -d /home/lwdauser -s /bin/bash -m lwdauser -u 1000 -g 1000
USER lwdauser
ELW HOME /home/lwdauser
ELW VULKAN_SDK /usr/local

CMD ["/bin/bash"]