FROM lwpu/lwca:10.1-devel-ubuntu16.04
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
  libfile-find-object-perl \
  lsb-release \
  axel \
  autoconf debhelper chrpath pkg-config autotools-dev dpatch lsof m4 graphviz ethtool swig python-libxml2 libnl-3-dev libnl-route-3-200 libnl-route-3-dev libltdl-dev kmod pciutils automake libglib2.0-0 linux-headers-4.4.0-141-generic linux-headers-4.4.0-142-generic libnuma1 libnl-3-200 libmnl0 libmnl-dev \
  libnuma-dev \
  linux-headers-generic \
  lwpu-410-dev \
  libhdf5-dev libhdf5-serial-dev libhdf5-10 \
  unzip lwrl wget doxygen git \
&& rm -rf /var/lib/apt/lists/*

RUN cd /usr/local \
&& wget -q https://github.com/Kitware/CMake/releases/download/v3.14.0-rc4/cmake-3.14.0-rc4-Linux-x86_64.tar.gz \
&& tar xf cmake-3.14.0-rc4-Linux-x86_64.tar.gz --strip-components=1 \
&& rm cmake-3.14.0-rc4-Linux-x86_64.tar.gz

