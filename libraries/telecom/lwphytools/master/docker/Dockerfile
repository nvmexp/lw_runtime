FROM lwpu/lwca:10.1-devel-ubuntu16.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
  libfile-find-object-perl \
  lsb-release \
  axel wget \
  autoconf debhelper chrpath pkg-config autotools-dev dpatch lsof m4 graphviz ethtool swig python-libxml2 libnl-3-dev libnl-route-3-200 libnl-route-3-dev libltdl-dev kmod pciutils automake libglib2.0-0 linux-headers-4.4.0-141-generic linux-headers-4.4.0-142-generic libnuma1 libnl-3-200 libmnl0 libmnl-dev \
  libnuma-dev \
  linux-headers-generic \
  libjsoncpp-dev zlib1g zlib1g-dev \
  udev unzip lwrl \
&& rm -rf /var/lib/apt/lists/*


RUN axel -q -n 10 http://content.mellanox.com/ofed/MLNX_OFED-4.7-1.0.0.1/MLNX_OFED_LINUX-4.7-1.0.0.1-ubuntu16.04-x86_64.tgz              \
&& tar xf MLNX_OFED_LINUX-4.7-1.0.0.1-ubuntu16.04-x86_64.tgz                                                                             \
&& rm MLNX_OFED_LINUX-4.7-1.0.0.1-ubuntu16.04-x86_64.tgz

RUN ./MLNX_OFED_LINUX-4.7-1.0.0.1-ubuntu16.04-x86_64/mlnxofedinstall --dpdk --upstream-libs --user-space-only --force --without-fw-update \
&& rm -rf MLNX_OFED_LINUX-4.7-1.0.0.1-ubuntu16.04-x86_64

RUN wget  https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz                                    \
&& tar -xzf hdf5-1.10.5.tar.gz                                                                                                           \
&& cd hdf5-1.10.5                                                                                                                        \
&& ./configure --prefix=/usr/local --enable-cxx --enable-build-mode=production                                                           \
&& make install
