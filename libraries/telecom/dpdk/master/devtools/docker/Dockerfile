FROM lwpu/lwca:10.0-devel-ubuntu16.04
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
  libfile-find-object-perl \
  lsb-release \
  axel \
  autoconf debhelper chrpath pkg-config autotools-dev dpatch lsof m4 graphviz ethtool swig python-libxml2 libnl-3-dev libnl-route-3-200 libnl-route-3-dev libltdl-dev kmod pciutils automake libglib2.0-0 linux-headers-$(uname -r) linux-headers-$(uname -r) libnuma1 libnl-3-200 libmnl0 libmnl-dev \
  libnuma-dev \
  linux-headers-generic \
  lwpu-410-dev \
&& rm -rf /var/lib/apt/lists/*

RUN axel -q -n 10 http://content.mellanox.com/ofed/MLNX_OFED-5.0-2.1.8.0/MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu16.04-x86_64.tgz \
&& tar xf MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu16.04-x86_64.tgz \
&& rm MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu16.04-x86_64.tgz \
&& ./MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu16.04-x86_64/mlnxofedinstall --dpdk --upstream-libs --user-space-only --force --without-fw-update \
&& rm -rf MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu16.04-x86_64
