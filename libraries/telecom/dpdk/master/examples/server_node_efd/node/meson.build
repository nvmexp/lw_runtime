# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2019 Intel Corporation

# meson file, for building this example as part of a main DPDK build.
#
# To build this example as a standalone application with an already-installed
# DPDK instance, use 'make'

name = 'efd_node'

allow_experimental_apis = true
deps += ['hash']
sources += files('node.c')
includes += include_directories('../shared')
