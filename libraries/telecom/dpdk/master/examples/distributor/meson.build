# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2017 Intel Corporation

# meson file, for building this example as part of a main DPDK build.
#
# To build this example as a standalone application with an already-installed
# DPDK instance, use 'make'

# require the power library
build = dpdk_conf.has('RTE_LIB_POWER')
if not build
	subdir_done()
endif

allow_experimental_apis = true
deps += ['distributor', 'power']
sources = files(
	'main.c'
)
