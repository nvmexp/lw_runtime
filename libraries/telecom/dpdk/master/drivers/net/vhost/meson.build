# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2018 Intel Corporation

build = dpdk_conf.has('RTE_LIB_VHOST')
reason = 'missing dependency, DPDK vhost library'
sources = files('rte_eth_vhost.c')
headers = files('rte_eth_vhost.h')
deps += 'vhost'
