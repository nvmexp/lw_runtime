# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2019 Intel Corporation

sources = files('rte_eth_af_xdp.c')

bpf_dep = dependency('libbpf', required: false)
if not bpf_dep.found()
	bpf_dep = cc.find_library('bpf', required: false)
endif

if bpf_dep.found() and cc.has_header('bpf/xsk.h') and cc.has_header('linux/if_xdp.h')
	ext_deps += bpf_dep
	bpf_ver_dep = dependency('libbpf', version : '>=0.2.0', required: false)
	if bpf_ver_dep.found()
		dpdk_conf.set('RTE_LIBRTE_AF_XDP_PMD_SHARED_UMEM', 1)
	endif
else
	build = false
	reason = 'missing dependency, "libbpf"'
endif
