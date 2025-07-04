# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright(c) 2019-2020 Xilinx, Inc.
# Copyright(c) 2016-2019 Solarflare Communications Inc.
#
# This software was jointly developed between OKTET Labs (under contract
# for Solarflare) and Solarflare Communications, Inc.

if (arch_subdir != 'x86' or not dpdk_conf.get('RTE_ARCH_64')) and (arch_subdir != 'arm' or not host_machine.cpu_family().startswith('aarch64'))
	build = false
	reason = 'only supported on x86_64 and aarch64'
endif

extra_flags = []

# Strict-aliasing rules are violated by rte_eth_link to uint64_t casts
extra_flags += '-Wno-strict-aliasing'

# Enable more warnings
extra_flags += [
	'-Wdisabled-optimization'
]

# Compiler and version dependent flags
extra_flags += [
	'-Waggregate-return',
	'-Wbad-function-cast'
]

foreach flag: extra_flags
	if cc.has_argument(flag)
		cflags += flag
	endif
endforeach

deps += ['common_sfc_efx', 'bus_pci']
sources = files(
	'sfc_ethdev.c',
	'sfc_kvargs.c',
	'sfc.c',
	'sfc_mcdi.c',
	'sfc_sriov.c',
	'sfc_intr.c',
	'sfc_ev.c',
	'sfc_port.c',
	'sfc_rx.c',
	'sfc_tx.c',
	'sfc_tso.c',
	'sfc_filter.c',
	'sfc_switch.c',
	'sfc_mae.c',
	'sfc_flow.c',
	'sfc_dp.c',
	'sfc_ef10_rx.c',
	'sfc_ef10_essb_rx.c',
	'sfc_ef10_tx.c',
	'sfc_ef100_rx.c',
	'sfc_ef100_tx.c',
)
