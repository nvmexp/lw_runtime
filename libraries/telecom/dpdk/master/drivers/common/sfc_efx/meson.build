# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright(c) 2019-2020 Xilinx, Inc.
#
# This software was jointly developed between OKTET Labs (under contract
# for Solarflare) and Solarflare Communications, Inc.

if (arch_subdir != 'x86' or not dpdk_conf.get('RTE_ARCH_64')) and (arch_subdir != 'arm' or not host_machine.cpu_family().startswith('aarch64'))
	build = false
	reason = 'only supported on x86_64 and aarch64'
endif

extra_flags = []

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

subdir('base')
objs = [base_objs]

sources = files(
	'sfc_efx.c',
	'sfc_efx_mcdi.c',
)

includes += include_directories('base')
