# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2018 Intel Corporation

subdir('base')
objs = [base_objs]

sources = files(
	'ice_ethdev.c',
	'ice_rxtx.c',
	'ice_switch_filter.c',
	'ice_generic_flow.c',
	'ice_fdir_filter.c',
	'ice_hash.c',
	'ice_acl_filter.c'
	)

deps += ['hash', 'net', 'common_iavf']
includes += include_directories('base', '../../common/iavf')

if arch_subdir == 'x86'
	sources += files('ice_rxtx_vec_sse.c')

	# compile AVX2 version if either:
	# a. we have AVX supported in minimum instruction set baseline
	# b. it's not minimum instruction set, but supported by compiler
	if cc.get_define('__AVX2__', args: machine_args) != ''
		sources += files('ice_rxtx_vec_avx2.c')
	elif cc.has_argument('-mavx2')
		ice_avx2_lib = static_library('ice_avx2_lib',
				'ice_rxtx_vec_avx2.c',
				dependencies: [static_rte_ethdev,
					static_rte_kvargs, static_rte_hash],
				include_directories: includes,
				c_args: [cflags, '-mavx2'])
		objs += ice_avx2_lib.extract_objects('ice_rxtx_vec_avx2.c')
	endif

	ice_avx512_cpu_support = (
		cc.get_define('__AVX512F__', args: machine_args) != '' and
		cc.get_define('__AVX512BW__', args: machine_args) != '')

	ice_avx512_cc_support = (
		not machine_args.contains('-mno-avx512f') and
		cc.has_argument('-mavx512f') and
		cc.has_argument('-mavx512bw'))

	if ice_avx512_cpu_support == true or ice_avx512_cc_support == true
		cflags += ['-DCC_AVX512_SUPPORT']
		avx512_args = [cflags, '-mavx512f', '-mavx512bw']
		if cc.has_argument('-march=skylake-avx512')
			avx512_args += '-march=skylake-avx512'
		endif
		ice_avx512_lib = static_library('ice_avx512_lib',
				'ice_rxtx_vec_avx512.c',
				dependencies: [static_rte_ethdev,
					static_rte_kvargs, static_rte_hash],
				include_directories: includes,
				c_args: avx512_args)
		objs += ice_avx512_lib.extract_objects('ice_rxtx_vec_avx512.c')
	endif
endif

sources += files('ice_dcf.c',
		 'ice_dcf_ethdev.c',
		 'ice_dcf_parent.c')

headers = files('rte_pmd_ice.h')
