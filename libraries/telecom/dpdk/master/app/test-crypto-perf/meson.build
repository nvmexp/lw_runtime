# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2018 Intel Corporation

sources = files('cperf_ops.c',
		'cperf_options_parsing.c',
		'cperf_test_common.c',
		'cperf_test_latency.c',
		'cperf_test_pmd_cyclecount.c',
		'cperf_test_throughput.c',
		'cperf_test_vector_parsing.c',
		'cperf_test_vectors.c',
		'cperf_test_verify.c',
		'main.c')
deps += ['cryptodev', 'net', 'security']
if dpdk_conf.has('RTE_CRYPTO_SCHEDULER')
	deps += 'crypto_scheduler'
endif
