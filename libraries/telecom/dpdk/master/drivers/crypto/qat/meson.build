# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2017-2018 Intel Corporation

# this does not build the QAT driver, instead that is done in the compression
# driver which comes later. Here we just add our sources files to the list
build = false
reason = '' # sentinal value to suppress printout
dep = dependency('libcrypto', required: false)
qat_includes += include_directories('.')
qat_deps += 'cryptodev'
qat_deps += 'net'
qat_deps += 'security'
if dep.found()
	# Add our sources files to the list
	qat_sources += files('qat_sym_pmd.c',
			     'qat_sym.c',
			     'qat_sym_session.c',
			     'qat_sym_hw_dp.c',
			     'qat_asym_pmd.c',
			     'qat_asym.c')
	qat_ext_deps += dep
	qat_cflags += '-DBUILD_QAT_SYM'
	qat_cflags += '-DBUILD_QAT_ASYM'
endif
