# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2019 Cesnet
# Copyright(c) 2019 Netcope Technologies, a.s. <info@netcope.com>
# All rights reserved.

dep = dependency('netcope-common', required: false)
reason = 'missing dependency, "libnfb"'
build = dep.found()
ext_deps += dep

sources = files('nfb_rx.c', 'nfb_tx.c', 'nfb_stats.c', 'nfb_ethdev.c', 'nfb_rxmode.c')
