#
# Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
# Copyright (C) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
# Copyright (C) The University of Tennessee and The University
#               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
# Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

SUBDIRS = lwca rocm lib
CC      = $(UCX_PERFTEST_CC)

noinst_HEADERS        = api/libperf.h
bin_PROGRAMS          = ucx_perftest
ucx_perftest_SOURCES  = perftest.c
ucx_perftest_CPPFLAGS = $(BASE_CPPFLAGS) $(RTE_CPPFLAGS)
ucx_perftest_CFLAGS   = $(BASE_CFLAGS) $(OPENMP_CFLAGS)
ucx_perftest_LDFLAGS  = $(RTE_LDFLAGS) -Wl,-dynamic-list-data
ucx_perftest_LDADD    = \
	$(abs_top_builddir)/src/uct/libuct.la \
	$(abs_top_builddir)/src/ucp/libucp.la \
	$(abs_top_builddir)/src/ucs/libucs.la \
	lib/libucxperf.la

perftestdir = $(pkgdatadir)/perftest
dist_perftest_DATA = \
	$(top_srcdir)/contrib/ucx_perftest_config/msg_pow2 \
	$(top_srcdir)/contrib/ucx_perftest_config/msg_pow2_large \
	$(top_srcdir)/contrib/ucx_perftest_config/README \
	$(top_srcdir)/contrib/ucx_perftest_config/test_types_uct \
	$(top_srcdir)/contrib/ucx_perftest_config/test_types_ucp \
	$(top_srcdir)/contrib/ucx_perftest_config/transports

if HAVE_MPIRUN
.PHONY: ucx test help

MPI_EXTRA =
MPI_ARGS  = -n 2 -map-by node -mca pml ob1 -mca btl self,tcp,sm $(MPI_EXTRA)

ucx:
	$(MAKE) -C $(top_builddir)

test: ucx ucx_perftest
	$(MPIRUN) $(MPI_ARGS) $(abs_builddir)/ucx_perftest$(EXEEXT) $(TEST_ARGS)

help:
	@echo
	@echo "Targets:"
	@echo "  test       : Run performance test."
	@echo
	@echo "Environment variables:"
	@echo "  MPI_ARGS   : Arguments for mpirun (\"$(MPI_ARGS)\")"
	@echo "  TEST_ARGS  : Arguments for performance test (\"$(TEST_ARGS)\")"
	@echo

endif
