#
# Copyright (c) 2013-2016 Mellanox Technologies, Inc.
#                         All rights reserved
# Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

include $(top_srcdir)/Makefile.ompi-rules

#
# OSHMEM_PROFILING flag is enabled when we want our shmem_* symbols
# to be replaced by pshmem_*. In other words, this flag decides
# whether "profile/defines.h" is included or not. "profile/defines.h"
# replaces all shmem_* symbols with pshmem_* symbols. In this directory
# we definately need it to be 1.
#
AM_CPPFLAGS = -DOSHMEM_PROFILING=1

noinst_LTLIBRARIES =
if PROJECT_OSHMEM
# Only build if we're building OSHMEM
noinst_LTLIBRARIES += liboshmem_c_pshmem.la
endif

headers = defines.h


OSHMEM_API_SOURCES = \
	pshmem_init.c \
	pshmem_finalize.c \
	pshmem_free.c \
	pshmem_alloc.c \
	pshmem_realloc.c \
	pshmem_align.c \
	pshmem_query.c \
	pshmem_p.c \
	pshmem_context.c \
	pshmem_put.c \
	pshmem_g.c \
	pshmem_get.c \
	pshmem_alltoall.c \
	pshmem_broadcast.c \
	pshmem_collect.c \
	pshmem_ptr.c \
	pshmem_pe_accessible.c \
	pshmem_addr_accessible.c \
	pshmem_barrier.c \
	pshmem_sync.c \
	pshmem_fence.c \
	pshmem_quiet.c \
	pshmem_wait.c \
	pshmem_iget.c \
	pshmem_iput.c \
	pshmem_get_nb.c \
	pshmem_put_nb.c \
	pshmem_udcflush.c \
	pshmem_udcflush_line.c \
	pshmem_set_cache_ilw.c \
	pshmem_set_cache_line_ilw.c \
	pshmem_clear_cache_ilw.c \
	pshmem_clear_cache_line_ilw.c \
	pshmem_reduce.c \
	pshmem_swap.c \
	pshmem_set.c \
	pshmem_cswap.c \
	pshmem_fadd.c \
	pshmem_fand.c \
	pshmem_for.c \
	pshmem_fxor.c \
	pshmem_fetch.c \
	pshmem_finc.c \
	pshmem_add.c \
	pshmem_and.c \
	pshmem_or.c \
	pshmem_xor.c \
	pshmem_inc.c \
	pshmem_clear_lock.c \
	pshmem_set_lock.c \
	pshmem_test_lock.c \
	pshmem_global_exit.c \
	pshmem_info.c

nodist_liboshmem_c_pshmem_la_SOURCES = \
	$(OSHMEM_API_SOURCES)

#
# Sym link in the sources from the real OSHMEM directory
#
$(nodist_liboshmem_c_pshmem_la_SOURCES):
	$(OMPI_V_LN_S) if test ! -r $@ ; then \
		pname=`echo $@ | cut -b '2-'` ; \
		$(LN_S) $(top_srcdir)/oshmem/shmem/c/$$pname $@ ; \
	fi

if PROJECT_OSHMEM
if WANT_INSTALL_HEADERS
oshmemdir = $(oshmemincludedir)/$(subdir)
oshmem_HEADERS = $(headers)
endif
endif

# These files were created by targets above

MAINTAINERCLEANFILES = $(nodist_liboshmem_c_pshmem_la_SOURCES)

# Don't want these targets in here

tags-relwrsive:
tags:
TAGS:
GTAGS:
ID:
