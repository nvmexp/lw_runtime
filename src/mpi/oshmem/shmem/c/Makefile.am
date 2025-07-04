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


if OSHMEM_PROFILING
    SUBDIRS = profile
endif

OSHMEM_AUX_SOURCES = \
	shmem_lock.c

OSHMEM_API_SOURCES = \
	shmem_init.c \
	shmem_finalize.c \
	shmem_free.c \
	shmem_alloc.c \
	shmem_realloc.c \
	shmem_align.c \
	shmem_query.c \
	shmem_p.c \
	shmem_context.c \
	shmem_put.c \
	shmem_g.c \
	shmem_get.c \
	shmem_broadcast.c \
	shmem_collect.c \
	shmem_alltoall.c \
	shmem_ptr.c \
	shmem_pe_accessible.c \
	shmem_addr_accessible.c \
	shmem_barrier.c \
	shmem_sync.c \
	shmem_fence.c \
	shmem_quiet.c \
	shmem_wait.c \
	shmem_iget.c \
	shmem_iput.c \
	shmem_get_nb.c \
	shmem_put_nb.c \
	shmem_udcflush.c \
	shmem_udcflush_line.c \
	shmem_set_cache_ilw.c \
	shmem_set_cache_line_ilw.c \
	shmem_clear_cache_ilw.c \
	shmem_clear_cache_line_ilw.c \
	shmem_reduce.c \
	shmem_swap.c \
	shmem_set.c \
	shmem_cswap.c \
	shmem_fadd.c \
	shmem_fand.c \
	shmem_for.c \
	shmem_fxor.c \
	shmem_fetch.c \
	shmem_finc.c \
	shmem_add.c \
	shmem_and.c \
	shmem_or.c \
	shmem_xor.c \
	shmem_inc.c \
	shmem_clear_lock.c \
	shmem_set_lock.c \
	shmem_test_lock.c \
	shmem_global_exit.c \
	shmem_info.c

AM_CPPFLAGS     = -DOSHMEM_PROFILING=0

noinst_LTLIBRARIES =
if PROJECT_OSHMEM
# Only build if we're building OSHMEM
noinst_LTLIBRARIES += liboshmem_c.la
endif

headers =

# In case when OSHMEM_PROFILING enabled
# We build api from c/profile folder
liboshmem_c_la_SOURCES = $(OSHMEM_AUX_SOURCES)
if ! OSHMEM_PROFILING
liboshmem_c_la_SOURCES += $(OSHMEM_API_SOURCES)
endif

if PROJECT_OSHMEM
if WANT_INSTALL_HEADERS
oshmemdir = $(oshmemincludedir)/$(subdir)
oshmem_HEADERS = $(headers)
endif
endif
