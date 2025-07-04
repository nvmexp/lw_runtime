#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2006      Sandia National Laboratories. All rights
#                         reserved.
# Copyright (c) 2010-2019 Cisco Systems, Inc.  All rights reserved
# Copyright (c) 2015      Intel, Inc. All rights reserved.
# Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
# Copyright (c) 2017      Los Alamos National Security, LLC.  All rights
#                         reserved.
# Copyright (c) 2019      Research Organization for Information Science
#                         and Technology (RIST).  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = -DBTL_IN_OPAL=1 $(opal_ofi_CPPFLAGS) -DOMPI_LIBMPI_NAME=\"$(OMPI_LIBMPI_NAME)\"

EXTRA_DIST = README.txt README.test

dist_opaldata_DATA = \
    help-mpi-btl-usnic.txt

test_sources = \
    test/btl_usnic_component_test.h

sources = \
    btl_usnic_compat.h \
    btl_usnic_compat.c \
    btl_usnic_module.c \
    btl_usnic_module.h \
    btl_usnic.h \
    btl_usnic_ack.c \
    btl_usnic_ack.h \
    btl_usnic_component.c \
    btl_usnic_connectivity.h \
    btl_usnic_cclient.c \
    btl_usnic_cagent.c \
    btl_usnic_endpoint.c \
    btl_usnic_endpoint.h \
    btl_usnic_frag.c \
    btl_usnic_frag.h \
    btl_usnic_hwloc.c \
    btl_usnic_hwloc.h \
    btl_usnic_map.c \
    btl_usnic_mca.c \
    btl_usnic_proc.c \
    btl_usnic_proc.h \
    btl_usnic_recv.c \
    btl_usnic_recv.h \
    btl_usnic_send.c \
    btl_usnic_send.h \
    btl_usnic_stats.h \
    btl_usnic_stats.c \
    btl_usnic_util.c \
    btl_usnic_util.h \
    btl_usnic_test.c \
    btl_usnic_test.h \
    $(test_sources)

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_btl_usnic_DSO
lib =
lib_sources =
component = mca_btl_usnic.la
component_sources = $(sources)
else
lib = libmca_btl_usnic.la
lib_sources = $(sources)
component =
component_sources =
endif

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component)
mca_btl_usnic_la_SOURCES = $(component_sources)
mca_btl_usnic_la_LDFLAGS = \
        $(opal_btl_usnic_LDFLAGS) \
        $(opal_ofi_LDFLAGS) \
        -module -avoid-version
mca_btl_usnic_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la \
        $(opal_ofi_LIBS)

noinst_LTLIBRARIES = $(lib)
libmca_btl_usnic_la_SOURCES = $(lib_sources)
libmca_btl_usnic_la_LDFLAGS = \
        $(opal_btl_usnic_LDFLAGS) \
        $(opal_ofi_LDFLAGS) \
        -module -avoid-version
libmca_btl_usnic_la_LIBADD = $(opal_ofi_LIBS)

if OPAL_BTL_USNIC_BUILD_UNIT_TESTS
usnic_btl_run_tests_CPPFLAGS = $(AM_CPPFLAGS) \
    -DBTL_USNIC_RUN_TESTS_SYMBOL=\"opal_btl_usnic_run_tests\"
usnic_btl_run_tests_SOURCES = test/usnic_btl_run_tests.c
usnic_btl_run_tests_LDADD = -ldl
bin_PROGRAMS = usnic_btl_run_tests
endif OPAL_BTL_USNIC_BUILD_UNIT_TESTS
