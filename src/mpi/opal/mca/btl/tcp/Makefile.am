#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2009-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2013      LWPU Corporation.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

dist_opaldata_DATA = help-mpi-btl-tcp.txt

sources = \
    btl_tcp.c \
    btl_tcp.h \
    btl_tcp_addr.h \
    btl_tcp_component.c \
    btl_tcp_endpoint.c \
    btl_tcp_endpoint.h \
    btl_tcp_frag.c \
    btl_tcp_frag.h \
    btl_tcp_hdr.h \
    btl_tcp_proc.c \
    btl_tcp_proc.h \
    btl_tcp_ft.c \
    btl_tcp_ft.h

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_btl_tcp_DSO
lib =
lib_sources =
component = mca_btl_tcp.la
component_sources = $(sources)
else
lib = libmca_btl_tcp.la
lib_sources = $(sources)
component =
component_sources =
endif

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component)
mca_btl_tcp_la_SOURCES = $(component_sources)
mca_btl_tcp_la_LDFLAGS = -module -avoid-version
if OPAL_lwda_support
mca_btl_tcp_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la \
    $(OPAL_TOP_BUILDDIR)/opal/mca/common/lwca/lib@OPAL_LIB_PREFIX@mca_common_lwda.la
endif

noinst_LTLIBRARIES = $(lib)
libmca_btl_tcp_la_SOURCES = $(lib_sources)
libmca_btl_tcp_la_LDFLAGS = -module -avoid-version
