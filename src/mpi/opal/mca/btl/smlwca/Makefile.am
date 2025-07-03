#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2009 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2009-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2012      LWPU Corporation.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

dist_opaldata_DATA = help-mpi-btl-smlwda.txt

libmca_btl_smlwda_la_sources = \
    btl_smlwda.c \
    btl_smlwda.h \
    btl_smlwda_component.c \
    btl_smlwda_endpoint.h \
    btl_smlwda_fifo.h \
    btl_smlwda_frag.c \
    btl_smlwda_frag.h

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_btl_smlwda_DSO
component_noinst =
component_install = mca_btl_smlwda.la
else
component_noinst = libmca_btl_smlwda.la
component_install =
endif

# See opal/mca/common/lwca/Makefile.am for an explanation of
# libmca_common_sm.la.

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_btl_smlwda_la_SOURCES = $(libmca_btl_smlwda_la_sources)
mca_btl_smlwda_la_LDFLAGS = -module -avoid-version
mca_btl_smlwda_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la \
    $(OPAL_TOP_BUILDDIR)/opal/mca/common/sm/lib@OPAL_LIB_PREFIX@mca_common_sm.la
mca_btl_smlwda_la_CPPFLAGS = $(btl_smlwda_CPPFLAGS)
if OPAL_lwda_support
mca_btl_smlwda_la_LIBADD += \
    $(OPAL_TOP_BUILDDIR)/opal/mca/common/lwca/lib@OPAL_LIB_PREFIX@mca_common_lwda.la
endif

noinst_LTLIBRARIES = $(component_noinst)
libmca_btl_smlwda_la_SOURCES = $(libmca_btl_smlwda_la_sources)
libmca_btl_smlwda_la_LDFLAGS = -module -avoid-version
libmca_btl_smlwda_la_CPPFLAGS = $(btl_smlwda_CPPFLAGS)
