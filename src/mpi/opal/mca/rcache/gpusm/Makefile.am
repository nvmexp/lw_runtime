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
# Copyright (c) 2010-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2012      LWPU Corporation.  All rights reserved.
# Copyright (c) 2015      Los Alamos National Security, LLC. All rights
#                         reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = $(rcache_gpusm_CPPFLAGS)

sources = \
    rcache_gpusm_module.c \
    rcache_gpusm_component.c

if WANT_INSTALL_HEADERS
opaldir = $(opalincludedir)/$(subdir)
opal_HEADERS = rcache_gpusm.h
endif

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_rcache_gpusm_DSO
component_noinst =
component_install = mca_rcache_gpusm.la
else
component_noinst = libmca_rcache_gpusm.la
component_install =
endif

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_rcache_gpusm_la_SOURCES = $(sources)
mca_rcache_gpusm_la_LDFLAGS = -module -avoid-version
mca_rcache_gpusm_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la \
	$(rcache_gpusm_LIBS)
if OPAL_lwda_support
mca_rcache_gpusm_la_LIBADD += \
    $(OPAL_TOP_BUILDDIR)/opal/mca/common/lwca/lib@OPAL_LIB_PREFIX@mca_common_lwda.la
endif

noinst_LTLIBRARIES = $(component_noinst)
libmca_rcache_gpusm_la_SOURCES = $(sources)
libmca_rcache_gpusm_la_LDFLAGS = -module -avoid-version
libmca_rcache_gpusm_la_LIBADD = $(rcache_gpusm_LIBS)
