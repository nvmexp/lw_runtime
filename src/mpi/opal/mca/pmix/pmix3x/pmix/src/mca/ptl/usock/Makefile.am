# -*- makefile -*-
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
# Copyright (c) 2012      Los Alamos National Security, Inc.  All rights reserved.
# Copyright (c) 2013-2019 Intel, Inc.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

headers = ptl_usock.h
sources = \
        ptl_usock_component.c \
        ptl_usock.c

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_pmix_ptl_usock_DSO
lib =
lib_sources =
component = mca_ptl_usock.la
component_sources = $(headers) $(sources)
else
lib = libmca_ptl_usock.la
lib_sources = $(headers) $(sources)
component =
component_sources =
endif

mcacomponentdir = $(pmixlibdir)
mcacomponent_LTLIBRARIES = $(component)
mca_ptl_usock_la_SOURCES = $(component_sources)
mca_ptl_usock_la_LDFLAGS = -module -avoid-version
if NEED_LIBPMIX
mca_ptl_usock_la_LIBADD = $(top_builddir)/src/libpmix.la
endif

noinst_LTLIBRARIES = $(lib)
libmca_ptl_usock_la_SOURCES = $(lib_sources)
libmca_ptl_usock_la_LDFLAGS = -module -avoid-version
