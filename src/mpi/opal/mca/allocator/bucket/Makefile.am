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
# Copyright (c) 2010      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
        allocator_bucket.c \
        allocator_bucket_alloc.c \
        allocator_bucket_alloc.h

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_allocator_bucket_DSO
component_noinst =
component_install = mca_allocator_bucket.la
else
component_noinst = libmca_allocator_bucket.la
component_install =
endif

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_allocator_bucket_la_SOURCES = $(sources)
mca_allocator_bucket_la_LDFLAGS = -module -avoid-version
mca_allocator_bucket_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_allocator_bucket_la_SOURCES = $(sources)
libmca_allocator_bucket_la_LDFLAGS = -module -avoid-version

