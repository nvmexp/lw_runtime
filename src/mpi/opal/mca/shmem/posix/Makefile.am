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
# Copyright (c) 2010      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2010-2011 Los Alamos National Security, LLC.
#                         All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
        shmem_posix.h \
        shmem_posix_common_utils.h shmem_posix_common_utils.c \
        shmem_posix_component.c \
        shmem_posix_module.c

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_shmem_posix_DSO
component_noinst =
component_install = mca_shmem_posix.la
else
component_noinst = libmca_shmem_posix.la
component_install =
endif

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_shmem_posix_la_SOURCES = $(sources)
mca_shmem_posix_la_LDFLAGS = -module -avoid-version
mca_shmem_posix_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_shmem_posix_la_SOURCES = $(sources)
libmca_shmem_posix_la_LDFLAGS = -module -avoid-version

# help file
dist_opaldata_DATA = help-opal-shmem-posix.txt
