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
# Copyright (c) 2008-2018 University of Houston. All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_fs_lustre_DSO
component_noinst =
component_install = mca_fs_lustre.la
else
component_noinst = libmca_fs_lustre.la
component_install =
endif

# Source files

fs_lustre_sources = \
        fs_lustre.h \
        fs_lustre.c \
        fs_lustre_component.c \
        fs_lustre_file_open.c 

AM_CPPFLAGS = $(fs_lustre_CPPFLAGS)

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_fs_lustre_la_SOURCES = $(fs_lustre_sources)
mca_fs_lustre_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la \
	$(fs_lustre_LIBS)
mca_fs_lustre_la_LDFLAGS = -module -avoid-version $(fs_lustre_LDFLAGS)

noinst_LTLIBRARIES = $(component_noinst)
libmca_fs_lustre_la_SOURCES = $(fs_lustre_sources)
libmca_fs_lustre_la_LIBADD = $(fs_lustre_LIBS)
libmca_fs_lustre_la_LDFLAGS = -module -avoid-version $(fs_lustre_LDFLAGS)
