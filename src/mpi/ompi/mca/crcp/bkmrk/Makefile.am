#
# Copyright (c) 2004-2007 The Trustees of Indiana University.
#                         All rights reserved.
# Copyright (c) 2004-2005 The Trustees of the University of Tennessee.
#                         All rights reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2010-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
        crcp_bkmrk.h \
        crcp_bkmrk_pml.h \
        crcp_bkmrk_component.c \
        crcp_bkmrk_module.c \
        crcp_bkmrk_pml.c

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_crcp_bkmrk_DSO
component_noinst =
component_install = mca_crcp_bkmrk.la
else
component_noinst = libmca_crcp_bkmrk.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_crcp_bkmrk_la_SOURCES = $(sources)
mca_crcp_bkmrk_la_LDFLAGS = -module -avoid-version
mca_crcp_bkmrk_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_crcp_bkmrk_la_SOURCES = $(sources)
libmca_crcp_bkmrk_la_LDFLAGS = -module -avoid-version
