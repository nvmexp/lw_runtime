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

libmca_btl_self_la_sources = \
    btl_self.c \
    btl_self.h \
    btl_self_component.c \
    btl_self_frag.c \
    btl_self_frag.h

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_btl_self_DSO
component_noinst =
component_install = mca_btl_self.la
else
component_noinst = libmca_btl_self.la
component_install =
endif

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_btl_self_la_SOURCES = $(libmca_btl_self_la_sources)
mca_btl_self_la_LDFLAGS = -module -avoid-version
mca_btl_self_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_btl_self_la_SOURCES = $(libmca_btl_self_la_sources)
libmca_btl_self_la_LDFLAGS = -module -avoid-version
