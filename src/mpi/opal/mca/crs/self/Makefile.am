#
# Copyright (c) 2004-2007 The Trustees of Indiana University.
#                         All rights reserved.
# Copyright (c) 2004-2005 The Trustees of the University of Tennessee.
#                         All rights reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2010-2015 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

dist_opaldata_DATA = help-opal-crs-self.txt

sources = \
        crs_self.h \
        crs_self_component.c \
        crs_self_module.c

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_crs_self_DSO
component_noinst =
component_install = mca_crs_self.la
else
component_noinst = libmca_crs_self.la
component_install =
endif

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_crs_self_la_SOURCES = $(sources)
mca_crs_self_la_LDFLAGS = -module -avoid-version
mca_crs_self_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_crs_self_la_SOURCES = $(sources)
libmca_crs_self_la_LDFLAGS = -module -avoid-version
