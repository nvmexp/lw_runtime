#
# Copyright (c) 2004-2005 The Trustees of Indiana University.
#						 All rights reserved.
# Copyright (c) 2004-2005 The Trustees of the University of Tennessee.
#						 All rights reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#						 University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#						 All rights reserved.
# Copyright (c) 2010      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

r2_sources  = \
	bml_r2.c \
	bml_r2.h \
	bml_r2_component.c \
        bml_r2_ft.c \
        bml_r2_ft.h

dist_ompidata_DATA = help-mca-bml-r2.txt

if MCA_BUILD_ompi_bml_r2_DSO
component_noinst =
component_install = mca_bml_r2.la
else
component_noinst = libmca_bml_r2.la
component_install =
endif


mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_bml_r2_la_SOURCES = $(r2_sources)
mca_bml_r2_la_LDFLAGS = -module -avoid-version
mca_bml_r2_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_bml_r2_la_SOURCES = $(r2_sources)
libmca_bml_r2_la_LDFLAGS = -module -avoid-version
