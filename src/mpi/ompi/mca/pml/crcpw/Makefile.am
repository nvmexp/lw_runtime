#
# Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
#
# Copyright (c) 2010      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

crcpw_sources  = \
	pml_crcpw.h \
	pml_crcpw_component.c \
	pml_crcpw_module.c

if MCA_BUILD_ompi_pml_crcpw_DSO
component_noinst =
component_install = mca_pml_crcpw.la
else
component_noinst = libmca_pml_crcpw.la
component_install =
endif


mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_pml_crcpw_la_SOURCES = $(crcpw_sources)
mca_pml_crcpw_la_LDFLAGS = -module -avoid-version
mca_pml_crcpw_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_pml_crcpw_la_SOURCES = $(crcpw_sources)
libmca_pml_crcpw_la_LDFLAGS = -module -avoid-version
