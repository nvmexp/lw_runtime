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
# Copyright (c) 2008      UT-Battelle, LLC
# Copyright (c) 2008-2010 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = $(ras_alps_CPPFLAGS)

dist_ortedata_DATA = help-ras-alps.txt \
		  ras-alps-command.sh

install-data-hook:
	chmod +x $(ortedatadir)/ras-alps-command.sh

sources = \
        ras_alps.h \
        ras_alps_component.c \
        ras_alps_module.c


# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_orte_ras_alps_DSO
lib =
lib_sources =
component = mca_ras_alps.la
component_sources = $(sources)
else
lib = libmca_ras_alps.la
lib_sources = $(sources)
component =
component_sources =
endif

mcacomponentdir = $(ortelibdir)
mcacomponent_LTLIBRARIES = $(component)
mca_ras_alps_la_SOURCES = $(component_sources)
mca_ras_alps_la_LDFLAGS = -module -avoid-version $(ras_alps_LDFLAGS)
mca_ras_alps_la_LIBADD = $(top_builddir)/orte/lib@ORTE_LIB_PREFIX@open-rte.la \
	$(ras_alps_LIBS)
mca_ras_alps_la_CPPFLAGS = $(ras_alps_CPPFLAGS)

noinst_LTLIBRARIES = $(lib)
libmca_ras_alps_la_SOURCES = $(lib_sources)
libmca_ras_alps_la_LDFLAGS = -module -avoid-version $(ras_alps_LDFLAGS)
libmca_ras_alps_la_LIBADD = $(ras_alps_LIBS)
libmca_ras_alps_la_CPPFLAGS = $(ras_alps_CPPFLAGS)
