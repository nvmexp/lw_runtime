#
# Copyright (c) 2013-2015 Sandia National Laboratories. All rights reserved.
# Copyright (c) 2015      Bull SAS.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

local_sources = \
	coll_portals4.h \
	coll_portals4_allreduce.c \
	coll_portals4_component.c \
	coll_portals4_barrier.c \
	coll_portals4_bcast.c \
	coll_portals4_reduce.c \
	coll_portals4_gather.c \
	coll_portals4_scatter.c \
	coll_portals4_request.h \
	coll_portals4_request.c

if MCA_BUILD_ompi_coll_portals4_DSO
component_noinst =
component_install = mca_coll_portals4.la
else
component_noinst = libmca_coll_portals4.la
component_install =
endif

AM_CPPFLAGS = $(coll_portals4_CPPFLAGS)
mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_portals4_la_SOURCES = $(local_sources)
mca_coll_portals4_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la \
	$(coll_portals4_LIBS)
mca_coll_portals4_la_LDFLAGS = -module -avoid-version $(coll_portals4_LDFLAGS)

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_portals4_la_SOURCES = $(local_sources)
libmca_coll_portals4_la_LIBADD = $(coll_portals4_LIBS)
libmca_coll_portals4_la_LDFLAGS = -module -avoid-version $(coll_portals4_LDFLAGS)
