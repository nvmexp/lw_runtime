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

if MCA_BUILD_ompi_coll_inter_DSO
component_noinst =
component_install = mca_coll_inter.la
else
component_noinst = libmca_coll_inter.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_inter_la_SOURCES = $(sources)
mca_coll_inter_la_LDFLAGS = -module -avoid-version
mca_coll_inter_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_inter_la_SOURCES = $(sources)
libmca_coll_inter_la_LDFLAGS = -module -avoid-version

# Source files

sources = \
        coll_inter.h \
        coll_inter.c \
        coll_inter_allreduce.c \
	coll_inter_allgather.c \
	coll_inter_allgatherv.c \
	coll_inter_gather.c \
	coll_inter_gatherv.c \
        coll_inter_scatter.c \
	coll_inter_scatterv.c \
        coll_inter_bcast.c \
        coll_inter_component.c \
        coll_inter_reduce.c
