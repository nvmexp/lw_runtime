#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2015 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2010      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# Copyright (c) 2018      Research Organization for Information Science
#                         and Technology (RIST).  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
        coll_tuned.h \
        coll_tuned_dynamic_file.h \
        coll_tuned_dynamic_rules.h \
        coll_tuned_decision_fixed.c \
        coll_tuned_decision_dynamic.c \
        coll_tuned_dynamic_file.c \
        coll_tuned_dynamic_rules.c \
        coll_tuned_component.c \
        coll_tuned_module.c \
        coll_tuned_allgather_decision.c \
        coll_tuned_allgatherv_decision.c \
        coll_tuned_allreduce_decision.c \
        coll_tuned_alltoall_decision.c \
        coll_tuned_gather_decision.c \
        coll_tuned_alltoallv_decision.c \
        coll_tuned_barrier_decision.c \
        coll_tuned_reduce_decision.c \
        coll_tuned_bcast_decision.c \
        coll_tuned_reduce_scatter_decision.c \
        coll_tuned_scatter_decision.c \
        coll_tuned_reduce_scatter_block_decision.c \
        coll_tuned_exscan_decision.c \
        coll_tuned_scan_decision.c

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_coll_tuned_DSO
component_noinst =
component_install = mca_coll_tuned.la
else
component_noinst = libmca_coll_tuned.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_tuned_la_SOURCES = $(sources)
mca_coll_tuned_la_LDFLAGS = -module -avoid-version
mca_coll_tuned_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_tuned_la_SOURCES =$(sources)
libmca_coll_tuned_la_LDFLAGS = -module -avoid-version
