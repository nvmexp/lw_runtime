#
# Copyright (c) 2014      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2014      LWPU Corporation.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#
dist_ompidata_DATA = help-mpi-coll-lwca.txt

sources = coll_lwda_module.c coll_lwda_reduce.c coll_lwda_allreduce.c \
          coll_lwda_reduce_scatter_block.c coll_lwda_component.c \
          coll_lwda_scan.c coll_lwda_exscan.c coll_lwda.h

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_coll_lwda_DSO
component_noinst =
component_install = mca_coll_lwda.la
else
component_noinst = libmca_coll_lwda.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_lwda_la_SOURCES = $(sources)
mca_coll_lwda_la_LDFLAGS = -module -avoid-version
mca_coll_lwda_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_lwda_la_SOURCES =$(sources)
libmca_coll_lwda_la_LDFLAGS = -module -avoid-version

