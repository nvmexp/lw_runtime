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
# Copyright (c) 2008-2015 University of Houston. All rights reserved.
# Copyright (c) 2012      Cisco Systems, Inc.  All rights reserved.
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
        fcoll_individual.h \
        fcoll_individual_module.c \
        fcoll_individual_component.c \
        fcoll_individual_file_read_all.c \
        fcoll_individual_file_write_all.c

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_fcoll_individual_DSO
component_noinst =
component_install = mca_fcoll_individual.la
else
component_noinst = libmca_fcoll_individual.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_fcoll_individual_la_SOURCES = $(sources)
mca_fcoll_individual_la_LDFLAGS = -module -avoid-version
mca_fcoll_individual_la_LIBADD = $(OMPI_TOP_BUILDDIR)/ompi/mca/common/ompio/libmca_common_ompio.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_fcoll_individual_la_SOURCES =$(sources)
libmca_fcoll_individual_la_LDFLAGS = -module -avoid-version
