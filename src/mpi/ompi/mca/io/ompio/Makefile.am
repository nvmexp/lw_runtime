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
# Copyright (c) 2008-2012 University of Houston. All rights reserved.
# Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
# Copyright (c) 2018      Research Organization for Information Science
#                         and Technology (RIST).  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_io_ompio_DSO
component_noinst =
component_install = mca_io_ompio.la
else
component_noinst = libmca_io_ompio.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_io_ompio_la_SOURCES = $(headers) $(sources)
mca_io_ompio_la_LDFLAGS = -module -avoid-version
mca_io_ompio_la_LIBADD = $(io_ompio_LIBS) \
    $(OMPI_TOP_BUILDDIR)/ompi/mca/common/ompio/libmca_common_ompio.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_io_ompio_la_SOURCES = $(headers) $(sources)
libmca_io_ompio_la_LDFLAGS = -module -avoid-version

# Source files

headers = \
	io_ompio.h

sources = \
        io_ompio.c \
        io_ompio_component.c \
        io_ompio_module.c \
        io_ompio_file_set_view.c \
        io_ompio_file_open.c \
        io_ompio_file_write.c \
        io_ompio_file_read.c
