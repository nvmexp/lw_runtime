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
# Copyright (c) 2010-2015 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# Copyright (c) 2017      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

EXTRA_DIST = autogen.subdirs

sources =
include src/Makefile.extra

if MCA_io_romio321_SHOULD_BUILD
# Need to set both SUBDIRS and DIST_SUBDIRS due to oddities with AM
SUBDIRS = romio
DIST_SUBDIRS = romio
endif

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_io_romio321_DSO
lib =
lib_sources =
component = mca_io_romio321.la
component_sources = $(sources)
else
lib = libmca_io_romio321.la
lib_sources = $(sources)
component =
component_sources =
endif

libs = romio/libromio_dist.la

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component)
mca_io_romio321_la_SOURCES = $(component_sources)
mca_io_romio321_la_LIBADD = $(libs)
mca_io_romio321_la_DEPENDENCIES = $(libs)
mca_io_romio321_la_LDFLAGS = -module -avoid-version

noinst_LTLIBRARIES = $(lib)
libmca_io_romio321_la_SOURCES = $(lib_sources)
libmca_io_romio321_la_LIBADD = $(libs)
libmca_io_romio321_la_DEPENDENCIES = $(libs)
libmca_io_romio321_la_LDFLAGS = -module -avoid-version
