#
# Copyright (c) 2004-2009 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2008-2019 Cisco Systems, Inc.  All rights reserved
# Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
# Copyright (c) 2010-2011 Sandia National Laboratories. All rights reserved.
# Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
#                         reserved.
# Copyright (c) 2015-2017 Intel, Inc. All rights reserved.
# Copyright (c) 2015-2018 Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# Copyright (c) 2016      IBM Corporation.  All rights reserved.
# Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# We always have C bindings, but do we have profiling?

if BUILD_MPI_BINDINGS_LAYER
c_mpi_lib = mpi/c/libmpi_c_mpi.la mpi/tool/libmpi_mpit.la
else
c_mpi_lib =
endif
c_pmpi_lib = mpi/c/profile/libmpi_c_pmpi.la mpi/tool/profile/libmpi_pmpit.la

# See if we have Fortran mpif.h MPI bindings

if OMPI_BUILD_FORTRAN_MPIFH_BINDINGS
mpi_fortran_base_lib = mpi/fortran/base/libmpi_fortran_base.la
else
mpi_fortran_base_lib =
endif

# Note that the ordering of "." in SUBDIRS is important: the C++,
# Fortran mpif.h, and use mpi/use mpi_f08 bindings are all in
# standalone .la files that depend on libmpi.la.  So we must fully
# build libmpi.la first.

# NOTE: A handful of files in mpi/fortran/base must be included in
# libmpi.la.  But we wanted to keep all the Fortran sources together
# in the same tree, so we moved those sources to a separate
# subdirectory with its own Makefile.include that is included in this
# Makefile.am (NOTE: it did *not* work to put all the files -- base
# and non-base -- into mpi/fortran/mpif-h and have both a regular
# Makefile.am for building the mpif.h bindings library and a separate
# Makefile.include that was included in this top-level Makefile.am;
# problems oclwrred with "make distclean" and files in the
# ompi/mpi/fortran/mpif-h/.deps directory -- it's not clear whether
# this is an AM bug or whether this behavior is simply not supported).
# This ompi/mpi/fortran/base/Makefile.include file makes a colwenience
# LT library that is then sucked into libmpi.la (the
# ompi/mpi/fortran/base sources must be compiled with special
# CPPFLAGS; we can't just add the raw sources to libmpi_la_SOURCES,
# unfortunately).

# The end of the result is that libmpi.la -- including a few sources
# from mpi/fortran/base -- is fully built before the C++, mpif.h, and
# use mpi/use mpi_f08 bindings are built.  Therefore, the C++, mpif.h
# and use mpi/use mpi_f08 bindings libraries can all link against
# libmpi.la.

# NOTE: The ordering of this list is VERY CAREFULLY CRAFTED!  Do not
# change the ordering unless you really know what you're doing.  :-)

SUBDIRS = \
        include \
        datatype \
        debuggers \
        etc \
        mpi/c \
        mpi/tool \
        $(OMPI_MPIEXT_C_DIRS) \
        mpi/fortran/base/ \
        $(MCA_ompi_FRAMEWORKS_SUBDIRS) \
        $(MCA_ompi_FRAMEWORK_COMPONENT_STATIC_SUBDIRS) \
        . \
        mpi/cxx \
        $(OMPI_MPIEXT_MPIFH_DIRS) \
        mpi/fortran/mpif-h \
        $(OMPI_MPIEXT_USEMPI_DIR) \
        $(OMPI_FORTRAN_USEMPI_DIR) \
        mpi/fortran/mpiext-use-mpi \
        mpi/fortran/use-mpi-f08/base \
        mpi/fortran/use-mpi-f08/mod \
        mpi/fortran/use-mpi-f08/bindings \
        $(OMPI_MPIEXT_USEMPIF08_DIRS) \
        mpi/fortran/use-mpi-f08 \
        mpi/fortran/mpiext-use-mpi-f08 \
        $(MCA_ompi_FRAMEWORK_COMPONENT_DSO_SUBDIRS) \
        $(OMPI_CONTRIB_SUBDIRS)

if OMPI_WANT_JAVA_BINDINGS
SUBDIRS += \
        mpi/java
endif

# The ordering of the DIST_SUBDIRS isn't as important, but note that
# its contents *are* different than SUBDIRS.  In particular, the
# MPIEXT subdirs has a different value that is *not* equivalent to the
# union of the various $(OMPI_MPIEXT_*_DIRS) variables from SUBDIRS!

DIST_SUBDIRS = \
        include \
        datatype \
        debuggers \
        etc \
        mpi/c \
        mpi/tool \
        mpi/cxx \
        mpi/fortran/base \
        mpi/fortran/mpif-h \
        mpi/fortran/use-mpi-tkr \
        mpi/fortran/use-mpi-ignore-tkr \
        mpi/fortran/mpiext-use-mpi \
        mpi/fortran/use-mpi-f08 \
        mpi/fortran/use-mpi-f08/base \
        mpi/fortran/use-mpi-f08/mod \
        mpi/fortran/use-mpi-f08/bindings \
        mpi/fortran/mpiext-use-mpi-f08 \
        mpi/java \
        $(OMPI_MPIEXT_ALL_SUBDIRS) \
        $(MCA_ompi_FRAMEWORKS_SUBDIRS) \
        $(MCA_ompi_FRAMEWORK_COMPONENT_ALL_SUBDIRS) \
        $(OMPI_CONTRIB_DIST_SUBDIRS)

# Build the main MPI library

lib_LTLIBRARIES = lib@OMPI_LIBMPI_NAME@.la
lib@OMPI_LIBMPI_NAME@_la_SOURCES =
lib@OMPI_LIBMPI_NAME@_la_LIBADD = \
        datatype/libdatatype.la \
        debuggers/libdebuggers.la \
        mpi/c/libmpi_c.la \
        mpi/tool/libmpi_mpit_common.la \
        $(c_mpi_lib) \
        $(c_pmpi_lib) \
        $(mpi_fortran_base_lib) \
        $(MCA_ompi_FRAMEWORK_LIBS) \
        $(OMPI_MPIEXT_C_LIBS) \
        $(OMPI_LIBMPI_EXTRA_LIBS)


if OMPI_RTE_ORTE
lib@OMPI_LIBMPI_NAME@_la_LIBADD += \
        $(OMPI_TOP_BUILDDIR)/orte/lib@ORTE_LIB_PREFIX@open-rte.la
endif
lib@OMPI_LIBMPI_NAME@_la_LIBADD += \
        $(OMPI_TOP_BUILDDIR)/opal/lib@OPAL_LIB_PREFIX@open-pal.la
lib@OMPI_LIBMPI_NAME@_la_DEPENDENCIES = $(lib@OMPI_LIBMPI_NAME@_la_LIBADD)
lib@OMPI_LIBMPI_NAME@_la_LDFLAGS = \
       -version-info $(libmpi_so_version) \
        $(OMPI_LIBMPI_EXTRA_LDFLAGS)

# included subdirectory Makefile.am's and appended-to variables
headers =
noinst_LTLIBRARIES =
include_HEADERS =
dist_ompidata_DATA =
lib@OMPI_LIBMPI_NAME@_la_SOURCES += $(headers)
nodist_man_MANS =

# Conditionally install the header files

if WANT_INSTALL_HEADERS
ompidir = $(ompiincludedir)/$(subdir)
nobase_ompi_HEADERS = $(headers)
endif

include class/Makefile.am
include attribute/Makefile.am
include communicator/Makefile.am
include dpm/Makefile.am
include errhandler/Makefile.am
include file/Makefile.am
include group/Makefile.am
include info/Makefile.am
include interlib/Makefile.am
include message/Makefile.am
include op/Makefile.am
include peruse/Makefile.am
include proc/Makefile.am
include request/Makefile.am
include runtime/Makefile.am
include win/Makefile.am
include tools/Makefile.am
include mpi/Makefile.am
include mpi/man/man3/Makefile.extra
include mpiext/Makefile.am
include patterns/net/Makefile.am
include patterns/comm/Makefile.am
include mca/Makefile.am
include util/Makefile.am

# Ensure that the man page directory exists before we try to make man
# page files (because ompi/mpi/man/man3 has no config.status-generated
# Makefile)
dir_stamp = $(top_builddir)/$(subdir)/mpi/man/man3/.dir-stamp

# Also ensure that the man pages are rebuilt if the opal_config.h file
# changes (e.g., configure was run again, meaning that the release
# date or version may have changed)
$(nodist_man_MANS): $(dir_stamp) $(top_builddir)/opal/include/opal_config.h

$(dir_stamp):
	$(MKDIR_P) `dirname $@`
	touch "$@"

# Remove the generated man pages
distclean-local:
	rm -f $(nodist_man_MANS) $(dir_stamp) mpiext/static-components.h
