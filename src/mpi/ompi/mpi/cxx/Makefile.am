# -*- makefile -*-
#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2007-2012 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2016      IBM Corporation.  All rights reserved.
# Copyright (c) 2017      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# Need the first so that we can get the path names correct inside the
# MPI C++ library.  The second is necessary so that mpi.h doesn't
# include mpicxx.h through the incorrect pathname in any of the C++
# bindings .c files.  Just use the define for this purpose from user
# code.
AM_CPPFLAGS = -DOMPI_BUILDING_CXX_BINDINGS_LIBRARY=1 -DOMPI_SKIP_MPICXX=1

if OMPI_BUILD_MPI_CXX_BINDINGS

mpi_lib = lib@OMPI_LIBMPI_NAME@_cxx.la

lib_LTLIBRARIES = lib@OMPI_LIBMPI_NAME@_cxx.la
lib@OMPI_LIBMPI_NAME@_cxx_la_SOURCES = \
	mpicxx.cc \
	intercepts.cc \
	comm.cc \
	datatype.cc \
	file.cc \
	win.cc \
	cxx_glue.c

lib@OMPI_LIBMPI_NAME@_cxx_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la
lib@OMPI_LIBMPI_NAME@_cxx_la_LDFLAGS = -version-info $(libmpi_cxx_so_version)

headers = \
	mpicxx.h \
        constants.h \
	file.h \
        functions.h \
        datatype.h \
        exception.h \
        op.h \
        status.h \
        request.h \
        group.h \
        comm.h \
        errhandler.h \
        intracomm.h \
        info.h \
        win.h \
        topology.h \
        intercomm.h \
        datatype_inln.h \
	file_inln.h \
        functions_inln.h \
        request_inln.h \
        comm_inln.h \
        intracomm_inln.h \
        info_inln.h \
        win_inln.h \
        topology_inln.h \
        intercomm_inln.h \
        group_inln.h \
        op_inln.h \
        errhandler_inln.h \
        status_inln.h \
        cxx_glue.h

ompidir = $(ompiincludedir)/ompi/mpi/cxx
ompi_HEADERS = \
        $(headers)
endif
