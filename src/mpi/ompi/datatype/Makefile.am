# -*- makefile -*-
#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2011 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2007-2013 Los Alamos National Security, LLC.  All rights
#                         reserved.
# Copyright (c) 2010 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2016      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

headers = \
        ompi_datatype.h \
        ompi_datatype_internal.h

noinst_LTLIBRARIES = \
        libdatatype.la

# these sources will be compiled with the normal CFLAGS only
libdatatype_la_SOURCES = \
        ompi_datatype_args.c \
        ompi_datatype_create.c \
        ompi_datatype_create_contiguous.c \
        ompi_datatype_create_indexed.c \
        ompi_datatype_create_struct.c \
        ompi_datatype_create_vector.c \
        ompi_datatype_create_darray.c \
        ompi_datatype_create_subarray.c \
        ompi_datatype_external.c \
        ompi_datatype_external32.c \
        ompi_datatype_match_size.c \
        ompi_datatype_module.c \
        ompi_datatype_sndrcv.c \
        ompi_datatype_get_elements.c

libdatatype_la_LIBADD =

# Conditionally install the header files
if WANT_INSTALL_HEADERS
ompidir = $(ompiincludedir)/$(subdir)
ompi_HEADERS = $(headers)
endif
