# -*- makefile -*-
#
# Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2014      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2015      Intel, Inc. All rights reserved.
# Copyright (c) 2018      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# This makefile.am does not stand on its own - it is included from orte/Makefile.am

dist_ortedata_DATA += orted/help-orted.txt

headers += \
	orted/orted.h \
	orted/orted_submit.h

lib@ORTE_LIB_PREFIX@open_rte_la_SOURCES += \
        orted/orted_main.c \
        orted/orted_comm.c

# The MPIR portion of the library must be built with -g, even if
# the rest of the library has other optimization flags.
# Use an intermediate library to isolate the debug object.
noinst_LTLIBRARIES += liborted_mpir.la
liborted_mpir_la_SOURCES = \
	orted/orted_submit.c
liborted_mpir_la_CFLAGS = $(CFLAGS_WITHOUT_OPTFLAGS) $(DEBUGGER_CFLAGS)

lib@ORTE_LIB_PREFIX@open_rte_la_LIBADD += liborted_mpir.la


include orted/pmix/Makefile.am
