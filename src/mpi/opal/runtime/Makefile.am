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
# Copyright (c) 2012      Los Alamos National Security, LLC.
#                         All rights reserved.
# Copyright (c) 2014      Intel, Inc. All rights reserved
# Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# This makefile.am does not stand on its own - it is included from opal/Makefile.am

AM_CFLAGS = \
            -DOPAL_CONFIGURE_HOST="\"@OPAL_CONFIGURE_HOST@\""

# Source code files
dist_opaldata_DATA += runtime/help-opal-runtime.txt \
                     runtime/help-opal_info.txt

headers += \
        runtime/opal_progress.h \
        runtime/opal.h \
        runtime/opal_cr.h \
        runtime/opal_info_support.h \
        runtime/opal_params.h \
        runtime/opal_progress_threads.h

lib@OPAL_LIB_PREFIX@open_pal_la_SOURCES += \
        runtime/opal_progress.c \
        runtime/opal_finalize.c \
        runtime/opal_init.c \
        runtime/opal_params.c \
        runtime/opal_cr.c \
        runtime/opal_info_support.c \
        runtime/opal_progress_threads.c
