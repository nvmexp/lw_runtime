#
# Copyright (c) 2004-2008 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2010 Cisco Systems, Inc.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# Need this so that the base knows where to load dynamic components from
# (by default)

AM_CPPFLAGS = \
	$(LTDLINCL)

noinst_LTLIBRARIES = libmca_base.la

dist_opaldata_DATA = help-mca-base.txt help-mca-var.txt

# Source code files

headers = \
        base.h \
        mca_base_component_repository.h \
        mca_base_var.h \
        mca_base_pvar.h \
	mca_base_var_enum.h \
        mca_base_var_group.h \
        mca_base_vari.h \
	mca_base_framework.h

# Library

libmca_base_la_SOURCES = \
        $(headers) \
        mca_base_close.c \
        mca_base_cmd_line.c \
        mca_base_component_compare.c \
        mca_base_component_find.c \
        mca_base_component_repository.c \
        mca_base_components_open.c \
        mca_base_components_close.c \
        mca_base_components_select.c \
        mca_base_list.c \
        mca_base_open.c \
        mca_base_var.c \
        mca_base_pvar.c \
	mca_base_var_enum.c \
        mca_base_var_group.c \
        mca_base_parse_paramfile.c \
	mca_base_components_register.c \
	mca_base_framework.c

# Conditionally install the header files

if WANT_INSTALL_HEADERS
opaldir = $(opalincludedir)/$(subdir)
opal_HEADERS = $(headers)
endif
