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
# Copyright (c) 2009-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2015      Los Alamos National Security, LLC. All rights
#                         reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

SUBDIRS = \
	include \
	$(MCA_orte_FRAMEWORKS_SUBDIRS) \
	$(MCA_orte_FRAMEWORK_COMPONENT_STATIC_SUBDIRS) \
	etc \
        . \
	$(MCA_orte_FRAMEWORK_COMPONENT_DSO_SUBDIRS)

DIST_SUBDIRS = \
	include \
	etc \
	$(MCA_orte_FRAMEWORKS_SUBDIRS) \
	$(MCA_orte_FRAMEWORK_COMPONENT_ALL_SUBDIRS)

# Build the main ORTE library

lib_LTLIBRARIES = lib@ORTE_LIB_PREFIX@open-rte.la
lib@ORTE_LIB_PREFIX@open_rte_la_SOURCES =
lib@ORTE_LIB_PREFIX@open_rte_la_LIBADD = \
	$(MCA_orte_FRAMEWORK_LIBS) \
	$(ORTE_TOP_BUILDDIR)/opal/lib@OPAL_LIB_PREFIX@open-pal.la
lib@ORTE_LIB_PREFIX@open_rte_la_DEPENDENCIES = $(libopen_rte_la_LIBADD)
lib@ORTE_LIB_PREFIX@open_rte_la_LDFLAGS = -version-info $(libopen_rte_so_version)

# included subdirectory Makefile.am's and appended-to variables
headers =
noinst_LTLIBRARIES =
dist_ortedata_DATA =
lib@ORTE_LIB_PREFIX@open_rte_la_SOURCES += $(headers)

# Conditionally install the header files

if WANT_INSTALL_HEADERS
ortedir = $(orteincludedir)/$(subdir)
nobase_orte_HEADERS = $(headers)
endif

# setup to catch extra files to be included in tarballs
EXTRA_DIST = Doxyfile

include runtime/Makefile.am
include util/Makefile.am
include tools/Makefile.am
include orted/Makefile.am
include test/mpi/Makefile.include
include test/system/Makefile.include
include mca/Makefile.am
