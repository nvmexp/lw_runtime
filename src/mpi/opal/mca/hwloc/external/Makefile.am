#
# Copyright (c) 2011-2012 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2016      Intel, Inc. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# This is a special component -- its main purpose in life is to have
# its configure.m4 add some things to CPPFLAGS, LDFLAGS, and
# LIBS.  Hence, there's nothing to build here.  :-)

# We only ever build this component statically

noinst_LTLIBRARIES = libmca_hwloc_external.la

libmca_hwloc_external_la_SOURCES = \
        external.h \
        hwloc_external_component.c
libmca_hwloc_external_la_LDFLAGS = \
    -module -avoid-version \
    $(opal_hwloc_external_LDFLAGS)
libmca_hwloc_external_la_LIBADD = $(hwloc_external_LIBS) -lhwloc

# Conditionally install the header files
if WANT_INSTALL_HEADERS
opaldir = $(opalincludedir)/$(subdir)
nobase_opal_HEADERS = external.h
endif
