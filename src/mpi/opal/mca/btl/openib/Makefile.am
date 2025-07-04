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
# Copyright (c) 2007-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2010      Oracle and/or its affiliates.  All rights reserved.
# Copyright (c) 2011      LWPU Corporation.  All rights reserved.
# Copyright (c) 2011      Mellanox Technologies.  All rights reserved.
# Copyright (c) 2012      Oak Ridge National Laboratory.  All rights reserved
# Copyright (c) 2013      Intel, Inc. All rights reserved.
# Copyright (c) 2016      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = $(btl_openib_CPPFLAGS)
AM_LFLAGS = -Pbtl_openib_ini_yy
LEX_OUTPUT_ROOT = lex.btl_openib_ini_yy

amca_paramdir = $(AMCA_PARAM_SETS_DIR)
dist_amca_param_DATA = btl-openib-benchmark

dist_opaldata_DATA = \
        help-mpi-btl-openib.txt \
        connect/help-mpi-btl-openib-cpc-base.txt \
        mca-btl-openib-device-params.ini

sources = \
    btl_openib.c \
    btl_openib.h \
    btl_openib_component.c \
    btl_openib_endpoint.c \
    btl_openib_endpoint.h \
    btl_openib_frag.c \
    btl_openib_frag.h \
    btl_openib_proc.c \
    btl_openib_proc.h \
    btl_openib_eager_rdma.h \
    btl_openib_lex.h \
    btl_openib_lex.l \
    btl_openib_mca.c \
    btl_openib_mca.h \
    btl_openib_ini.c \
    btl_openib_ini.h \
    btl_openib_async.c \
    btl_openib_async.h \
    btl_openib_xrc.c \
    btl_openib_xrc.h \
    btl_openib_ip.h \
    btl_openib_ip.c \
    btl_openib_put.c \
    btl_openib_get.c \
    btl_openib_atomic.c \
    connect/base.h \
    connect/btl_openib_connect_base.c \
    connect/btl_openib_connect_empty.c \
    connect/btl_openib_connect_empty.h \
    connect/connect.h

# If we have rdmacm support, build that CPC
if MCA_btl_openib_have_rdmacm
sources += \
    connect/btl_openib_connect_rdmacm.c \
    connect/btl_openib_connect_rdmacm.h

dist_opaldata_DATA += connect/help-mpi-btl-openib-cpc-rdmacm.txt
endif

# If we have udcm support, build that CPC
if MCA_btl_openib_have_udcm
sources += \
    connect/btl_openib_connect_udcm.c \
    connect/btl_openib_connect_udcm.h

# dist_opaldata_DATA += connect/help-mpi-btl-openib-cpc-ud.txt
endif

# If we have dynamic SL support, build those files
if MCA_btl_openib_have_dynamic_sl
sources += \
    connect/btl_openib_connect_sl.c \
    connect/btl_openib_connect_sl.h
endif

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_opal_btl_openib_DSO
lib =
lib_sources =
component = mca_btl_openib.la
component_sources = $(sources)
else
lib = libmca_btl_openib.la
lib_sources = $(sources)
component =
component_sources =
endif

mcacomponentdir = $(opallibdir)
mcacomponent_LTLIBRARIES = $(component)
mca_btl_openib_la_SOURCES = $(component_sources)
mca_btl_openib_la_LDFLAGS = -module -avoid-version $(btl_openib_LDFLAGS)
mca_btl_openib_la_LIBADD = $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la \
	$(btl_openib_LIBS) \
    $(OPAL_TOP_BUILDDIR)/opal/mca/common/verbs/lib@OPAL_LIB_PREFIX@mca_common_verbs.la
if OPAL_lwda_support
mca_btl_openib_la_LIBADD += \
    $(OPAL_TOP_BUILDDIR)/opal/mca/common/lwca/lib@OPAL_LIB_PREFIX@mca_common_lwda.la
endif

noinst_LTLIBRARIES = $(lib)
libmca_btl_openib_la_SOURCES = $(lib_sources)
libmca_btl_openib_la_LDFLAGS= -module -avoid-version $(btl_openib_LDFLAGS)
libmca_btl_openib_la_LIBADD = $(btl_openib_LIBS)

maintainer-clean-local:
	rm -f btl_openib_lex.c
