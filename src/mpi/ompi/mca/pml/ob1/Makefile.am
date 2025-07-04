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
# Copyright (c) 2009      Sun Microsystems, Inc.  All rights reserved.
# Copyright (c) 2009-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2012      LWPU Corporation.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

dist_ompidata_DATA = \
	help-mpi-pml-ob1.txt

EXTRA_DIST = post_configure.sh

ob1_sources  = \
	pml_ob1.c \
	pml_ob1.h \
	pml_ob1_comm.c \
	pml_ob1_comm.h \
	pml_ob1_component.c \
	pml_ob1_component.h \
	pml_ob1_hdr.h \
	pml_ob1_iprobe.c \
	pml_ob1_irecv.c \
	pml_ob1_isend.c \
	pml_ob1_progress.c \
	pml_ob1_rdma.c \
	pml_ob1_rdma.h \
	pml_ob1_rdmafrag.c \
	pml_ob1_rdmafrag.h \
	pml_ob1_recvfrag.c \
	pml_ob1_recvfrag.h \
	pml_ob1_recvreq.c \
	pml_ob1_recvreq.h \
	pml_ob1_sendreq.c \
	pml_ob1_sendreq.h \
	pml_ob1_start.c

# If we have LWCA support requested, build the LWCA file also
if OPAL_lwda_support
ob1_sources += \
    pml_ob1_lwda.c
endif

if MCA_BUILD_ompi_pml_ob1_DSO
component_noinst =
component_install = mca_pml_ob1.la
else
component_noinst = libmca_pml_ob1.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_pml_ob1_la_SOURCES = $(ob1_sources)
mca_pml_ob1_la_LDFLAGS = -module -avoid-version

if OPAL_lwda_support
mca_pml_ob1_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la \
    $(OMPI_TOP_BUILDDIR)/opal/mca/common/lwca/lib@OPAL_LIB_PREFIX@mca_common_lwda.la
endif

noinst_LTLIBRARIES = $(component_noinst)
libmca_pml_ob1_la_SOURCES = $(ob1_sources)
libmca_pml_ob1_la_LDFLAGS = -module -avoid-version
