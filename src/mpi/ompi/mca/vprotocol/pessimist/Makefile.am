#
# Copyright (c) 2004-2007 The Trustees of the University of Tennessee.
#                         All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# Make the output library in this directory, and name it either
# protocol_<name>.la (for DSO builds) or libprotocol_<name>.la
# (for static builds).

local_sources = \
        vprotocol_pessimist.h \
        vprotocol_pessimist.c \
        vprotocol_pessimist_component.c \
        vprotocol_pessimist_request.c \
        vprotocol_pessimist_request.h \
        vprotocol_pessimist_proc.c \
        vprotocol_pessimist_comm.c \
        vprotocol_pessimist_progress.c \
        vprotocol_pessimist_start.h \
        vprotocol_pessimist_start.c \
        vprotocol_pessimist_recv.c \
        vprotocol_pessimist_send.c \
        vprotocol_pessimist_probe.c \
        vprotocol_pessimist_wait.h \
        vprotocol_pessimist_wait.c \
        vprotocol_pessimist_event.h \
        vprotocol_pessimist_event.c \
        vprotocol_pessimist_eventlog.h \
        vprotocol_pessimist_eventlog.c \
        vprotocol_pessimist_eventlog_protocol.h \
        vprotocol_pessimist_sender_based_types.h \
        vprotocol_pessimist_sender_based.h \
        vprotocol_pessimist_sender_based.c

if MCA_BUILD_ompi_vprotocol_pessimist_DSO
component_noinst =
component_install = mca_vprotocol_pessimist.la
else
component_noinst = libmca_vprotocol_pessimist.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_vprotocol_pessimist_la_SOURCES = $(local_sources)
mca_vprotocol_pessimist_la_LDFLAGS = -module -avoid-version
mca_vprotocol_pessimist_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_vprotocol_pessimist_la_SOURCES = $(local_sources)
libmca_vprotocol_pessimist_la_LDFLAGS = -module -avoid-version
