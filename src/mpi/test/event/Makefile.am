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
# Copyright (c) 2015-2016 Cisco Systems, Inc.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#



check_PROGRAMS = \
	signal-test \
	time-test \
	event-test

TESTS = $(check_PROGRAMS)

signal_test_SOURCES = signal-test.c
signal_test_LDFLAGS = $(OPAL_PKG_CONFIG_LDFLAGS)
signal_test_LDADD = \
        $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la
signal_test_DEPENDENCIES = $(signal_test_LDADD)

time_test_SOURCES = time-test.c
time_test_LDFLAGS = $(OPAL_PKG_CONFIG_LDFLAGS)
time_test_LDADD = \
        $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la
time_test_DEPENDENCIES = $(time_test_LDADD)

event_test_SOURCES = event-test.c
event_test_LDFLAGS = $(OPAL_PKG_CONFIG_LDFLAGS)
event_test_LDADD = \
        $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la
event_test_DEPENDENCIES = $(event_test_LDADD)

distclean:
	rm -rf *.dSYM .deps .libs *.log *.o *.trs $(check_PROGRAMS) Makefile
