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
# Copyright (c) 2008-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
# Copyright (c) 2015      Intel, Inc.  All rights reserved.
# Copyright (c) 2018      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

include $(top_srcdir)/Makefile.ompi-rules

man_pages = orterun.1
EXTRA_DIST = $(man_pages:.1=.1in)

if OPAL_INSTALL_BINARIES

bin_PROGRAMS = orterun

nodist_man_MANS = $(man_pages)

# Ensure that the man pages are rebuilt if the opal_config.h file
# changes; a "good enough" way to know if configure was run again (and
# therefore the release date or version may have changed)
$(nodist_man_MANS): $(top_builddir)/opal/include/opal_config.h

dist_ortedata_DATA = help-orterun.txt

endif # OPAL_INSTALL_BINARIES

orterun_SOURCES = \
        main.c \
        orterun.c \
        orterun.h

orterun_LDADD = \
	$(top_builddir)/orte/lib@ORTE_LIB_PREFIX@open-rte.la \
	$(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la

distclean-local:
	rm -f $(man_pages)
