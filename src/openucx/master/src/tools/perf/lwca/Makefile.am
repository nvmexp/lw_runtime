#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_LWDA

module_LTLIBRARIES               = libucx_perftest_lwda.la
libucx_perftest_lwda_la_CPPFLAGS = $(BASE_CPPFLAGS) $(LWDA_CPPFLAGS)
libucx_perftest_lwda_la_CFLAGS   = $(BASE_CFLAGS) $(LWDA_CFLAGS)
libucx_perftest_lwda_la_LDFLAGS  = $(LWDA_LDFLAGS) -version-info $(SOVERSION)
libucx_perftest_lwda_la_SOURCES  = lwda_alloc.c

include $(top_srcdir)/config/module.am

endif
