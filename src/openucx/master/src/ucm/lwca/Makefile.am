#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

if HAVE_LWDA

module_LTLIBRARIES      = libucm_lwda.la
libucm_lwda_la_CPPFLAGS = $(BASE_CPPFLAGS) $(LWDA_CPPFLAGS)
libucm_lwda_la_CFLAGS   = $(BASE_CFLAGS) $(LWDA_CFLAGS)
libucm_lwda_la_LIBADD   = ../libucm.la
libucm_lwda_la_LDFLAGS  = $(UCM_MODULE_LDFLAGS) \
                          $(patsubst %, -Xlinker %, $(LWDA_LDFLAGS)) \
                          -version-info $(SOVERSION)

noinst_HEADERS = \
	lwdamem.h

libucm_lwda_la_SOURCES = \
	lwdamem.c

include $(top_srcdir)/config/module.am

endif
