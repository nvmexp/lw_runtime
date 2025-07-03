#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_LWDA

SUBDIRS = . gdr_copy

module_LTLIBRARIES      = libuct_lwda.la
libuct_lwda_la_CPPFLAGS = $(BASE_CPPFLAGS) $(LWDA_CPPFLAGS)
libuct_lwda_la_CFLAGS   = $(BASE_CFLAGS) $(LWDA_CFLAGS)
libuct_lwda_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                          $(top_builddir)/src/uct/libuct.la
libuct_lwda_la_LDFLAGS  = $(LWDA_LDFLAGS) -version-info $(SOVERSION)

noinst_HEADERS = \
	base/lwda_md.h \
	base/lwda_iface.h \
	lwda_copy/lwda_copy_md.h \
	lwda_copy/lwda_copy_iface.h \
	lwda_copy/lwda_copy_ep.h \
	lwda_ipc/lwda_ipc_md.h \
	lwda_ipc/lwda_ipc_iface.h \
	lwda_ipc/lwda_ipc_ep.h \
	lwda_ipc/lwda_ipc_cache.h

libuct_lwda_la_SOURCES = \
	base/lwda_iface.c \
	base/lwda_md.c \
	lwda_copy/lwda_copy_md.c \
	lwda_copy/lwda_copy_iface.c \
	lwda_copy/lwda_copy_ep.c \
	lwda_ipc/lwda_ipc_md.c \
	lwda_ipc/lwda_ipc_iface.c \
	lwda_ipc/lwda_ipc_ep.c \
	lwda_ipc/lwda_ipc_cache.c

include $(top_srcdir)/config/module.am

endif
