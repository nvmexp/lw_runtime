#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

if HAVE_GDR_COPY

module_LTLIBRARIES              = libuct_lwda_gdrcopy.la
libuct_lwda_gdrcopy_la_CPPFLAGS = $(BASE_CPPFLAGS) $(LWDA_CPPFLAGS) $(GDR_COPY_CPPFLAGS)
libuct_lwda_gdrcopy_la_CFLAGS   = $(BASE_CFLAGS)
libuct_lwda_gdrcopy_la_LIBADD   = $(top_builddir)/src/ucs/libucs.la \
                                  $(top_builddir)/src/uct/lwca/libuct_lwda.la
libuct_lwda_gdrcopy_la_LDFLAGS  = $(LWDA_LDFLAGS) $(GDR_COPY_LDFLAGS) -version-info $(SOVERSION)

noinst_HEADERS = \
	gdr_copy_md.h \
	gdr_copy_iface.h \
	gdr_copy_ep.h

libuct_lwda_gdrcopy_la_SOURCES = \
	gdr_copy_md.c \
	gdr_copy_iface.c \
	gdr_copy_ep.c

include $(top_srcdir)/config/module.am

endif
