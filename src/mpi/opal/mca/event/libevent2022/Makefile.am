#
# Copyright (c) 2010-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2010      Oracle and/or its affiliates.  All rights reserved.
# Copyright (c) 2012-2013 Los Alamos National Security, LLC.
#                         All rights reserved.
# Copyright (c) 2015      Intel, Inc. All rights reserved
# Copyright (c) 2016      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

EXTRA_DIST = autogen.subdirs

AM_CPPFLAGS = -I$(srcdir)/libevent -I$(srcdir)/libevent/include -I$(builddir)/libevent/include -I$(srcdir)/libevent/compat

SUBDIRS = libevent

headers = libevent2022.h

sources = \
        libevent2022_component.c \
        libevent2022_module.c

# Conditionally install the header files
if WANT_INSTALL_HEADERS
headers += libevent/opal_rename.h libevent/event.h libevent/evutil.h libevent/util-internal.h \
           libevent/mm-internal.h libevent/ipv6-internal.h \
	   libevent/strlcpy-internal.h libevent/evbuffer-internal.h \
	   libevent/bufferevent-internal.h libevent/event-internal.h \
	   libevent/evthread-internal.h libevent/defer-internal.h \
	   libevent/minheap-internal.h libevent/log-internal.h \
           libevent/evsignal-internal.h libevent/evmap-internal.h \
	   libevent/changelist-internal.h libevent/iocp-internal.h \
	   libevent/ratelim-internal.h \
	   libevent/WIN32-Code/event2/event-config.h \
	   libevent/WIN32-Code/tree.h \
	   libevent/compat/sys/queue.h \
           libevent/evhttp.h libevent/http-internal.h libevent/ht-internal.h \
           libevent/evrpc.h libevent/evrpc-internal.h \
           libevent/evdns.h libevent/include/event2/buffer_compat.h \
           libevent/include/event2/buffer.h \
           libevent/include/event2/bufferevent_compat.h \
           libevent/include/event2/bufferevent_ssl.h \
           libevent/include/event2/bufferevent_struct.h \
           libevent/include/event2/bufferevent.h \
           libevent/include/event2/dns_compat.h \
           libevent/include/event2/dns_struct.h \
           libevent/include/event2/event_compat.h \
           libevent/include/event2/event_struct.h \
           libevent/include/event2/event.h \
           libevent/include/event2/http_compat.h \
           libevent/include/event2/http_struct.h \
           libevent/include/event2/http.h \
           libevent/include/event2/keyvalq_struct.h \
           libevent/include/event2/listener.h \
           libevent/include/event2/rpc_compat.h \
           libevent/include/event2/rpc_struct.h \
           libevent/include/event2/rpc.h \
           libevent/include/event2/tag_compat.h \
           libevent/include/event2/tag.h \
           libevent/include/event2/thread.h \
           libevent/include/event2/util.h
opaldir = $(opalincludedir)/$(subdir)
nobase_opal_HEADERS = $(headers)

nobase_nodist_opal_HEADERS = libevent/include/event2/event-config.h
endif

# Make the output library in this directory, and name it
# libmca_<type>_<name>.la because build is forced to be static-only

component_noinst = libmca_event_libevent2022.la
component_install =


# We only ever build this component statically

noinst_LTLIBRARIES = $(component_noinst)
libmca_event_libevent2022_la_SOURCES =$(sources)
libmca_event_libevent2022_la_LDFLAGS = -module -avoid-version
libmca_event_libevent2022_la_LIBADD = $(builddir)/libevent/libevent.la
libmca_event_libevent2022_la_DEPENDENCIES = $(builddir)/libevent/libevent.la

