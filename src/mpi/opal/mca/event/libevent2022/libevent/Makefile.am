# Makefile.am for libevent
# Copyright 2000-2007 Niels Provos
# Copyright 2007-2012 Niels Provos and Nick Mathewson
#
# See LICENSE for copying information.

# 'foreign' means that we're not enforcing GNU package rules strictly.
# '1.7' means that we need automake 1.7 or later (and we do).
AUTOMAKE_OPTIONS = foreign 1.7

ACLOCAL_AMFLAGS = -I m4

# This is the "Release" of the Libevent ABI.  It takes precedence over
# the VERSION_INFO, so that two versions of Libevent with the same
# "Release" are never binary-compatible.
#
# This number incremented once for the 2.0 release candidate, and
# will increment for each series until we revise our interfaces enough
# that we can seriously expect ABI compatibility between series.
#
RELEASE = -release 2.0

# This is the version info for the libevent binary API.  It has three
# numbers:
#   Current  -- the number of the binary API that we're implementing
#   Revision -- which iteration of the implementation of the binary
#               API are we supplying?
#   Age      -- How many previous binary API versions do we also
#               support?
#
# To increment a VERSION_INFO (current:revision:age):
#    If the ABI didn't change:
#        Return (current:revision+1:age)
#    If the ABI changed, but it's backward-compatible:
#        Return (current+1:0:age+1)
#    If the ABI changed and it isn't backward-compatible:
#        Return (current+1:0:0)
#
# Once an RC is out, DO NOT MAKE ANY ABI-BREAKING CHANGES IN THAT SERIES
# UNLESS YOU REALLY REALLY HAVE TO.
VERSION_INFO = 6:9:1

# History:          RELEASE    VERSION_INFO
#  2.0.1-alpha --     2.0        1:0:0
#  2.0.2-alpha --                2:0:0
#  2.0.3-alpha --                2:0:0  (should have incremented; didn't.)
#  2.0.4-alpha --                3:0:0
#  2.0.5-beta  --                4:0:0
#  2.0.6-rc    --     2.0        2:0:0
#  2.0.7-rc    --     2.0        3:0:1
#  2.0.8-rc    --     2.0        4:0:2
#  2.0.9-rc    --     2.0        5:0:0 (ABI changed slightly)
#  2.0.10-stable--    2.0        5:1:0 (No ABI change)
#  2.0.11-stable--    2.0        6:0:1 (ABI changed, backward-compatible)
#  2.0.12-stable--    2.0        6:1:1 (No ABI change)
#  2.0.13-stable--    2.0        6:2:1 (No ABI change)
#  2.0.14-stable--    2.0        6:3:1 (No ABI change)
#  2.0.15-stable--    2.0        6:3:1 (Forgot to update :( )
#  2.0.16-stable--    2.0        6:4:1 (No ABI change)
#  2.0.17-stable--    2.0        6:5:1 (No ABI change)
#  2.0.18-stable--    2.0        6:6:1 (No ABI change)
#  2.0.19-stable--    2.0        6:7:1 (No ABI change)
#  2.0.20-stable--    2.0        6:8:1 (No ABI change)
#  2.0.21-stable--    2.0        6:9:1 (No ABI change)
#
# For Libevent 2.1:
#  2.1.1-alpha --     2.1        1:0:0


# ABI version history for this package effectively restarts every time
# we change RELEASE.  Version 1.4.x had RELEASE of 1.4.
#
# Ideally, we would not be using RELEASE at all; instead we could just
# use the VERSION_INFO field to label our backward-incompatible ABI
# changes, and those would be few and far between.  Unfortunately,
# Libevent still exposes far too many volatile structures in its
# headers, so we pretty much have to assume that most development
# series will break ABI compatibility.  For now, it's simplest just to
# keep incrementing the RELEASE between series and resetting VERSION_INFO.
#
# Eventually, when we get to the point where the structures in the
# headers are all non-changing (or not there at all!), we can shift to
# a more normal worldview where backward-incompatible ABI changes are
# nice and rare.  For the next couple of years, though, 'struct event'
# is user-visible, and so we can pretty much guarantee that release
# series won't be binary-compatible.

# Open MPI: Don't install this, ever
#if INSTALL_LIBEVENT
#dist_bin_SCRIPTS = event_rpcgen.py
#endif

# Open MPI: disable pkg-config stuff (left commented for reference)
#pkgconfigdir=$(libdir)/pkgconfig
#LIBEVENT_PKGCONFIG=libevent.pc

# These sources are conditionally added by configure.in or conditionally
# included from other files.
PLATFORM_DEPENDENT_SRC = \
	epoll_sub.c \
	arc4random.c

# Open MPI: removed test/Makefile.nmake and make-event-config.sed from
# EXTRA_DIST
EXTRA_DIST = \
	LICENSE \
	autogen.sh \
	event_rpcgen.py \
	libevent.pc.in \
	Doxyfile \
	whatsnew-2.0.txt \
	Makefile.nmake \
	$(PLATFORM_DEPENDENT_SRC)

# Open MPI: disable these lines
#LIBEVENT_LIBS_LA = libevent.la libevent_core.la libevent_extra.la
#if PTHREADS
#LIBEVENT_LIBS_LA += libevent_pthreads.la
#LIBEVENT_PKGCONFIG += libevent_pthreads.pc
#endif
#if OPENSSL
#LIBEVENT_LIBS_LA += libevent_openssl.la
#LIBEVENT_PKGCONFIG += libevent_openssl.pc
#endif

# Open MPI: Changed to noinst and libevent.la
noinst_LTLIBRARIES = libevent.la

# Open MPI: removed sample and test dirs
SUBDIRS = . include

if BUILD_WIN32

SYS_LIBS = -lws2_32 -lshell32 -ladvapi32
SYS_SRC = win32select.c evthread_win32.c buffer_iocp.c event_iocp.c \
	bufferevent_async.c
SYS_INCLUDES = -IWIN32-Code

else

SYS_LIBS =
SYS_SRC =
SYS_INCLUDES =

endif

if SELECT_BACKEND
SYS_SRC += select.c
endif
if POLL_BACKEND
SYS_SRC += poll.c
endif
if DEVPOLL_BACKEND
SYS_SRC += devpoll.c
endif
if KQUEUE_BACKEND
SYS_SRC += kqueue.c
endif
if EPOLL_BACKEND
SYS_SRC += epoll.c
endif
if EVPORT_BACKEND
SYS_SRC += evport.c
endif
if SIGNAL_SUPPORT
SYS_SRC += signal.c
endif

# Open MPI: commented this out.  We generate event-config.h in the
# component configure.m4.
#BUILT_SOURCES = include/event2/event-config.h

# Open MPI: commented this out.  We generate event-config.h in the
# component configure.m4
#include/event2/event-config.h: config.h make-event-config.sed
#	test -d include/event2 || $(MKDIR_P) include/event2
#	$(SED) -f $(srcdir)/make-event-config.sed < config.h > $@T
#	mv -f $@T $@

# Open MPI: -- eliminate extra libevent libraries. We compile
# everything that is enabled into the core. Do not include any
# code not lwrrently in use (buffer, tagging, and listener code).
CORE_SRC = event.c evthread.c evmap.c log.c evutil.c evutil_rand.c \
           strlcpy.c $(SYS_SRC)
CORE_LIBS =

if HTTP
CORE_SRC += http.c
endif
if DNS
CORE_SRC += evdns.c
endif
if RPC
CORE_SRC += evrpc.c
endif
if PTHREADS
CORE_SRC += evthread_pthread.c
endif
if OPENSSL
CORE_SRC += bufferevent_openssl.c
CORE_LIBS += -lcrypto -lssl
endif


# Open MPI - comment out all unused extra libs
#if BUILD_WITH_NO_UNDEFINED
#NO_UNDEFINED = -no-undefined
#MAYBE_CORE = libevent_core.la
#else
#NO_UNDEFINED =
#MAYBE_CORE =
#endif
#
#GENERIC_LDFLAGS = -version-info $(VERSION_INFO) $(RELEASE) $(NO_UNDEFINED)
#
#libevent_la_SOURCES = $(CORE_SRC) $(EXTRA_SRC)l
#libevent_la_LIBADD = @LTLIBOBJS@ $(SYS_LIBS)
#libevent_la_LDFLAGS = $(GENERIC_LDFLAGS)
#
#libevent_core_la_SOURCES = $(CORE_SRC)
#libevent_core_la_LIBADD = @LTLIBOBJS@ $(SYS_LIBS)
#libevent_core_la_LDFLAGS = $(GENERIC_LDFLAGS)
#
#if PTHREADS
#libevent_pthreads_la_SOURCES = evthread_pthread.c
#libevent_pthreads_la_LIBADD = $(MAYBE_CORE)
#libevent_pthreads_la_LDFLAGS = $(GENERIC_LDFLAGS)
#endif
#
#libevent_extra_la_SOURCES = $(EXTRA_SRC)
#libevent_extra_la_LIBADD = $(MAYBE_CORE) $(SYS_LIBS)
#libevent_extra_la_LDFLAGS = $(GENERIC_LDFLAGS)
#
#if OPENSSL
#libevent_openssl_la_SOURCES = bufferevent_openssl.c
#libevent_openssl_la_LIBADD = $(MAYBE_CORE) $(OPENSSL_LIBS)
#libevent_openssl_la_LDFLAGS = $(GENERIC_LDFLAGS)
#endif
#
#noinst_HEADERS = util-internal.h mm-internal.h ipv6-internal.h \
#	evrpc-internal.h strlcpy-internal.h evbuffer-internal.h \
#	bufferevent-internal.h http-internal.h event-internal.h \
#	evthread-internal.h ht-internal.h defer-internal.h \
#	minheap-internal.h log-internal.h evsignal-internal.h evmap-internal.h \
#	changelist-internal.h iocp-internal.h \
#	ratelim-internal.h \
#	WIN32-Code/event2/event-config.h \
#	WIN32-Code/tree.h \
#	compat/sys/queue.h
#
#EVENT1_HDRS = event.h evhttp.h evdns.h evrpc.h evutil.h
#
#if INSTALL_LIBEVENT
#include_HEADERS = $(EVENT1_HDRS)
#else
#noinst_HEADERS += $(EVENT1_HDRS)
#endif

# Open MPI - define libs and headers
libevent_la_SOURCES = $(CORE_SRC) $(headers)
libevent_la_LIBADD = $(CORE_LIBS) $(SYS_LIBS)
EXTRA_DIST += event.h evutil.h util-internal.h mm-internal.h ipv6-internal.h \
	strlcpy-internal.h evbuffer-internal.h \
	bufferevent-internal.h event-internal.h \
	evthread-internal.h defer-internal.h \
	minheap-internal.h log-internal.h evsignal-internal.h evmap-internal.h \
	changelist-internal.h iocp-internal.h \
	ratelim-internal.h \
	WIN32-Code/event2/event-config.h \
	WIN32-Code/tree.h \
	compat/sys/queue.h $(SYS_INCLUDES) \
        evhttp.h http-internal.h ht-internal.h \
        evrpc.h evrpc-internal.h \
        evdns.h

AM_CPPFLAGS = -I$(srcdir)/compat -I$(srcdir)/include -I./include $(SYS_INCLUDES)

# Open MPI: remove verify, doxygen, and FORCE
#verify: check
#
#doxygen: FORCE
#	doxygen $(srcdir)/Doxyfile
#FORCE:

DISTCLEANFILES = *~ libevent.pc ./include/event2/event-config.h

