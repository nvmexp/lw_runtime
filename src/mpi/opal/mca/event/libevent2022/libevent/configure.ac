dnl configure.in for libevent
dnl Copyright 2000-2007 Niels Provos
dnl Copyright 2007-2012 Niels Provos and Nick Mathewson
dnl
dnl See LICENSE for copying information.
dnl
dnl Original version Dug Song <dugsong@monkey.org>

AC_PREREQ(2.59c)
# Open MPI: changed to 4 arg AC INIT
AC_INIT([libevent], [2.0.21-stable], [http://libevent.org], [libevent])

AC_CONFIG_MACRO_DIR([m4])

# Open MPI: changed to one 1 arg AM INIT_AUTOMAKE
AM_INIT_AUTOMAKE([foreign subdir-objects])
# Open MPI: changed AM CONFIG_HEADER to AC CONFIG_HEADERS
AC_CONFIG_HEADERS(config.h)
AC_DEFINE(NUMERIC_VERSION, 0x02001500, [Numeric representation of the version])

dnl Initialize prefix.
if test "$prefix" = "NONE"; then
   prefix="/usr/local"
fi

AC_CANONICAL_BUILD
AC_CANONICAL_HOST
dnl the 'build' machine is where we run configure and compile
dnl the 'host' machine is where the resulting stuff runs.

case "$host_os" in

 osf5*)
    CFLAGS="$CFLAGS -D_OSF_SOURCE"
    ;;
esac

dnl Checks for programs.
AC_PROG_CC
AM_PROG_CC_C_O
AC_PROG_SED
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MKDIR_P

AC_PROG_GCC_TRADITIONAL

# We need to test for at least gcc 2.95 here, because older versions don't
# have -fno-strict-aliasing
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [
#if !defined(__GNUC__) || (__GNUC__ < 2) || (__GNUC__ == 2 && __GNUC_MINOR__ < 95)
#error
#endif])], have_gcc295=yes, have_gcc295=no)

if test "$GCC" = "yes" ; then
        # Enable many gcc warnings by default...
        CFLAGS="$CFLAGS -Wall"
	# And disable the strict-aliasing optimization, since it breaks
	# our sockaddr-handling code in strange ways.
	if test x$have_gcc295 = xyes; then
		CFLAGS="$CFLAGS -fno-strict-aliasing"
	fi
fi

# OS X Lion started deprecating the system openssl. Let's just disable
# all deprecation warnings on OS X.
if test "$GCC" = "yes" ; then
case "$host_os" in

 darwin*)
    CFLAGS="$CFLAGS -Wno-deprecated-declarations"
    ;;
esac
fi

AC_ARG_ENABLE(gcc-warnings,
     AS_HELP_STRING(--enable-gcc-warnings, enable verbose warnings with GCC))
AC_ARG_ENABLE(thread-support,
     AS_HELP_STRING(--disable-thread-support, disable support for threading),
	[], [enable_thread_support=yes])
AC_ARG_ENABLE(malloc-replacement,
     AS_HELP_STRING(--disable-malloc-replacement, disable support for replacing the memory mgt functions),
        [], [enable_malloc_replacement=yes])
AC_ARG_ENABLE(openssl,
     AS_HELP_STRING(--disable-openssl, disable support for openssl encryption),
        [], [enable_openssl=yes])
AC_ARG_ENABLE(debug-mode,
     AS_HELP_STRING(--disable-debug-mode, disable support for running in debug mode),
        [], [enable_debug_mode=yes])
AC_ARG_ENABLE([libevent-install],
     AS_HELP_STRING([--disable-libevent-install, disable installation of libevent]),
	[], [enable_libevent_install=yes])
AC_ARG_ENABLE([libevent-regress],
     AS_HELP_STRING([--disable-libevent-regress, skip regress in make check]),
	[], [enable_libevent_regress=yes])
AC_ARG_ENABLE([function-sections],
     AS_HELP_STRING([--enable-function-sections, make static library allow smaller binaries with --gc-sections]),
	[], [enable_function_sections=no])


dnl   *******  Open MPI Changes  ******
dnl Disable many options unless they are specifically enabled.  The
dnl defaults for these may change upstream.
AC_ARG_ENABLE([dns],
              [AS_HELP_STRING([--disable-dns], [disable DNS support])],
              [], [enable_dns=no])
AM_CONDITIONAL(DNS, [test "$enable_dns" = "yes"])
AC_DEFINE(HAVE_DNS, test "$enable_dns" = "yes", [whether we want DNS support])

AC_ARG_ENABLE([http],
              [AS_HELP_STRING([--disable-http], [disable HTTP support])],
              [], [enable_http=no])
AM_CONDITIONAL(HTTP, [test "$enable_http" = "yes"])
AC_DEFINE(HAVE_HTTP, test "$enable_http" = "yes", [whether we want HTTP suport])

AC_ARG_ENABLE([rpc],
              [AS_HELP_STRING([--disable-rpc], [disable RPC support])],
              [], [enable_rpc=no])
AM_CONDITIONAL(RPC, [test "$enable_rpc" = "yes"])
AC_DEFINE(HAVE_RPC, [test "$enable_rpc" = "yes"], [whether we want RPC suport])

AC_ARG_ENABLE([select],
              [AS_HELP_STRING([--disable-select], [disable select support])],
              [], [enable_select=yes])

AC_ARG_ENABLE([poll],
              [AS_HELP_STRING([--disable-poll], [disable poll support])],
              [], [enable_poll=yes])

AC_ARG_ENABLE([devpoll],
              [AS_HELP_STRING([--disable-devpoll], [disable devpoll support])],
              [], [enable_devpoll=yes])

AC_ARG_ENABLE([kqueue],
              [AS_HELP_STRING([--disable-kqueue], [disable kqueue support])],
              [], [enable_kqueue=yes])

AC_ARG_ENABLE([epoll],
              [AS_HELP_STRING([--disable-epoll], [disable epoll support])],
              [], [enable_epoll=yes])

AC_ARG_ENABLE([evport],
              [AS_HELP_STRING([--disable-evport], [disable evport support])],
              [], [enable_evport=yes])

AC_ARG_ENABLE([signal],
              [AS_HELP_STRING([--disable-signal], [disable signal support])],
              [], [enable_signal=yes])

dnl   ****** END Open MPI Changes ******

AC_PROG_LIBTOOL

dnl   Uncomment "AC_DISABLE_SHARED" to make shared librraries not get
dnl   built by default.  You can also turn shared libs on and off from
dnl   the command line with --enable-shared and --disable-shared.
dnl AC_DISABLE_SHARED
AC_SUBST(LIBTOOL_DEPS)

AM_CONDITIONAL([BUILD_REGRESS], [test "$enable_libevent_regress" = "yes"])

dnl Checks for libraries.
AC_SEARCH_LIBS([inet_ntoa], [nsl])
AC_SEARCH_LIBS([socket], [socket])
AC_SEARCH_LIBS([inet_aton], [resolv])
AC_SEARCH_LIBS([clock_gettime], [rt])
AC_SEARCH_LIBS([sendfile], [sendfile])

dnl   ****** Open MPI Changes (autoconf 2.68) ******
dnl - check if the macro WIN32 is defined on this compiler.
dnl - (this is how we check for a windows version of GCC)
AC_MSG_CHECKING(for WIN32)
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#ifndef WIN32
die horribly
#endif
]], [[int i;]])],
	[bwin32=true; AC_MSG_RESULT(yes)],
	[bwin32=false; AC_MSG_RESULT(no)]
)
dnl   ****** END Open MPI Changes (autoconf 2.68) ******

dnl - check if the macro __CYGWIN__ is defined on this compiler.
dnl - (this is how we check for a cygwin version of GCC)
AC_MSG_CHECKING(for CYGWIN)
AC_TRY_COMPILE(,
	[
#ifndef __CYGWIN__
die horribly
#endif
	],
	cygwin=true; AC_MSG_RESULT(yes),
	cygwin=false; AC_MSG_RESULT(no),
)

AC_CHECK_HEADERS([zlib.h])

if test "x$ac_cv_header_zlib_h" = "xyes"; then
dnl Determine if we have zlib for regression tests
dnl Don't put this one in LIBS
save_LIBS="$LIBS"
LIBS=""
ZLIB_LIBS=""
have_zlib=no
AC_SEARCH_LIBS([inflateEnd], [z],
	[have_zlib=yes
	ZLIB_LIBS="$LIBS"
	AC_DEFINE(HAVE_LIBZ, 1, [Define if the system has zlib])])
LIBS="$save_LIBS"
AC_SUBST(ZLIB_LIBS)
fi
AM_CONDITIONAL(ZLIB_REGRESS, [test "$have_zlib" = "yes"])

dnl See if we have openssl.  This doesn't go in LIBS either.
if test "$bwin32" = true; then
  EV_LIB_WS32=-lws2_32
  EV_LIB_GDI=-lgdi32
else
  EV_LIB_WS32=
  EV_LIB_GDI=
fi
AC_SUBST(EV_LIB_WS32)
AC_SUBST(EV_LIB_GDI)
AC_SUBST(OPENSSL_LIBADD)

AC_CHECK_HEADERS([openssl/bio.h])

if test "$enable_openssl" = "yes"; then
save_LIBS="$LIBS"
LIBS=""
OPENSSL_LIBS=""
have_openssl=no
AC_SEARCH_LIBS([SSL_new], [ssl],
	[have_openssl=yes
	OPENSSL_LIBS="$LIBS -lcrypto $EV_LIB_GDI $EV_LIB_WS32 $OPENSSL_LIBADD"
	AC_DEFINE(HAVE_OPENSSL, 1, [Define if the system has openssl])],
	[have_openssl=no],
	[-lcrypto $EV_LIB_GDI $EV_LIB_WS32 $OPENSSL_LIBADD])
LIBS="$save_LIBS"
AC_SUBST(OPENSSL_LIBS)
fi

dnl Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([fcntl.h stdarg.h inttypes.h stdint.h stddef.h poll.h unistd.h sys/epoll.h sys/time.h sys/queue.h sys/event.h sys/param.h sys/ioctl.h sys/select.h sys/devpoll.h port.h netinet/in.h netinet/in6.h sys/socket.h sys/uio.h arpa/inet.h sys/eventfd.h sys/mman.h sys/sendfile.h sys/wait.h netdb.h])
AC_CHECK_HEADERS([sys/stat.h])
AC_CHECK_HEADERS(sys/sysctl.h, [], [], [
#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
])
if test "x$ac_cv_header_sys_queue_h" = "xyes"; then
	AC_MSG_CHECKING(for TAILQ_FOREACH in sys/queue.h)
	AC_EGREP_CPP(yes,
[
#include <sys/queue.h>
#ifdef TAILQ_FOREACH
 yes
#endif
],	[AC_MSG_RESULT(yes)
	 AC_DEFINE(HAVE_TAILQFOREACH, 1,
		[Define if TAILQ_FOREACH is defined in <sys/queue.h>])],
	AC_MSG_RESULT(no)
	)
fi

if test "x$ac_cv_header_sys_time_h" = "xyes"; then
	AC_MSG_CHECKING(for timeradd in sys/time.h)
	AC_EGREP_CPP(yes,
[
#include <sys/time.h>
#ifdef timeradd
 yes
#endif
],	[ AC_DEFINE(HAVE_TIMERADD, 1,
		[Define if timeradd is defined in <sys/time.h>])
	  AC_MSG_RESULT(yes)] ,AC_MSG_RESULT(no)
)
fi

if test "x$ac_cv_header_sys_time_h" = "xyes"; then
	AC_MSG_CHECKING(for timercmp in sys/time.h)
	AC_EGREP_CPP(yes,
[
#include <sys/time.h>
#ifdef timercmp
 yes
#endif
],	[ AC_DEFINE(HAVE_TIMERCMP, 1,
		[Define if timercmp is defined in <sys/time.h>])
	  AC_MSG_RESULT(yes)] ,AC_MSG_RESULT(no)
)
fi

if test "x$ac_cv_header_sys_time_h" = "xyes"; then
	AC_MSG_CHECKING(for timerclear in sys/time.h)
	AC_EGREP_CPP(yes,
[
#include <sys/time.h>
#ifdef timerclear
 yes
#endif
],	[ AC_DEFINE(HAVE_TIMERCLEAR, 1,
		[Define if timerclear is defined in <sys/time.h>])
	  AC_MSG_RESULT(yes)] ,AC_MSG_RESULT(no)
)
fi

if test "x$ac_cv_header_sys_time_h" = "xyes"; then
	AC_MSG_CHECKING(for timerisset in sys/time.h)
	AC_EGREP_CPP(yes,
[
#include <sys/time.h>
#ifdef timerisset
 yes
#endif
],	[ AC_DEFINE(HAVE_TIMERISSET, 1,
		[Define if timerisset is defined in <sys/time.h>])
	  AC_MSG_RESULT(yes)] ,AC_MSG_RESULT(no)
)
fi

if test "x$ac_cv_header_sys_sysctl_h" = "xyes"; then
	AC_CHECK_DECLS([CTL_KERN, KERN_RANDOM, RANDOM_UUID, KERN_ARND], [], [],
	   [[#include <sys/types.h>
	     #include <sys/sysctl.h>]]
	)
fi

AM_CONDITIONAL(BUILD_WIN32, test x$bwin32 = xtrue)
AM_CONDITIONAL(BUILD_CYGWIN, test x$cygwin = xtrue)
AM_CONDITIONAL(BUILD_WITH_NO_UNDEFINED, test x$bwin32 = xtrue || test x$cygwin = xtrue)

if test x$bwin32 = xtrue; then
   AC_SEARCH_LIBS([getservbyname],[ws2_32])
fi

dnl Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_HEADER_TIME

dnl Checks for library functions.
AC_CHECK_FUNCS([gettimeofday vasprintf fcntl clock_gettime strtok_r strsep])
AC_CHECK_FUNCS([getnameinfo strlcpy inet_ntop inet_pton signal sigaction strtoll inet_aton pipe eventfd sendfile mmap splice arc4random arc4random_buf issetugid geteuid getegid getprotobynumber setelw unsetelw putelw sysctl])
AC_CHECK_FUNCS([umask])

AC_CACHE_CHECK(
    [for getaddrinfo],
    [libevent_cv_getaddrinfo],
    [AC_LINK_IFELSE(
	[AC_LANG_PROGRAM(
	    [[
		#ifdef HAVE_NETDB_H
		#include <netdb.h>
		#endif
	    ]],
	    [[
		getaddrinfo;
	    ]]
	)],
	[libevent_cv_getaddrinfo=yes],
	[libevent_cv_getaddrinfo=no]
    )]
)
if test "$libevent_cv_getaddrinfo" = "yes" ; then
    AC_DEFINE([HAVE_GETADDRINFO], [1], [Do we have getaddrinfo()?])
else

AC_CHECK_FUNCS([getservbyname])
# Check for gethostbyname_r in all its glorious incompatible versions.
#   (This is cut-and-pasted from Tor, which based its logic on
#   Python's configure.in.)
AH_TEMPLATE(HAVE_GETHOSTBYNAME_R,
  [Define this if you have any gethostbyname_r()])

AC_CHECK_FUNC(gethostbyname_r, [
  AC_MSG_CHECKING([how many arguments gethostbyname_r() wants])
  OLD_CFLAGS=$CFLAGS
  CFLAGS="$CFLAGS $MY_CPPFLAGS $MY_THREAD_CPPFLAGS $MY_CFLAGS"
  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
#include <netdb.h>
  ], [[
    char *cp1, *cp2;
    struct hostent *h1, *h2;
    int i1, i2;
    (void)gethostbyname_r(cp1,h1,cp2,i1,&h2,&i2);
  ]])],[
    AC_DEFINE(HAVE_GETHOSTBYNAME_R)
    AC_DEFINE(HAVE_GETHOSTBYNAME_R_6_ARG, 1,
     [Define this if gethostbyname_r takes 6 arguments])
    AC_MSG_RESULT(6)
  ], [
    AC_TRY_COMPILE([
#include <netdb.h>
    ], [
      char *cp1, *cp2;
      struct hostent *h1;
      int i1, i2;
      (void)gethostbyname_r(cp1,h1,cp2,i1,&i2);
    ], [
      AC_DEFINE(HAVE_GETHOSTBYNAME_R)
      AC_DEFINE(HAVE_GETHOSTBYNAME_R_5_ARG, 1,
        [Define this if gethostbyname_r takes 5 arguments])
      AC_MSG_RESULT(5)
   ], [
      AC_TRY_COMPILE([
#include <netdb.h>
     ], [
       char *cp1;
       struct hostent *h1;
       struct hostent_data hd;
       (void) gethostbyname_r(cp1,h1,&hd);
     ], [
       AC_DEFINE(HAVE_GETHOSTBYNAME_R)
       AC_DEFINE(HAVE_GETHOSTBYNAME_R_3_ARG, 1,
         [Define this if gethostbyname_r takes 3 arguments])
       AC_MSG_RESULT(3)
     ], [
       AC_MSG_RESULT(0)
     ])
  ])
 ])
 CFLAGS=$OLD_CFLAGS
])

fi

AC_CHECK_SIZEOF(long)

AC_MSG_CHECKING(for F_SETFD in fcntl.h)
AC_EGREP_CPP(yes,
[
#define _GNU_SOURCE
#include <fcntl.h>
#ifdef F_SETFD
yes
#endif
],	[ AC_DEFINE(HAVE_SETFD, 1,
	      [Define if F_SETFD is defined in <fcntl.h>])
	  AC_MSG_RESULT(yes) ], AC_MSG_RESULT(no))

needsignal=no
haveselect=no
if test x$bwin32 != xtrue; then
    AC_CHECK_FUNCS(select, [haveselect=yes], )
    if test "x$haveselect" = "xyes" ; then
 	needsignal=yes
    fi
fi
AM_CONDITIONAL(SELECT_BACKEND, [test "x$haveselect" = "xyes"])

dnl   ****** Open MPI Changes ******
AM_CONDITIONAL(SELECT_BACKEND, [test "x$haveselect" = "xyes" -a "$enable_select" != "no"])
AC_MSG_CHECKING([for select support])
AS_IF([test "$enable_select" != "no" && test "x$haveselect" = "xyes"],
      [AC_DEFINE(HAVE_SELECT, 1, [Have select support])
       AC_MSG_RESULT([yes]) ],
      [AC_DEFINE(HAVE_SELECT, 0, [Have select support])
       AC_MSG_RESULT([no])])

dnl Check for poll.  Acknowledge the dolwmented break of "poll" for
dnl Mac Darwin.
if test "$enable_poll" != "no" ; then
    if test "$host" = "apple-darwin" ; then
        havepoll=no
    else
      havepoll=no
      AC_CHECK_FUNCS(poll, [havepoll=yes], )
    fi
    if test "x$havepoll" = "xyes" ; then
	needsignal=yes
    fi
fi
AM_CONDITIONAL(POLL_BACKEND, [test "x$havepoll" = "xyes" -a "$enable_poll" != "no"])
AC_MSG_CHECKING([for poll support])
AS_IF([test "$enable_poll" != "no" && test "x$havepoll" = "xyes"],
      [AC_DEFINE(HAVE_POLL, 1, [Have poll support])
       AC_MSG_RESULT([yes])],
      [AC_DEFINE(HAVE_POLL, 0, [No poll support])
       AC_MSG_RESULT([no])])


AC_MSG_CHECKING([for /dev/poll support])
AM_CONDITIONAL(DEVPOLL_BACKEND, [test "x$ac_cv_header_sys_devpoll_h" = "xyes" -a "$enable_devpoll" != "no"])
AS_IF([test "$enable_devpoll" != "no" && test "x$ac_cv_header_sys_devpoll_h" = "xyes"],
      [AC_DEFINE(HAVE_DEVPOLL, 1,
                 [Define if /dev/poll is available])
       AC_MSG_RESULT([yes])],
      [AC_DEFINE(HAVE_DEVPOLL, 0,
                 [Devpoll support available])
       AC_MSG_RESULT([no])])
dnl   ****** END Open MPI Changes ******

havekqueue=no
if test "x$ac_cv_header_sys_event_h" = "xyes"; then
dnl   ****** Open MPI Changes ******
    # All versions of Mac OS X before at least 10.5.2 are completely
    # broken when kqueue is used with pty. So, until they get fixed,
    # completely disable kqueue on Mac OS X (note: kqueue/pty support
    # has not been tested with 10.6 or beyond).
    case "$host" in
        *apple-darwin*)
                  AC_MSG_CHECKING(for working kqueue)
            AC_MSG_RESULT([no (MAC OS X)])
        ;;
        *)
            AC_CHECK_FUNCS(kqueue, [havekqueue=yes], )
        ;;
    esac
dnl   ****** End Open MPI Changes ******
	if test "x$havekqueue" = "xyes" ; then
		AC_MSG_CHECKING(for working kqueue)
		AC_TRY_RUN(
#include <sys/types.h>
#include <sys/time.h>
#include <sys/event.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

int
main(int argc, char **argv)
{
	int kq;
	int n;
	int fd[[2]];
	struct kevent ev;
	struct timespec ts;
	char buf[[8000]];

	if (pipe(fd) == -1)
		exit(1);
	if (fcntl(fd[[1]], F_SETFL, O_NONBLOCK) == -1)
		exit(1);

	while ((n = write(fd[[1]], buf, sizeof(buf))) == sizeof(buf))
		;

        if ((kq = kqueue()) == -1)
		exit(1);

	memset(&ev, 0, sizeof(ev));
	ev.ident = fd[[1]];
	ev.filter = EVFILT_WRITE;
	ev.flags = EV_ADD | EV_ENABLE;
	n = kevent(kq, &ev, 1, NULL, 0, NULL);
	if (n == -1)
		exit(1);

	read(fd[[0]], buf, sizeof(buf));

	ts.tv_sec = 0;
	ts.tv_nsec = 0;
	n = kevent(kq, NULL, 0, &ev, 1, &ts);
	if (n == -1 || n == 0)
		exit(1);

	exit(0);
}, [AC_MSG_RESULT(yes)
    AC_DEFINE(HAVE_WORKING_KQUEUE, 1,
		[Define if kqueue works correctly with pipes])
    havekqueue=yes
    ], AC_MSG_RESULT(no), AC_MSG_RESULT(no))
	fi
fi
AM_CONDITIONAL(KQUEUE_BACKEND, [test "x$havekqueue" = "xyes"])

haveepollsyscall=no
haveepoll=no
dnl   ****** Open MPI Changes ******
AC_MSG_CHECKING([for epoll support])
if test "$enable_epoll" != "no" ; then
    AC_MSG_RESULT([enabled])
    AC_CHECK_FUNCS(epoll_ctl, [haveepoll=yes], )
    if test "x$haveepoll" = "xyes" -a "$cross_compiling" != "yes" ; then
        # Unfortunately, it's not sufficient to just test for the
        # existence of the epoll_ctl symbol on some Linux distros
        # (e.g., Fedora 9), where the function is defined and you can
        # link against it, but it's hardwired to return ENOSYS -- and
        # /usr/include/gnu/stubs.h fails to define __stub_epoll_ctl
        # (the usual mechanism in glibc to indicate that a function is
        # a stub and isn't really implemented).  Hence, checking for
        # the symbol succeeds because it thinks it can use epoll_ctl
        # (and friends).  So we have to do an actual test after we
        # determine that epoll_ctl is linkable.  Grumble.  If we are
        # cross compiling, all we can do is trust AC_CHECK_FUNCS and
        # pray.

        # Unfortunately, there's also another potential
        # incompatibility.  The event_poll struct is defined in the
        # sys/epoll.h file.  The structure is the interface between
        # the application and the kernel and is therefore compiled
        # into both.  The event_poll struct is defined with a compiler
        # directive __attribute__ ((__packed__).  It turns out that
        # there is at least one compiler (Sun Studio) that does not
        # lwrrently recognize this directive.  This means that the
        # event_poll struct may be packed in the kernel, but not in
        # the libevent library.  Badness ensues.  Therefore, check to
        # see that this struct gets correctly passed between userspace
        # and the kernel.

        # In this test, we use epoll in Level Triggered mode. We create
        # a pipe and the write only file descriptor of the pipe is
        # added to the epoll set. The test is successful if
        # epoll_wait() returns 1 indicating that the fd is ready to be
        # written to.

        haveepoll=no
        AC_RUN_IFELSE([AC_LANG_PROGRAM([
AC_INCLUDES_DEFAULT
#include <sys/epoll.h>
],[[
    struct epoll_event epevin;
    struct epoll_event epevout;
    int res;
    int epfd;
    int fildes[2];

    if ((epfd = epoll_create(1)) == -1)
        exit(1);
    if (pipe(&fildes[0]) < 0)
        exit(1);
    memset(&epevin, 0, sizeof(epevin));
    memset(&epevout, 0, sizeof(epevout));
    memset(&epevin.data.ptr, 5, sizeof(epevin.data.ptr));
    epevin.events = EPOLLIN | EPOLLOUT;

    if (epoll_ctl(epfd, EPOLL_CTL_ADD, fildes[1], &epevin) == -1)
        exit(1);

    res = epoll_wait(epfd, &epevout, 1, 0);
    if (res != 1) {
        exit(1);
    } else {
        if (epevout.data.ptr != epevin.data.ptr) {
            exit(1);
        }
    }
    /* SUCCESS */
]])],
        [haveepoll=yes
        # OMPI: Don't use AC_LIBOBJ
        needsignal=yes],
        [haveepoll=no], [])
	AC_MSG_CHECKING([if epoll can build])
        AC_MSG_RESULT([$haveepoll])
    fi
else
    AC_MSG_RESULT([disabled])
fi
AM_CONDITIONAL(EPOLL_BACKEND, [test "x$haveepoll" = "xyes" -a "$enable_epoll" != "no"])
AC_MSG_CHECKING([for working epoll library interface])
AS_IF([test "$enable_epoll" != "no" && test "x$haveepoll" = "xyes"],
      [AC_DEFINE(HAVE_EPOLL, 1,
                 [Define if your system supports the epoll interface])
       AC_MSG_RESULT([yes])],
      [AC_DEFINE(HAVE_EPOLL, 0,
                 [No epoll interface support])
       AC_MSG_RESULT([no])])

dnl Check for epollsyscall if epoll not found.
haveepollsyscall=no
if test "x$ac_cv_header_sys_epoll_h" = "xyes" -a "x$haveepoll" = "xno" -a "$cross_compiling" != "yes"; then
        # See comment above.  This test uses the epoll syscalls
        # instead of the library interface.
        AC_MSG_CHECKING(for working epoll system call)
        AC_RUN_IFELSE([AC_LANG_PROGRAM([
AC_INCLUDES_DEFAULT
#include <sys/syscall.h>
#include <sys/epoll.h>
],[[
    struct epoll_event epevin;
    struct epoll_event epevout;
    int res;
    int epfd;
    int fildes[[2]];

    if ((epfd = syscall(__NR_epoll_create, 1)) == -1)
        exit(1);
    if (pipe(&fildes[[0]]) < 0)
        exit(1);
    memset(&epevin, 0, sizeof(epevin));
    memset(&epevout, 0, sizeof(epevout));
    memset(&epevin.data.ptr, 5, sizeof(epevin.data.ptr));
    epevin.events = EPOLLIN | EPOLLOUT;

    if (syscall(__NR_epoll_ctl, epfd,
        EPOLL_CTL_ADD, fildes[[1]], &epevin) == -1)
        exit(1);

    res = syscall(__NR_epoll_wait, epfd, &epevout, 1, 0);
    if (res != 1) {
        exit(1);
    } else {
        if (epevout.data.ptr != epevin.data.ptr) {
            exit(1);
        }
    }
    /* SUCCESS */
]])],
        [haveepollsyscall=yes
        # OMPI: don't use AC_LIBOBJ
        needsignal=yes])
        AC_MSG_RESULT([$haveepollsyscall])
fi
AC_MSG_CHECKING([for epoll syscall support])
AS_IF([test "$enable_epoll" != "no" && test "x$haveepollsyscall" = "xyes"],
      [AC_DEFINE(HAVE_EPOLL, 1,
                 [Define if your system supports the epoll interface])
       AC_MSG_RESULT([yes])],
      [AC_MSG_RESULT([no])])

haveeventports=no
if test "$enable_evport" != "no" ; then
    AC_CHECK_FUNCS(port_create, [haveeventports=yes], )
    if test "x$haveeventports" = "xyes" -a "$enable_evport" != "no" ; then
	needsignal=yes
    fi
fi
AM_CONDITIONAL(EVPORT_BACKEND, [test "x$haveeventports" = "xyes" -a "$enable_evport" != "no"])
AC_MSG_CHECKING([for evport support])
AS_IF([test "$enable_evport" != "no" && test "x$haveeventports" = "xyes"],
      [AC_DEFINE(HAVE_EVENT_PORTS, 1,
		[Define if your system supports event ports])
       AC_MSG_RESULT([yes])],
      [AC_DEFINE(HAVE_EVENT_PORTS, 0,
		[No event port support])
       AC_MSG_RESULT([no])])

dnl Check for eventops
AC_MSG_CHECKING(event_ops)
if test "$enable_select" != "no" && test "x$haveselect" = "xyes" ; then
    have_ops=yes
elif test "$enable_poll" != "no" && test "x$havepoll" = "xyes" ; then
    have_ops=yes
elif test "$enable_kqueue" != "no" && test "x$havekqueue" = "xyes" ; then
    have_ops=yes
elif test "$enable_epoll" != "no" && test "x$haveepoll" = "xyes" ; then
    have_ops=yes
else
    have_ops=no
fi

AS_IF([test "$have_ops" = "yes"],
      [AC_MSG_RESULT(yes)
       AC_DEFINE(HAVE_WORKING_EVENTOPS, 1, [Define if there is a working event op])],
      [AC_MSG_RESULT(no)])
AC_MSG_CHECKING([for working ops])
AS_IF([test "$have_ops" = "yes"],
      [AC_MSG_RESULT([yes])],
      [AC_MSG_RESULT([no])])

dnl   ****** End OPEN MPI Changes ******

AM_CONDITIONAL(SIGNAL_SUPPORT, [test "x$needsignal" = "xyes"])

AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T

AC_CHECK_TYPES([uint64_t, uint32_t, uint16_t, uint8_t, uintptr_t], , ,
[#ifdef HAVE_STDINT_H
#include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
#include <inttypes.h>
#endif
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif])

AC_CHECK_TYPES([fd_mask], , ,
[#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SELECT_H
#include <sys/select.h>
#endif])

AC_CHECK_SIZEOF(long long)
AC_CHECK_SIZEOF(long)
AC_CHECK_SIZEOF(int)
AC_CHECK_SIZEOF(short)
AC_CHECK_SIZEOF(size_t)
AC_CHECK_SIZEOF(void *)

AC_CHECK_TYPES([struct in6_addr, struct sockaddr_in6, sa_family_t, struct addrinfo, struct sockaddr_storage], , ,
[#define _GNU_SOURCE
#include <sys/types.h>
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_IN6_H
#include <netinet/in6.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef WIN32
#define WIN32_WINNT 0x400
#define _WIN32_WINNT 0x400
#define WIN32_LEAN_AND_MEAN
#if defined(_MSC_VER) && (_MSC_VER < 1300)
#include <winsock.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#endif
])
AC_CHECK_MEMBERS([struct in6_addr.s6_addr32, struct in6_addr.s6_addr16, struct sockaddr_in.sin_len, struct sockaddr_in6.sin6_len, struct sockaddr_storage.ss_family, struct sockaddr_storage.__ss_family], , ,
[#include <sys/types.h>
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_NETINET_IN6_H
#include <netinet/in6.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef WIN32
#define WIN32_WINNT 0x400
#define _WIN32_WINNT 0x400
#define WIN32_LEAN_AND_MEAN
#if defined(_MSC_VER) && (_MSC_VER < 1300)
#include <winsock.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#endif
#endif
])

AC_MSG_CHECKING([for socklen_t])
AC_TRY_COMPILE([
 #include <sys/types.h>
 #include <sys/socket.h>],
  [socklen_t x;],
  AC_MSG_RESULT([yes]),
  [AC_MSG_RESULT([no])
  AC_DEFINE(socklen_t, unsigned int,
	[Define to unsigned int if you dont have it])]
)

AC_MSG_CHECKING([whether our compiler supports __func__])
AC_TRY_COMPILE([],
 [ const char *cp = __func__; ],
 AC_MSG_RESULT([yes]),
 AC_MSG_RESULT([no])
 AC_MSG_CHECKING([whether our compiler supports __FUNCTION__])
 AC_TRY_COMPILE([],
   [ const char *cp = __FUNCTION__; ],
   AC_MSG_RESULT([yes])
   AC_DEFINE(__func__, __FUNCTION__,
         [Define to appropriate substitue if compiler doesnt have __func__]),
   AC_MSG_RESULT([no])
   AC_DEFINE(__func__, __FILE__,
         [Define to appropriate substitue if compiler doesnt have __func__])))


# check if we can compile with pthreads
have_pthreads=no
if test x$bwin32 != xtrue && test "$enable_thread_support" != "no"; then
  ACX_PTHREAD([
	AC_DEFINE(HAVE_PTHREADS, 1,
		[Define if we have pthreads on this system])
	have_pthreads=yes])
  CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
  AC_CHECK_SIZEOF(pthread_t, ,
     [AC_INCLUDES_DEFAULT()
      #include <pthread.h> ]
  )
fi
AM_CONDITIONAL(PTHREADS, [test "$have_pthreads" != "no" && test "$enable_thread_support" != "no"])

# check if we should compile locking into the library
if test x$enable_thread_support = xno; then
   AC_DEFINE(DISABLE_THREAD_SUPPORT, 1,
	[Define if libevent should not be compiled with thread support])
fi

# check if we should hard-code the mm functions.
if test x$enable_malloc_replacement = xno; then
  AC_DEFINE(DISABLE_MM_REPLACEMENT, 1,
        [Define if libevent should not allow replacing the mm functions])
fi

# check if we should hard-code debugging out
if test x$enable_debug_mode = xno; then
  AC_DEFINE(DISABLE_DEBUG_MODE, 1,
        [Define if libevent should build without support for a debug mode])
fi

# check if we have and should use openssl
AM_CONDITIONAL(OPENSSL, [test "$enable_openssl" != "no" && test "$have_openssl" = "yes"])

# Add some more warnings which we use in development but not in the
# released versions.  (Some relevant gcc versions can't handle these.)
if test x$enable_gcc_warnings = xyes && test "$GCC" = "yes"; then

  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [
#if !defined(__GNUC__) || (__GNUC__ < 4)
#error
#endif])], have_gcc4=yes, have_gcc4=no)

  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [
#if !defined(__GNUC__) || (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 2)
#error
#endif])], have_gcc42=yes, have_gcc42=no)

  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [
#if !defined(__GNUC__) || (__GNUC__ < 4) || (__GNUC__ == 4 && __GNUC_MINOR__ < 5)
#error
#endif])], have_gcc45=yes, have_gcc45=no)

  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [
#if !defined(__clang__)
#error
#endif])], have_clang=yes, have_clang=no)

  CFLAGS="$CFLAGS -W -Wfloat-equal -Wundef -Wpointer-arith -Wstrict-prototypes -Wmissing-prototypes -Wwrite-strings -Wredundant-decls -Wchar-subscripts -Wcomment -Wformat -Wwrite-strings -Wmissing-declarations -Wredundant-decls -Wnested-externs -Wbad-function-cast -Wswitch-enum -Werror"
  CFLAGS="$CFLAGS -Wno-unused-parameter -Wstrict-aliasing"

  if test x$have_gcc4 = xyes ; then
    # These warnings break gcc 3.3.5 and work on gcc 4.0.2
    CFLAGS="$CFLAGS -Winit-self -Wmissing-field-initializers -Wdeclaration-after-statement"
    #CFLAGS="$CFLAGS -Wold-style-definition"
  fi

  if test x$have_gcc42 = xyes ; then
    # These warnings break gcc 4.0.2 and work on gcc 4.2
    CFLAGS="$CFLAGS -Waddress"
  fi

  if test x$have_gcc42 = xyes && test x$have_clang = xno; then
    # These warnings break gcc 4.0.2 and clang, but work on gcc 4.2
    CFLAGS="$CFLAGS -Wnormalized=id -Woverride-init"
  fi

  if test x$have_gcc45 = xyes ; then
    # These warnings work on gcc 4.5
    CFLAGS="$CFLAGS -Wlogical-op"
  fi

  if test x$have_clang = xyes; then
    # Disable the unused-function warnings, because these trigger
    # for minheap-internal.h related code.
    CFLAGS="$CFLAGS -Wno-unused-function"
  fi

##This will break the world on some 64-bit architectures
# CFLAGS="$CFLAGS -Winline"

fi

LIBEVENT_GC_SECTIONS=
if test "$GCC" = yes && test "$enable_function_sections" = yes ; then
    AC_CACHE_CHECK(
	[if linker supports omitting unused code and data],
	[libevent_cv_gc_sections_runs],
	[
	    dnl  NetBSD will link but likely not run with --gc-sections
	    dnl  http://bugs.ntp.org/1844
	    dnl  http://gnats.netbsd.org/40401
	    dnl  --gc-sections causes attempt to load as linux elf, with
	    dnl  wrong syscalls in place.  Test a little gauntlet of
	    dnl  simple stdio read code checking for errors, expecting
	    dnl  enough syscall differences that the NetBSD code will
	    dnl  fail even with Linux emulation working as designed.
	    dnl  A shorter test could be refined by someone with access
	    dnl  to a NetBSD host with Linux emulation working.
	    origCFLAGS="$CFLAGS"
	    CFLAGS="$CFLAGS -Wl,--gc-sections"
	    AC_LINK_IFELSE(
		[AC_LANG_PROGRAM(
		    [[
			#include <stdlib.h>
			#include <stdio.h>
		    ]],
		    [[
			FILE *	fpC;
			char	buf[32];
			size_t	cch;
			int	read_success_once;

			fpC = fopen("conftest.c", "r");
			if (NULL == fpC)
				exit(1);
			do {
				cch = fread(buf, sizeof(buf), 1, fpC);
				read_success_once |= (0 != cch);
			} while (0 != cch);
			if (!read_success_once)
				exit(2);
			if (!feof(fpC))
				exit(3);
			if (0 != fclose(fpC))
				exit(4);

			exit(EXIT_SUCCESS);
		    ]]
		)],
		[
                    dnl We have to do this invocation manually so that we can
                    dnl get the output of conftest.err to make sure it doesn't
                    dnl mention gc-sections.
		    if test "X$cross_compiling" = "Xyes" || grep gc-sections conftest.err ; then
			libevent_cv_gc_sections_runs=no
		    else
			libevent_cv_gc_sections_runs=no
			./conftest >/dev/null 2>&1 && libevent_cv_gc_sections_runs=yes
		    fi
		],
		[libevent_cv_gc_sections_runs=no]
	    )
	    CFLAGS="$origCFLAGS"
	    AS_UNSET([origCFLAGS])
	]
    )
    case "$libevent_cv_gc_sections_runs" in
     yes)
	CFLAGS="-ffunction-sections -fdata-sections $CFLAGS"
	LIBEVENT_GC_SECTIONS="-Wl,--gc-sections"
	;;
    esac
fi
AC_SUBST([LIBEVENT_GC_SECTIONS])

AM_CONDITIONAL([INSTALL_LIBEVENT], [test "$enable_libevent_install" = "yes"])

AC_CONFIG_FILES( [libevent.pc libevent_openssl.pc libevent_pthreads.pc] )
######    Open MPI CHANGE - remove test/Makefile and sample/Makefile from line below    #####
AC_OUTPUT(Makefile include/Makefile)
