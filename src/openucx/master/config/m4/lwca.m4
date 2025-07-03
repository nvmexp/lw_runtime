#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([UCX_CHECK_LWDA],[

AS_IF([test "x$lwda_checked" != "xyes"],
   [
    AC_ARG_WITH([lwca],
                [AS_HELP_STRING([--with-lwca=(DIR)], [Enable the use of LWCA (default is guess).])],
                [], [with_lwda=guess])

    AS_IF([test "x$with_lwda" = "xno"],
        [lwda_happy=no],
        [
         save_CPPFLAGS="$CPPFLAGS"
         save_LDFLAGS="$LDFLAGS"

         LWDA_CPPFLAGS=""
         LWDA_LDFLAGS=""

         AS_IF([test ! -z "$with_lwda" -a "x$with_lwda" != "xyes" -a "x$with_lwda" != "xguess"],
               [ucx_check_lwda_dir="$with_lwda"
                AS_IF([test -d "$with_lwda/lib64"], [libsuff="64"], [libsuff=""])
                ucx_check_lwda_libdir="$with_lwda/lib$libsuff"
                LWDA_CPPFLAGS="-I$with_lwda/include"
                LWDA_LDFLAGS="-L$ucx_check_lwda_libdir -L$ucx_check_lwda_libdir/stubs"])

         AS_IF([test ! -z "$with_lwda_libdir" -a "x$with_lwda_libdir" != "xyes"],
               [ucx_check_lwda_libdir="$with_lwda_libdir"
                LWDA_LDFLAGS="-L$ucx_check_lwda_libdir -L$ucx_check_lwda_libdir/stubs"])

         CPPFLAGS="$CPPFLAGS $LWDA_CPPFLAGS"
         LDFLAGS="$LDFLAGS $LWDA_LDFLAGS"

         # Check lwca header files
         AC_CHECK_HEADERS([lwca.h lwda_runtime.h],
                          [lwda_happy="yes"], [lwda_happy="no"])

         # Check lwca libraries
         AS_IF([test "x$lwda_happy" = "xyes"],
                [AC_CHECK_LIB([lwca], [lwDeviceGetUuid],
                              [LWDA_LDFLAGS="$LWDA_LDFLAGS -llwda"], [lwda_happy="no"])])
         AS_IF([test "x$lwda_happy" = "xyes"],
                [AC_CHECK_LIB([lwdart], [lwdaGetDeviceCount],
                              [LWDA_LDFLAGS="$LWDA_LDFLAGS -llwdart"], [lwda_happy="no"])])

         CPPFLAGS="$save_CPPFLAGS"
         LDFLAGS="$save_LDFLAGS"

         AS_IF([test "x$lwda_happy" = "xyes"],
               [AC_SUBST([LWDA_CPPFLAGS], ["$LWDA_CPPFLAGS"])
                AC_SUBST([LWDA_LDFLAGS], ["$LWDA_LDFLAGS"])
                AC_DEFINE([HAVE_LWDA], 1, [Enable LWCA support])],
               [AS_IF([test "x$with_lwda" != "xguess"],
                      [AC_MSG_ERROR([LWCA support is requested but lwca packages cannot be found])],
                      [AC_MSG_WARN([LWCA not found])])])

        ]) # "x$with_lwda" = "xno"

        lwda_checked=yes
        AM_CONDITIONAL([HAVE_LWDA], [test "x$lwda_happy" != xno])

   ]) # "x$lwda_checked" != "xyes"

]) # UCX_CHECK_LWDA
