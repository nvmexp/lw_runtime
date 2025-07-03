dnl -*- shell-script -*-
dnl
dnl Copyright (c) 2004-2010 The Trustees of Indiana University and Indiana
dnl                         University Research and Technology
dnl                         Corporation.  All rights reserved.
dnl Copyright (c) 2004-2005 The University of Tennessee and The University
dnl                         of Tennessee Research Foundation.  All rights
dnl                         reserved.
dnl Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
dnl                         University of Stuttgart.  All rights reserved.
dnl Copyright (c) 2004-2005 The Regents of the University of California.
dnl                         All rights reserved.
dnl Copyright (c) 2006-2016 Cisco Systems, Inc.  All rights reserved.
dnl Copyright (c) 2007      Sun Microsystems, Inc.  All rights reserved.
dnl Copyright (c) 2009      IBM Corporation.  All rights reserved.
dnl Copyright (c) 2009      Los Alamos National Security, LLC.  All rights
dnl                         reserved.
dnl Copyright (c) 2009-2011 Oak Ridge National Labs.  All rights reserved.
dnl Copyright (c) 2011-2015 LWPU Corporation.  All rights reserved.
dnl Copyright (c) 2015      Research Organization for Information Science
dnl                         and Technology (RIST). All rights reserved.
dnl
dnl $COPYRIGHT$
dnl
dnl Additional copyrights may follow
dnl
dnl $HEADER$
dnl

AC_DEFUN([OPAL_CHECK_LWDA],[
#
# Check to see if user wants LWCA support
#
AC_ARG_WITH([lwca],
            [AC_HELP_STRING([--with-lwca(=DIR)],
            [Build lwca support, optionally adding DIR/include])])
AC_MSG_CHECKING([if --with-lwca is set])

# Note that LWCA support is off by default.  To turn it on, the user has to
# request it.  The user can just ask for --with-lwca and it that case we
# look for the lwca.h file in /usr/local/lwca.  Otherwise, they can give
# us a directory.  If they provide a directory, we will look in that directory
# as well as the directory with the "include" string appended to it.  The fact
# that we check in two directories precludes us from using the OMPI_CHECK_DIR
# macro as that would error out after not finding it in the first directory.
# Note that anywhere LWCA aware code is in the Open MPI repository requires
# us to make use of AC_REQUIRE to ensure this check has been done.
AS_IF([test "$with_lwda" = "no" || test "x$with_lwda" = "x"],
      [opal_check_lwda_happy="no"
       AC_MSG_RESULT([not set (--with-lwca=$with_lwda)])],
      [AS_IF([test "$with_lwda" = "yes"],
             [AS_IF([test "x`ls /usr/local/lwca/include/lwca.h 2> /dev/null`" = "x"],
                    [AC_MSG_RESULT([not found in standard location])
                     AC_MSG_WARN([Expected file /usr/local/lwca/include/lwca.h not found])
                     AC_MSG_ERROR([Cannot continue])],
                    [AC_MSG_RESULT([found])
                     opal_check_lwda_happy=yes
                     opal_lwda_incdir=/usr/local/lwca/include])],
             [AS_IF([test ! -d "$with_lwda"],
                    [AC_MSG_RESULT([not found])
                     AC_MSG_WARN([Directory $with_lwda not found])
                     AC_MSG_ERROR([Cannot continue])],
                    [AS_IF([test "x`ls $with_lwda/include/lwca.h 2> /dev/null`" = "x"],
                           [AS_IF([test "x`ls $with_lwda/lwca.h 2> /dev/null`" = "x"],
                                  [AC_MSG_RESULT([not found])
                                   AC_MSG_WARN([Could not find lwca.h in $with_lwda/include or $with_lwda])
                                   AC_MSG_ERROR([Cannot continue])],
                                  [opal_check_lwda_happy=yes
                                   opal_lwda_incdir=$with_lwda
                                   AC_MSG_RESULT([found ($with_lwda/lwca.h)])])],
                           [opal_check_lwda_happy=yes
                            opal_lwda_incdir="$with_lwda/include"
                            AC_MSG_RESULT([found ($opal_lwda_incdir/lwca.h)])])])])])

dnl We cannot have LWCA support without dlopen support.  HOWEVER, at
dnl this point in configure, we can't know whether the DL framework
dnl has been configured or not yet (it likely hasn't, since LWCA is a
dnl common framework, and likely configured first).  So we have to
dnl defer this check until later (see the OPAL_CHECK_LWDA_AFTER_OPAL_DL m4
dnl macro, below).  :-(

# We require LWCA IPC support which started in LWCA 4.1. Error
# out if the support is not there.
AS_IF([test "$opal_check_lwda_happy" = "yes"],
    [AC_CHECK_MEMBER([struct LWipcMemHandle_st.reserved],
        [],
        [AC_MSG_ERROR([Cannot continue because LWCA 4.1 or later is required])],
        [#include <$opal_lwda_incdir/lwca.h>])],
    [])

# If we have LWCA support, check to see if we have support for SYNC_MEMOPS
# which was first introduced in LWCA 6.0.
AS_IF([test "$opal_check_lwda_happy"="yes"],
    AC_CHECK_DECL([LW_POINTER_ATTRIBUTE_SYNC_MEMOPS], [LWDA_SYNC_MEMOPS=1], [LWDA_SYNC_MEMOPS=0],
        [#include <$opal_lwda_incdir/lwca.h>]),
    [])

# If we have LWCA support, check to see if we have LWCA 6.0 or later.
AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[#include <$opal_lwda_incdir/lwca.h>]],
        [[
#if LWDA_VERSION < 6000
#error "LWDA_VERSION is less than 6000"
#endif
        ]])],
        [LWDA_VERSION_60_OR_GREATER=1],
        [LWDA_VERSION_60_OR_GREATER=0])

# If we have LWCA support, check to see if we have support for lwPointerGetAttributes
# which was first introduced in LWCA 7.0.
AS_IF([test "$opal_check_lwda_happy"="yes"],
    AC_CHECK_DECL([lwPointerGetAttributes], [LWDA_GET_ATTRIBUTES=1], [LWDA_GET_ATTRIBUTES=0],
        [#include <$opal_lwda_incdir/lwca.h>]),
    [])

AC_MSG_CHECKING([if have lwca support])
if test "$opal_check_lwda_happy" = "yes"; then
    AC_MSG_RESULT([yes (-I$opal_lwda_incdir)])
    LWDA_SUPPORT=1
    opal_datatype_lwda_CPPFLAGS="-I$opal_lwda_incdir"
    AC_SUBST([opal_datatype_lwda_CPPFLAGS])
else
    AC_MSG_RESULT([no])
    LWDA_SUPPORT=0
fi

OPAL_SUMMARY_ADD([[Miscellaneous]],[[LWCA support]],[opal_lwda], [$opal_check_lwda_happy])

AM_CONDITIONAL([OPAL_lwda_support], [test "x$LWDA_SUPPORT" = "x1"])
AC_DEFINE_UNQUOTED([OPAL_LWDA_SUPPORT],$LWDA_SUPPORT,
                   [Whether we want lwca device pointer support])

AM_CONDITIONAL([OPAL_lwda_sync_memops], [test "x$LWDA_SYNC_MEMOPS" = "x1"])
AC_DEFINE_UNQUOTED([OPAL_LWDA_SYNC_MEMOPS],$LWDA_SYNC_MEMOPS,
                   [Whether we have LWCA LW_POINTER_ATTRIBUTE_SYNC_MEMOPS support available])

AM_CONDITIONAL([OPAL_lwda_get_attributes], [test "x$LWDA_GET_ATTRIBUTES" = "x1"])
AC_DEFINE_UNQUOTED([OPAL_LWDA_GET_ATTRIBUTES],$LWDA_GET_ATTRIBUTES,
                   [Whether we have LWCA lwPointerGetAttributes function available])

# There is nothing specific we can check for to see if GPU Direct RDMA is available.
# Therefore, we check to see whether we have LWCA 6.0 or later.
AM_CONDITIONAL([OPAL_lwda_gdr_support], [test "x$LWDA_VERSION_60_OR_GREATER" = "x1"])
AC_DEFINE_UNQUOTED([OPAL_LWDA_GDR_SUPPORT],$LWDA_VERSION_60_OR_GREATER,
                   [Whether we have LWCA GDR support available])

])

dnl
dnl LWCA support requires DL support (it dynamically opens the LWCA
dnl library at run time).  But we do not check for OPAL DL support
dnl until lafter the initial OPAL_CHECK_LWDA is called.  So put the
dnl LWCA+DL check in a separate macro that can be called after the DL MCA
dnl framework checks in the top-level configure.ac.
dnl
AC_DEFUN([OPAL_CHECK_LWDA_AFTER_OPAL_DL],[

    # We cannot have LWCA support without OPAL DL support.  Error out
    # if the user wants LWCA but we do not have OPAL DL support.
    AS_IF([test $OPAL_HAVE_DL_SUPPORT -eq 0 && \
           test "$opal_check_lwda_happy" = "yes"],
          [AC_MSG_WARN([--with-lwca was specified, but dlopen support is disabled.])
           AC_MSG_WARN([You must reconfigure Open MPI with dlopen ("dl") support.])
           AC_MSG_ERROR([Cannot continue.])])
])
