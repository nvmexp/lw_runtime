# -*- shell-script -*-
#
# Copyright (c) 2011-2013 LWPU Corporation.  All rights reserved.
# Copyright (c) 2013      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

#
# If LWCA support was requested, then build the LWCA support library.
# This code checks just makes sure the check was done earlier by the
# opal_check_lwda.m4 code.
#

AC_DEFUN([MCA_opal_common_lwda_CONFIG],[
    AC_CONFIG_FILES([opal/mca/common/lwca/Makefile])

    # make sure that LWCA-aware checks have been done
    AC_REQUIRE([OPAL_CHECK_LWDA])

    AS_IF([test "x$LWDA_SUPPORT" = "x1"],
          [$1],
          [$2])

    # Copy over the includes needed to build LWCA
    common_lwda_CPPFLAGS=$opal_datatype_lwda_CPPFLAGS
    AC_SUBST([common_lwda_CPPFLAGS])

])dnl
