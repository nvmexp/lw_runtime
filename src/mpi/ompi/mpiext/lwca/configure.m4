# -*- shell-script -*-
#
# Copyright (c) 2004-2010 The Trustees of Indiana University.
#                         All rights reserved.
# Copyright (c) 2012-2015 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2015      Intel, Inc. All rights reserved.
# Copyright (c) 2015      LWPU, Inc.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# OMPI_MPIEXT_lwda_CONFIG([action-if-found], [action-if-not-found])
# -----------------------------------------------------------
AC_DEFUN([OMPI_MPIEXT_lwda_CONFIG],[
    AC_CONFIG_FILES([ompi/mpiext/lwca/Makefile])
    AC_CONFIG_FILES([ompi/mpiext/lwca/c/Makefile])
    AC_CONFIG_HEADER([ompi/mpiext/lwca/c/mpiext_lwda_c.h])

    AC_DEFINE_UNQUOTED([MPIX_LWDA_AWARE_SUPPORT],[$LWDA_SUPPORT],
                       [Macro that is set to 1 when LWCA-aware support is configured in and 0 when it is not])

    # We compile this whether LWCA support was requested or not. It allows
    # us to to detect if we have LWCA support.
    AS_IF([test "$ENABLE_lwda" = "1" || \
           test "$ENABLE_EXT_ALL" = "1"],
          [$1],
          [$2])
])
