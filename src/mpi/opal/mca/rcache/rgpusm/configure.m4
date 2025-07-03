# -*- shell-script -*-
#
# Copyright (c) 2012-2015 LWPU Corporation.  All rights reserved.
# Copyright (c) 2015      Los Alamos National Security, LLC. All rights
#                         reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

#
# If LWCA support was requested, then build the LWCA memory pools.
# This code checks the variable LWDA_SUPPORT which was set earlier in
# the configure sequence by the opal_configure_options.m4 code.
#

AC_DEFUN([MCA_opal_rcache_rgpusm_CONFIG],[
    AC_CONFIG_FILES([opal/mca/rcache/rgpusm/Makefile])

    # Use LWDA_SUPPORT which was filled in by the opal configure code.
    AS_IF([test "x$LWDA_SUPPORT" = "x1"],
          [$1],
          [$2])

])dnl
