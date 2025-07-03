#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
#
# See file LICENSE for terms.
#

UCX_CHECK_LWDA
AS_IF([test "x$lwda_happy" = "xyes"], [ucm_modules="${ucm_modules}:lwca"])
AC_CONFIG_FILES([src/ucm/lwca/Makefile])
