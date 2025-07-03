#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_LWDA

AS_IF([test "x$lwda_happy" = "xyes"], [uct_modules="${uct_modules}:lwca"])
uct_lwda_modules=""
m4_include([src/uct/lwca/gdr_copy/configure.m4])
AC_DEFINE_UNQUOTED([uct_lwda_MODULES], ["${uct_lwda_modules}"], [LWCA loadable modules])
AC_CONFIG_FILES([src/uct/lwca/Makefile])
