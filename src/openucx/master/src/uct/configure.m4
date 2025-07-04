#
# Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

uct_modules=""
m4_include([src/uct/lwca/configure.m4])
m4_include([src/uct/ib/configure.m4])
m4_include([src/uct/rocm/configure.m4])
m4_include([src/uct/sm/configure.m4])
m4_include([src/uct/ugni/configure.m4])

AC_DEFINE_UNQUOTED([uct_MODULES], ["${uct_modules}"], [UCT loadable modules])

AC_CONFIG_FILES([src/uct/Makefile])
