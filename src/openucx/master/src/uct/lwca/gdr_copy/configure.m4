#
# Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
# Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

UCX_CHECK_GDRCOPY

AS_IF([test "x$gdrcopy_happy" = "xyes"], [uct_lwda_modules="${uct_lwda_modules}:gdrcopy"])
AC_CONFIG_FILES([src/uct/lwca/gdr_copy/Makefile])
