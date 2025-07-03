# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2019 LWPU CORPORATION. All rights reserved

include $(RTE_SDK)/mk/rte.vars.mk

ifeq ($(CONFIG_RTE_TEST_PMD),y)

#
# library name
#
APP = flow

CFLAGS += -DALLOW_EXPERIMENTAL_API
CFLAGS += -O3
CFLAGS += $(WERROR_FLAGS)

CFLAGS += -I$(CONFIG_RTE_LW_LWDAHEADERS_DIR)
LDFLAGS += -L$(CONFIG_RTE_LW_LIBLWDA_DIR) -llwda

#
# all source are stored in SRCS-y
#
SRCS-y := main.c

CFLAGS_cmdline.o := -D_GNU_SOURCE

include $(RTE_SDK)/mk/rte.app.mk

endif
