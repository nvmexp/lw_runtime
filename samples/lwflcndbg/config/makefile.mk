#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2008-2013 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#
# drivers/resman/lwflcndbg/config/makefile.mk
#
# Run lwwatch-config.pl to build lwwatch-config.h and lwwatch-config.mk
#
# *** Used by lwwatch builds that are lwwatch-config aware
#
# Each platform build will set LWFLCNDBGCFG_DIR, LWFLCNDBGCFG_OUTPUT_DIR and LWFLCNDBGCFG_VERBOSE variables, then 'include' this file.
#
# Makefiles that 'include'/call this file:
#
#    lwalloc/mods/Makefile
#    lwwatch/GNUMakefile
#    lwwatch/Makefile_mac

BRANCH_ROOT:=$(shell cd ../../..; pwd)

LWFLCNDBGCFG_PERL            ?= $(if $(findstring undefined,$(origin PERL)), perl, $(PERL))
LWFLCNDBGCFG_PROFILE         ?= shipping-gpus-mods
LWFLCNDBGCFG_LWFLCNDBG_ROOT  ?= $(BRANCH_ROOT)/apps/lwflcndbg
LWFLCNDBGCFG_DIR             ?= $(LWFLCNDBGCFG_LWFLCNDBG_ROOT)/config
LWFLCNDBGCFG_OUTPUTDIR       ?= $(LWFLCNDBGCFG_DIR)
LWFLCNDBGCFG_VERBOSE         ?= 
LWFLCNDBGCFG_RESMAN_ROOT     ?= $(BRANCH_ROOT)/drivers/resman
LWFLCNDBGCFG_CHIPCFG_ROOT    ?= $(BRANCH_ROOT)/drivers/common/chip-config
KERNEL_TOOLS_DIR             ?= $(LWFLCNDBGCFG_RESMAN_ROOT)/kernel/inc/tools
LWWATCHCFG_DIR               := $(LWFLCNDBGCFG_LWFLCNDBG_ROOT)/../lwwatch/config

# Files we create that should be deleted by 'make clean'.
# May be used by "calling" makefile.
LWFLCNDBGCFG_CLEAN          =
LWFLCNDBGCFG_CLEAN         += $(LWFLCNDBGCFG_OUTPUTDIR)/g_*.h

# table to colwert LWFLCNDBGCFG_VERBOSE into an option for lwwatch-config.pl
_lwflcndbgcfgVerbosity_quiet       = --quiet
_lwflcndbgcfgVerbosity_@           = --quiet
_lwflcndbgcfgVerbosity_default     = 
_lwflcndbgcfgVerbosity_verbose     = --verbose
_lwflcndbgcfgVerbosity_veryverbose = --verbose --verbose
_lwflcndbgcfgVerbosity_            = $(_lwflcndbgcfgVerbosity_default)
lwflcndbgcfgVerbosityFlag := $(_lwflcndbgcfgVerbosity_$(LWFLCNDBGCFG_VERBOSE))

# files that implement lwwatch-config
lwflcndbgcfgSrcFiles += $(LWFLCNDBGCFG_DIR)/Apis.pm
lwflcndbgcfgSrcFiles += $(LWFLCNDBGCFG_DIR)/Classes.pm
lwflcndbgcfgSrcFiles += $(LWFLCNDBGCFG_DIR)/Engines.pm
lwflcndbgcfgSrcFiles += $(LWFLCNDBGCFG_DIR)/Features.pm
lwflcndbgcfgSrcFiles += $(LWFLCNDBGCFG_DIR)/makefile.mk
lwflcndbgcfgSrcFiles += $(LWFLCNDBGCFG_DIR)/lwwatch-config.cfg

# pull in implementations list of chip-config to supplement $(lwflcndbgcfgSrcFiles)
include $(LWFLCNDBGCFG_CHIPCFG_ROOT)/Implementation.mk
lwflcndbgcfgSrcFiles += $(addprefix $(LWFLCNDBGCFG_CHIPCFG_ROOT)/, $(CHIPCONFIG_IMPLEMENTATION))

# pull in haldef list to define $(HALDEFS)
include $(LWFLCNDBGCFG_DIR)/haldefs/Haldefs.mk
lwflcndbgcfgSrcFiles += $(foreach hd,$(LWFLCNDBGCFG_HALDEFS),$(LWFLCNDBGCFG_DIR)/haldefs/$(hd).def)

# pull in templates list to supplement $(lwflcndbgcfgSrcFiles)
include $(LWFLCNDBGCFG_DIR)/templates/Templates.mk
lwflcndbgcfgSrcFiles += $(addprefix $(LWFLCNDBGCFG_DIR)/templates/, $(LWFLCNDBGCFG_TEMPLATES))


### Run chip-config.pl to generate lwwatch-config.mk

# lwwatch-config GZIP option
ifdef GZIP_CMD
RMCFG_GZIP_OPT = --gzip-cmd "$(GZIP_CMD)"
else
RMCFG_GZIP_OPT =
endif

# lwwatch-config invocation args
lwflcndbgcfgArgs = $(lwflcndbgcfgVerbosityFlag) \
             --mode lwwatch-config \
             --config $(LWFLCNDBGCFG_DIR)/lwwatch-config.cfg \
             --profile $(LWFLCNDBGCFG_PROFILE) \
             --source-root $(LWFLCNDBGCFG_LWFLCNDBG_ROOT) \
             --output-dir $(LWFLCNDBGCFG_OUTPUTDIR) \
             $(RMCFG_GZIP_OPT)

# generate lwwatch-config.mk and lwwatch-config.h
build_all: $(lwflcndbgcfgSrcFiles)
	$(LWFLCNDBGCFG_PERL) -I$(LWFLCNDBGCFG_DIR) -I$(LWWATCHCFG_DIR) $(LWFLCNDBGCFG_CHIPCFG_ROOT)/chip-config.pl $(lwflcndbgcfgArgs)

check_lwwatch:
	rm -f check_lwwatch.succeeded
	$(LWFLCNDBGCFG_PERL) $(KERNEL_TOOLS_DIR)/mcheck.pl --lwwatch --root $(LWFLCNDBGCFG_LWFLCNDBG_ROOT)
	touch check_lwwatch.succeeded

# test targets
.PHONY: _lwflcndbg_config_all _lwflcndbg_cfg_clean 
_lwflcndbg_config_all: build_all

_lwflcndbg_cfg_clean: 
	rm -f $(LWFLCNDBGCFG_CLEAN)

