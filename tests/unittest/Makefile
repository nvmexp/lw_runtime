#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

#-------------------------------------------------------------------------------
# Main Unit test makefile
# Calls resman build and then builds unit tests 
#
# In the first-phase of build, all the relevent c-file from resman(sources.mk)
# are compiled to form a lib.
# All the unittest infra + actual files are compiled and linked with the rmlib
# to create allunittest.exe
#-------------------------------------------------------------------------------

# enable extra warnings
ENABLE_PEDANTIC   ?= true

DRIVERS_DIR ?= ../../drivers
UT_DIR = .
include makedefs.inc

defines += -DRM_UNITTEST
BRANCH = ../..
other_dir = $(BRANCH)/drivers/resman/arch/lwalloc/unittest/$(BUILD_CFG)
mk_dir = $(BRANCH)/drivers/resman/arch/lwalloc/unittest/$(BUILD_CFG)/makefiles

CLIENTPATHS += $(BRANCH)/diag/unittest/...
CLIENTPATHS += $(BRANCH)/drivers/common/...
CLIENTPATHS += $(BRANCH)/drivers/resman/...

#-------------------------------------------------------------------------------
# include directory paths
#-------------------------------------------------------------------------------

INCLUDES += $(BRANCH)/drivers/resman/arch/lwalloc/unittest/$(BUILD_CFG)
INCLUDES += resman/harness/inc
INCLUDES += resman/suites
INCLUDES += common/inc
INCLUDES += $(BRANCH)/drivers/resman/kernel/inc/fermi 
INCLUDES += $(BRANCH)/sdk/lwpu/inc
INCLUDES += $(BRANCH)/drivers/resman/kernel/inc
INCLUDES += $(BRANCH)/drivers/resman/kernel/inc/rmtest
INCLUDES += $(BRANCH)/drivers/resman/kernel/inc/fermi/gf100
INCLUDES += $(BRANCH)/drivers/common/inc
INCLUDES += $(BRANCH)/drivers/resman/arch/lwalloc/common/inc
INCLUDES += $(BRANCH)/drivers/resman/interface
INCLUDES += $(BRANCH)/drivers/resman/arch/lwalloc/unittest/oslayer/inc
ifeq ($(shell uname),Linux)
INCLUDES += $(BUILD_TOOLS_DIR)/linux/ruby1.8.6/32bit/lib/ruby/1.8/i686-linux
else
INCLUDES += $(BUILD_TOOLS_DIR)/win32/msvc90/PlatformSDK/Include
endif

#-------------------------------------------------------------------------------
# Unittest Infra C Files 
#-------------------------------------------------------------------------------

CFILES += common/mock/lwmock.c
CFILES += common/regops/regops.c
CFILES += common/testops/lwtest.c
CFILES += common/utility/utility.c
CFILES += resman/harness/utility/rmassert.c
CFILES += resman/harness/utility/fixtureutil.c
CFILES += resman/harness/odbsetup/infoblkinfra.c
CFILES += resman/harness/odbsetup/odbinfra.c
CFILES += resman/harness/odbsetup/unitodb.c
CFILES += common/utility/errorLogger.c
CFILES += common/utility/dbgprints.c

#-------------------------------------------------------------------------------
# Adding test files
#-------------------------------------------------------------------------------

include $(wildcard $(mk_dir)/*.mk)

CFILES += $(unitTestSrc)

#-------------------------------------------------------------------------------
# Temporarily conditional code till testrunner gets enabled on windows
#-------------------------------------------------------------------------------

UT_CFILES += resman/suites/testrunner.c

ifeq ($(shell uname),Linux)
CFILES += unizen/extensions.c

$(BUILD_CFG)/odbinfra.o :warnings += -Wno-implicit

else
CFILES += $(UT_CFILES)
endif

#-------------------------------------------------------------------------------
# Rm library to be linked
#-------------------------------------------------------------------------------
LIBS += $(BRANCH)/drivers/resman/arch/lwalloc/unittest/$(BUILD_CFG)/librm$(lib_suffix)

exe_name = allunittest

#-------------------------------------------------------------------------------
# Targets...
#-------------------------------------------------------------------------------
j ?= 1

.PHONY: build_all clean_all build_install clean clean_unizen run_unizen uproc_build uproc_clean

# using make3.81 internally as it is not the default make used on io-xterms and dvs
# will have to update make version on dvs to remove this from here

build_all:
	$(MAKE_PATH)/make build -j $j
 
build: submake.resman.build 
	$(Q)$(MAKE) build_install

% :
	$(MAKE_PATH)/make build_$@ -j $j

build_%: submake.resman.%
	$(Q)$(MAKE) build_install

submake.resman.build : submake.unizen.build 

submake.unizen.%:
	$(Q)$(MAKE) -C unizen $*

submake.resman.%:
	$(Q)echo building $@
	$(Q)$(MAKE) -C $(DRIVERS_DIR)/resman/arch/lwalloc/unittest $*

ifeq ($(shell uname),Linux)
build_install: $(o_dir)/unizen.so
else
build_install: $(o_dir)/$(exe_name)$(exe_suffix)
endif

$(o_dir)/unizen.so: $(OBJS)
	$(Q)echo "linking..."
	$(Q)echo "creating unizen.so..."
	$(Q)$(CC) -shared $(OBJS) $(LIBS) -o $(o_dir)/unizen.so

$(o_dir)/$(exe_name)$(exe_suffix) : $(LIBS)
$(o_dir)/unizen.so : $(LIBS)

uproc_run : submake.uproc.run

uproc_build : submake.uproc.build

submake.uproc.%:
	$(Q)$(MAKE) -C uproc $*

clean_all: clean submake.resman.clean clean_unizen

clean_unizen : submake.unizen.clean

run_unizen : submake.unizen.run

clean: 
	$(Q)echo Removing solution files
	$(Q)rm -f $(o_dir)/*$(obj_suffix)
	$(Q)rm -f $(o_dir)/*.d
	$(Q)rm -f $(o_dir)/*.pdb
	$(Q)rm -f $(o_dir)/*$(exe_suffix)
	$(Q)rm -f $(o_dir)/*$(lib_suffix)
	$(Q)rm -f $(o_dir)/*.suo
	$(Q)rm -f $(o_dir)/*.exp

uproc_clean: submake.uproc.clean

run : build_all
ifeq ($(shell uname),Linux)
	$(Q)$(MAKE) run_unizen
else
	@cp $(o_dir)/$(exe_name)$(exe_suffix) $(exe_name)$(exe_suffix)
	./$(exe_name)$(exe_suffix)
	@rm $(exe_name)$(exe_suffix)
endif

client :
	@echo The following paths must be in your client:
	@echo $(CLIENTPATHS) | tr ' ' '\n' | sort

include makerule.inc

$(LIBS): ;

