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
# Ruby script to generate mock files (o_dir)
#-------------------------------------------------------------------------------

include ../makedefs.inc
BRANCH = $(LWRDIR)/../../..
UT_DIR = ..
o_dir    = ../../../drivers/resman/arch/lwalloc/unittest/$(BUILD_CFG)
mock_dir = ../../../drivers/resman/arch/lwalloc/unittest/mocks
mk_dir   = ../../../drivers/resman/arch/lwalloc/unittest/$(BUILD_CFG)/makefiles

include defines.mk

rb_files = $(notdir $(rbDefs))
alltest_files = $(addprefix $(o_dir)/,$(rb_files:.rb=.h))
mk_files = $(addprefix $(mk_dir)/file_,$(rb_files:.rb=.mk))

PHONY: build clean dirs

build : $(alltest_files) $(mk_files)

$(alltest_files) $(mk_files): $(UT_DIR)/mocksrc.mk

# rule to create various output folders: $(o_dir), $(mock_dir), $(mk_dir)
dirs: $(o_dir)/dummy.txt $(mock_dir)/dummy.txt $(mk_dir)/dummy.txt
%/dummy.txt:
	$(Q)mkdir -p $(dir $@)
	$(Q)touch $@

# creating list of mocked files
$(UT_DIR)/mocksrc.mk : $(rbDefs)
	$(Q)echo generating mocksrc.mk...
	$(Q)$(ruby) driverDbGen.rb $?

# creating makefiles for auto adding tests
	$(Q)echo generating makefiles...
	$(Q)$(ruby) driverGenerateTestMk.rb $?

# generates list of all available definition files
	@echo generating list of definition files...
	$(Q)$(ruby) allDefs.rb

# generates ruby extensions for all suites
	@echo gennerating ruby extensoins...
	$(Q)$(ruby) genext.rb $(rbDefs)

$(rbDefs) : $(o_dir)/dummy.txt $(mock_dir)/dummy.txt $(mk_dir)/dummy.txt

run :
	$(Q)$(ruby) runnerZen.rb --run unit

clean :
	@echo Deleting generated ruby files database/mocksrc...
	$(Q)rm -f generatedDocDb.rb
	$(Q)rm -f ../mocksrc.mk
	$(Q)rm -f $(o_dir)/*.rb
	$(Q)rm -f $(o_dir)/exp*.*
	$(Q)rm -f $(o_dir)/all*.*
	$(Q)rm -f $(mk_dir)/*.mk
	$(Q)rm -f ../$(BUILD_CFG)/extensions.*
