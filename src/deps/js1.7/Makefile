#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2009, 2011, 2014-2018 by LWPU Corporation. All rights reserved.
# All information contained herein is proprietary and confidential to LWPU
# Corporation. Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

#-------------------------------------------------------------------------------
# JavaScript make file.
# See //sw/<tree>/diag/mods/makedefs.inc for more information about the MODS
# makefiles.
#-------------------------------------------------------------------------------


MODS_DIR = ../mods
LWDIAG_UTILS_DIR = ../utils

dll_build = false

# TODO: always shuffle opcodes for every MODS build

ifneq "$(SHUFFLE_OPCODES)" "true"
base_name = libjs17
else
base_name = libejs17
include_dirs += .
endif

include_dirs += $(LWDIAG_UTILS_DIR)

include $(MODS_DIR)/makedefs.inc
include ./makesrc.inc

include_dirs += $(BOOST_DIR)
include_dirs += $(MODS_DIR)
include_dirs += $(gen_cpp_dir)

ifeq ($(toolchain),msvc)
   CFLAGS += /EHsc
   defines += -DEXPORT_JS_API -DNOMINMAX
   ifeq "$(BUILD_ARCH)" "ia64"
      defines += -D_WIN64 -D_WINDOWS -D_M_X64 -D_IA64_
   else
      ifeq "$(BUILD_ARCH)" "amd64"
         defines += -D_WIN64 -D_AMD64_
      else
         ifneq (,$(filter $(BUILD_ARCH), armv7 aarch64))
            ifeq "$(BUILD_OS)" "win32"
                defines += -D_ARM_
            endif
         else
             defines += -D_X86_ -D_WIN32 -D_WINDOWS
         endif
      endif
   endif
endif

#-------------------------------------------------------------------------------
# Targets:
#-------------------------------------------------------------------------------

.PHONY : all clean install jsautocfg jsautokw jsc opshuffle

build: $(o_dir)/dummy.txt $(o_dir)/$(lib_name)

all: build

clean :
	@echo "Deleting object, lib, and dependency files."
	@-rm -f $(o_dir)/*$(obj_suffix)
	@-rm -f $(o_dir)/*$(lib_suffix)
	@-rm -f $(o_dir)/*$(dll_suffix)
	@-rm -f $(o_dir)/*$(exe_suffix)
	@-rm -f $(o_dir)/*.cpp
	@-rm -f $(o_dir)/*.h
	@-rm -f $(o_dir)/*.d
	@-rm -f $(o_dir)/*.pdb
	@-rm -f *.d

install:
	@echo "Copying $(o_dir)/$(lib_name) to $(MODS_DIR)/$(base_mods_lib_dir)/$(lib_name)..."
	@cp $(o_dir)/$(lib_name) $(MODS_DIR)/$(base_mods_lib_dir)/$(lib_name)

jsautocfg : $(o_dir)/jsautocfg.h

$(o_dir)/jsautocfg.h: $(o_dir)/jscpucfg$(exe_suffix)
	@-rm -f $@
	$(o_dir)/jscpucfg$(exe_suffix) > $@

$(o_dir)/jscpucfg$(exe_suffix): link_def=
$(o_dir)/jscpucfg$(exe_suffix): $(o_dir)/jscpucfg$(obj_suffix)
	@echo "Linking $@"
	$(Q)$(LD) $(LDFLAGS_EXE) $^

jsautokw : $(o_dir)/jsautokw.h

$(o_dir)/jsautokw.h: $(o_dir)/jskwgen$(exe_suffix) jskeyword.tbl
	@echo "Generating $@"
	$(Q)$(o_dir)/jskwgen$(exe_suffix) $@

$(o_dir)/jskwgen$(exe_suffix): link_def=
$(o_dir)/jskwgen$(exe_suffix): $(o_dir)/jskwgen$(obj_suffix)
	@echo "Linking $@"
	$(Q)$(LD) $(LDFLAGS_EXE) $^

$(MODS_OUTPUT_DIR_UNIX)/g_jsopcode.tbl: jsopcode.tbl $(MODS_OUTPUT_DIR_UNIX)
ifeq "$(SHUFFLE_OPCODES)" "true"
	@echo "Generating shuffled opcode table $@"
	$(Q)$(PYTHON3) ../jsprotect/opshuffle/opshuffle.py < $< > $@
else
	$(Q)$(CP) $< $@
endif

$(MODS_OUTPUT_DIR_UNIX):
	$(Q)$(MKDIR_P) $@

$(gen_cpp_dir)/g_jsopcode.tbl: $(MODS_OUTPUT_DIR_UNIX)/g_jsopcode.tbl
	$(Q)$(CP) $< $@

$(o_dir)/jsopcode$(obj_suffix): $(gen_cpp_dir)/g_jsopcode.tbl

jsc: $(gen_cpp_dir)/g_jsopcode.tbl
	$(Q)$(MAKE) -C ../jsprotect/jsc OPCODE_TABLE_DIR=$(gen_cpp_dir)

include $(MODS_DIR_UNIX)/makerule.inc

# Include dependency information.
-include $(depend_files)
