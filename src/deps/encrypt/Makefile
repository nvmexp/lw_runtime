#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 1999-2022 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#


#-------------------------------------------------------------------------------
# MODS encrypt utility make file.
# See //sw/<tree>/diag/mods/makedefs.inc for more information about the MODS
# makefiles.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
# Environment variables: (set default values if not defined by user)
#
#-------------------------------------------------------------------------------
# Set variables to their default state if they are not already set.

INCLUDE_LWDA      ?= false
INCLUDE_MDIAG     ?= false
INCLUDE_RMTEST    ?= false
INCLUDE_REFS      ?= false
INCLUDE_LWOGTEST  ?= false
INCLUDE_OGL       ?= true
INCLUDE_OGL_APIC  ?= false
INCLUDE_GPU       ?= true
INCLUDE_MINIRM    ?= false
INCLUDE_SDI       ?= true
INCLUDE_LWWATCH   ?= false
INCLUDE_VULKAN    ?= true
BOUND_JS          ?= false
CHANGELIST        ?= 0
VERSION           ?= 000
DEBUG_TRACE_LEVEL ?= 0
BUILD_DECRYPT     ?= false
DECRYPT_LOG_ONLY  ?= true
STEALTH_MODE      = ""
ENFORCE_LW_NETWORK ?= false

ENCRYPT_DIR     ?= .
ENCRYPTION_DIR  ?= $(ENCRYPT_DIR)/../encryption
UTIL_DIR        ?= $(ENCRYPT_DIR)/../utils
MODS_DIR        ?= $(ENCRYPT_DIR)/../mods
DRIVERS_DIR     ?= $(ENCRYPT_DIR)/../../drivers
SDK_DIR         ?= $(ENCRYPT_DIR)/../../sdk

exe_build = true

INCLUDE_BOARDS_DB ?= true

ifeq "$(INCLUDE_BOARDS_DB)" "true"
   defines += -DINCLUDE_BOARDS_DB
endif

ifeq "$(BUILD_DECRYPT)" "true"
    # Any usage of the decryptor *must* be on the LW network for added security
    ENFORCE_LW_NETWORK = true
endif

ifeq "$(BUILD_DECRYPT)" "true"
    defines += -DDECRYPTOR=1
    ifeq "$(DECRYPT_LOG_ONLY)" "true"
        base_name = decrypt
        defines += -DDECRYPT_LOG_ONLY=1
    else
        base_name = decryptall
    endif
else
    base_name = encrypt
    defines += -DENCRYPTOR=1
    ifeq "$(ENFORCE_LW_NETWORK)" "true"
        base_name = encrypt.INTERNAL
    endif
endif


ifeq "$(ENFORCE_LW_NETWORK)" "true"
   defines += -DENFORCE_LW_NETWORK=1
else
    ifeq "$(BUILD_DECRYPT)" "true"
        ifneq "$(DECRYPT_LOG_ONLY)" "true"
            $(error JS decryptor must enforce LWPU network)
        endif
    endif
endif

include $(MODS_DIR)/makedefs.inc

include_dirs += $(o_dir)
include_dirs += $(ENCRYPTION_DIR)
include_dirs += $(UTIL_DIR)
include_dirs += $(MODS_DIR) # for version.h, types.h
include_dirs += $(SDK_DIR)/lwpu/inc # for lwtypes.h
include_dirs += $(BOOST_DIR)
force_include_files += core/include/jsprotect.h

ifeq "$(use_pthreads)" "true"
    # Android supports POSIX threads, but not libpthread.so
    ifeq (,$(filter $(BUILD_OS_SUBTYPE), bionic qnx))
        libraries += pthread
    endif
endif

libraries += encryption cryptopp lwdiag_utils

ifeq "$(base_os)" "linux"
    libraries += dl
endif

libraries += z boost

lib_dirs     = $(MODS_DIR)/$(mods_lib_dir) $(MODS_DIR)/$(base_mods_lib_dir)
lib_dirs += $(MODS_OUTPUT_DIR)/$(rel_o_dir)/lwmake/diag/utils

cpp_files += encryptmain.cpp
cpp_files += $(VERSION_CPP)

cpp_files     += $(gen_cpp_dir)/aes_file_key.cpp
cpp_files     += $(gen_cpp_dir)/aes_log_key.cpp
gen_cpp_files += $(gen_cpp_dir)/aes_file_key.cpp
gen_cpp_files += $(gen_cpp_dir)/aes_log_key.cpp

defines += -DDEBUG_TRACE_LEVEL=$(DEBUG_TRACE_LEVEL)

ifeq "$(toolchain)" "gnu"
   libraries += m
endif

ifeq "$(base_os)" "win32"
   defines += -DREGEX_MALLOC -DSTDC_HEADERS -DHAVE_CONFIG_H
   LDFLAGS += /NODEFAULTLIB:libc.lib

   ifeq "$(BUILD_CFG)" "debug"
      LDFLAGS += /NODEFAULTLIB:libcmt.lib
   else
      LDFLAGS += /NODEFAULTLIB:libcmtd.lib
   endif
   extra_libs = ws2_32.lib crypt32.lib selwr32.lib gdi32.lib user32.lib advapi32.lib shell32.lib
endif

ifeq "$(toolchain)" "msvc"
   # Must turn on C++ exception handling when using the C++ standard library.
   # Would really prefer to not have to turn this on... unwinding logic is not
   # free, and we don't really want C++ exceptions.
   CFLAGS += /EHsc

   # Windows builds have some libraries that are different in debug/release
   # builds.  Include those paths here
   lib_dirs    += $(MODS_DIR)/$(base_mods_lib_dir)/$(BUILD_CFG)

   ifeq "$(BUILD_CFG)" "debug"
       optimize = /Od /Z7
   endif
else
   # Warn about various picky things
   warnings += -Wundef -Wsign-compare
   cc_warnings += -Wmissing-prototypes
   cc_warnings += -Wnested-externs -Wmissing-declarations

   ifeq "$(BUILD_CFG)" "release"
      optimize = -O2
      ifeq "$(base_os)" "macosx"
         #OSX doesn't support -s, we will strip it later
         strip =
      else
         strip = -s
      endif
   else
      strip    =
   endif

   LDFLAGS += $(strip)
endif


#-------------------------------------------------------------------------------
# Targets:
#-------------------------------------------------------------------------------

# The default target makes the the JavaScript library (libjs.a).
all :  $(o_dir)/$(exe_name)

.PHONY : all clean install FORCE keys

$(MODS_OUTPUT_DIR_UNIX):
	$(MKDIR_P) $(MODS_OUTPUT_DIR_UNIX)

# Creates rules for generating one type of AES key
# Usage: CREATE_KEY <key_name> <variable_name>
define CREATE_KEY
$(gen_cpp_dir)/aes_$1_key.cpp: $(MODS_OUTPUT_DIR_UNIX)/aes_$1_key.cpp.gen $(gen_cpp_dir)/dummy.txt
	$(Q)$(CP_IF_CHANGED) $$< $$@
    ifdef CYGWIN_MAKE
	$(Q)chmod 644 $$@
    endif

$(MODS_OUTPUT_DIR_UNIX)/aes_$1_key.cpp.gen: gen_aes_key.py | $(MODS_OUTPUT_DIR_UNIX)
	@$(ECHO) Creating $$@
	$(Q)$(PYTHON3) gen_aes_key.py $2 $(MODS_OUTPUT_DIR)/aes_$1_key.cpp.gen
    ifdef CYGWIN_MAKE
	$(Q)chmod 644 $$@
    endif
endef

$(eval $(call CREATE_KEY,file,X_1))
$(eval $(call CREATE_KEY,log,m_6))

keys: $(MODS_OUTPUT_DIR_UNIX)/aes_file_key.cpp.gen $(MODS_OUTPUT_DIR_UNIX)/aes_log_key.cpp.gen

include $(MODS_DIR_UNIX)/makerule.inc

ifneq "$(BUILD_ARCHS)" ""
   ifeq "$(base_os)" "macosx"
      $(o_dir)/$(exe_name): $(SUBMAKE_FAT_ARCHS:%=%.all) $(o_dir)/dummy.txt
	$(Q)echo "Creating universal $(exe_name)..."
	$(Q)lipo -create -output $@ $(foreach arch,$(BUILD_ARCHS),$(BUILD_CFG)/$(BUILD_OS)/$(arch)/$(exe_name))

      submake.fat.%.all:
	$(Q)echo "Building $* $(exe_name)..."
	$(Q)$(MAKE) BUILD_ARCHS="" BUILD_ARCH=$* mods_lib_dir=$(mods_lib_dir) base_mods_lib_dir=$(base_mods_lib_dir) all
   endif
else

$(o_dir)/$(exe_name) : $(o_files)
	$(Q)$(ECHO) "Linking $(o_dir)/$(exe_name) version $(VERSION)..."
	$(Q)$(ECHO) "File key for $(exe_name):"
	$(Q)$(GREP) "^ " $(gen_cpp_dir)/aes_file_key.cpp
	$(Q)$(ECHO) "Log key for $(exe_name):"
	$(Q)$(GREP) "^ " $(gen_cpp_dir)/aes_log_key.cpp
	$(Q)$(LD) $(LDFLAGS_EXE) $(o_files) $(link_libs)

$(o_dir)/$(exe_name): $(o_dir)/bypass.decrypt.bin

$(o_dir)/$(exe_name): submake.lwdiag_utils.build

submake.lwdiag_utils.%:
	$(Q)$(MAKE) -C $(MODS_DIR)/../utils -f makefile.lwmk $(subst submake.lwdiag_utils.,,$@) $(LWMAKE_ARGS) $*

$(o_dir)/bypass.decrypt.bin:

$(o_dir)/bypass.decrypt.bin: $(MODS_DIR_UNIX)/tools/genfiles.py FORCE
	@$(ECHO) "Creating bypass files"
	$(Q)$(PYTHON3) $(MODS_DIR)/tools/genfiles.py        \
             -b$(o_dir)/bypass.decrypt.bin          \
             g_Version,$(VERSION)                   \
             g_Changelist,$(CHANGELIST)

endif # ifeq "$(BUILD_ARCHS)" ""

clean : $(SUBMAKE_FAT_ARCHS:%=%.clean) submake.lwdiag_utils.clean
	@echo "Deleting object, lib, and dependency files."
	@-$(RM_RF) $(gen_cpp_dir)
	@-$(RM_RF) $(o_dir)

clobber: clean

# Include dependency information.
-include $(depend_files)

submake.fat.%.clean:
	@$(MAKE) BUILD_ARCH=$* BUILD_ARCHS="" clean
