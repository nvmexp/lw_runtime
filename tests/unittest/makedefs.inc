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
# UnitTest shared makefile definitions.
#
# Portability rules:
#    Use "/" as path separator.
#    Use UNIX shell commands.
#
# Code rules:
#    Use tabs only in shell command lines.
#    BEWARE that tabs MUST be used at the beginning of command lines!
#
# Environment variables:
#
# BUILD_CFG : Build release or debug library.
#             Defaults to "debug".
#
# LW_BULLSEYE : Set to 1 for bullseye build 
#
# SHOW_OUTPUT : Causes all compiler commands to be printed verbosely
#               Defaults to "false". 
#
# ENABLE_PEDANTIC : Adds extra warnings. Default false.
#
#-------------------------------------------------------------------------------

P4ROOT := $(subst \,/,$(P4ROOT))
LW_BULLSEYE ?= 0
SHOW_OUTPUT ?= false
ENABLE_PEDANTIC ?= false 
RM_CFILES =
MOCKED_CFILES =


ifdef BUILD_TOOLS_DIR
    BUILD_TOOLS_DIR := $(subst \,/,$(BUILD_TOOLS_DIR))
else
    BUILD_TOOLS_DIR := $(P4ROOT)/sw/tools
endif

BUILD_CFG ?= debug
o_dir = $(BUILD_CFG)

# Allow secondary expansion
.SECONDEXPANSION:

LIBS =
MOCK_DIR =
CFILES =
OBJS =
DEPS =

#-------------------------------------------------------------------------------
# include directory paths
#-------------------------------------------------------------------------------

INCLUDES =

ifeq ($(shell uname),Linux)
    base_os = linux
    BUILD_ARCH := $(shell uname -m)
    LW_BULLSEYE_BIN ?= $(BUILD_TOOLS_DIR)/Bullseye/linux/7.13.29/bin

    # using make3.81 internally as it is not the default make used on io-xterms and dvs
    # will have to update make version on dvs to remove this from here
    MAKE_PATH := $(BUILD_TOOLS_DIR)/linux/make3.81/whitebox

    lib_suffix = .a
    obj_suffix = .o
    exe_suffix =

    defines += -DUNIT_LINUX
    CLIENTPATHS += //sw/tools/linux/make3.81/...
else
    # using make3.81 internally as it is not the default make used on io-xterms and dvs
    # will have to update make version on dvs to remove this from here
    MAKE_PATH := $(BUILD_TOOLS_DIR)/win32/make3.81

    base_os = win32
    BUILD_ARCH := x86
    defines += -D_WIN32
    defines += -DUNIT_WINDOWS
    LW_BULLSEYE_BIN ?= $(BUILD_TOOLS_DIR)/Bullseye/win32/7.13.25/BullseyeCoverage/bin

    INCLUDES += $(BUILD_TOOLS_DIR)/win32/msvc90/include
    INCLUDES += $(BUILD_TOOLS_DIR)/win32/msvc90/lib
    INCLUDES += $(BUILD_TOOLS_DIR)/win32/msvc90/bin

    lib_suffix = .lib
    obj_suffix = .obj
    exe_suffix = .exe

    CLIENTPATHS += //sw/tools/win32/msvc90/...
    CLIENTPATHS += //sw/tools/win32/ActivePerl/5.10.0.1004...
    CLIENTPATHS += //sw/tools/ruby1.8.6/...
    CLIENTPATHS += //sw/tools/win32/python/254/...
    CLIENTPATHS += //sw/tools/win32/make3.81/...
endif

INCS = $(addprefix -I, $(INCLUDES))

#-------------------------------------------------------------------------------
# Flags required while compilation
#-------------------------------------------------------------------------------

ifeq ($(base_os),linux)
    PERL = perl
    CC := gcc
    ifeq "$(BUILD_ARCH)" "x86_64"
        ruby = $(BUILD_TOOLS_DIR)/linux/ruby1.8.6/bin/ruby

        CLIENTPATHS += //sw/tools/linux/mods/gcc-4.2.4-i686/...
        CLIENTPATHS += //sw/tools/linux/ruby1.8.6/...
    else
        ruby = $(BUILD_TOOLS_DIR)/linux/ruby1.8.6/redhat/bin/ruby

        CLIENTPATHS += //sw/tools/linux/gcc-2.96/Linux-x86/...
        CLIENTPATHS += //sw/tools/linux/ruby1.8.6/redhat/...
    endif

    COV_DIR ?= /../../
    ifeq "$(LW_BULLSEYE)" "1"
        CLIENTPATHS += //sw/tools/Bullseye/linux/7.13.29/...
        CC := $(LW_BULLSEYE_BIN)/covc -a -t$(COV_DIR) --no-banner $(CC)
    endif

    LWCFG_PROFILE ?= unix_global_internal_profile

    ifeq "$(BUILD_CFG)" "debug"
        defines += -DUNIT_DEBUG_BUILD
        CFLAGS += -g
    endif

    CFLAGS += -fpic

    AR = ar
    ARFLAGS = rcs 

else
    COV_DIR ?= $(P4ROOT)/sw/
    ifeq "$(LW_BULLSEYE)" "1"
        CLIENTPATHS += //sw/tools/Bullseye/win32/7.13.25/...
        CC := $(LW_BULLSEYE_BIN)/covc.exe -a -t$(COV_DIR) --no-banner $(BUILD_TOOLS_DIR)/win32/msvc90/bin/cl
        LINK = $(LW_BULLSEYE_BIN)/covlink.exe $(BUILD_TOOLS_DIR)/win32/msvc90/bin/link
    else
        CC := $(BUILD_TOOLS_DIR)/win32/msvc90/bin/cl.exe
        LINK := $(BUILD_TOOLS_DIR)/win32/msvc90/bin/link.exe
    endif

    PYTHON = $(BUILD_TOOLS_DIR)/win32/python/254/python.exe
    LWCFG_PROFILE ?= windows_global_internal_profile
    PERL = $(BUILD_TOOLS_DIR)/win32/ActivePerl/5.10.0.1004/bin/perl.exe
    ruby = $(BUILD_TOOLS_DIR)/ruby1.8.6/bin/ruby.exe

    AR = $(BUILD_TOOLS_DIR)/win32/msvc90/bin/lib
    ARFLAGS = /nologo /out:

    defines += -D_WIN32

    #eliminate deprication warnings for sprinf, strcpy 
    defines += -D_CRT_SELWRE_NO_WARNINGS
    defines += -D_SCL_SELWRE_NO_WARNINGS

    CFLAGS += -nologo

    ifeq "$(BUILD_CFG)" "release"
        CFLAGS += /MT /O2
    else
        CFLAGS += /MTd /Od /Zi
        defines += -DUNIT_DEBUG_BUILD
    endif

    TARGET_CFLAGS += /Fo$(o_dir)/ /Fd$(o_dir)/$(notdir $(@:.obj=.pdb))
    LINKFLAGS = -nodefaultlib:libcd -nologo -incremental:no 

    ifeq "$(BUILD_CFG)" "debug"
        LINKFLAGS += -debug
    endif

    LINKFLAGS += -libpath:$(BUILD_TOOLS_DIR)/win32/msvc90/lib \
           -libpath:$(BUILD_TOOLS_DIR)/win32/msvc90/PlatformSDK/Lib \
           -libpath:$(BUILD_TOOLS_DIR)/win32/msvc90/atlmfc/lib \
           -libpath:$(BUILD_TOOLS_DIR)/win32/msvc90/sdk/lwpu/lib/x86
endif

ECHO  = echo

# Add some extra warnings that are known to be useful
warnings =

ifeq ($(base_os),linux)
    ifeq "$(ENABLE_PEDANTIC)" "true"

        # Enables some easy to avoid warnings
        warnings += -Wall

        # Warn about anything that depends on the *size of* a function type or of void
        warnings += -Wpointer-arith 

        # Warn if a char * is cast to an int * (required alignment of the target is increased).
        warnings += -Wcast-align

    endif

    # treat warnings as errors
    # warnings += -Werror

    # Disable GCC 4 warning: deprecated colwersion from string constant to 'char*'
    warnings += -Wno-write-strings 
    warnings +=-Wno-unknown-pragmas

else
    ifeq "$(ENABLE_PEDANTIC)" "true"
        # Warn about most things
        warnings += /W2
        warnings += /w34063 # case 'identifier' is not a valid value for switch of enum 'enumeration'
        warnings += /w34130 # 'operator' : logical operation on address of string constant
        warnings += /w34131 # 'function' : uses old-style declarator
        warnings += /w34189 # 'identifier' : local variable is initialized but not referenced
        warnings += /w34205 # nonstandard extension used : static function declaration in function scope
        warnings += /w34211 # nonstandard extension used : redefined extern to static
        warnings += /w34310 # cast truncates constant value
        warnings += /w34505 # 'function' : unreferenced local function has been removed
    endif
endif

defines += -DLW_UNITTEST
CFLAGS += $(INCS) $(warnings) $(defines)

OBJS = $(addprefix $(o_dir)/, $(notdir $(CFILES:.c=$(obj_suffix))))
DEPS = $(OBJS:$(obj_suffix)=.d)

# Suppress compile command output. Allow it with SHOW_OUTPUT flag
ifeq "$(SHOW_OUTPUT)" "true"
   Q =
else
   Q = @
endif

#-------------------------------------------------------------------------------
# Turn off old-style suffix rules
#-------------------------------------------------------------------------------
.SUFFIXES:
