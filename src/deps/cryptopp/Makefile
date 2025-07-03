#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2019 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

base_name := libcryptopp
dll_build := false
MODS_DIR := ../mods

include $(MODS_DIR)/makedefs.inc

CRYPTOPP_DIR_ABS := $(COMMON_WS_DIR)/sw/tools/cryptopp-8.2
ifeq ($(toolchain),msvc)
    CFLAGS += /EHsc
    ifeq "$(BUILD_CFG)" "debug"
        optimize = /Od /Z7
    endif
endif
ifdef CYGWIN_MAKE
    # MSVC can't handle Cygwin paths, use relative path instead
    CRYPTOPP_DIR := $(shell $(MODS_DIR)/tools/abs2rel.sh '$(CRYPTOPP_DIR_ABS)')
else
    CRYPTOPP_DIR := $(CRYPTOPP_DIR_ABS)
endif

include_dirs += $(CRYPTOPP_DIR)

cryptopp_files += authenc.cpp
cryptopp_files += cryptlib.cpp
cryptopp_files += cpu.cpp
cryptopp_files += integer.cpp
cryptopp_files += algebra.cpp
cryptopp_files += algparam.cpp
cryptopp_files += asn.cpp
cryptopp_files += basecode.cpp
cryptopp_files += crc_simd.cpp
cryptopp_files += des.cpp
cryptopp_files += dessp.cpp
cryptopp_files += dll.cpp
cryptopp_files += ec2n.cpp
cryptopp_files += eccrypto.cpp
cryptopp_files += ecp.cpp
cryptopp_files += filters.cpp
cryptopp_files += fips140.cpp
cryptopp_files += gcm.cpp
cryptopp_files += gcm_simd.cpp
cryptopp_files += gf2n.cpp
cryptopp_files += gf2n_simd.cpp
cryptopp_files += gfpcrypt.cpp
cryptopp_files += hex.cpp
cryptopp_files += hmac.cpp
cryptopp_files += hrtimer.cpp
cryptopp_files += integer.cpp
cryptopp_files += iterhash.cpp
cryptopp_files += keccak_core.cpp
cryptopp_files += misc.cpp
cryptopp_files += modes.cpp
cryptopp_files += mqueue.cpp
cryptopp_files += neon_simd.cpp
cryptopp_files += nbtheory.cpp
cryptopp_files += oaep.cpp
cryptopp_files += osrng.cpp
cryptopp_files += ppc_power7.cpp
cryptopp_files += ppc_power8.cpp
cryptopp_files += ppc_power9.cpp
cryptopp_files += ppc_simd.cpp
cryptopp_files += pubkey.cpp
cryptopp_files += queue.cpp
cryptopp_files += randpool.cpp
cryptopp_files += rdtables.cpp
cryptopp_files += rijndael.cpp
cryptopp_files += rijndael_simd.cpp
cryptopp_files += rng.cpp
cryptopp_files += sha.cpp
cryptopp_files += sha3.cpp
cryptopp_files += sha_simd.cpp
cryptopp_files += shake.cpp
cryptopp_files += sse_simd.cpp
cryptopp_files += xtr.cpp

cpp_files += $(addprefix $(CRYPTOPP_DIR)/, $(cryptopp_files))

ifeq "$(base_os)" "win32"
ifeq "$(BUILD_ARCH)" "amd64"
asm_files := $(addprefix $(CRYPTOPP_DIR)/, x64dll.asm x64masm.asm)
ASFLAGS := /I$(mssdk_include_dir)/shared
endif
endif

ifneq (,$(filter $(BUILD_ARCH),x86 amd64))
ifneq "$(base_os)" "win32"
CFLAGS += -maes -msse4.2 -mpclmul -msha
endif
endif
ifeq "$(BUILD_ARCH)" "aarch64"
CFLAGS += -march=armv8-a+crypto+crc
endif
ifeq "$(BUILD_ARCH)" "ppc64le"
CFLAGS += -mcpu=power8
defines += -DCRYPTOPP_DISABLE_ALTIVEC # current toolchain doesn't handle it
endif
ifeq "$(base_os)" "macosx"
defines += -DCRYPTOPP_DISABLE_ASM
endif
ifeq "$(BUILD_OS_SUBTYPE)" "bionic"
include_dirs += . # for cpu-features.h
endif

all: $(o_dir)/$(lib_name)

install:
ifeq ($(toolchain),msvc)
	@echo "Copying $(o_dir)/$(lib_name) to $(MODS_DIR)/$(base_mods_lib_dir)/$(BUILD_CFG)/$(lib_name)..."
	@cp $(o_dir)/$(lib_name) $(MODS_DIR)/$(base_mods_lib_dir)/$(BUILD_CFG)/$(lib_name)
else
	@echo "Copying $(o_dir)/$(lib_name) to $(MODS_DIR)/$(base_mods_lib_dir)/$(lib_name)..."
	@cp $(o_dir)/$(lib_name) $(MODS_DIR)/$(base_mods_lib_dir)/$(lib_name)
endif

build_install: all
	$(Q)$(MAKE) --no-print-directory install

include $(MODS_DIR_UNIX)/makelib.inc
include $(MODS_DIR_UNIX)/makerule.inc
-include $(depend_files)
