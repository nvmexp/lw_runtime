# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2018 by LWPU Corporation. All rights reserved. All information
# contained herein is proprietary and confidential to LWPU Corporation. Any
# use, reproduction, or disclosure without the written permission of LWPU
# Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END

MODS_DIR ?= ../mods
SDK_DIR= ?= ../../sdk
BUILD_OS_SUBTYPE ?= gcc73

include $(MODS_DIR)/makedefs.inc

exe_build = true
base_name = lwenlwtil

include_dirs += $(BOOST_DIR)
include_dirs += $(SDK_DIR)/lwpu/inc
include_dirs += ../js1.7
include_dirs += ../utils
include_dirs += $(MODS_DIR)
include_dirs += $(MODS_DIR)/gpu/tests/lwencoders

pch_h_name = pch.hpp
pch_name   = pch.hpp.gch
depend_files += $(o_dir)/$(notdir $(pch_h_name)).d

# it's diffilwlt to reuse MODS code with DEBUG on
defines := $(filter-out -DDEBUG, $(defines))
defines += -DINCLUDE_STD_NAMESPACE
defines += -DMODS_FUNCTION=\"\"
defines += -DFUSION_MAX_VECTOR_SIZE=20
defines += -DBOOST_RESULT_OF_USE_DECLTYPE
defines += -DBOOST_PP_VARIADICS=1
defines += -DBOOST_WAVE_STRINGTYPE_USE_STDSTRING
defines += -DBOOST_WAVE_SUPPORT_THREADING=0

cpp_files = lwenlwtil.cpp
cpp_files += clexer.cpp
cpp_files += $(MODS_DIR)/gpu/tests/lwencoders/h264parser.cpp
cpp_files += $(MODS_DIR)/gpu/tests/lwencoders/h264syntax.cpp
cpp_files += $(MODS_DIR)/gpu/tests/lwencoders/h265parser.cpp
cpp_files += $(MODS_DIR)/gpu/tests/lwencoders/h265syntax.cpp
cpp_files += $(BOOST_DIR)/libs/filesystem/src/operations.cpp
cpp_files += $(BOOST_DIR)/libs/filesystem/src/path.cpp
cpp_files += $(BOOST_DIR)/libs/program_options/src/cmdline.cpp
cpp_files += $(BOOST_DIR)/libs/program_options/src/colwert.cpp
cpp_files += $(BOOST_DIR)/libs/program_options/src/options_description.cpp
cpp_files += $(BOOST_DIR)/libs/program_options/src/positional_options.cpp
cpp_files += $(BOOST_DIR)/libs/program_options/src/value_semantic.cpp
cpp_files += $(BOOST_DIR)/libs/program_options/src/variables_map.cpp
cpp_files += $(BOOST_DIR)/libs/system/src/error_code.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_re2c_lexer.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_re2c_lexer_str.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_cpp_exprgrammar.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_cpp_literalgrs.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/cpplexer/re2clex/cpp_re.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/cpplexer/re2clex/aq.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/token_ids.cpp

$(gen_cpp_dir)/filesystem_utf8_codecvt_facet.cpp: $(BOOST_DIR)/libs/filesystem/src/utf8_codecvt_facet.cpp $(gen_cpp_dir)/dummy.txt
	$(Q)$(CP) $< $@

$(gen_cpp_dir)/program_options_utf8_codecvt_facet.cpp: $(BOOST_DIR)/libs/program_options/src/utf8_codecvt_facet.cpp $(gen_cpp_dir)/dummy.txt
	$(Q)$(CP) $< $@

cpp_files += $(gen_cpp_dir)/filesystem_utf8_codecvt_facet.cpp
cpp_files += $(gen_cpp_dir)/program_options_utf8_codecvt_facet.cpp

gen_cpp_files += $(gen_cpp_dir)/filesystem_utf8_codecvt_facet.cpp
gen_cpp_files += $(gen_cpp_dir)/program_options_utf8_codecvt_facet.cpp

include $(MODS_DIR_UNIX)/makerule.inc

.PHONY : all clean

LDFLAGS += -static-libgcc

.DEFAULT_GOAL := all

$(o_dir)/$(notdir $(pch_h_name)): $(pch_h_name)
	$(CP) $(pch_h_name) $(o_dir)/$(notdir $(pch_h_name))

$(o_dir)/$(pch_name): $(o_dir)/$(notdir $(pch_h_name))
	$(call COMPILE_PCH,$<)

$(o_files): $(o_dir)/$(pch_name)

all:  $(o_dir)/dummy.txt $(o_dir)/$(exe_name)

clean :
	@-rm -f $(o_dir)/*$(obj_suffix)
	@-rm -f $(o_dir)/*.d
	@-rm -f $(o_dir)/*.pdb
	@-rm -f *.d
	@-rm -f $(o_dir)/$(exe_name)
	@-rm -f $(boost_files_to_clean)

$(o_dir)/$(exe_name) : $(o_files)
	$(Q)$(LD) $(LDFLAGS_EXE) $(o_files)

-include $(depend_files)
