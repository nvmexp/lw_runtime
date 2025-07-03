# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2014-2018 by LWPU Corporation. All rights reserved. All
# information contained herein is proprietary and confidential to LWPU
# Corporation. Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END

include_dirs += $(BOOST_DIR)

# boost is trying to workaround this glibc bug:
# http://sourceware.org/bugzilla/show_bug.cgi?id=2445, but our toolchain is
# missing felw.h and fails to compile with this workaround. The define below
# turns off the workaround.
defines += -DBOOST_NO_FELW_H
defines += -DBOOST_WAVE_SUPPORT_THREADING=0

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
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_cpp_grammar.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_cpp_exprgrammar.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_defined_grammar.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_cpp_literalgrs.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/instantiate_predef_macros.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/cpplexer/re2clex/cpp_re.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/cpplexer/re2clex/aq.cpp
cpp_files += $(BOOST_DIR)/libs/wave/src/token_ids.cpp

boost_o_files += $(o_dir)/program_options/utf8_codecvt_facet$(obj_suffix)
boost_o_files += $(o_dir)/filesystem/utf8_codecvt_facet$(obj_suffix)

#manually resolve filename clash between program_options and filesystem
$(o_dir)/program_options/utf8_codecvt_facet$(obj_suffix) : \
    $(BOOST_DIR)/libs/program_options/src/utf8_codecvt_facet.cpp \
	$(o_dir)/program_options/dummy.txt
	$(call COMPILE_CPP,$<)

$(o_dir)/filesystem/utf8_codecvt_facet$(obj_suffix) : \
    $(BOOST_DIR)/libs/filesystem/src/utf8_codecvt_facet.cpp \
    $(o_dir)/filesystem/dummy.txt
	$(call COMPILE_CPP,$<)

$(o_dir)/program_options/dummy.txt:
	$(Q)$(MKDIR_P) $(o_dir)/program_options
	$(Q)$(TOUCH) $(o_dir)/program_options/dummy.txt

$(o_dir)/filesystem/dummy.txt:
	$(Q)$(MKDIR_P) $(o_dir)/filesystem
	$(Q)$(TOUCH) $(o_dir)/filesystem/dummy.txt

boost_files_to_clean += $(o_dir)/program_options/*.d
boost_files_to_clean += $(o_dir)/program_options/*$(obj_suffix)
boost_files_to_clean += $(o_dir)/filesystem/*.d
boost_files_to_clean += $(o_dir)/filesystem/*$(obj_suffix)
