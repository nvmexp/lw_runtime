#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

cc_files += $(PROTOBUF_DIR)/src/google/protobuf/stubs/atomicops_internals_x86_gcc.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/stubs/atomicops_internals_x86_msvc.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/stubs/common.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/stubs/once.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/stubs/stringprintf.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/extension_set.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/generated_message_util.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/message_lite.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/repeated_field.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/wire_format_lite.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/io/coded_stream.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/io/zero_copy_stream.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/io/zero_copy_stream_impl_lite.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/stubs/strutil.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/stubs/substitute.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/stubs/structurally_valid.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/descriptor.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/descriptor.pb.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/descriptor_database.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/dynamic_message.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/extension_set_heavy.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/generated_message_reflection.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/message.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/reflection_ops.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/service.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/text_format.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/unknown_field_set.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/wire_format.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/io/gzip_stream.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/io/printer.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/io/strtod.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/io/tokenizer.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/io/zero_copy_stream_impl.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/compiler/importer.cc
cc_files += $(PROTOBUF_DIR)/src/google/protobuf/compiler/parser.cc

gen_cc_files += $(gen_cpp_dir)/topology.pb.cc
all_gen_files += $(gen_cpp_dir)/topology.proto
cc_files += $(gen_cc_files)

PROTOC = $(BUILD_TOOLS_DIR)/unix/hosts/Linux-x86/unix-build/bin/protoc_2.6.0

$(gen_cpp_dir)/topology.proto : $(PROTOBUF_DIR)/../../common/protobuf/topology.proto.precomp
	$(CPP) $(CPPFLAGS) -P -DBUILD_LR_TOPOLOGY -o $(gen_cpp_dir)/topology.proto - < $<

$(gen_cpp_dir)/topology.pb.cc : $(gen_cpp_dir)/topology.proto
	@$(ECHO) Creating $@
	$(PROTOC) -I$(gen_cpp_dir) --cpp_out=$(gen_cpp_dir) $<

include_dirs += $(PROTOBUF_DIR)/src $(PROTOBUF_DIR)/include
