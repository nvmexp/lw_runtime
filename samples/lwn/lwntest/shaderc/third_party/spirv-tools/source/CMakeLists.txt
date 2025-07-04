# Copyright (c) 2015-2016 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(GRAMMAR_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_grammar_tables.py")
set(VIMSYNTAX_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_vim_syntax.py")
set(XML_REGISTRY_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_registry_tables.py")
set(LANG_HEADER_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_language_headers.py")

# For now, assume the DebugInfo grammar file is in the current directory.
# It might migrate to SPIRV-Headers.
set(DEBUGINFO_GRAMMAR_JSON_FILE "${CMAKE_LWRRENT_SOURCE_DIR}/extinst.debuginfo.grammar.json")
set(CLDEBUGINFO100_GRAMMAR_JSON_FILE "${CMAKE_LWRRENT_SOURCE_DIR}/extinst.opencl.debuginfo.100.grammar.json")

# macro() definitions are used in the following because we need to append .inc
# file paths into some global lists (*_CPP_DEPENDS). And those global lists are
# later used by set_source_files_properties() calls.
# function() definitions are not suitable because they create new scopes.
macro(spvtools_core_tables CONFIG_VERSION)
  set(GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(GRAMMAR_INSTS_INC_FILE "${spirv-tools_BINARY_DIR}/core.insts-${CONFIG_VERSION}.inc")
  set(GRAMMAR_KINDS_INC_FILE "${spirv-tools_BINARY_DIR}/operand.kinds-${CONFIG_VERSION}.inc")
  add_lwstom_command(OUTPUT ${GRAMMAR_INSTS_INC_FILE} ${GRAMMAR_KINDS_INC_FILE}
    COMMAND ${PYTHON_EXELWTABLE} ${GRAMMAR_PROCESSING_SCRIPT}
      --spirv-core-grammar=${GRAMMAR_JSON_FILE}
      --extinst-debuginfo-grammar=${DEBUGINFO_GRAMMAR_JSON_FILE}
      --extinst-cldebuginfo100-grammar=${CLDEBUGINFO100_GRAMMAR_JSON_FILE}
      --core-insts-output=${GRAMMAR_INSTS_INC_FILE}
      --operand-kinds-output=${GRAMMAR_KINDS_INC_FILE}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT}
            ${GRAMMAR_JSON_FILE}
            ${DEBUGINFO_GRAMMAR_JSON_FILE}
            ${CLDEBUGINFO100_GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for SPIR-V v${CONFIG_VERSION} core instructions and operands.")
  list(APPEND OPCODE_CPP_DEPENDS ${GRAMMAR_INSTS_INC_FILE})
  list(APPEND OPERAND_CPP_DEPENDS ${GRAMMAR_KINDS_INC_FILE})
endmacro(spvtools_core_tables)

macro(spvtools_enum_string_mapping CONFIG_VERSION)
  set(GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(GRAMMAR_EXTENSION_ENUM_INC_FILE "${spirv-tools_BINARY_DIR}/extension_enum.inc")
  set(GRAMMAR_ENUM_STRING_MAPPING_INC_FILE "${spirv-tools_BINARY_DIR}/enum_string_mapping.inc")
  add_lwstom_command(OUTPUT ${GRAMMAR_EXTENSION_ENUM_INC_FILE}
     ${GRAMMAR_ENUM_STRING_MAPPING_INC_FILE}
    COMMAND ${PYTHON_EXELWTABLE} ${GRAMMAR_PROCESSING_SCRIPT}
      --spirv-core-grammar=${GRAMMAR_JSON_FILE}
      --extinst-debuginfo-grammar=${DEBUGINFO_GRAMMAR_JSON_FILE}
      --extinst-cldebuginfo100-grammar=${CLDEBUGINFO100_GRAMMAR_JSON_FILE}
      --extension-enum-output=${GRAMMAR_EXTENSION_ENUM_INC_FILE}
      --enum-string-mapping-output=${GRAMMAR_ENUM_STRING_MAPPING_INC_FILE}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT}
            ${GRAMMAR_JSON_FILE}
            ${DEBUGINFO_GRAMMAR_JSON_FILE}
            ${CLDEBUGINFO100_GRAMMAR_JSON_FILE}
    COMMENT "Generate enum-string mapping for SPIR-V v${CONFIG_VERSION}.")
  list(APPEND EXTENSION_H_DEPENDS ${GRAMMAR_EXTENSION_ENUM_INC_FILE})
  list(APPEND ENUM_STRING_MAPPING_CPP_DEPENDS ${GRAMMAR_ENUM_STRING_MAPPING_INC_FILE})
endmacro(spvtools_enum_string_mapping)

macro(spvtools_vimsyntax CONFIG_VERSION CLVERSION)
  set(GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(GLSL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/extinst.glsl.std.450.grammar.json")
  set(OPENCL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/extinst.opencl.std.100.grammar.json")
  set(VIMSYNTAX_FILE "${spirv-tools_BINARY_DIR}/spvasm.vim")
  add_lwstom_command(OUTPUT ${VIMSYNTAX_FILE}
      COMMAND ${PYTHON_EXELWTABLE} ${VIMSYNTAX_PROCESSING_SCRIPT}
      --spirv-core-grammar=${GRAMMAR_JSON_FILE}
      --extinst-debuginfo-grammar=${DEBUGINFO_GRAMMAR_JSON_FILE}
      --extinst-glsl-grammar=${GLSL_GRAMMAR_JSON_FILE}
      --extinst-opencl-grammar=${OPENCL_GRAMMAR_JSON_FILE}
      >${VIMSYNTAX_FILE}
    DEPENDS ${VIMSYNTAX_PROCESSING_SCRIPT} ${GRAMMAR_JSON_FILE}
            ${GLSL_GRAMMAR_JSON_FILE} ${OPENCL_GRAMMAR_JSON_FILE} ${DEBUGINFO_GRAMMAR_JSON_FILE}
    COMMENT "Generate spvasm.vim: Vim syntax file for SPIR-V assembly.")
endmacro(spvtools_vimsyntax)

macro(spvtools_glsl_tables CONFIG_VERSION)
  set(CORE_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(GLSL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/extinst.glsl.std.450.grammar.json")
  set(GRAMMAR_INC_FILE "${spirv-tools_BINARY_DIR}/glsl.std.450.insts.inc")
  add_lwstom_command(OUTPUT ${GRAMMAR_INC_FILE}
    COMMAND ${PYTHON_EXELWTABLE} ${GRAMMAR_PROCESSING_SCRIPT}
      --extinst-glsl-grammar=${GLSL_GRAMMAR_JSON_FILE}
      --glsl-insts-output=${GRAMMAR_INC_FILE}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${CORE_GRAMMAR_JSON_FILE} ${GLSL_GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for GLSL extended instructions and operands v${CONFIG_VERSION}.")
  list(APPEND EXTINST_CPP_DEPENDS ${GRAMMAR_INC_FILE})
endmacro(spvtools_glsl_tables)

macro(spvtools_opencl_tables CONFIG_VERSION)
  set(CORE_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(OPENCL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/extinst.opencl.std.100.grammar.json")
  set(GRAMMAR_INC_FILE "${spirv-tools_BINARY_DIR}/opencl.std.insts.inc")
  add_lwstom_command(OUTPUT ${GRAMMAR_INC_FILE}
    COMMAND ${PYTHON_EXELWTABLE} ${GRAMMAR_PROCESSING_SCRIPT}
      --extinst-opencl-grammar=${OPENCL_GRAMMAR_JSON_FILE}
      --opencl-insts-output=${GRAMMAR_INC_FILE}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${CORE_GRAMMAR_JSON_FILE} ${OPENCL_GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for OpenCL extended instructions and operands v${CONFIG_VERSION}.")
  list(APPEND EXTINST_CPP_DEPENDS ${GRAMMAR_INC_FILE})
endmacro(spvtools_opencl_tables)

macro(spvtools_vendor_tables VENDOR_TABLE SHORT_NAME OPERAND_KIND_PREFIX)
  set(INSTS_FILE "${spirv-tools_BINARY_DIR}/${VENDOR_TABLE}.insts.inc")
  set(GRAMMAR_FILE "${spirv-tools_SOURCE_DIR}/source/extinst.${VENDOR_TABLE}.grammar.json")
  add_lwstom_command(OUTPUT ${INSTS_FILE}
    COMMAND ${PYTHON_EXELWTABLE} ${GRAMMAR_PROCESSING_SCRIPT}
      --extinst-vendor-grammar=${GRAMMAR_FILE}
      --vendor-insts-output=${INSTS_FILE}
      --vendor-operand-kind-prefix=${OPERAND_KIND_PREFIX}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${GRAMMAR_FILE}
    COMMENT "Generate extended instruction tables for ${VENDOR_TABLE}.")
  add_lwstom_target(spv-tools-${SHORT_NAME} DEPENDS ${INSTS_FILE})
  set_property(TARGET spv-tools-${SHORT_NAME} PROPERTY FOLDER "SPIRV-Tools build")
  list(APPEND EXTINST_CPP_DEPENDS spv-tools-${SHORT_NAME})
endmacro(spvtools_vendor_tables)

macro(spvtools_extinst_lang_headers NAME GRAMMAR_FILE)
  set(OUT_H ${spirv-tools_BINARY_DIR}/${NAME}.h)
  add_lwstom_command(OUTPUT ${OUT_H}
    COMMAND ${PYTHON_EXELWTABLE} ${LANG_HEADER_PROCESSING_SCRIPT}
      --extinst-grammar=${GRAMMAR_FILE}
      --extinst-output-path=${OUT_H}
    DEPENDS ${LANG_HEADER_PROCESSING_SCRIPT} ${GRAMMAR_FILE}
    COMMENT "Generate language specific header for ${NAME}.")
  add_lwstom_target(spirv-tools-header-${NAME} DEPENDS ${OUT_H})
  set_property(TARGET spirv-tools-header-${NAME} PROPERTY FOLDER "SPIRV-Tools build")
  list(APPEND EXTINST_CPP_DEPENDS spirv-tools-header-${NAME})
endmacro(spvtools_extinst_lang_headers)

spvtools_core_tables("unified1")
spvtools_enum_string_mapping("unified1")
spvtools_opencl_tables("unified1")
spvtools_glsl_tables("unified1")
spvtools_vendor_tables("spv-amd-shader-explicit-vertex-parameter" "spv-amd-sevp" "")
spvtools_vendor_tables("spv-amd-shader-trinary-minmax" "spv-amd-stm" "")
spvtools_vendor_tables("spv-amd-gcn-shader" "spv-amd-gs" "")
spvtools_vendor_tables("spv-amd-shader-ballot" "spv-amd-sb" "")
spvtools_vendor_tables("debuginfo" "debuginfo" "")
spvtools_vendor_tables("opencl.debuginfo.100" "cldi100" "CLDEBUG100_")
spvtools_extinst_lang_headers("DebugInfo" ${DEBUGINFO_GRAMMAR_JSON_FILE})
spvtools_extinst_lang_headers("OpenCLDebugInfo100" ${CLDEBUGINFO100_GRAMMAR_JSON_FILE})

spvtools_vimsyntax("unified1" "1.0")
add_lwstom_target(spirv-tools-vimsyntax DEPENDS ${VIMSYNTAX_FILE})
set_property(TARGET spirv-tools-vimsyntax PROPERTY FOLDER "SPIRV-Tools utilities")

# Extract the list of known generators from the SPIR-V XML registry file.
set(GENERATOR_INC_FILE ${spirv-tools_BINARY_DIR}/generators.inc)
set(SPIRV_XML_REGISTRY_FILE ${SPIRV_HEADER_INCLUDE_DIR}/spirv/spir-v.xml)
add_lwstom_command(OUTPUT ${GENERATOR_INC_FILE}
  COMMAND ${PYTHON_EXELWTABLE} ${XML_REGISTRY_PROCESSING_SCRIPT}
    --xml=${SPIRV_XML_REGISTRY_FILE}
    --generator-output=${GENERATOR_INC_FILE}
  DEPENDS ${XML_REGISTRY_PROCESSING_SCRIPT} ${SPIRV_XML_REGISTRY_FILE}
  COMMENT "Generate tables based on the SPIR-V XML registry.")
list(APPEND OPCODE_CPP_DEPENDS ${GENERATOR_INC_FILE})

# The following .cpp files include the above generated .inc files.
# Add those .inc files as their dependencies.
#
# We need to wrap the .inc files with a custom target to avoid problems when
# multiple targets depend on the same custom command.
add_lwstom_target(core_tables
  DEPENDS ${OPCODE_CPP_DEPENDS} ${OPERAND_CPP_DEPENDS})
add_lwstom_target(enum_string_mapping
  DEPENDS ${EXTENSION_H_DEPENDS} ${ENUM_STRING_MAPPING_CPP_DEPENDS})
add_lwstom_target(extinst_tables
  DEPENDS ${EXTINST_CPP_DEPENDS})

set_source_files_properties(
  ${CMAKE_LWRRENT_SOURCE_DIR}/extensions.h
  PROPERTIES HEADER_FILE_ONLY TRUE)

set(SPIRV_TOOLS_BUILD_VERSION_INC
  ${spirv-tools_BINARY_DIR}/build-version.inc)
set(SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR
  ${spirv-tools_SOURCE_DIR}/utils/update_build_version.py)
set(SPIRV_TOOLS_CHANGES_FILE
  ${spirv-tools_SOURCE_DIR}/CHANGES)
add_lwstom_command(OUTPUT ${SPIRV_TOOLS_BUILD_VERSION_INC}
   COMMAND ${PYTHON_EXELWTABLE}
           ${SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR}
           ${spirv-tools_SOURCE_DIR} ${SPIRV_TOOLS_BUILD_VERSION_INC}
   DEPENDS ${SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR}
           ${SPIRV_TOOLS_CHANGES_FILE}
   COMMENT "Update build-version.inc in the SPIRV-Tools build directory (if necessary).")
# Colwenience target for standalone generation of the build-version.inc file.
# This is not required for any dependence chain.
add_lwstom_target(spirv-tools-build-version
   DEPENDS ${SPIRV_TOOLS_BUILD_VERSION_INC})
set_property(TARGET spirv-tools-build-version PROPERTY FOLDER "SPIRV-Tools build")

list(APPEND PCH_DEPENDS ${ENUM_STRING_MAPPING_CPP_DEPENDS} ${OPCODE_CPP_DEPENDS} ${OPERAND_CPP_DEPENDS} ${EXTENSION_H_DEPENDS} ${EXTINST_CPP_DEPENDS} ${SPIRV_TOOLS_BUILD_VERSION_INC})
set_source_files_properties(
  ${CMAKE_LWRRENT_SOURCE_DIR}/pch_source.cpp
  PROPERTIES OBJECT_DEPENDS "${PCH_DEPENDS}")

add_subdirectory(opt)
add_subdirectory(reduce)
add_subdirectory(fuzz)
add_subdirectory(link)

set(SPIRV_SOURCES
  ${spirv-tools_SOURCE_DIR}/include/spirv-tools/libspirv.h

  ${CMAKE_LWRRENT_SOURCE_DIR}/util/bitutils.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/bit_vector.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/hex_float.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/make_unique.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/parse_number.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/small_vector.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/string_utils.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/timer.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/assembly_grammar.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/binary.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/cfa.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/diagnostic.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/disassemble.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/enum_set.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/enum_string_mapping.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/ext_inst.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/extensions.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/instruction.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/latest_version_glsl_std_450_header.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/latest_version_opencl_std_header.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/latest_version_spirv_header.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/macro.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/name_mapper.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/opcode.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/operand.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/parsed_operand.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/print.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_constant.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_definition.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_endian.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_fuzzer_options.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_optimizer_options.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_reducer_options.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_target_elw.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_validator_options.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/table.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/text.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/text_handler.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate.h

  ${CMAKE_LWRRENT_SOURCE_DIR}/util/bit_vector.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/parse_number.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/util/string_utils.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/assembly_grammar.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/binary.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/diagnostic.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/disassemble.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/enum_string_mapping.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/ext_inst.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/extensions.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/libspirv.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/name_mapper.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/opcode.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/operand.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/parsed_operand.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/print.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/software_version.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_endian.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_fuzzer_options.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_optimizer_options.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_reducer_options.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_target_elw.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/spirv_validator_options.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/table.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/text.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/text_handler.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_adjacency.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_annotation.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_arithmetics.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_atomics.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_barriers.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_bitwise.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_builtins.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_capability.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_cfg.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_composites.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_constants.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_colwersion.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_debug.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_decorations.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_derivatives.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_extensions.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_exelwtion_limitations.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_function.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_id.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_image.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_interfaces.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_instruction.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_layout.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_literals.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_logicals.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_memory.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_memory_semantics.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_misc.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_mode_setting.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_non_uniform.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_primitives.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_scopes.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_small_type_uses.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validate_type.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/decoration.h
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/basic_block.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/construct.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/function.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/instruction.cpp
  ${CMAKE_LWRRENT_SOURCE_DIR}/val/validation_state.cpp)

if (${SPIRV_TIMER_ENABLED})
  set(SPIRV_SOURCES
    ${SPIRV_SOURCES}
    ${CMAKE_LWRRENT_SOURCE_DIR}/util/timer.cpp)
endif()

# The software_version.cpp file includes build-version.inc.
# Rebuild the software_version.cpp object file if it is older than
# build-version.inc or whenever build-version.inc itself is out of
# date.  In the latter case, rebuild build-version.inc first.
# CMake is not smart enough to detect this dependency automatically.
# Without this, the dependency detection system for #included files
# does not kick in on a clean build for the following reason:  The
# build will fail early because it doesn't know how to build the
# missing source file build-version.inc. That oclwrs before the
# preprocessor is run on software_version.cpp to detect the
# #include dependency.
set_source_files_properties(
  ${CMAKE_LWRRENT_SOURCE_DIR}/software_version.cpp
  PROPERTIES OBJECT_DEPENDS "${SPIRV_TOOLS_BUILD_VERSION_INC}")

spvtools_pch(SPIRV_SOURCES pch_source)

add_library(${SPIRV_TOOLS} ${SPIRV_SOURCES})
spvtools_default_compile_options(${SPIRV_TOOLS})
target_include_directories(${SPIRV_TOOLS}
  PUBLIC
    $<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${spirv-tools_BINARY_DIR}
  PRIVATE ${SPIRV_HEADER_INCLUDE_DIR}
  )
set_property(TARGET ${SPIRV_TOOLS} PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(${SPIRV_TOOLS})
add_dependencies( ${SPIRV_TOOLS} core_tables enum_string_mapping extinst_tables )

add_library(${SPIRV_TOOLS}-shared SHARED ${SPIRV_SOURCES})
spvtools_default_compile_options(${SPIRV_TOOLS}-shared)
target_include_directories(${SPIRV_TOOLS}-shared
  PUBLIC
    $<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE ${spirv-tools_BINARY_DIR}
  PRIVATE ${SPIRV_HEADER_INCLUDE_DIR}
  )
set_target_properties(${SPIRV_TOOLS}-shared PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_property(TARGET ${SPIRV_TOOLS}-shared PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(${SPIRV_TOOLS}-shared)
target_compile_definitions(${SPIRV_TOOLS}-shared
  PRIVATE SPIRV_TOOLS_IMPLEMENTATION
  PUBLIC SPIRV_TOOLS_SHAREDLIB
)
add_dependencies( ${SPIRV_TOOLS}-shared core_tables enum_string_mapping extinst_tables )

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  find_library(LIBRT rt)
  if(LIBRT)
    target_link_libraries(${SPIRV_TOOLS} ${LIBRT})
    target_link_libraries(${SPIRV_TOOLS}-shared ${LIBRT})
  endif()
endif()

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS ${SPIRV_TOOLS} ${SPIRV_TOOLS}-shared EXPORT ${SPIRV_TOOLS}Targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  export(EXPORT ${SPIRV_TOOLS}Targets FILE ${SPIRV_TOOLS}Target.cmake)

  spvtools_config_package_dir(${SPIRV_TOOLS} PACKAGE_DIR)
  install(EXPORT ${SPIRV_TOOLS}Targets FILE ${SPIRV_TOOLS}Target.cmake DESTINATION ${PACKAGE_DIR})

  # Special config file for root library compared to other libs.
  file(WRITE ${CMAKE_BINARY_DIR}/${SPIRV_TOOLS}Config.cmake
    "include(\${CMAKE_LWRRENT_LIST_DIR}/${SPIRV_TOOLS}Target.cmake)\n"
    "set(${SPIRV_TOOLS}_LIBRARIES ${SPIRV_TOOLS})\n"
    "get_target_property(${SPIRV_TOOLS}_INCLUDE_DIRS ${SPIRV_TOOLS} INTERFACE_INCLUDE_DIRECTORIES)\n")
  install(FILES ${CMAKE_BINARY_DIR}/${SPIRV_TOOLS}Config.cmake DESTINATION ${PACKAGE_DIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)

if(MSVC)
  # Enable parallel builds across four cores for this lib
  add_definitions(/MP4)
endif()
