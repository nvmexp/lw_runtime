// Copyright (c) 2018 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Validates correctness of extension SPIR-V instructions.

#include <sstream>
#include <string>
#include <vector>

#include "OpenCLDebugInfo100.h"
#include "source/diagnostic.h"
#include "source/enum_string_mapping.h"
#include "source/extensions.h"
#include "source/latest_version_glsl_std_450_header.h"
#include "source/latest_version_opencl_std_header.h"
#include "source/opcode.h"
#include "source/spirv_target_elw.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

uint32_t GetSizeTBitWidth(const ValidationState_t& _) {
  if (_.addressing_model() == SpvAddressingModelPhysical32) return 32;

  if (_.addressing_model() == SpvAddressingModelPhysical64) return 64;

  return 0;
}

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of an instruction with |expected_opcode|.
spv_result_t ValidateOperandForDebugInfo(
    ValidationState_t& _, const std::string& operand_name,
    SpvOp expected_opcode, const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name) {
  auto* operand = _.FindDef(inst->word(word_index));
  if (operand->opcode() != expected_opcode) {
    spv_opcode_desc desc = nullptr;
    if (_.grammar().lookupOpcode(expected_opcode, &desc) != SPV_SUCCESS ||
        !desc) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << ext_inst_name() << ": "
             << "expected operand " << operand_name << " is invalid";
    }
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << ext_inst_name() << ": "
           << "expected operand " << operand_name << " must be a result id of "
           << "Op" << desc->name;
  }
  return SPV_SUCCESS;
}

#define CHECK_OPERAND(NAME, opcode, index)                                  \
  do {                                                                      \
    auto result = ValidateOperandForDebugInfo(_, NAME, opcode, inst, index, \
                                              ext_inst_name);               \
    if (result != SPV_SUCCESS) return result;                               \
  } while (0)

// True if the operand of a debug info instruction |inst| at |word_index|
// satisifies |expectation| that is given as a function. Otherwise,
// returns false.
bool DoesDebugInfoOperandMatchExpectation(
    const ValidationState_t& _,
    const std::function<bool(OpenCLDebugInfo100Instructions)>& expectation,
    const Instruction* inst, uint32_t word_index) {
  auto* debug_inst = _.FindDef(inst->word(word_index));
  if (debug_inst->opcode() != SpvOpExtInst ||
      debug_inst->ext_inst_type() != SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100 ||
      !expectation(OpenCLDebugInfo100Instructions(debug_inst->word(4)))) {
    return false;
  }
  return true;
}

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of an debug info instruction whose debug instruction type
// is |expected_debug_inst|.
spv_result_t ValidateDebugInfoOperand(
    ValidationState_t& _, const std::string& debug_inst_name,
    OpenCLDebugInfo100Instructions expected_debug_inst, const Instruction* inst,
    uint32_t word_index, const std::function<std::string()>& ext_inst_name) {
  std::function<bool(OpenCLDebugInfo100Instructions)> expectation =
      [expected_debug_inst](OpenCLDebugInfo100Instructions dbg_inst) {
        return dbg_inst == expected_debug_inst;
      };
  if (DoesDebugInfoOperandMatchExpectation(_, expectation, inst, word_index))
    return SPV_SUCCESS;

  spv_ext_inst_desc desc = nullptr;
  _.grammar().lookupExtInst(SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100,
                            expected_debug_inst, &desc);
  if (_.grammar().lookupExtInst(SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100,
                                expected_debug_inst, &desc) != SPV_SUCCESS ||
      !desc) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << ext_inst_name() << ": "
           << "expected operand " << debug_inst_name << " is invalid";
  }
  return _.diag(SPV_ERROR_ILWALID_DATA, inst)
         << ext_inst_name() << ": "
         << "expected operand " << debug_inst_name << " must be a result id of "
         << desc->name;
}

#define CHECK_DEBUG_OPERAND(NAME, debug_opcode, index)                         \
  do {                                                                         \
    auto result = ValidateDebugInfoOperand(_, NAME, debug_opcode, inst, index, \
                                           ext_inst_name);                     \
    if (result != SPV_SUCCESS) return result;                                  \
  } while (0)

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of an debug info instruction with DebugTypeBasic.
spv_result_t ValidateOperandBaseType(
    ValidationState_t& _, const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name) {
  return ValidateDebugInfoOperand(_, "Base Type",
                                  OpenCLDebugInfo100DebugTypeBasic, inst,
                                  word_index, ext_inst_name);
}

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of a debug lexical scope instruction which is one of
// DebugCompilationUnit, DebugFunction, DebugLexicalBlock, or
// DebugTypeComposite.
spv_result_t ValidateOperandLexicalScope(
    ValidationState_t& _, const std::string& debug_inst_name,
    const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name) {
  std::function<bool(OpenCLDebugInfo100Instructions)> expectation =
      [](OpenCLDebugInfo100Instructions dbg_inst) {
        return dbg_inst == OpenCLDebugInfo100DebugCompilationUnit ||
               dbg_inst == OpenCLDebugInfo100DebugFunction ||
               dbg_inst == OpenCLDebugInfo100DebugLexicalBlock ||
               dbg_inst == OpenCLDebugInfo100DebugTypeComposite;
      };
  if (DoesDebugInfoOperandMatchExpectation(_, expectation, inst, word_index))
    return SPV_SUCCESS;

  return _.diag(SPV_ERROR_ILWALID_DATA, inst)
         << ext_inst_name() << ": "
         << "expected operand " << debug_inst_name
         << " must be a result id of a lexical scope";
}

// Check that the operand of a debug info instruction |inst| at |word_index|
// is a result id of a debug type instruction (See DebugTypeXXX in
// "4.3. Type instructions" section of OpenCL.DebugInfo.100 spec.
spv_result_t ValidateOperandDebugType(
    ValidationState_t& _, const std::string& debug_inst_name,
    const Instruction* inst, uint32_t word_index,
    const std::function<std::string()>& ext_inst_name) {
  std::function<bool(OpenCLDebugInfo100Instructions)> expectation =
      [](OpenCLDebugInfo100Instructions dbg_inst) {
        return OpenCLDebugInfo100DebugTypeBasic <= dbg_inst &&
               dbg_inst <= OpenCLDebugInfo100DebugTypePtrToMember;
      };
  if (DoesDebugInfoOperandMatchExpectation(_, expectation, inst, word_index))
    return SPV_SUCCESS;

  return _.diag(SPV_ERROR_ILWALID_DATA, inst)
         << ext_inst_name() << ": "
         << "expected operand " << debug_inst_name
         << " is not a valid debug type";
}

}  // anonymous namespace

spv_result_t ValidateExtension(ValidationState_t& _, const Instruction* inst) {
  if (spvIsWebGPUElw(_.context()->target_elw)) {
    std::string extension = GetExtensionString(&(inst->c_inst()));

    if (extension != ExtensionToString(kSPV_KHR_vulkan_memory_model)) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "For WebGPU, the only valid parameter to OpExtension is "
             << "\"" << ExtensionToString(kSPV_KHR_vulkan_memory_model)
             << "\".";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateExtInstImport(ValidationState_t& _,
                                   const Instruction* inst) {
  const auto name_id = 1;
  if (spvIsWebGPUElw(_.context()->target_elw)) {
    const std::string name(reinterpret_cast<const char*>(
        inst->words().data() + inst->operands()[name_id].offset));
    if (name != "GLSL.std.450") {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "For WebGPU, the only valid parameter to OpExtInstImport is "
                "\"GLSL.std.450\".";
    }
  }

  if (!_.HasExtension(kSPV_KHR_non_semantic_info)) {
    const std::string name(reinterpret_cast<const char*>(
        inst->words().data() + inst->operands()[name_id].offset));
    if (name.find("NonSemantic.") == 0) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "NonSemantic extended instruction sets cannot be declared "
                "without SPV_KHR_non_semantic_info.";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateExtInst(ValidationState_t& _, const Instruction* inst) {
  const uint32_t result_type = inst->type_id();
  const uint32_t num_operands = static_cast<uint32_t>(inst->operands().size());

  const uint32_t ext_inst_set = inst->word(3);
  const uint32_t ext_inst_index = inst->word(4);
  const spv_ext_inst_type_t ext_inst_type =
      spv_ext_inst_type_t(inst->ext_inst_type());

  auto ext_inst_name = [&_, ext_inst_set, ext_inst_type, ext_inst_index]() {
    spv_ext_inst_desc desc = nullptr;
    if (_.grammar().lookupExtInst(ext_inst_type, ext_inst_index, &desc) !=
            SPV_SUCCESS ||
        !desc) {
      return std::string("Unknown ExtInst");
    }

    auto* import_inst = _.FindDef(ext_inst_set);
    assert(import_inst);

    std::ostringstream ss;
    ss << reinterpret_cast<const char*>(import_inst->words().data() + 2);
    ss << " ";
    ss << desc->name;

    return ss.str();
  };

  if (ext_inst_type == SPV_EXT_INST_TYPE_GLSL_STD_450) {
    const GLSLstd450 ext_inst_key = GLSLstd450(ext_inst_index);
    switch (ext_inst_key) {
      case GLSLstd450Round:
      case GLSLstd450RoundEven:
      case GLSLstd450FAbs:
      case GLSLstd450Trunc:
      case GLSLstd450FSign:
      case GLSLstd450Floor:
      case GLSLstd450Ceil:
      case GLSLstd450Fract:
      case GLSLstd450Sqrt:
      case GLSLstd450IlwerseSqrt:
      case GLSLstd450FMin:
      case GLSLstd450FMax:
      case GLSLstd450FClamp:
      case GLSLstd450FMix:
      case GLSLstd450Step:
      case GLSLstd450SmoothStep:
      case GLSLstd450Fma:
      case GLSLstd450Normalize:
      case GLSLstd450FaceForward:
      case GLSLstd450Reflect:
      case GLSLstd450NMin:
      case GLSLstd450NMax:
      case GLSLstd450NClamp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case GLSLstd450SAbs:
      case GLSLstd450SSign:
      case GLSLstd450UMin:
      case GLSLstd450SMin:
      case GLSLstd450UMax:
      case GLSLstd450SMax:
      case GLSLstd450UClamp:
      case GLSLstd450SClamp:
      case GLSLstd450FindILsb:
      case GLSLstd450FindUMsb:
      case GLSLstd450FindSMsb: {
        if (!_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int scalar or vector type";
        }

        const uint32_t result_type_bit_width = _.GetBitWidth(result_type);
        const uint32_t result_type_dimension = _.GetDimension(result_type);

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (!_.IsIntScalarOrVectorType(operand_type)) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected all operands to be int scalars or vectors";
          }

          if (result_type_dimension != _.GetDimension(operand_type)) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected all operands to have the same dimension as "
                   << "Result Type";
          }

          if (result_type_bit_width != _.GetBitWidth(operand_type)) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected all operands to have the same bit width as "
                   << "Result Type";
          }

          if (ext_inst_key == GLSLstd450FindUMsb ||
              ext_inst_key == GLSLstd450FindSMsb) {
            if (result_type_bit_width != 32) {
              return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                     << ext_inst_name() << ": "
                     << "this instruction is lwrrently limited to 32-bit width "
                     << "components";
            }
          }
        }
        break;
      }

      case GLSLstd450Radians:
      case GLSLstd450Degrees:
      case GLSLstd450Sin:
      case GLSLstd450Cos:
      case GLSLstd450Tan:
      case GLSLstd450Asin:
      case GLSLstd450Acos:
      case GLSLstd450Atan:
      case GLSLstd450Sinh:
      case GLSLstd450Cosh:
      case GLSLstd450Tanh:
      case GLSLstd450Asinh:
      case GLSLstd450Acosh:
      case GLSLstd450Atanh:
      case GLSLstd450Exp:
      case GLSLstd450Exp2:
      case GLSLstd450Log:
      case GLSLstd450Log2:
      case GLSLstd450Atan2:
      case GLSLstd450Pow: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 16 or 32-bit scalar or "
                    "vector float type";
        }

        const uint32_t result_type_bit_width = _.GetBitWidth(result_type);
        if (result_type_bit_width != 16 && result_type_bit_width != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 16 or 32-bit scalar or "
                    "vector float type";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case GLSLstd450Determinant: {
        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;
        uint32_t col_type = 0;
        uint32_t component_type = 0;
        if (!_.GetMatrixTypeInfo(x_type, &num_rows, &num_cols, &col_type,
                                 &component_type) ||
            num_rows != num_cols) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to be a square matrix";
        }

        if (result_type != component_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X component type to be equal to "
                 << "Result Type";
        }
        break;
      }

      case GLSLstd450MatrixIlwerse: {
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;
        uint32_t col_type = 0;
        uint32_t component_type = 0;
        if (!_.GetMatrixTypeInfo(result_type, &num_rows, &num_cols, &col_type,
                                 &component_type) ||
            num_rows != num_cols) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a square matrix";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }
        break;
      }

      case GLSLstd450Modf: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t i_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        uint32_t i_storage_class = 0;
        uint32_t i_data_type = 0;
        if (!_.GetPointerTypeInfo(i_type, &i_data_type, &i_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand I to be a pointer";
        }

        if (i_data_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand I data type to be equal to Result Type";
        }

        break;
      }

      case GLSLstd450ModfStruct: {
        std::vector<uint32_t> result_types;
        if (!_.GetStructMemberTypes(result_type, &result_types) ||
            result_types.size() != 2 ||
            !_.IsFloatScalarOrVectorType(result_types[0]) ||
            result_types[1] != result_types[0]) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a struct with two identical "
                 << "scalar or vector float type members";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (x_type != result_types[0]) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to members of "
                 << "Result Type struct";
        }
        break;
      }

      case GLSLstd450Frexp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t exp_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        uint32_t exp_storage_class = 0;
        uint32_t exp_data_type = 0;
        if (!_.GetPointerTypeInfo(exp_type, &exp_data_type,
                                  &exp_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to be a pointer";
        }

        if (!_.IsIntScalarOrVectorType(exp_data_type) ||
            (!_.HasExtension(kSPV_AMD_gpu_shader_int16) &&
             _.GetBitWidth(exp_data_type) != 32) ||
            (_.HasExtension(kSPV_AMD_gpu_shader_int16) &&
             _.GetBitWidth(exp_data_type) != 16 &&
             _.GetBitWidth(exp_data_type) != 32)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp data type to be a "
                 << (_.HasExtension(kSPV_AMD_gpu_shader_int16)
                         ? "16-bit or 32-bit "
                         : "32-bit ")
                 << "int scalar or vector type";
        }

        if (_.GetDimension(result_type) != _.GetDimension(exp_data_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp data type to have the same component "
                 << "number as Result Type";
        }

        break;
      }

      case GLSLstd450Ldexp: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or vector float type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t exp_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        if (!_.IsIntScalarOrVectorType(exp_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to be a 32-bit int scalar "
                 << "or vector type";
        }

        if (_.GetDimension(result_type) != _.GetDimension(exp_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Exp to have the same component "
                 << "number as Result Type";
        }

        break;
      }

      case GLSLstd450FrexpStruct: {
        std::vector<uint32_t> result_types;
        if (!_.GetStructMemberTypes(result_type, &result_types) ||
            result_types.size() != 2 ||
            !_.IsFloatScalarOrVectorType(result_types[0]) ||
            !_.IsIntScalarOrVectorType(result_types[1]) ||
            (!_.HasExtension(kSPV_AMD_gpu_shader_int16) &&
             _.GetBitWidth(result_types[1]) != 32) ||
            (_.HasExtension(kSPV_AMD_gpu_shader_int16) &&
             _.GetBitWidth(result_types[1]) != 16 &&
             _.GetBitWidth(result_types[1]) != 32) ||
            _.GetDimension(result_types[0]) !=
                _.GetDimension(result_types[1])) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a struct with two members, "
                 << "first member a float scalar or vector, second member a "
                 << (_.HasExtension(kSPV_AMD_gpu_shader_int16)
                         ? "16-bit or 32-bit "
                         : "32-bit ")
                 << "int scalar or vector with the same number of "
                 << "components as the first member";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (x_type != result_types[0]) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to the first member "
                 << "of Result Type struct";
        }
        break;
      }

      case GLSLstd450PackSnorm4x8:
      case GLSLstd450PackUnorm4x8: {
        if (!_.IsIntScalarType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 32-bit int scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatVectorType(v_type) || _.GetDimension(v_type) != 4 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit float vector of size 4";
        }
        break;
      }

      case GLSLstd450PackSnorm2x16:
      case GLSLstd450PackUnorm2x16:
      case GLSLstd450PackHalf2x16: {
        if (!_.IsIntScalarType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 32-bit int scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatVectorType(v_type) || _.GetDimension(v_type) != 2 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit float vector of size 2";
        }
        break;
      }

      case GLSLstd450PackDouble2x32: {
        if (!_.IsFloatScalarType(result_type) ||
            _.GetBitWidth(result_type) != 64) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be 64-bit float scalar type";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntVectorType(v_type) || _.GetDimension(v_type) != 2 ||
            _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 32-bit int vector of size 2";
        }
        break;
      }

      case GLSLstd450UnpackSnorm4x8:
      case GLSLstd450UnpackUnorm4x8: {
        if (!_.IsFloatVectorType(result_type) ||
            _.GetDimension(result_type) != 4 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float vector of size "
                    "4";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntScalarType(v_type) || _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a 32-bit int scalar";
        }
        break;
      }

      case GLSLstd450UnpackSnorm2x16:
      case GLSLstd450UnpackUnorm2x16:
      case GLSLstd450UnpackHalf2x16: {
        if (!_.IsFloatVectorType(result_type) ||
            _.GetDimension(result_type) != 2 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float vector of size "
                    "2";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntScalarType(v_type) || _.GetBitWidth(v_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a 32-bit int scalar";
        }
        break;
      }

      case GLSLstd450UnpackDouble2x32: {
        if (!_.IsIntVectorType(result_type) ||
            _.GetDimension(result_type) != 2 ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int vector of size "
                    "2";
        }

        const uint32_t v_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarType(v_type) || _.GetBitWidth(v_type) != 64) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand V to be a 64-bit float scalar";
        }
        break;
      }

      case GLSLstd450Length: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(x_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(x_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X component type to be equal to Result "
                    "Type";
        }
        break;
      }

      case GLSLstd450Distance: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t p0_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(p0_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(p0_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 component type to be equal to "
                 << "Result Type";
        }

        const uint32_t p1_type = _.GetOperandTypeId(inst, 5);
        if (!_.IsFloatScalarOrVectorType(p1_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P1 to be of float scalar or vector type";
        }

        if (result_type != _.GetComponentType(p1_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P1 component type to be equal to "
                 << "Result Type";
        }

        if (_.GetDimension(p0_type) != _.GetDimension(p1_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operands P0 and P1 to have the same number of "
                 << "components";
        }
        break;
      }

      case GLSLstd450Cross: {
        if (!_.IsFloatVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float vector type";
        }

        if (_.GetDimension(result_type) != 3) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 3 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t y_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        if (y_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Y type to be equal to Result Type";
        }
        break;
      }

      case GLSLstd450Refract: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t i_type = _.GetOperandTypeId(inst, 4);
        const uint32_t n_type = _.GetOperandTypeId(inst, 5);
        const uint32_t eta_type = _.GetOperandTypeId(inst, 6);

        if (result_type != i_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand I to be of type equal to Result Type";
        }

        if (result_type != n_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand N to be of type equal to Result Type";
        }

        if (!_.IsFloatScalarType(eta_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Eta to be a float scalar";
        }
        break;
      }

      case GLSLstd450InterpolateAtCentroid:
      case GLSLstd450InterpolateAtSample:
      case GLSLstd450InterpolateAtOffset: {
        if (!_.HasCapability(SpvCapabilityInterpolationFunction)) {
          return _.diag(SPV_ERROR_ILWALID_CAPABILITY, inst)
                 << ext_inst_name()
                 << " requires capability InterpolationFunction";
        }

        if (!_.IsFloatScalarOrVectorType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit float scalar "
                 << "or vector type";
        }

        const uint32_t interpolant_type = _.GetOperandTypeId(inst, 4);
        uint32_t interpolant_storage_class = 0;
        uint32_t interpolant_data_type = 0;
        if (!_.GetPointerTypeInfo(interpolant_type, &interpolant_data_type,
                                  &interpolant_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Interpolant to be a pointer";
        }

        if (result_type != interpolant_data_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Interpolant data type to be equal to Result Type";
        }

        if (interpolant_storage_class != SpvStorageClassInput) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Interpolant storage class to be Input";
        }

        if (ext_inst_key == GLSLstd450InterpolateAtSample) {
          const uint32_t sample_type = _.GetOperandTypeId(inst, 5);
          if (!_.IsIntScalarType(sample_type) ||
              _.GetBitWidth(sample_type) != 32) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Sample to be 32-bit integer";
          }
        }

        if (ext_inst_key == GLSLstd450InterpolateAtOffset) {
          const uint32_t offset_type = _.GetOperandTypeId(inst, 5);
          if (!_.IsFloatVectorType(offset_type) ||
              _.GetDimension(offset_type) != 2 ||
              _.GetBitWidth(offset_type) != 32) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Offset to be a vector of 2 32-bit floats";
          }
        }

        _.function(inst->function()->id())
            ->RegisterExelwtionModelLimitation(
                SpvExelwtionModelFragment,
                ext_inst_name() +
                    std::string(" requires Fragment exelwtion model"));
        break;
      }

      case GLSLstd450IMix: {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << "Extended instruction GLSLstd450IMix is not supported";
      }

      case GLSLstd450Bad: {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << "Encountered extended instruction GLSLstd450Bad";
      }

      case GLSLstd450Count: {
        assert(0);
        break;
      }
    }
  } else if (ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_STD) {
    const OpenCLLIB::Entrypoints ext_inst_key =
        OpenCLLIB::Entrypoints(ext_inst_index);
    switch (ext_inst_key) {
      case OpenCLLIB::Acos:
      case OpenCLLIB::Acosh:
      case OpenCLLIB::Acospi:
      case OpenCLLIB::Asin:
      case OpenCLLIB::Asinh:
      case OpenCLLIB::Asinpi:
      case OpenCLLIB::Atan:
      case OpenCLLIB::Atan2:
      case OpenCLLIB::Atanh:
      case OpenCLLIB::Atanpi:
      case OpenCLLIB::Atan2pi:
      case OpenCLLIB::Cbrt:
      case OpenCLLIB::Ceil:
      case OpenCLLIB::Copysign:
      case OpenCLLIB::Cos:
      case OpenCLLIB::Cosh:
      case OpenCLLIB::Cospi:
      case OpenCLLIB::Erfc:
      case OpenCLLIB::Erf:
      case OpenCLLIB::Exp:
      case OpenCLLIB::Exp2:
      case OpenCLLIB::Exp10:
      case OpenCLLIB::Expm1:
      case OpenCLLIB::Fabs:
      case OpenCLLIB::Fdim:
      case OpenCLLIB::Floor:
      case OpenCLLIB::Fma:
      case OpenCLLIB::Fmax:
      case OpenCLLIB::Fmin:
      case OpenCLLIB::Fmod:
      case OpenCLLIB::Hypot:
      case OpenCLLIB::Lgamma:
      case OpenCLLIB::Log:
      case OpenCLLIB::Log2:
      case OpenCLLIB::Log10:
      case OpenCLLIB::Log1p:
      case OpenCLLIB::Logb:
      case OpenCLLIB::Mad:
      case OpenCLLIB::Maxmag:
      case OpenCLLIB::Minmag:
      case OpenCLLIB::Nextafter:
      case OpenCLLIB::Pow:
      case OpenCLLIB::Powr:
      case OpenCLLIB::Remainder:
      case OpenCLLIB::Rint:
      case OpenCLLIB::Round:
      case OpenCLLIB::Rsqrt:
      case OpenCLLIB::Sin:
      case OpenCLLIB::Sinh:
      case OpenCLLIB::Sinpi:
      case OpenCLLIB::Sqrt:
      case OpenCLLIB::Tan:
      case OpenCLLIB::Tanh:
      case OpenCLLIB::Tanpi:
      case OpenCLLIB::Tgamma:
      case OpenCLLIB::Trunc:
      case OpenCLLIB::Half_cos:
      case OpenCLLIB::Half_divide:
      case OpenCLLIB::Half_exp:
      case OpenCLLIB::Half_exp2:
      case OpenCLLIB::Half_exp10:
      case OpenCLLIB::Half_log:
      case OpenCLLIB::Half_log2:
      case OpenCLLIB::Half_log10:
      case OpenCLLIB::Half_powr:
      case OpenCLLIB::Half_recip:
      case OpenCLLIB::Half_rsqrt:
      case OpenCLLIB::Half_sin:
      case OpenCLLIB::Half_sqrt:
      case OpenCLLIB::Half_tan:
      case OpenCLLIB::Native_cos:
      case OpenCLLIB::Native_divide:
      case OpenCLLIB::Native_exp:
      case OpenCLLIB::Native_exp2:
      case OpenCLLIB::Native_exp10:
      case OpenCLLIB::Native_log:
      case OpenCLLIB::Native_log2:
      case OpenCLLIB::Native_log10:
      case OpenCLLIB::Native_powr:
      case OpenCLLIB::Native_recip:
      case OpenCLLIB::Native_rsqrt:
      case OpenCLLIB::Native_sin:
      case OpenCLLIB::Native_sqrt:
      case OpenCLLIB::Native_tan:
      case OpenCLLIB::FClamp:
      case OpenCLLIB::Degrees:
      case OpenCLLIB::FMax_common:
      case OpenCLLIB::FMin_common:
      case OpenCLLIB::Mix:
      case OpenCLLIB::Radians:
      case OpenCLLIB::Step:
      case OpenCLLIB::Smoothstep:
      case OpenCLLIB::Sign: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case OpenCLLIB::Fract:
      case OpenCLLIB::Modf:
      case OpenCLLIB::Sincos: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected type of operand X to be equal to Result Type";
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, 5);
        uint32_t p_storage_class = 0;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected the last operand to be a pointer";
        }

        if (p_storage_class != SpvStorageClassGeneric &&
            p_storage_class != SpvStorageClassCrossWorkgroup &&
            p_storage_class != SpvStorageClassWorkgroup &&
            p_storage_class != SpvStorageClassFunction) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected storage class of the pointer to be Generic, "
                    "CrossWorkgroup, Workgroup or Function";
        }

        if (result_type != p_data_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected data type of the pointer to be equal to Result "
                    "Type";
        }
        break;
      }

      case OpenCLLIB::Frexp:
      case OpenCLLIB::Lgamma_r:
      case OpenCLLIB::Remquo: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        uint32_t operand_index = 4;
        const uint32_t x_type = _.GetOperandTypeId(inst, operand_index++);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected type of operand X to be equal to Result Type";
        }

        if (ext_inst_key == OpenCLLIB::Remquo) {
          const uint32_t y_type = _.GetOperandTypeId(inst, operand_index++);
          if (result_type != y_type) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected type of operand Y to be equal to Result Type";
          }
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, operand_index++);
        uint32_t p_storage_class = 0;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected the last operand to be a pointer";
        }

        if (p_storage_class != SpvStorageClassGeneric &&
            p_storage_class != SpvStorageClassCrossWorkgroup &&
            p_storage_class != SpvStorageClassWorkgroup &&
            p_storage_class != SpvStorageClassFunction) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected storage class of the pointer to be Generic, "
                    "CrossWorkgroup, Workgroup or Function";
        }

        if (!_.IsIntScalarOrVectorType(p_data_type) ||
            _.GetBitWidth(p_data_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected data type of the pointer to be a 32-bit int "
                    "scalar or vector type";
        }

        if (_.GetDimension(p_data_type) != num_components) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected data type of the pointer to have the same number "
                    "of components as Result Type";
        }
        break;
      }

      case OpenCLLIB::Ilogb: {
        if (!_.IsIntScalarOrVectorType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int scalar or vector "
                    "type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(x_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to be a float scalar or vector";
        }

        if (_.GetDimension(x_type) != num_components) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to have the same number of components "
                    "as Result Type";
        }
        break;
      }

      case OpenCLLIB::Ldexp:
      case OpenCLLIB::Pown:
      case OpenCLLIB::Rootn: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        if (result_type != x_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected type of operand X to be equal to Result Type";
        }

        const uint32_t exp_type = _.GetOperandTypeId(inst, 5);
        if (!_.IsIntScalarOrVectorType(exp_type) ||
            _.GetBitWidth(exp_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected the exponent to be a 32-bit int scalar or vector";
        }

        if (_.GetDimension(exp_type) != num_components) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected the exponent to have the same number of "
                    "components as Result Type";
        }
        break;
      }

      case OpenCLLIB::Nan: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t nancode_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsIntScalarOrVectorType(nancode_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Nancode to be an int scalar or vector type";
        }

        if (_.GetDimension(nancode_type) != num_components) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Nancode to have the same number of components as "
                    "Result Type";
        }

        if (_.GetBitWidth(result_type) != _.GetBitWidth(nancode_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Nancode to have the same bit width as Result "
                    "Type";
        }
        break;
      }

      case OpenCLLIB::SAbs:
      case OpenCLLIB::SAbs_diff:
      case OpenCLLIB::SAdd_sat:
      case OpenCLLIB::UAdd_sat:
      case OpenCLLIB::SHadd:
      case OpenCLLIB::UHadd:
      case OpenCLLIB::SRhadd:
      case OpenCLLIB::URhadd:
      case OpenCLLIB::SClamp:
      case OpenCLLIB::UClamp:
      case OpenCLLIB::Clz:
      case OpenCLLIB::Ctz:
      case OpenCLLIB::SMad_hi:
      case OpenCLLIB::UMad_sat:
      case OpenCLLIB::SMad_sat:
      case OpenCLLIB::SMax:
      case OpenCLLIB::UMax:
      case OpenCLLIB::SMin:
      case OpenCLLIB::UMin:
      case OpenCLLIB::SMul_hi:
      case OpenCLLIB::Rotate:
      case OpenCLLIB::SSub_sat:
      case OpenCLLIB::USub_sat:
      case OpenCLLIB::Popcount:
      case OpenCLLIB::UAbs:
      case OpenCLLIB::UAbs_diff:
      case OpenCLLIB::UMul_hi:
      case OpenCLLIB::UMad_hi: {
        if (!_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case OpenCLLIB::U_Upsample:
      case OpenCLLIB::S_Upsample: {
        if (!_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int scalar or vector "
                    "type";
        }

        const uint32_t result_num_components = _.GetDimension(result_type);
        if (result_num_components > 4 && result_num_components != 8 &&
            result_num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t result_bit_width = _.GetBitWidth(result_type);
        if (result_bit_width != 16 && result_bit_width != 32 &&
            result_bit_width != 64) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected bit width of Result Type components to be 16, 32 "
                    "or 64";
        }

        const uint32_t hi_type = _.GetOperandTypeId(inst, 4);
        const uint32_t lo_type = _.GetOperandTypeId(inst, 5);

        if (hi_type != lo_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Hi and Lo operands to have the same type";
        }

        if (result_num_components != _.GetDimension(hi_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Hi and Lo operands to have the same number of "
                    "components as Result Type";
        }

        if (result_bit_width != 2 * _.GetBitWidth(hi_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected bit width of components of Hi and Lo operands to "
                    "be half of the bit width of components of Result Type";
        }
        break;
      }

      case OpenCLLIB::SMad24:
      case OpenCLLIB::UMad24:
      case OpenCLLIB::SMul24:
      case OpenCLLIB::UMul24: {
        if (!_.IsIntScalarOrVectorType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int scalar or vector "
                    "type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case OpenCLLIB::Cross: {
        if (!_.IsFloatVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components != 3 && num_components != 4) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 3 or 4 components";
        }

        const uint32_t x_type = _.GetOperandTypeId(inst, 4);
        const uint32_t y_type = _.GetOperandTypeId(inst, 5);

        if (x_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X type to be equal to Result Type";
        }

        if (y_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Y type to be equal to Result Type";
        }
        break;
      }

      case OpenCLLIB::Distance:
      case OpenCLLIB::Fast_distance: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t p0_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(p0_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 to be of float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(p0_type);
        if (num_components > 4) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 to have no more than 4 components";
        }

        if (result_type != _.GetComponentType(p0_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P0 component type to be equal to "
                 << "Result Type";
        }

        const uint32_t p1_type = _.GetOperandTypeId(inst, 5);
        if (p0_type != p1_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operands P0 and P1 to be of the same type";
        }
        break;
      }

      case OpenCLLIB::Length:
      case OpenCLLIB::Fast_length: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, 4);
        if (!_.IsFloatScalarOrVectorType(p_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a float scalar or vector";
        }

        const uint32_t num_components = _.GetDimension(p_type);
        if (num_components > 4) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to have no more than 4 components";
        }

        if (result_type != _.GetComponentType(p_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P component type to be equal to Result "
                    "Type";
        }
        break;
      }

      case OpenCLLIB::Normalize:
      case OpenCLLIB::Fast_normalize: {
        if (!_.IsFloatScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar or vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have no more than 4 components";
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, 4);
        if (p_type != result_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P type to be equal to Result Type";
        }
        break;
      }

      case OpenCLLIB::Bitselect: {
        if (!_.IsFloatScalarOrVectorType(result_type) &&
            !_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int or float scalar or "
                    "vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        for (uint32_t operand_index = 4; operand_index < num_operands;
             ++operand_index) {
          const uint32_t operand_type = _.GetOperandTypeId(inst, operand_index);
          if (result_type != operand_type) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected types of all operands to be equal to Result "
                      "Type";
          }
        }
        break;
      }

      case OpenCLLIB::Select: {
        if (!_.IsFloatScalarOrVectorType(result_type) &&
            !_.IsIntScalarOrVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int or float scalar or "
                    "vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t a_type = _.GetOperandTypeId(inst, 4);
        const uint32_t b_type = _.GetOperandTypeId(inst, 5);
        const uint32_t c_type = _.GetOperandTypeId(inst, 6);

        if (result_type != a_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand A type to be equal to Result Type";
        }

        if (result_type != b_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand B type to be equal to Result Type";
        }

        if (!_.IsIntScalarOrVectorType(c_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand C to be an int scalar or vector";
        }

        if (num_components != _.GetDimension(c_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand C to have the same number of components "
                    "as Result Type";
        }

        if (_.GetBitWidth(result_type) != _.GetBitWidth(c_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand C to have the same bit width as Result "
                    "Type";
        }
        break;
      }

      case OpenCLLIB::Vloadn: {
        if (!_.IsFloatVectorType(result_type) &&
            !_.IsIntVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int or float vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 2, 3, 4, 8 or 16 components";
        }

        const uint32_t offset_type = _.GetOperandTypeId(inst, 4);
        const uint32_t p_type = _.GetOperandTypeId(inst, 5);

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        uint32_t p_storage_class = 0;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != SpvStorageClassUniformConstant &&
            p_storage_class != SpvStorageClassGeneric &&
            p_storage_class != SpvStorageClassCrossWorkgroup &&
            p_storage_class != SpvStorageClassWorkgroup &&
            p_storage_class != SpvStorageClassFunction) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be UniformConstant, "
                    "Generic, CrossWorkgroup, Workgroup or Function";
        }

        if (_.GetComponentType(result_type) != p_data_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be equal to component "
                    "type of Result Type";
        }

        const uint32_t n_value = inst->word(7);
        if (num_components != n_value) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected literal N to be equal to the number of "
                    "components of Result Type";
        }
        break;
      }

      case OpenCLLIB::Vstoren: {
        if (_.GetIdOpcode(result_type) != SpvOpTypeVoid) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": expected Result Type to be void";
        }

        const uint32_t data_type = _.GetOperandTypeId(inst, 4);
        const uint32_t offset_type = _.GetOperandTypeId(inst, 5);
        const uint32_t p_type = _.GetOperandTypeId(inst, 6);

        if (!_.IsFloatVectorType(data_type) && !_.IsIntVectorType(data_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Data to be an int or float vector";
        }

        const uint32_t num_components = _.GetDimension(data_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Data to have 2, 3, 4, 8 or 16 components";
        }

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        uint32_t p_storage_class = 0;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != SpvStorageClassGeneric &&
            p_storage_class != SpvStorageClassCrossWorkgroup &&
            p_storage_class != SpvStorageClassWorkgroup &&
            p_storage_class != SpvStorageClassFunction) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be Generic, "
                    "CrossWorkgroup, Workgroup or Function";
        }

        if (_.GetComponentType(data_type) != p_data_type) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be equal to the type of "
                    "operand Data components";
        }
        break;
      }

      case OpenCLLIB::Vload_half: {
        if (!_.IsFloatScalarType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float scalar type";
        }

        const uint32_t offset_type = _.GetOperandTypeId(inst, 4);
        const uint32_t p_type = _.GetOperandTypeId(inst, 5);

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        uint32_t p_storage_class = 0;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != SpvStorageClassUniformConstant &&
            p_storage_class != SpvStorageClassGeneric &&
            p_storage_class != SpvStorageClassCrossWorkgroup &&
            p_storage_class != SpvStorageClassWorkgroup &&
            p_storage_class != SpvStorageClassFunction) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be UniformConstant, "
                    "Generic, CrossWorkgroup, Workgroup or Function";
        }

        if (!_.IsFloatScalarType(p_data_type) ||
            _.GetBitWidth(p_data_type) != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be 16-bit float scalar";
        }
        break;
      }

      case OpenCLLIB::Vload_halfn:
      case OpenCLLIB::Vloada_halfn: {
        if (!_.IsFloatVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a float vector type";
        }

        const uint32_t num_components = _.GetDimension(result_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 2, 3, 4, 8 or 16 components";
        }

        const uint32_t offset_type = _.GetOperandTypeId(inst, 4);
        const uint32_t p_type = _.GetOperandTypeId(inst, 5);

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        uint32_t p_storage_class = 0;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != SpvStorageClassUniformConstant &&
            p_storage_class != SpvStorageClassGeneric &&
            p_storage_class != SpvStorageClassCrossWorkgroup &&
            p_storage_class != SpvStorageClassWorkgroup &&
            p_storage_class != SpvStorageClassFunction) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be UniformConstant, "
                    "Generic, CrossWorkgroup, Workgroup or Function";
        }

        if (!_.IsFloatScalarType(p_data_type) ||
            _.GetBitWidth(p_data_type) != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be 16-bit float scalar";
        }

        const uint32_t n_value = inst->word(7);
        if (num_components != n_value) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected literal N to be equal to the number of "
                    "components of Result Type";
        }
        break;
      }

      case OpenCLLIB::Vstore_half:
      case OpenCLLIB::Vstore_half_r:
      case OpenCLLIB::Vstore_halfn:
      case OpenCLLIB::Vstore_halfn_r:
      case OpenCLLIB::Vstorea_halfn:
      case OpenCLLIB::Vstorea_halfn_r: {
        if (_.GetIdOpcode(result_type) != SpvOpTypeVoid) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": expected Result Type to be void";
        }

        const uint32_t data_type = _.GetOperandTypeId(inst, 4);
        const uint32_t offset_type = _.GetOperandTypeId(inst, 5);
        const uint32_t p_type = _.GetOperandTypeId(inst, 6);
        const uint32_t data_type_bit_width = _.GetBitWidth(data_type);

        if (ext_inst_key == OpenCLLIB::Vstore_half ||
            ext_inst_key == OpenCLLIB::Vstore_half_r) {
          if (!_.IsFloatScalarType(data_type) ||
              (data_type_bit_width != 32 && data_type_bit_width != 64)) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Data to be a 32 or 64-bit float scalar";
          }
        } else {
          if (!_.IsFloatVectorType(data_type) ||
              (data_type_bit_width != 32 && data_type_bit_width != 64)) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Data to be a 32 or 64-bit float vector";
          }

          const uint32_t num_components = _.GetDimension(data_type);
          if (num_components > 4 && num_components != 8 &&
              num_components != 16) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected Data to have 2, 3, 4, 8 or 16 components";
          }
        }

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(offset_type) ||
            _.GetBitWidth(offset_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Offset to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }

        uint32_t p_storage_class = 0;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P to be a pointer";
        }

        if (p_storage_class != SpvStorageClassGeneric &&
            p_storage_class != SpvStorageClassCrossWorkgroup &&
            p_storage_class != SpvStorageClassWorkgroup &&
            p_storage_class != SpvStorageClassFunction) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P storage class to be Generic, "
                    "CrossWorkgroup, Workgroup or Function";
        }

        if (!_.IsFloatScalarType(p_data_type) ||
            _.GetBitWidth(p_data_type) != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand P data type to be 16-bit float scalar";
        }

        // Rounding mode enum is checked by assembler.
        break;
      }

      case OpenCLLIB::Shuffle:
      case OpenCLLIB::Shuffle2: {
        if (!_.IsFloatVectorType(result_type) &&
            !_.IsIntVectorType(result_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be an int or float vector type";
        }

        const uint32_t result_num_components = _.GetDimension(result_type);
        if (result_num_components != 2 && result_num_components != 4 &&
            result_num_components != 8 && result_num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to have 2, 4, 8 or 16 components";
        }

        uint32_t operand_index = 4;
        const uint32_t x_type = _.GetOperandTypeId(inst, operand_index++);

        if (ext_inst_key == OpenCLLIB::Shuffle2) {
          const uint32_t y_type = _.GetOperandTypeId(inst, operand_index++);
          if (x_type != y_type) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected operands X and Y to be of the same type";
          }
        }

        const uint32_t shuffle_mask_type =
            _.GetOperandTypeId(inst, operand_index++);

        if (!_.IsFloatVectorType(x_type) && !_.IsIntVectorType(x_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to be an int or float vector";
        }

        const uint32_t x_num_components = _.GetDimension(x_type);
        if (x_num_components != 2 && x_num_components != 4 &&
            x_num_components != 8 && x_num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X to have 2, 4, 8 or 16 components";
        }

        const uint32_t result_component_type = _.GetComponentType(result_type);

        if (result_component_type != _.GetComponentType(x_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand X and Result Type to have equal "
                    "component types";
        }

        if (!_.IsIntVectorType(shuffle_mask_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Shuffle Mask to be an int vector";
        }

        if (result_num_components != _.GetDimension(shuffle_mask_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Shuffle Mask to have the same number of "
                    "components as Result Type";
        }

        if (_.GetBitWidth(result_component_type) !=
            _.GetBitWidth(shuffle_mask_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Shuffle Mask components to have the same "
                    "bit width as Result Type components";
        }
        break;
      }

      case OpenCLLIB::Printf: {
        if (!_.IsIntScalarType(result_type) ||
            _.GetBitWidth(result_type) != 32) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a 32-bit int type";
        }

        const uint32_t format_type = _.GetOperandTypeId(inst, 4);
        uint32_t format_storage_class = 0;
        uint32_t format_data_type = 0;
        if (!_.GetPointerTypeInfo(format_type, &format_data_type,
                                  &format_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Format to be a pointer";
        }

        if (format_storage_class != SpvStorageClassUniformConstant) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Format storage class to be UniformConstant";
        }

        if (!_.IsIntScalarType(format_data_type) ||
            _.GetBitWidth(format_data_type) != 8) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Format data type to be 8-bit int";
        }
        break;
      }

      case OpenCLLIB::Prefetch: {
        if (_.GetIdOpcode(result_type) != SpvOpTypeVoid) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": expected Result Type to be void";
        }

        const uint32_t p_type = _.GetOperandTypeId(inst, 4);
        const uint32_t num_elements_type = _.GetOperandTypeId(inst, 5);

        uint32_t p_storage_class = 0;
        uint32_t p_data_type = 0;
        if (!_.GetPointerTypeInfo(p_type, &p_data_type, &p_storage_class)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Ptr to be a pointer";
        }

        if (p_storage_class != SpvStorageClassCrossWorkgroup) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Ptr storage class to be CrossWorkgroup";
        }

        if (!_.IsFloatScalarOrVectorType(p_data_type) &&
            !_.IsIntScalarOrVectorType(p_data_type)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Ptr data type to be int or float scalar or "
                    "vector";
        }

        const uint32_t num_components = _.GetDimension(p_data_type);
        if (num_components > 4 && num_components != 8 && num_components != 16) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected Result Type to be a scalar or a vector with 2, "
                    "3, 4, 8 or 16 components";
        }

        const uint32_t size_t_bit_width = GetSizeTBitWidth(_);
        if (!size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name()
                 << " can only be used with physical addressing models";
        }

        if (!_.IsIntScalarType(num_elements_type) ||
            _.GetBitWidth(num_elements_type) != size_t_bit_width) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Num Elements to be of type size_t ("
                 << size_t_bit_width
                 << "-bit integer for the addressing model used in the module)";
        }
        break;
      }
    }
  } else if (ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100) {
    if (!_.IsVoidType(result_type)) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << ext_inst_name() << ": "
             << "expected result type must be a result id of "
             << "OpTypeVoid";
    }

    auto num_words = inst->words().size();

    const OpenCLDebugInfo100Instructions ext_inst_key =
        OpenCLDebugInfo100Instructions(ext_inst_index);
    switch (ext_inst_key) {
      case OpenCLDebugInfo100DebugInfoNone:
      case OpenCLDebugInfo100DebugNoScope:
      case OpenCLDebugInfo100DebugOperation:
        // The binary parser validates the opcode for DebugInfoNone,
        // DebugNoScope, DebugOperation, and the literal values don't need
        // further checks.
        break;
      case OpenCLDebugInfo100DebugCompilationUnit: {
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 7);
        break;
      }
      case OpenCLDebugInfo100DebugSource: {
        CHECK_OPERAND("File", SpvOpString, 5);
        if (num_words == 7) CHECK_OPERAND("Text", SpvOpString, 6);
        break;
      }
      case OpenCLDebugInfo100DebugTypeBasic: {
        CHECK_OPERAND("Name", SpvOpString, 5);
        CHECK_OPERAND("Size", SpvOpConstant, 6);
        // "Encoding" param is already validated by the binary parsing stage.
        break;
      }
      case OpenCLDebugInfo100DebugTypePointer:
      case OpenCLDebugInfo100DebugTypeQualifier: {
        auto validate_base_type =
            ValidateOperandBaseType(_, inst, 5, ext_inst_name);
        if (validate_base_type != SPV_SUCCESS) return validate_base_type;
        break;
      }
      case OpenCLDebugInfo100DebugTypeVector: {
        auto validate_base_type =
            ValidateOperandBaseType(_, inst, 5, ext_inst_name);
        if (validate_base_type != SPV_SUCCESS) return validate_base_type;

        uint32_t component_count = inst->word(6);
        if (!component_count || component_count > 4) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": Component Count must be positive "
                 << "integer less than or equal to 4";
        }
        break;
      }
      case OpenCLDebugInfo100DebugTypeArray: {
        auto validate_base_type =
            ValidateOperandDebugType(_, "Base Type", inst, 5, ext_inst_name);
        if (validate_base_type != SPV_SUCCESS) return validate_base_type;
        for (uint32_t i = 6; i < num_words; ++i) {
          CHECK_OPERAND("Component Count", SpvOpConstant, i);
          auto* component_count = _.FindDef(inst->word(i));
          if (!_.IsIntScalarType(component_count->type_id()) ||
              !component_count->word(3)) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": Component Count must be positive "
                   << "integer";
          }
        }
        break;
      }
      case OpenCLDebugInfo100DebugTypedef: {
        CHECK_OPERAND("Name", SpvOpString, 5);
        auto validate_base_type =
            ValidateOperandBaseType(_, inst, 6, ext_inst_name);
        if (validate_base_type != SPV_SUCCESS) return validate_base_type;
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 7);
        auto validate_parent =
            ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
        if (validate_parent != SPV_SUCCESS) return validate_parent;
        break;
      }
      case OpenCLDebugInfo100DebugTypeFunction: {
        auto* return_type = _.FindDef(inst->word(6));
        if (return_type->opcode() != SpvOpTypeVoid) {
          auto validate_return = ValidateOperandDebugType(
              _, "Return Type", inst, 6, ext_inst_name);
          if (validate_return != SPV_SUCCESS) return validate_return;
        }
        for (uint32_t word_index = 7; word_index < num_words; ++word_index) {
          auto validate_param = ValidateOperandDebugType(
              _, "Parameter Types", inst, word_index, ext_inst_name);
          if (validate_param != SPV_SUCCESS) return validate_param;
        }
        break;
      }
      case OpenCLDebugInfo100DebugTypeEnum: {
        CHECK_OPERAND("Name", SpvOpString, 5);
        if (!DoesDebugInfoOperandMatchExpectation(
                _,
                [](OpenCLDebugInfo100Instructions dbg_inst) {
                  return dbg_inst == OpenCLDebugInfo100DebugInfoNone;
                },
                inst, 6)) {
          auto validate_underlying_type = ValidateOperandDebugType(
              _, "Underlying Types", inst, 6, ext_inst_name);
          if (validate_underlying_type != SPV_SUCCESS)
            return validate_underlying_type;
        }
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 7);
        auto validate_parent =
            ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
        if (validate_parent != SPV_SUCCESS) return validate_parent;
        CHECK_OPERAND("Size", SpvOpConstant, 11);
        auto* size = _.FindDef(inst->word(11));
        if (!_.IsIntScalarType(size->type_id()) || !size->word(3)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": expected operand Size is a "
                 << "positive integer";
        }
        for (uint32_t word_index = 13; word_index + 1 < num_words;
             word_index += 2) {
          CHECK_OPERAND("Value", SpvOpConstant, word_index);
          CHECK_OPERAND("Name", SpvOpString, word_index + 1);
        }
        break;
      }
      case OpenCLDebugInfo100DebugTypeComposite: {
        CHECK_OPERAND("Name", SpvOpString, 5);
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 7);
        auto validate_parent =
            ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
        if (validate_parent != SPV_SUCCESS) return validate_parent;
        CHECK_OPERAND("Linkage Name", SpvOpString, 11);
        CHECK_OPERAND("Size", SpvOpConstant, 12);
        for (uint32_t word_index = 14; word_index < num_words; ++word_index) {
          if (!DoesDebugInfoOperandMatchExpectation(
                  _,
                  [](OpenCLDebugInfo100Instructions dbg_inst) {
                    return dbg_inst == OpenCLDebugInfo100DebugTypeMember ||
                           dbg_inst == OpenCLDebugInfo100DebugFunction ||
                           dbg_inst == OpenCLDebugInfo100DebugTypeInheritance;
                  },
                  inst, word_index)) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << ext_inst_name() << ": "
                   << "expected operand Members "
                   << "must be DebugTypeMember, DebugFunction, or "
                      "DebugTypeInheritance";
          }
        }
        break;
      }
      case OpenCLDebugInfo100DebugTypeMember: {
        CHECK_OPERAND("Name", SpvOpString, 5);
        auto validate_type =
            ValidateOperandDebugType(_, "Type", inst, 6, ext_inst_name);
        if (validate_type != SPV_SUCCESS) return validate_type;
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 7);
        CHECK_DEBUG_OPERAND("Parent", OpenCLDebugInfo100DebugTypeComposite, 10);
        CHECK_OPERAND("Offset", SpvOpConstant, 11);
        CHECK_OPERAND("Size", SpvOpConstant, 12);
        if (num_words == 15) CHECK_OPERAND("Value", SpvOpConstant, 14);
        break;
      }
      case OpenCLDebugInfo100DebugTypeInheritance: {
        CHECK_DEBUG_OPERAND("Child", OpenCLDebugInfo100DebugTypeComposite, 5);
        auto* debug_inst = _.FindDef(inst->word(5));
        auto composite_type =
            OpenCLDebugInfo100DebugCompositeType(debug_inst->word(6));
        if (composite_type != OpenCLDebugInfo100Class &&
            composite_type != OpenCLDebugInfo100Structure) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Child must be class or struct debug type";
        }
        CHECK_DEBUG_OPERAND("Parent", OpenCLDebugInfo100DebugTypeComposite, 6);
        debug_inst = _.FindDef(inst->word(6));
        composite_type =
            OpenCLDebugInfo100DebugCompositeType(debug_inst->word(6));
        if (composite_type != OpenCLDebugInfo100Class &&
            composite_type != OpenCLDebugInfo100Structure) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Parent must be class or struct debug "
                    "type";
        }
        CHECK_OPERAND("Offset", SpvOpConstant, 7);
        CHECK_OPERAND("Size", SpvOpConstant, 8);
        break;
      }
      case OpenCLDebugInfo100DebugFunction: {
        CHECK_OPERAND("Name", SpvOpString, 5);
        auto validate_type =
            ValidateOperandDebugType(_, "Type", inst, 6, ext_inst_name);
        if (validate_type != SPV_SUCCESS) return validate_type;
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 7);
        auto validate_parent =
            ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
        if (validate_parent != SPV_SUCCESS) return validate_parent;
        CHECK_OPERAND("Linkage Name", SpvOpString, 11);
        // TODO: The current OpenCL.100.DebugInfo spec says "Function
        // is an OpFunction which is described by this instruction.".
        // However, the function definition can be opted-out e.g.,
        // inlining. We assume that Function operand can be a
        // DebugInfoNone, but we must discuss it and update the spec.
        if (!DoesDebugInfoOperandMatchExpectation(
                _,
                [](OpenCLDebugInfo100Instructions dbg_inst) {
                  return dbg_inst == OpenCLDebugInfo100DebugInfoNone;
                },
                inst, 14)) {
          CHECK_OPERAND("Function", SpvOpFunction, 14);
        }
        if (num_words == 16) {
          CHECK_DEBUG_OPERAND("Declaration",
                              OpenCLDebugInfo100DebugFunctionDeclaration, 15);
        }
        break;
      }
      case OpenCLDebugInfo100DebugFunctionDeclaration: {
        CHECK_OPERAND("Name", SpvOpString, 5);
        auto validate_type =
            ValidateOperandDebugType(_, "Type", inst, 6, ext_inst_name);
        if (validate_type != SPV_SUCCESS) return validate_type;
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 7);
        auto validate_parent =
            ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
        if (validate_parent != SPV_SUCCESS) return validate_parent;
        CHECK_OPERAND("Linkage Name", SpvOpString, 11);
        break;
      }
      case OpenCLDebugInfo100DebugLexicalBlock: {
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 5);
        auto validate_parent =
            ValidateOperandLexicalScope(_, "Parent", inst, 8, ext_inst_name);
        if (validate_parent != SPV_SUCCESS) return validate_parent;
        if (num_words == 10) CHECK_OPERAND("Name", SpvOpString, 9);
        break;
      }
      case OpenCLDebugInfo100DebugScope: {
        // TODO(https://gitlab.khronos.org/spirv/SPIR-V/issues/533): We are
        // still in spec dislwssion about what must be "Scope" operand of
        // DebugScope. Update this code if the conclusion is different.
        auto validate_scope =
            ValidateOperandLexicalScope(_, "Scope", inst, 5, ext_inst_name);
        if (validate_scope != SPV_SUCCESS) return validate_scope;
        if (num_words == 7) {
          CHECK_DEBUG_OPERAND("Inlined At", OpenCLDebugInfo100DebugInlinedAt,
                              6);
        }
        break;
      }
      case OpenCLDebugInfo100DebugLocalVariable: {
        CHECK_OPERAND("Name", SpvOpString, 5);
        auto validate_type =
            ValidateOperandDebugType(_, "Type", inst, 6, ext_inst_name);
        if (validate_type != SPV_SUCCESS) return validate_type;
        CHECK_DEBUG_OPERAND("Source", OpenCLDebugInfo100DebugSource, 7);
        auto validate_parent =
            ValidateOperandLexicalScope(_, "Parent", inst, 10, ext_inst_name);
        if (validate_parent != SPV_SUCCESS) return validate_parent;
        break;
      }
      case OpenCLDebugInfo100DebugDeclare: {
        CHECK_DEBUG_OPERAND("Local Variable",
                            OpenCLDebugInfo100DebugLocalVariable, 5);

        // TODO: We must discuss DebugDeclare.Variable of OpenCL.100.DebugInfo.
        // Lwrrently, it says "Variable must be an id of OpVariable instruction
        // which defines the local variable.", but we want to allow
        // OpFunctionParameter as well.
        auto* operand = _.FindDef(inst->word(6));
        if (operand->opcode() != SpvOpVariable &&
            operand->opcode() != SpvOpFunctionParameter) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << ext_inst_name() << ": "
                 << "expected operand Variable must be a result id of "
                    "OpVariable or OpFunctionParameter";
        }

        CHECK_DEBUG_OPERAND("Expression", OpenCLDebugInfo100DebugExpression, 7);
        break;
      }
      case OpenCLDebugInfo100DebugExpression: {
        for (uint32_t word_index = 5; word_index < num_words; ++word_index) {
          CHECK_DEBUG_OPERAND("Operation", OpenCLDebugInfo100DebugOperation,
                              word_index);
        }
        break;
      }

      // TODO: Add validation rules for remaining cases as well.
      case OpenCLDebugInfo100DebugTypePtrToMember:
      case OpenCLDebugInfo100DebugTypeTemplate:
      case OpenCLDebugInfo100DebugTypeTemplateParameter:
      case OpenCLDebugInfo100DebugTypeTemplateTemplateParameter:
      case OpenCLDebugInfo100DebugTypeTemplateParameterPack:
      case OpenCLDebugInfo100DebugGlobalVariable:
      case OpenCLDebugInfo100DebugLexicalBlockDiscriminator:
      case OpenCLDebugInfo100DebugInlinedAt:
      case OpenCLDebugInfo100DebugInlinedVariable:
      case OpenCLDebugInfo100DebugValue:
      case OpenCLDebugInfo100DebugMacroDef:
      case OpenCLDebugInfo100DebugMacroUndef:
      case OpenCLDebugInfo100DebugImportedEntity:
        break;
      case OpenCLDebugInfo100InstructionsMax:
        assert(0);
        break;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ExtensionPass(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  if (opcode == SpvOpExtension) return ValidateExtension(_, inst);
  if (opcode == SpvOpExtInstImport) return ValidateExtInstImport(_, inst);
  if (opcode == SpvOpExtInst) return ValidateExtInst(_, inst);

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
