// Copyright (c) 2018 Google LLC.
// Copyright (c) 2019 LWPU Corporation
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

#include "source/val/validate.h"

#include "source/opcode.h"
#include "source/spirv_target_elw.h"
#include "source/val/instruction.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateUndef(ValidationState_t& _, const Instruction* inst) {
  if (_.HasCapability(SpvCapabilityShader) &&
      _.ContainsLimitedUseIntOrFloatType(inst->type_id()) &&
      !_.IsPointerType(inst->type_id())) {
    return _.diag(SPV_ERROR_ILWALID_ID, inst)
           << "Cannot create undefined values with 8- or 16-bit types";
  }

  if (spvIsWebGPUElw(_.context()->target_elw)) {
    return _.diag(SPV_ERROR_ILWALID_BINARY, inst) << "OpUndef is disallowed";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateShaderClock(ValidationState_t& _,
                                 const Instruction* inst) {
  const uint32_t scope = inst->GetOperandAs<uint32_t>(2);
  if (auto error = ValidateScope(_, inst, scope)) {
    return error;
  }

  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(scope);
  if (is_const_int32 && value != SpvScopeSubgroup && value != SpvScopeDevice) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << "Scope must be Subgroup or Device";
  }

  // Result Type must be a 64 - bit unsigned integer type or
  // a vector of two - components of 32 -
  // bit unsigned integer type
  const uint32_t result_type = inst->type_id();
  if (!(_.IsUnsignedIntScalarType(result_type) &&
        _.GetBitWidth(result_type) == 64) &&
      !(_.IsUnsignedIntVectorType(result_type) &&
        _.GetDimension(result_type) == 2 && _.GetBitWidth(result_type) == 32)) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst) << "Expected Value to be a "
                                                   "vector of two components"
                                                   " of unsigned integer"
                                                   " or 64bit unsigned integer";
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t MiscPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpUndef:
      if (auto error = ValidateUndef(_, inst)) return error;
      break;
    default:
      break;
  }
  switch (inst->opcode()) {
    case SpvOpBeginIlwocationInterlockEXT:
    case SpvOpEndIlwocationInterlockEXT:
      _.function(inst->function()->id())
          ->RegisterExelwtionModelLimitation(
              SpvExelwtionModelFragment,
              "OpBeginIlwocationInterlockEXT/OpEndIlwocationInterlockEXT "
              "require Fragment exelwtion model");

      _.function(inst->function()->id())
          ->RegisterLimitation([](const ValidationState_t& state,
                                  const Function* entry_point,
                                  std::string* message) {
            const auto* exelwtion_modes =
                state.GetExelwtionModes(entry_point->id());

            auto find_interlock = [](const SpvExelwtionMode& mode) {
              switch (mode) {
                case SpvExelwtionModePixelInterlockOrderedEXT:
                case SpvExelwtionModePixelInterlockUnorderedEXT:
                case SpvExelwtionModeSampleInterlockOrderedEXT:
                case SpvExelwtionModeSampleInterlockUnorderedEXT:
                case SpvExelwtionModeShadingRateInterlockOrderedEXT:
                case SpvExelwtionModeShadingRateInterlockUnorderedEXT:
                  return true;
                default:
                  return false;
              }
            };

            bool found = false;
            if (exelwtion_modes) {
              auto i = std::find_if(exelwtion_modes->begin(),
                                    exelwtion_modes->end(), find_interlock);
              found = (i != exelwtion_modes->end());
            }

            if (!found) {
              *message =
                  "OpBeginIlwocationInterlockEXT/OpEndIlwocationInterlockEXT "
                  "require a fragment shader interlock exelwtion mode.";
              return false;
            }
            return true;
          });
      break;
    case SpvOpDemoteToHelperIlwocationEXT:
      _.function(inst->function()->id())
          ->RegisterExelwtionModelLimitation(
              SpvExelwtionModelFragment,
              "OpDemoteToHelperIlwocationEXT requires Fragment exelwtion "
              "model");
      break;
    case SpvOpIsHelperIlwocationEXT: {
      const uint32_t result_type = inst->type_id();
      _.function(inst->function()->id())
          ->RegisterExelwtionModelLimitation(
              SpvExelwtionModelFragment,
              "OpIsHelperIlwocationEXT requires Fragment exelwtion model");
      if (!_.IsBoolScalarType(result_type))
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << "Expected bool scalar type as Result Type: "
               << spvOpcodeString(inst->opcode());
      break;
    }
    case SpvOpReadClockKHR:
      if (auto error = ValidateShaderClock(_, inst)) {
        return error;
      }
      break;
    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
