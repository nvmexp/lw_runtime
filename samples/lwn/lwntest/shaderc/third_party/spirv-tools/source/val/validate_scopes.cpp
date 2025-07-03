// Copyright (c) 2018 Google LLC.
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

#include "source/val/validate_scopes.h"

#include "source/diagnostic.h"
#include "source/spirv_target_elw.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

bool IsValidScope(uint32_t scope) {
  // Deliberately avoid a default case so we have to update the list when the
  // scopes list changes.
  switch (static_cast<SpvScope>(scope)) {
    case SpvScopeCrossDevice:
    case SpvScopeDevice:
    case SpvScopeWorkgroup:
    case SpvScopeSubgroup:
    case SpvScopeIlwocation:
    case SpvScopeQueueFamilyKHR:
    case SpvScopeShaderCallKHR:
      return true;
    case SpvScopeMax:
      break;
  }
  return false;
}

spv_result_t ValidateScope(ValidationState_t& _, const Instruction* inst,
                           uint32_t scope) {
  SpvOp opcode = inst->opcode();
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(scope);

  if (!is_int32) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << spvOpcodeString(opcode) << ": expected scope to be a 32-bit int";
  }

  if (!is_const_int32) {
    if (_.HasCapability(SpvCapabilityShader) &&
        !_.HasCapability(SpvCapabilityCooperativeMatrixLW)) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "Scope ids must be OpConstant when Shader capability is "
             << "present";
    }
    if (_.HasCapability(SpvCapabilityShader) &&
        _.HasCapability(SpvCapabilityCooperativeMatrixLW) &&
        !spvOpcodeIsConstant(_.GetIdOpcode(scope))) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "Scope ids must be constant or specialization constant when "
             << "CooperativeMatrixLW capability is present";
    }
  }

  if (is_const_int32 && !IsValidScope(value)) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << "Invalid scope value:\n " << _.Disassemble(*_.FindDef(scope));
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateExelwtionScope(ValidationState_t& _,
                                    const Instruction* inst, uint32_t scope) {
  SpvOp opcode = inst->opcode();
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(scope);

  if (auto error = ValidateScope(_, inst, scope)) {
    return error;
  }

  if (!is_const_int32) {
    return SPV_SUCCESS;
  }

  // Vulkan specific rules
  if (spvIsVulkanElw(_.context()->target_elw)) {
    // Vulkan 1.1 specific rules
    if (_.context()->target_elw != SPV_ELW_VULKAN_1_0) {
      // Scope for Non Uniform Group Operations must be limited to Subgroup
      if (spvOpcodeIsNonUniformGroupOperation(opcode) &&
          value != SpvScopeSubgroup) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": in Vulkan environment Exelwtion scope is limited to "
               << "Subgroup";
      }
    }

    // If OpControlBarrier is used in fragment, vertex, tessellation evaluation,
    // or geometry stages, the exelwtion Scope must be Subgroup.
    if (opcode == SpvOpControlBarrier && value != SpvScopeSubgroup) {
      _.function(inst->function()->id())
          ->RegisterExelwtionModelLimitation([](SpvExelwtionModel model,
                                                std::string* message) {
            if (model == SpvExelwtionModelFragment ||
                model == SpvExelwtionModelVertex ||
                model == SpvExelwtionModelGeometry ||
                model == SpvExelwtionModelTessellationEvaluation) {
              if (message) {
                *message =
                    "in Vulkan evironment, OpControlBarrier exelwtion scope "
                    "must be Subgroup for Fragment, Vertex, Geometry and "
                    "TessellationEvaluation exelwtion models";
              }
              return false;
            }
            return true;
          });
    }

    // Vulkan generic rules
    // Scope for exelwtion must be limited to Workgroup or Subgroup
    if (value != SpvScopeWorkgroup && value != SpvScopeSubgroup) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": in Vulkan environment Exelwtion Scope is limited to "
             << "Workgroup and Subgroup";
    }
  }

  // WebGPU Specific rules
  if (spvIsWebGPUElw(_.context()->target_elw)) {
    if (value != SpvScopeWorkgroup) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": in WebGPU environment Exelwtion Scope is limited to "
             << "Workgroup";
    } else {
      _.function(inst->function()->id())
          ->RegisterExelwtionModelLimitation(
              [](SpvExelwtionModel model, std::string* message) {
                if (model != SpvExelwtionModelGLCompute) {
                  if (message) {
                    *message =
                        ": in WebGPU environment, Workgroup Exelwtion Scope is "
                        "limited to GLCompute exelwtion model";
                  }
                  return false;
                }
                return true;
              });
    }
  }

  // TODO(atgoo@github.com) Add checks for OpenCL and OpenGL elwironments.

  // General SPIRV rules
  // Scope for exelwtion must be limited to Workgroup or Subgroup for
  // non-uniform operations
  if (spvOpcodeIsNonUniformGroupOperation(opcode) &&
      value != SpvScopeSubgroup && value != SpvScopeWorkgroup) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << spvOpcodeString(opcode)
           << ": Exelwtion scope is limited to Subgroup or Workgroup";
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateMemoryScope(ValidationState_t& _, const Instruction* inst,
                                 uint32_t scope) {
  const SpvOp opcode = inst->opcode();
  bool is_int32 = false, is_const_int32 = false;
  uint32_t value = 0;
  std::tie(is_int32, is_const_int32, value) = _.EvalInt32IfConst(scope);

  if (auto error = ValidateScope(_, inst, scope)) {
    return error;
  }

  if (!is_const_int32) {
    return SPV_SUCCESS;
  }

  if (value == SpvScopeQueueFamilyKHR) {
    if (_.HasCapability(SpvCapabilityVulkanMemoryModelKHR)) {
      return SPV_SUCCESS;
    } else {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": Memory Scope QueueFamilyKHR requires capability "
             << "VulkanMemoryModelKHR";
    }
  }

  if (value == SpvScopeDevice &&
      _.HasCapability(SpvCapabilityVulkanMemoryModelKHR) &&
      !_.HasCapability(SpvCapabilityVulkanMemoryModelDeviceScopeKHR)) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << "Use of device scope with VulkanKHR memory model requires the "
           << "VulkanMemoryModelDeviceScopeKHR capability";
  }

  // Vulkan Specific rules
  if (spvIsVulkanElw(_.context()->target_elw)) {
    if (value == SpvScopeCrossDevice) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": in Vulkan environment, Memory Scope cannot be CrossDevice";
    }
    // Vulkan 1.0 specifc rules
    if (_.context()->target_elw == SPV_ELW_VULKAN_1_0 &&
        value != SpvScopeDevice && value != SpvScopeWorkgroup &&
        value != SpvScopeIlwocation) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": in Vulkan 1.0 environment Memory Scope is limited to "
             << "Device, Workgroup and Invocation";
    }
    // Vulkan 1.1 specifc rules
    if ((_.context()->target_elw == SPV_ELW_VULKAN_1_1 ||
         _.context()->target_elw == SPV_ELW_VULKAN_1_2) &&
        value != SpvScopeDevice && value != SpvScopeWorkgroup &&
        value != SpvScopeSubgroup && value != SpvScopeIlwocation &&
        value != SpvScopeShaderCallKHR) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << spvOpcodeString(opcode)
             << ": in Vulkan 1.1 and 1.2 environment Memory Scope is limited "
             << "to Device, Workgroup, Invocation, and ShaderCall";
    }

    if (value == SpvScopeShaderCallKHR) {
      _.function(inst->function()->id())
          ->RegisterExelwtionModelLimitation(
              [](SpvExelwtionModel model, std::string* message) {
                if (model != SpvExelwtionModelRayGenerationKHR &&
                    model != SpvExelwtionModelIntersectionKHR &&
                    model != SpvExelwtionModelAnyHitKHR &&
                    model != SpvExelwtionModelClosestHitKHR &&
                    model != SpvExelwtionModelMissKHR &&
                    model != SpvExelwtionModelCallableKHR) {
                  if (message) {
                    *message =
                        "ShaderCallKHR Memory Scope requires a ray tracing "
                        "exelwtion model";
                  }
                  return false;
                }
                return true;
              });
    }
  }

  // WebGPU specific rules
  if (spvIsWebGPUElw(_.context()->target_elw)) {
    switch (inst->opcode()) {
      case SpvOpControlBarrier:
        if (value != SpvScopeWorkgroup) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": in WebGPU environment Memory Scope is limited to "
                 << "Workgroup for OpControlBarrier";
        }
        break;
      case SpvOpMemoryBarrier:
        if (value != SpvScopeWorkgroup) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": in WebGPU environment Memory Scope is limited to "
                 << "Workgroup for OpMemoryBarrier";
        }
        break;
      default:
        if (spvOpcodeIsAtomicOp(inst->opcode())) {
          if (value != SpvScopeQueueFamilyKHR) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << spvOpcodeString(opcode)
                   << ": in WebGPU environment Memory Scope is limited to "
                   << "QueueFamilyKHR for OpAtomic* operations";
          }
        }

        if (value != SpvScopeWorkgroup && value != SpvScopeIlwocation &&
            value != SpvScopeQueueFamilyKHR) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << spvOpcodeString(opcode)
                 << ": in WebGPU environment Memory Scope is limited to "
                 << "Workgroup, Invocation, and QueueFamilyKHR";
        }
        break;
    }

    if (value == SpvScopeWorkgroup) {
      _.function(inst->function()->id())
          ->RegisterExelwtionModelLimitation(
              [](SpvExelwtionModel model, std::string* message) {
                if (model != SpvExelwtionModelGLCompute) {
                  if (message) {
                    *message =
                        ": in WebGPU environment, Workgroup Memory Scope is "
                        "limited to GLCompute exelwtion model";
                  }
                  return false;
                }
                return true;
              });
    }
  }

  // TODO(atgoo@github.com) Add checks for OpenCL and OpenGL elwironments.

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
