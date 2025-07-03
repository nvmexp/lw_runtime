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
//
#include "source/val/validate.h"

#include <algorithm>

#include "source/opcode.h"
#include "source/spirv_target_elw.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

spv_result_t ValidateEntryPoint(ValidationState_t& _, const Instruction* inst) {
  const auto entry_point_id = inst->GetOperandAs<uint32_t>(1);
  auto entry_point = _.FindDef(entry_point_id);
  if (!entry_point || SpvOpFunction != entry_point->opcode()) {
    return _.diag(SPV_ERROR_ILWALID_ID, inst)
           << "OpEntryPoint Entry Point <id> '" << _.getIdName(entry_point_id)
           << "' is not a function.";
  }

  // Only check the shader exelwtion models
  const SpvExelwtionModel exelwtion_model =
      inst->GetOperandAs<SpvExelwtionModel>(0);
  if (exelwtion_model != SpvExelwtionModelKernel) {
    const auto entry_point_type_id = entry_point->GetOperandAs<uint32_t>(3);
    const auto entry_point_type = _.FindDef(entry_point_type_id);
    if (!entry_point_type || 3 != entry_point_type->words().size()) {
      return _.diag(SPV_ERROR_ILWALID_ID, inst)
             << "OpEntryPoint Entry Point <id> '" << _.getIdName(entry_point_id)
             << "'s function parameter count is not zero.";
    }
  }

  auto return_type = _.FindDef(entry_point->type_id());
  if (!return_type || SpvOpTypeVoid != return_type->opcode()) {
    return _.diag(SPV_ERROR_ILWALID_ID, inst)
           << "OpEntryPoint Entry Point <id> '" << _.getIdName(entry_point_id)
           << "'s function return type is not void.";
  }

  const auto* exelwtion_modes = _.GetExelwtionModes(entry_point_id);
  if (_.HasCapability(SpvCapabilityShader)) {
    switch (exelwtion_model) {
      case SpvExelwtionModelFragment:
        if (exelwtion_modes &&
            exelwtion_modes->count(SpvExelwtionModeOriginUpperLeft) &&
            exelwtion_modes->count(SpvExelwtionModeOriginLowerLeft)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Fragment exelwtion model entry points can only specify "
                    "one of OriginUpperLeft or OriginLowerLeft exelwtion "
                    "modes.";
        }
        if (!exelwtion_modes ||
            (!exelwtion_modes->count(SpvExelwtionModeOriginUpperLeft) &&
             !exelwtion_modes->count(SpvExelwtionModeOriginLowerLeft))) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Fragment exelwtion model entry points require either an "
                    "OriginUpperLeft or OriginLowerLeft exelwtion mode.";
        }
        if (exelwtion_modes &&
            1 < std::count_if(exelwtion_modes->begin(), exelwtion_modes->end(),
                              [](const SpvExelwtionMode& mode) {
                                switch (mode) {
                                  case SpvExelwtionModeDepthGreater:
                                  case SpvExelwtionModeDepthLess:
                                  case SpvExelwtionModeDepthUnchanged:
                                    return true;
                                  default:
                                    return false;
                                }
                              })) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Fragment exelwtion model entry points can specify at most "
                    "one of DepthGreater, DepthLess or DepthUnchanged "
                    "exelwtion modes.";
        }
        if (exelwtion_modes &&
            1 < std::count_if(
                    exelwtion_modes->begin(), exelwtion_modes->end(),
                    [](const SpvExelwtionMode& mode) {
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
                    })) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Fragment exelwtion model entry points can specify at most "
                    "one fragment shader interlock exelwtion mode.";
        }
        break;
      case SpvExelwtionModelTessellationControl:
      case SpvExelwtionModelTessellationEvaluation:
        if (exelwtion_modes &&
            1 < std::count_if(exelwtion_modes->begin(), exelwtion_modes->end(),
                              [](const SpvExelwtionMode& mode) {
                                switch (mode) {
                                  case SpvExelwtionModeSpacingEqual:
                                  case SpvExelwtionModeSpacingFractionalEven:
                                  case SpvExelwtionModeSpacingFractionalOdd:
                                    return true;
                                  default:
                                    return false;
                                }
                              })) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Tessellation exelwtion model entry points can specify at "
                    "most one of SpacingEqual, SpacingFractionalOdd or "
                    "SpacingFractionalEven exelwtion modes.";
        }
        if (exelwtion_modes &&
            1 < std::count_if(exelwtion_modes->begin(), exelwtion_modes->end(),
                              [](const SpvExelwtionMode& mode) {
                                switch (mode) {
                                  case SpvExelwtionModeTriangles:
                                  case SpvExelwtionModeQuads:
                                  case SpvExelwtionModeIsolines:
                                    return true;
                                  default:
                                    return false;
                                }
                              })) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Tessellation exelwtion model entry points can specify at "
                    "most one of Triangles, Quads or Isolines exelwtion modes.";
        }
        if (exelwtion_modes &&
            1 < std::count_if(exelwtion_modes->begin(), exelwtion_modes->end(),
                              [](const SpvExelwtionMode& mode) {
                                switch (mode) {
                                  case SpvExelwtionModeVertexOrderCw:
                                  case SpvExelwtionModeVertexOrderCcw:
                                    return true;
                                  default:
                                    return false;
                                }
                              })) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Tessellation exelwtion model entry points can specify at "
                    "most one of VertexOrderCw or VertexOrderCcw exelwtion "
                    "modes.";
        }
        break;
      case SpvExelwtionModelGeometry:
        if (!exelwtion_modes ||
            1 != std::count_if(exelwtion_modes->begin(), exelwtion_modes->end(),
                               [](const SpvExelwtionMode& mode) {
                                 switch (mode) {
                                   case SpvExelwtionModeInputPoints:
                                   case SpvExelwtionModeInputLines:
                                   case SpvExelwtionModeInputLinesAdjacency:
                                   case SpvExelwtionModeTriangles:
                                   case SpvExelwtionModeInputTrianglesAdjacency:
                                     return true;
                                   default:
                                     return false;
                                 }
                               })) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Geometry exelwtion model entry points must specify "
                    "exactly one of InputPoints, InputLines, "
                    "InputLinesAdjacency, Triangles or InputTrianglesAdjacency "
                    "exelwtion modes.";
        }
        if (!exelwtion_modes ||
            1 != std::count_if(exelwtion_modes->begin(), exelwtion_modes->end(),
                               [](const SpvExelwtionMode& mode) {
                                 switch (mode) {
                                   case SpvExelwtionModeOutputPoints:
                                   case SpvExelwtionModeOutputLineStrip:
                                   case SpvExelwtionModeOutputTriangleStrip:
                                     return true;
                                   default:
                                     return false;
                                 }
                               })) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Geometry exelwtion model entry points must specify "
                    "exactly one of OutputPoints, OutputLineStrip or "
                    "OutputTriangleStrip exelwtion modes.";
        }
        break;
      default:
        break;
    }
  }

  if (spvIsVulkanElw(_.context()->target_elw)) {
    switch (exelwtion_model) {
      case SpvExelwtionModelGLCompute:
        if (!exelwtion_modes ||
            !exelwtion_modes->count(SpvExelwtionModeLocalSize)) {
          bool ok = false;
          for (auto& i : _.ordered_instructions()) {
            if (i.opcode() == SpvOpDecorate) {
              if (i.operands().size() > 2) {
                if (i.GetOperandAs<SpvDecoration>(1) == SpvDecorationBuiltIn &&
                    i.GetOperandAs<SpvBuiltIn>(2) == SpvBuiltInWorkgroupSize) {
                  ok = true;
                  break;
                }
              }
            }
          }
          if (!ok) {
            return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                   << "In the Vulkan environment, GLCompute exelwtion model "
                      "entry points require either the LocalSize exelwtion "
                      "mode or an object decorated with WorkgroupSize must be "
                      "specified.";
          }
        }
        break;
      default:
        break;
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateExelwtionMode(ValidationState_t& _,
                                   const Instruction* inst) {
  const auto entry_point_id = inst->GetOperandAs<uint32_t>(0);
  const auto found = std::find(_.entry_points().cbegin(),
                               _.entry_points().cend(), entry_point_id);
  if (found == _.entry_points().cend()) {
    return _.diag(SPV_ERROR_ILWALID_ID, inst)
           << "OpExelwtionMode Entry Point <id> '"
           << _.getIdName(entry_point_id)
           << "' is not the Entry Point "
              "operand of an OpEntryPoint.";
  }

  const auto mode = inst->GetOperandAs<SpvExelwtionMode>(1);
  if (inst->opcode() == SpvOpExelwtionModeId) {
    size_t operand_count = inst->operands().size();
    for (size_t i = 2; i < operand_count; ++i) {
      const auto operand_id = inst->GetOperandAs<uint32_t>(2);
      const auto* operand_inst = _.FindDef(operand_id);
      if (mode == SpvExelwtionModeSubgroupsPerWorkgroupId ||
          mode == SpvExelwtionModeLocalSizeHintId ||
          mode == SpvExelwtionModeLocalSizeId) {
        if (!spvOpcodeIsConstant(operand_inst->opcode())) {
          return _.diag(SPV_ERROR_ILWALID_ID, inst)
                 << "For OpExelwtionModeId all Extra Operand ids must be "
                    "constant "
                    "instructions.";
        }
      } else {
        return _.diag(SPV_ERROR_ILWALID_ID, inst)
               << "OpExelwtionModeId is only valid when the Mode operand is an "
                  "exelwtion mode that takes Extra Operands that are id "
                  "operands.";
      }
    }
  } else if (mode == SpvExelwtionModeSubgroupsPerWorkgroupId ||
             mode == SpvExelwtionModeLocalSizeHintId ||
             mode == SpvExelwtionModeLocalSizeId) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << "OpExelwtionMode is only valid when the Mode operand is an "
              "exelwtion mode that takes no Extra Operands, or takes Extra "
              "Operands that are not id operands.";
  }

  const auto* models = _.GetExelwtionModels(entry_point_id);
  switch (mode) {
    case SpvExelwtionModeIlwocations:
    case SpvExelwtionModeInputPoints:
    case SpvExelwtionModeInputLines:
    case SpvExelwtionModeInputLinesAdjacency:
    case SpvExelwtionModeInputTrianglesAdjacency:
    case SpvExelwtionModeOutputLineStrip:
    case SpvExelwtionModeOutputTriangleStrip:
      if (!std::all_of(models->begin(), models->end(),
                       [](const SpvExelwtionModel& model) {
                         return model == SpvExelwtionModelGeometry;
                       })) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << "Exelwtion mode can only be used with the Geometry exelwtion "
                  "model.";
      }
      break;
    case SpvExelwtionModeOutputPoints:
      if (!std::all_of(models->begin(), models->end(),
                       [&_](const SpvExelwtionModel& model) {
                         switch (model) {
                           case SpvExelwtionModelGeometry:
                             return true;
                           case SpvExelwtionModelMeshLW:
                             return _.HasCapability(SpvCapabilityMeshShadingLW);
                           default:
                             return false;
                         }
                       })) {
        if (_.HasCapability(SpvCapabilityMeshShadingLW)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Exelwtion mode can only be used with the Geometry or "
                    "MeshLW exelwtion model.";
        } else {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Exelwtion mode can only be used with the Geometry "
                    "exelwtion "
                    "model.";
        }
      }
      break;
    case SpvExelwtionModeSpacingEqual:
    case SpvExelwtionModeSpacingFractionalEven:
    case SpvExelwtionModeSpacingFractionalOdd:
    case SpvExelwtionModeVertexOrderCw:
    case SpvExelwtionModeVertexOrderCcw:
    case SpvExelwtionModePointMode:
    case SpvExelwtionModeQuads:
    case SpvExelwtionModeIsolines:
      if (!std::all_of(
              models->begin(), models->end(),
              [](const SpvExelwtionModel& model) {
                return (model == SpvExelwtionModelTessellationControl) ||
                       (model == SpvExelwtionModelTessellationEvaluation);
              })) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << "Exelwtion mode can only be used with a tessellation "
                  "exelwtion model.";
      }
      break;
    case SpvExelwtionModeTriangles:
      if (!std::all_of(models->begin(), models->end(),
                       [](const SpvExelwtionModel& model) {
                         switch (model) {
                           case SpvExelwtionModelGeometry:
                           case SpvExelwtionModelTessellationControl:
                           case SpvExelwtionModelTessellationEvaluation:
                             return true;
                           default:
                             return false;
                         }
                       })) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << "Exelwtion mode can only be used with a Geometry or "
                  "tessellation exelwtion model.";
      }
      break;
    case SpvExelwtionModeOutputVertices:
      if (!std::all_of(models->begin(), models->end(),
                       [&_](const SpvExelwtionModel& model) {
                         switch (model) {
                           case SpvExelwtionModelGeometry:
                           case SpvExelwtionModelTessellationControl:
                           case SpvExelwtionModelTessellationEvaluation:
                             return true;
                           case SpvExelwtionModelMeshLW:
                             return _.HasCapability(SpvCapabilityMeshShadingLW);
                           default:
                             return false;
                         }
                       })) {
        if (_.HasCapability(SpvCapabilityMeshShadingLW)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Exelwtion mode can only be used with a Geometry, "
                    "tessellation or MeshLW exelwtion model.";
        } else {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Exelwtion mode can only be used with a Geometry or "
                    "tessellation exelwtion model.";
        }
      }
      break;
    case SpvExelwtionModePixelCenterInteger:
    case SpvExelwtionModeOriginUpperLeft:
    case SpvExelwtionModeOriginLowerLeft:
    case SpvExelwtionModeEarlyFragmentTests:
    case SpvExelwtionModeDepthReplacing:
    case SpvExelwtionModeDepthGreater:
    case SpvExelwtionModeDepthLess:
    case SpvExelwtionModeDepthUnchanged:
    case SpvExelwtionModePixelInterlockOrderedEXT:
    case SpvExelwtionModePixelInterlockUnorderedEXT:
    case SpvExelwtionModeSampleInterlockOrderedEXT:
    case SpvExelwtionModeSampleInterlockUnorderedEXT:
    case SpvExelwtionModeShadingRateInterlockOrderedEXT:
    case SpvExelwtionModeShadingRateInterlockUnorderedEXT:
      if (!std::all_of(models->begin(), models->end(),
                       [](const SpvExelwtionModel& model) {
                         return model == SpvExelwtionModelFragment;
                       })) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << "Exelwtion mode can only be used with the Fragment exelwtion "
                  "model.";
      }
      break;
    case SpvExelwtionModeLocalSizeHint:
    case SpvExelwtionModeVecTypeHint:
    case SpvExelwtionModeContractionOff:
    case SpvExelwtionModeLocalSizeHintId:
      if (!std::all_of(models->begin(), models->end(),
                       [](const SpvExelwtionModel& model) {
                         return model == SpvExelwtionModelKernel;
                       })) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << "Exelwtion mode can only be used with the Kernel exelwtion "
                  "model.";
      }
      break;
    case SpvExelwtionModeLocalSize:
    case SpvExelwtionModeLocalSizeId:
      if (!std::all_of(models->begin(), models->end(),
                       [&_](const SpvExelwtionModel& model) {
                         switch (model) {
                           case SpvExelwtionModelKernel:
                           case SpvExelwtionModelGLCompute:
                             return true;
                           case SpvExelwtionModelTaskLW:
                           case SpvExelwtionModelMeshLW:
                             return _.HasCapability(SpvCapabilityMeshShadingLW);
                           default:
                             return false;
                         }
                       })) {
        if (_.HasCapability(SpvCapabilityMeshShadingLW)) {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Exelwtion mode can only be used with a Kernel, GLCompute, "
                    "MeshLW, or TaskLW exelwtion model.";
        } else {
          return _.diag(SPV_ERROR_ILWALID_DATA, inst)
                 << "Exelwtion mode can only be used with a Kernel or "
                    "GLCompute "
                    "exelwtion model.";
        }
      }
    default:
      break;
  }

  if (spvIsVulkanElw(_.context()->target_elw)) {
    if (mode == SpvExelwtionModeOriginLowerLeft) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "In the Vulkan environment, the OriginLowerLeft exelwtion mode "
                "must not be used.";
    }
    if (mode == SpvExelwtionModePixelCenterInteger) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "In the Vulkan environment, the PixelCenterInteger exelwtion "
                "mode must not be used.";
    }
  }

  if (spvIsWebGPUElw(_.context()->target_elw)) {
    if (mode != SpvExelwtionModeOriginUpperLeft &&
        mode != SpvExelwtionModeDepthReplacing &&
        mode != SpvExelwtionModeDepthGreater &&
        mode != SpvExelwtionModeDepthLess &&
        mode != SpvExelwtionModeDepthUnchanged &&
        mode != SpvExelwtionModeLocalSize &&
        mode != SpvExelwtionModeLocalSizeHint) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "Exelwtion mode must be one of OriginUpperLeft, "
                "DepthReplacing, DepthGreater, DepthLess, DepthUnchanged, "
                "LocalSize, or LocalSizeHint for WebGPU environment.";
    }
  }

  return SPV_SUCCESS;
}

spv_result_t ValidateMemoryModel(ValidationState_t& _,
                                 const Instruction* inst) {
  // Already produced an error if multiple memory model instructions are
  // present.
  if (_.memory_model() != SpvMemoryModelVulkanKHR &&
      _.HasCapability(SpvCapabilityVulkanMemoryModelKHR)) {
    return _.diag(SPV_ERROR_ILWALID_DATA, inst)
           << "VulkanMemoryModelKHR capability must only be specified if "
              "the VulkanKHR memory model is used.";
  }

  if (spvIsWebGPUElw(_.context()->target_elw)) {
    if (_.addressing_model() != SpvAddressingModelLogical) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "Addressing model must be Logical for WebGPU environment.";
    }
    if (_.memory_model() != SpvMemoryModelVulkanKHR) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "Memory model must be VulkanKHR for WebGPU environment.";
    }
  }

  if (spvIsOpenCLElw(_.context()->target_elw)) {
    if ((_.addressing_model() != SpvAddressingModelPhysical32) &&
        (_.addressing_model() != SpvAddressingModelPhysical64)) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "Addressing model must be Physical32 or Physical64 "
             << "in the OpenCL environment.";
    }
    if (_.memory_model() != SpvMemoryModelOpenCL) {
      return _.diag(SPV_ERROR_ILWALID_DATA, inst)
             << "Memory model must be OpenCL in the OpenCL environment.";
    }
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t ModeSettingPass(ValidationState_t& _, const Instruction* inst) {
  switch (inst->opcode()) {
    case SpvOpEntryPoint:
      if (auto error = ValidateEntryPoint(_, inst)) return error;
      break;
    case SpvOpExelwtionMode:
    case SpvOpExelwtionModeId:
      if (auto error = ValidateExelwtionMode(_, inst)) return error;
      break;
    case SpvOpMemoryModel:
      if (auto error = ValidateMemoryModel(_, inst)) return error;
      break;
    default:
      break;
  }
  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
