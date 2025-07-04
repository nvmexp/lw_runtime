// Copyright (c) 2018 LunarG Inc.
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

// Validates correctness of the intra-block preconditions of SPIR-V
// instructions.

#include "source/val/validate.h"

#include <string>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

enum {
  // Status right after meeting OpFunction.
  IN_NEW_FUNCTION,
  // Status right after meeting the entry block.
  IN_ENTRY_BLOCK,
  // Status right after meeting non-entry blocks.
  PHI_VALID,
  // Status right after meeting non-OpVariable instructions in the entry block
  // or non-OpPhi instructions in non-entry blocks, except OpLine.
  PHI_AND_VAR_ILWALID,
};

spv_result_t ValidateAdjacency(ValidationState_t& _) {
  const auto& instructions = _.ordered_instructions();
  int adjacency_status = PHI_AND_VAR_ILWALID;

  for (size_t i = 0; i < instructions.size(); ++i) {
    const auto& inst = instructions[i];
    switch (inst.opcode()) {
      case SpvOpFunction:
      case SpvOpFunctionParameter:
        adjacency_status = IN_NEW_FUNCTION;
        break;
      case SpvOpLabel:
        adjacency_status =
            adjacency_status == IN_NEW_FUNCTION ? IN_ENTRY_BLOCK : PHI_VALID;
        break;
      case SpvOpExtInst:
        // If it is a debug info instruction, we do not change the status to
        // allow debug info instructions before OpVariable in a function.
        // TODO(https://gitlab.khronos.org/spirv/SPIR-V/issues/533): We need
        // to discuss the location of DebugScope, DebugNoScope, DebugDeclare,
        // and DebugValue.
        if (!spvExtInstIsDebugInfo(inst.ext_inst_type())) {
          adjacency_status = PHI_AND_VAR_ILWALID;
        }
        break;
      case SpvOpPhi:
        if (adjacency_status != PHI_VALID) {
          return _.diag(SPV_ERROR_ILWALID_DATA, &inst)
                 << "OpPhi must appear within a non-entry block before all "
                 << "non-OpPhi instructions "
                 << "(except for OpLine, which can be mixed with OpPhi).";
        }
        break;
      case SpvOpLine:
      case SpvOpNoLine:
        break;
      case SpvOpLoopMerge:
        adjacency_status = PHI_AND_VAR_ILWALID;
        if (i != (instructions.size() - 1)) {
          switch (instructions[i + 1].opcode()) {
            case SpvOpBranch:
            case SpvOpBranchConditional:
              break;
            default:
              return _.diag(SPV_ERROR_ILWALID_DATA, &inst)
                     << "OpLoopMerge must immediately precede either an "
                     << "OpBranch or OpBranchConditional instruction. "
                     << "OpLoopMerge must be the second-to-last instruction in "
                     << "its block.";
          }
        }
        break;
      case SpvOpSelectionMerge:
        adjacency_status = PHI_AND_VAR_ILWALID;
        if (i != (instructions.size() - 1)) {
          switch (instructions[i + 1].opcode()) {
            case SpvOpBranchConditional:
            case SpvOpSwitch:
              break;
            default:
              return _.diag(SPV_ERROR_ILWALID_DATA, &inst)
                     << "OpSelectionMerge must immediately precede either an "
                     << "OpBranchConditional or OpSwitch instruction. "
                     << "OpSelectionMerge must be the second-to-last "
                     << "instruction in its block.";
          }
        }
        break;
      case SpvOpVariable:
        if (inst.GetOperandAs<SpvStorageClass>(2) == SpvStorageClassFunction &&
            adjacency_status != IN_ENTRY_BLOCK) {
          return _.diag(SPV_ERROR_ILWALID_DATA, &inst)
                 << "All OpVariable instructions in a function must be the "
                    "first instructions in the first block.";
        }
        break;
      default:
        adjacency_status = PHI_AND_VAR_ILWALID;
        break;
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
