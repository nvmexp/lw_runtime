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

// Validates correctness of barrier SPIR-V instructions.

#include <string>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/spirv_constant.h"
#include "source/spirv_target_elw.h"
#include "source/util/bitutils.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validate_memory_semantics.h"
#include "source/val/validate_scopes.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

// Validates correctness of barrier instructions.
spv_result_t BarriersPass(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();
  const uint32_t result_type = inst->type_id();

  switch (opcode) {
    case SpvOpControlBarrier: {
      if (_.version() < SPV_SPIRV_VERSION_WORD(1, 3)) {
        _.function(inst->function()->id())
            ->RegisterExelwtionModelLimitation(
                [](SpvExelwtionModel model, std::string* message) {
                  if (model != SpvExelwtionModelTessellationControl &&
                      model != SpvExelwtionModelGLCompute &&
                      model != SpvExelwtionModelKernel &&
                      model != SpvExelwtionModelTaskLW &&
                      model != SpvExelwtionModelMeshLW) {
                    if (message) {
                      *message =
                          "OpControlBarrier requires one of the following "
                          "Exelwtion "
                          "Models: TessellationControl, GLCompute or Kernel";
                    }
                    return false;
                  }
                  return true;
                });
      }

      const uint32_t exelwtion_scope = inst->word(1);
      const uint32_t memory_scope = inst->word(2);

      if (auto error = ValidateExelwtionScope(_, inst, exelwtion_scope)) {
        return error;
      }

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, 2)) {
        return error;
      }
      break;
    }

    case SpvOpMemoryBarrier: {
      const uint32_t memory_scope = inst->word(1);

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, 1)) {
        return error;
      }
      break;
    }

    case SpvOpNamedBarrierInitialize: {
      if (_.GetIdOpcode(result_type) != SpvOpTypeNamedBarrier) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Result Type to be OpTypeNamedBarrier";
      }

      const uint32_t subgroup_count_type = _.GetOperandTypeId(inst, 2);
      if (!_.IsIntScalarType(subgroup_count_type) ||
          _.GetBitWidth(subgroup_count_type) != 32) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Subgroup Count to be a 32-bit int";
      }
      break;
    }

    case SpvOpMemoryNamedBarrier: {
      const uint32_t named_barrier_type = _.GetOperandTypeId(inst, 0);
      if (_.GetIdOpcode(named_barrier_type) != SpvOpTypeNamedBarrier) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Named Barrier to be of type OpTypeNamedBarrier";
      }

      const uint32_t memory_scope = inst->word(2);

      if (auto error = ValidateMemoryScope(_, inst, memory_scope)) {
        return error;
      }

      if (auto error = ValidateMemorySemantics(_, inst, 2)) {
        return error;
      }
      break;
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
