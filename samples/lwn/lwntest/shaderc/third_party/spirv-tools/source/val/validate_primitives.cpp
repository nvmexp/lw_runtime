// Copyright (c) 2017 LunarG Inc.
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

// Validates correctness of primitive SPIR-V instructions.

#include "source/val/validate.h"

#include <string>

#include "source/diagnostic.h"
#include "source/opcode.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

// Validates correctness of primitive instructions.
spv_result_t PrimitivesPass(ValidationState_t& _, const Instruction* inst) {
  const SpvOp opcode = inst->opcode();

  switch (opcode) {
    case SpvOpEmitVertex:
    case SpvOpEndPrimitive:
    case SpvOpEmitStreamVertex:
    case SpvOpEndStreamPrimitive:
      _.function(inst->function()->id())
          ->RegisterExelwtionModelLimitation(
              SpvExelwtionModelGeometry,
              std::string(spvOpcodeString(opcode)) +
                  " instructions require Geometry exelwtion model");
      break;
    default:
      break;
  }

  switch (opcode) {
    case SpvOpEmitStreamVertex:
    case SpvOpEndStreamPrimitive: {
      const uint32_t stream_id = inst->word(1);
      const uint32_t stream_type = _.GetTypeId(stream_id);
      if (!_.IsIntScalarType(stream_type)) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Stream to be int scalar";
      }

      const SpvOp stream_opcode = _.GetIdOpcode(stream_id);
      if (!spvOpcodeIsConstant(stream_opcode)) {
        return _.diag(SPV_ERROR_ILWALID_DATA, inst)
               << spvOpcodeString(opcode)
               << ": expected Stream to be constant instruction";
      }
    }

    default:
      break;
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools
