// Copyright (c) 2020 André Perez Maselco
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

#include "source/fuzz/transformation_toggle_access_chain_instruction.h"

#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/instruction_descriptor.h"

namespace spvtools {
namespace fuzz {

TransformationToggleAccessChainInstruction::
    TransformationToggleAccessChainInstruction(
        const spvtools::fuzz::protobufs::
            TransformationToggleAccessChainInstruction& message)
    : message_(message) {}

TransformationToggleAccessChainInstruction::
    TransformationToggleAccessChainInstruction(
        const protobufs::InstructionDescriptor& instruction_descriptor) {
  *message_.mutable_instruction_descriptor() = instruction_descriptor;
}

bool TransformationToggleAccessChainInstruction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/
    ) const {
  auto instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  if (instruction == nullptr) {
    return false;
  }

  SpvOp opcode = static_cast<SpvOp>(
      message_.instruction_descriptor().target_instruction_opcode());

  assert(instruction->opcode() == opcode &&
         "The located instruction must have the same opcode as in the "
         "descriptor.");

  if (opcode == SpvOpAccessChain || opcode == SpvOpInBoundsAccessChain) {
    return true;
  }

  return false;
}

void TransformationToggleAccessChainInstruction::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/
    ) const {
  auto instruction =
      FindInstruction(message_.instruction_descriptor(), ir_context);
  SpvOp opcode = instruction->opcode();

  if (opcode == SpvOpAccessChain) {
    instruction->SetOpcode(SpvOpInBoundsAccessChain);
  } else {
    assert(opcode == SpvOpInBoundsAccessChain &&
           "The located instruction must be an OpInBoundsAccessChain "
           "instruction.");
    instruction->SetOpcode(SpvOpAccessChain);
  }
}

protobufs::Transformation
TransformationToggleAccessChainInstruction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_toggle_access_chain_instruction() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
