// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_set_selection_control.h"

namespace spvtools {
namespace fuzz {

TransformationSetSelectionControl::TransformationSetSelectionControl(
    const spvtools::fuzz::protobufs::TransformationSetSelectionControl& message)
    : message_(message) {}

TransformationSetSelectionControl::TransformationSetSelectionControl(
    uint32_t block_id, uint32_t selection_control) {
  message_.set_block_id(block_id);
  message_.set_selection_control(selection_control);
}

bool TransformationSetSelectionControl::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  assert((message_.selection_control() == SpvSelectionControlMaskNone ||
          message_.selection_control() == SpvSelectionControlFlattenMask ||
          message_.selection_control() == SpvSelectionControlDontFlattenMask) &&
         "Selection control should never be set to something other than "
         "'None', 'Flatten' or 'DontFlatten'");
  if (auto block = ir_context->get_instr_block(message_.block_id())) {
    if (auto merge_inst = block->GetMergeInst()) {
      return merge_inst->opcode() == SpvOpSelectionMerge;
    }
  }
  // Either the block did not exit, or did not end with OpSelectionMerge.
  return false;
}

void TransformationSetSelectionControl::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  ir_context->get_instr_block(message_.block_id())
      ->GetMergeInst()
      ->SetInOperand(1, {message_.selection_control()});
}

protobufs::Transformation TransformationSetSelectionControl::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_set_selection_control() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
