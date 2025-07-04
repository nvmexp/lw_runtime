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

#include "source/fuzz/fuzzer_pass_adjust_selection_controls.h"

#include "source/fuzz/transformation_set_selection_control.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAdjustSelectionControls::FuzzerPassAdjustSelectionControls(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAdjustSelectionControls::~FuzzerPassAdjustSelectionControls() =
    default;

void FuzzerPassAdjustSelectionControls::Apply() {
  // Consider every merge instruction in the module (via looking through all
  // functions and blocks).
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      if (auto merge_inst = block.GetMergeInst()) {
        // Ignore the instruction if it is not a selection merge.
        if (merge_inst->opcode() != SpvOpSelectionMerge) {
          continue;
        }

        // Choose randomly whether to change the selection control for this
        // instruction.
        if (!GetFuzzerContext()->ChoosePercentage(
                GetFuzzerContext()->GetChanceOfAdjustingSelectionControl())) {
          continue;
        }

        // The choices to change the selection control to are the set of valid
        // controls, minus the current control.
        std::vector<uint32_t> choices;
        for (auto control :
             {SpvSelectionControlMaskNone, SpvSelectionControlFlattenMask,
              SpvSelectionControlDontFlattenMask}) {
          if (control == merge_inst->GetSingleWordOperand(1)) {
            continue;
          }
          choices.push_back(control);
        }

        // Apply the transformation and add it to the output transformation
        // sequence.
        TransformationSetSelectionControl transformation(
            block.id(), choices[GetFuzzerContext()->RandomIndex(choices)]);
        ApplyTransformation(transformation);
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
