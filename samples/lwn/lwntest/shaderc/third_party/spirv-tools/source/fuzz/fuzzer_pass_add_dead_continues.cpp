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

#include "source/fuzz/fuzzer_pass_add_dead_continues.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_add_dead_continue.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

FuzzerPassAddDeadContinues::FuzzerPassAddDeadContinues(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassAddDeadContinues::~FuzzerPassAddDeadContinues() = default;

void FuzzerPassAddDeadContinues::Apply() {
  // Consider every block in every function.
  for (auto& function : *GetIRContext()->module()) {
    for (auto& block : function) {
      // Get the label id of the continue target of the innermost loop.
      auto continue_block_id =
          block.IsLoopHeader()
              ? block.ContinueBlockId()
              : GetIRContext()->GetStructuredCFGAnalysis()->LoopContinueBlock(
                    block.id());

      // This transformation is not applicable if current block is not inside a
      // loop.
      if (continue_block_id == 0) {
        continue;
      }

      auto* continue_block =
          fuzzerutil::MaybeFindBlock(GetIRContext(), continue_block_id);
      assert(continue_block && "Continue block is null");

      // Analyze return type of each OpPhi instruction in the continue target
      // and provide an id for the transformation if needed.
      std::vector<uint32_t> phi_ids;
      // Check whether current block has an edge to the continue target.
      // If this is the case, we don't need to do anything.
      if (!block.IsSuccessor(continue_block)) {
        continue_block->ForEachPhiInst([this, &phi_ids](opt::Instruction* phi) {
          // Add an additional operand for OpPhi instruction.
          //
          // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3177):
          // If we have a way to communicate to the fact manager
          // that a specific id use is irrelevant and could be replaced with
          // something else, we should add such a fact about the zero
          // provided as an OpPhi operand
          phi_ids.push_back(FindOrCreateZeroConstant(phi->type_id()));
        });
      }

      // Make a transformation to add a dead continue from this node; if the
      // node turns out to be inappropriate (e.g. by not being in a loop) the
      // precondition for the transformation will fail and it will be ignored.
      auto candidate_transformation = TransformationAddDeadContinue(
          block.id(), GetFuzzerContext()->ChooseEven(), std::move(phi_ids));
      // Probabilistically decide whether to apply the transformation in the
      // case that it is applicable.
      if (candidate_transformation.IsApplicable(GetIRContext(),
                                                *GetTransformationContext()) &&
          GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfAddingDeadContinue())) {
        candidate_transformation.Apply(GetIRContext(),
                                       GetTransformationContext());
        *GetTransformations()->add_transformation() =
            candidate_transformation.ToMessage();
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
