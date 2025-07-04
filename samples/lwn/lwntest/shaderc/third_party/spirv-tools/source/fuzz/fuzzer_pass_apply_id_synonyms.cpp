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

#include "source/fuzz/fuzzer_pass_apply_id_synonyms.h"

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/id_use_descriptor.h"
#include "source/fuzz/instruction_descriptor.h"
#include "source/fuzz/transformation_composite_extract.h"
#include "source/fuzz/transformation_compute_data_synonym_fact_closure.h"
#include "source/fuzz/transformation_replace_id_with_synonym.h"

namespace spvtools {
namespace fuzz {

FuzzerPassApplyIdSynonyms::FuzzerPassApplyIdSynonyms(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations) {}

FuzzerPassApplyIdSynonyms::~FuzzerPassApplyIdSynonyms() = default;

void FuzzerPassApplyIdSynonyms::Apply() {
  // Compute a closure of data synonym facts, to enrich the pool of synonyms
  // that are available.
  ApplyTransformation(TransformationComputeDataSynonymFactClosure(
      GetFuzzerContext()
          ->GetMaximumEquivalenceClassSizeForDataSynonymFactClosure()));

  for (auto id_with_known_synonyms : GetTransformationContext()
                                         ->GetFactManager()
                                         ->GetIdsForWhichSynonymsAreKnown()) {
    // Gather up all uses of |id_with_known_synonym| as a regular id, and
    // subsequently iterate over these uses.  We use this separation because,
    // when considering a given use, we might apply a transformation that will
    // ilwalidate the def-use manager.
    std::vector<std::pair<opt::Instruction*, uint32_t>> uses;
    GetIRContext()->get_def_use_mgr()->ForEachUse(
        id_with_known_synonyms,
        [&uses](opt::Instruction* use_inst, uint32_t use_index) -> void {
          // We only gather up regular id uses; e.g. we do not include a use of
          // the id as the scope for an atomic operation.
          if (use_inst->GetOperand(use_index).type == SPV_OPERAND_TYPE_ID) {
            uses.emplace_back(
                std::pair<opt::Instruction*, uint32_t>(use_inst, use_index));
          }
        });

    for (auto& use : uses) {
      auto use_inst = use.first;
      auto use_index = use.second;
      auto block_containing_use = GetIRContext()->get_instr_block(use_inst);
      // The use might not be in a block; e.g. it could be a decoration.
      if (!block_containing_use) {
        continue;
      }
      if (!GetFuzzerContext()->ChoosePercentage(
              GetFuzzerContext()->GetChanceOfReplacingIdWithSynonym())) {
        continue;
      }
      // |use_index| is the absolute index of the operand.  We require
      // the index of the operand restricted to input operands only, so
      // we subtract the number of non-input operands from |use_index|.
      uint32_t use_in_operand_index =
          use_index - use_inst->NumOperands() + use_inst->NumInOperands();
      if (!TransformationReplaceIdWithSynonym::UseCanBeReplacedWithSynonym(
              GetIRContext(), use_inst, use_in_operand_index)) {
        continue;
      }

      std::vector<const protobufs::DataDescriptor*> synonyms_to_try;
      for (auto& data_descriptor :
           GetTransformationContext()->GetFactManager()->GetSynonymsForId(
               id_with_known_synonyms)) {
        protobufs::DataDescriptor descriptor_for_this_id =
            MakeDataDescriptor(id_with_known_synonyms, {});
        if (DataDescriptorEquals()(data_descriptor, &descriptor_for_this_id)) {
          // Exclude the fact that the id is synonymous with itself.
          continue;
        }
        synonyms_to_try.push_back(data_descriptor);
      }
      while (!synonyms_to_try.empty()) {
        auto synonym_to_try =
            GetFuzzerContext()->RemoveAtRandomIndex(&synonyms_to_try);

        // If the synonym's |index_size| is zero, the synonym represents an id.
        // Otherwise it represents some element of a composite structure, in
        // which case we need to be able to add an extract instruction to get
        // that element out.
        if (synonym_to_try->index_size() > 0 &&
            !fuzzerutil::CanInsertOpcodeBeforeInstruction(SpvOpCompositeExtract,
                                                          use_inst) &&
            use_inst->opcode() != SpvOpPhi) {
          // We cannot insert an extract before this instruction, so this
          // synonym is no good.
          continue;
        }

        if (!fuzzerutil::IdIsAvailableAtUse(GetIRContext(), use_inst,
                                            use_in_operand_index,
                                            synonym_to_try->object())) {
          continue;
        }

        // We either replace the use with an id known to be synonymous (when
        // the synonym's |index_size| is 0), or an id that will hold the result
        // of extracting a synonym from a composite (when the synonym's
        // |index_size| is > 0).
        uint32_t id_with_which_to_replace_use;
        if (synonym_to_try->index_size() == 0) {
          id_with_which_to_replace_use = synonym_to_try->object();
        } else {
          id_with_which_to_replace_use = GetFuzzerContext()->GetFreshId();
          opt::Instruction* instruction_to_insert_before = nullptr;

          if (use_inst->opcode() != SpvOpPhi) {
            instruction_to_insert_before = use_inst;
          } else {
            auto parent_block_id =
                use_inst->GetSingleWordInOperand(use_in_operand_index + 1);
            auto parent_block_instruction =
                GetIRContext()->get_def_use_mgr()->GetDef(parent_block_id);
            auto parent_block =
                GetIRContext()->get_instr_block(parent_block_instruction);

            instruction_to_insert_before = parent_block->GetMergeInst()
                                               ? parent_block->GetMergeInst()
                                               : parent_block->terminator();
          }

          ApplyTransformation(TransformationCompositeExtract(
              MakeInstructionDescriptor(GetIRContext(),
                                        instruction_to_insert_before),
              id_with_which_to_replace_use, synonym_to_try->object(),
              fuzzerutil::RepeatedFieldToVector(synonym_to_try->index())));
        }

        ApplyTransformation(TransformationReplaceIdWithSynonym(
            MakeIdUseDescriptorFromUse(GetIRContext(), use_inst,
                                       use_in_operand_index),
            id_with_which_to_replace_use));
        break;
      }
    }
  }
}

}  // namespace fuzz
}  // namespace spvtools
