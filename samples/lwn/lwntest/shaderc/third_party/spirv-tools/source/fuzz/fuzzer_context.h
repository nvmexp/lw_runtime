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

#ifndef SOURCE_FUZZ_FUZZER_CONTEXT_H_
#define SOURCE_FUZZ_FUZZER_CONTEXT_H_

#include <functional>
#include <utility>

#include "source/fuzz/random_generator.h"
#include "source/opt/function.h"

namespace spvtools {
namespace fuzz {

// Encapsulates all parameters that control the fuzzing process, such as the
// source of randomness and the probabilities with which transformations are
// applied.
class FuzzerContext {
 public:
  // Constructs a fuzzer context with a given random generator and the minimum
  // value that can be used for fresh ids.
  FuzzerContext(RandomGenerator* random_generator, uint32_t min_fresh_id);

  ~FuzzerContext();

  // Returns a random boolean.
  bool ChooseEven();

  // Returns true if and only if a randomly-chosen integer in the range [0, 100]
  // is less than |percentage_chance|.
  bool ChoosePercentage(uint32_t percentage_chance);

  // Returns a random index into |sequence|, which is expected to have a 'size'
  // method, and which must be non-empty.  Typically 'HasSizeMethod' will be an
  // std::vector.
  template <typename HasSizeMethod>
  uint32_t RandomIndex(const HasSizeMethod& sequence) const {
    assert(sequence.size() > 0);
    return random_generator_->RandomUint32(
        static_cast<uint32_t>(sequence.size()));
  }

  // Selects a random index into |sequence|, removes the element at that index
  // and returns it.
  template <typename T>
  T RemoveAtRandomIndex(std::vector<T>* sequence) const {
    uint32_t index = RandomIndex(*sequence);
    T result = sequence->at(index);
    sequence->erase(sequence->begin() + index);
    return result;
  }

  // Randomly shuffles a |sequence| between |lo| and |hi| indices inclusively.
  // |lo| and |hi| must be valid indices to the |sequence|
  template <typename T>
  void Shuffle(std::vector<T>* sequence, size_t lo, size_t hi) const {
    auto& array = *sequence;

    if (array.empty()) {
      return;
    }

    assert(lo <= hi && hi < array.size() && "lo and/or hi indices are invalid");

    // i > lo to account for potential infinite loop when lo == 0
    for (size_t i = hi; i > lo; --i) {
      auto index =
          random_generator_->RandomUint32(static_cast<uint32_t>(i - lo + 1));

      if (lo + index != i) {
        // Introduce std::swap to the scope but don't use it
        // directly since there might be a better overload
        using std::swap;
        swap(array[lo + index], array[i]);
      }
    }
  }

  // Ramdomly shuffles a |sequence|
  template <typename T>
  void Shuffle(std::vector<T>* sequence) const {
    if (!sequence->empty()) {
      Shuffle(sequence, 0, sequence->size() - 1);
    }
  }

  // Yields an id that is guaranteed not to be used in the module being fuzzed,
  // or to have been issued before.
  uint32_t GetFreshId();

  // Probabilities associated with applying various transformations.
  // Keep them in alphabetical order.
  uint32_t GetChanceOfAddingAccessChain() {
    return chance_of_adding_access_chain_;
  }
  uint32_t GetChanceOfAddingAnotherStructField() {
    return chance_of_adding_another_struct_field_;
  }
  uint32_t GetChanceOfAddingArrayOrStructType() {
    return chance_of_adding_array_or_struct_type_;
  }
  uint32_t GetChanceOfAddingDeadBlock() { return chance_of_adding_dead_block_; }
  uint32_t GetChanceOfAddingDeadBreak() { return chance_of_adding_dead_break_; }
  uint32_t GetChanceOfAddingDeadContinue() {
    return chance_of_adding_dead_continue_;
  }
  uint32_t GetChanceOfAddingEquationInstruction() {
    return chance_of_adding_equation_instruction_;
  }
  uint32_t GetChanceOfAddingGlobalVariable() {
    return chance_of_adding_global_variable_;
  }
  uint32_t GetChanceOfAddingLoad() { return chance_of_adding_load_; }
  uint32_t GetChanceOfAddingLocalVariable() {
    return chance_of_adding_local_variable_;
  }
  uint32_t GetChanceOfAddingMatrixType() {
    return chance_of_adding_matrix_type_;
  }
  uint32_t GetChanceOfAddingNoContractionDecoration() {
    return chance_of_adding_no_contraction_decoration_;
  }
  uint32_t GetChanceOfAddingStore() { return chance_of_adding_store_; }
  uint32_t GetChanceOfAddingVectorType() {
    return chance_of_adding_vector_type_;
  }
  uint32_t GetChanceOfAdjustingFunctionControl() {
    return chance_of_adjusting_function_control_;
  }
  uint32_t GetChanceOfAdjustingLoopControl() {
    return chance_of_adjusting_loop_control_;
  }
  uint32_t GetChanceOfAdjustingMemoryOperandsMask() {
    return chance_of_adjusting_memory_operands_mask_;
  }
  uint32_t GetChanceOfAdjustingSelectionControl() {
    return chance_of_adjusting_selection_control_;
  }
  uint32_t GetChanceOfCallingFunction() { return chance_of_calling_function_; }
  uint32_t GetChanceOfChoosingStructTypeVsArrayType() {
    return chance_of_choosing_struct_type_vs_array_type_;
  }
  uint32_t GetChanceOfConstructingComposite() {
    return chance_of_constructing_composite_;
  }
  uint32_t GetChanceOfCopyingObject() { return chance_of_copying_object_; }
  uint32_t GetChanceOfDonatingAdditionalModule() {
    return chance_of_donating_additional_module_;
  }
  uint32_t GetChanceOfGoingDeeperWhenMakingAccessChain() {
    return chance_of_going_deeper_when_making_access_chain_;
  }
  uint32_t ChanceOfMakingDonorLivesafe() {
    return chance_of_making_donor_livesafe_;
  }
  uint32_t GetChanceOfMergingBlocks() { return chance_of_merging_blocks_; }
  uint32_t GetChanceOfMovingBlockDown() { return chance_of_moving_block_down_; }
  uint32_t GetChanceOfObfuscatingConstant() {
    return chance_of_obfuscating_constant_;
  }
  uint32_t GetChanceOfOutliningFunction() {
    return chance_of_outlining_function_;
  }
  uint32_t GetChanceOfPermutingParameters() {
    return chance_of_permuting_parameters_;
  }
  uint32_t GetChanceOfReplacingIdWithSynonym() {
    return chance_of_replacing_id_with_synonym_;
  }
  uint32_t GetChanceOfSplittingBlock() { return chance_of_splitting_block_; }
  uint32_t GetChanceOfTogglingAccessChainInstruction() {
    return chance_of_toggling_access_chain_instruction_;
  }

  // Other functions to control transformations. Keep them in alphabetical
  // order.
  uint32_t GetMaximumEquivalenceClassSizeForDataSynonymFactClosure() {
    return max_equivalence_class_size_for_data_synonym_fact_closure_;
  }
  uint32_t GetRandomIndexForAccessChain(uint32_t composite_size_bound) {
    return random_generator_->RandomUint32(composite_size_bound);
  }
  uint32_t GetRandomLoopControlPartialCount() {
    return random_generator_->RandomUint32(max_loop_control_partial_count_);
  }
  uint32_t GetRandomLoopControlPeelCount() {
    return random_generator_->RandomUint32(max_loop_control_peel_count_);
  }
  uint32_t GetRandomLoopLimit() {
    return random_generator_->RandomUint32(max_loop_limit_);
  }
  uint32_t GetRandomSizeForNewArray() {
    // Ensure that the array size is non-zero.
    return random_generator_->RandomUint32(max_new_array_size_limit_ - 1) + 1;
  }
  bool GoDeeperInConstantObfuscation(uint32_t depth) {
    return go_deeper_in_constant_obfuscation_(depth, random_generator_);
  }

 private:
  // The source of randomness.
  RandomGenerator* random_generator_;
  // The next fresh id to be issued.
  uint32_t next_fresh_id_;

  // Probabilities associated with applying various transformations.
  // Keep them in alphabetical order.
  uint32_t chance_of_adding_access_chain_;
  uint32_t chance_of_adding_another_struct_field_;
  uint32_t chance_of_adding_array_or_struct_type_;
  uint32_t chance_of_adding_dead_block_;
  uint32_t chance_of_adding_dead_break_;
  uint32_t chance_of_adding_dead_continue_;
  uint32_t chance_of_adding_equation_instruction_;
  uint32_t chance_of_adding_global_variable_;
  uint32_t chance_of_adding_load_;
  uint32_t chance_of_adding_local_variable_;
  uint32_t chance_of_adding_matrix_type_;
  uint32_t chance_of_adding_no_contraction_decoration_;
  uint32_t chance_of_adding_store_;
  uint32_t chance_of_adding_vector_type_;
  uint32_t chance_of_adjusting_function_control_;
  uint32_t chance_of_adjusting_loop_control_;
  uint32_t chance_of_adjusting_memory_operands_mask_;
  uint32_t chance_of_adjusting_selection_control_;
  uint32_t chance_of_calling_function_;
  uint32_t chance_of_choosing_struct_type_vs_array_type_;
  uint32_t chance_of_constructing_composite_;
  uint32_t chance_of_copying_object_;
  uint32_t chance_of_donating_additional_module_;
  uint32_t chance_of_going_deeper_when_making_access_chain_;
  uint32_t chance_of_making_donor_livesafe_;
  uint32_t chance_of_merging_blocks_;
  uint32_t chance_of_moving_block_down_;
  uint32_t chance_of_obfuscating_constant_;
  uint32_t chance_of_outlining_function_;
  uint32_t chance_of_permuting_parameters_;
  uint32_t chance_of_replacing_id_with_synonym_;
  uint32_t chance_of_splitting_block_;
  uint32_t chance_of_toggling_access_chain_instruction_;

  // Limits associated with various quantities for which random values are
  // chosen during fuzzing.
  // Keep them in alphabetical order.
  uint32_t max_equivalence_class_size_for_data_synonym_fact_closure_;
  uint32_t max_loop_control_partial_count_;
  uint32_t max_loop_control_peel_count_;
  uint32_t max_loop_limit_;
  uint32_t max_new_array_size_limit_;

  // Functions to determine with what probability to go deeper when generating
  // or mutating constructs relwrsively.
  const std::function<bool(uint32_t, RandomGenerator*)>&
      go_deeper_in_constant_obfuscation_;

  // Requires |min_max.first| <= |min_max.second|, and returns a value in the
  // range [ |min_max.first|, |min_max.second| ]
  uint32_t ChooseBetweenMinAndMax(const std::pair<uint32_t, uint32_t>& min_max);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_CONTEXT_H_
