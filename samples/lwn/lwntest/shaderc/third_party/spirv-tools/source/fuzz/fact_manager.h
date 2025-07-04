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

#ifndef SOURCE_FUZZ_FACT_MANAGER_H_
#define SOURCE_FUZZ_FACT_MANAGER_H_

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "source/fuzz/data_descriptor.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/constants.h"

namespace spvtools {
namespace fuzz {

// Keeps track of facts about the module being transformed on which the fuzzing
// process can depend. Some initial facts can be provided, for example about
// guarantees on the values of inputs to SPIR-V entry points. Transformations
// may then rely on these facts, can add further facts that they establish.
// Facts are intended to be simple properties that either cannot be deduced from
// the module (such as properties that are guaranteed to hold for entry point
// inputs), or that are established by transformations, likely to be useful for
// future transformations, and not completely trivial to deduce straight from
// the module.
class FactManager {
 public:
  FactManager();

  ~FactManager();

  // Adds all the facts from |facts|, checking them for validity with respect to
  // |context|.  Warnings about invalid facts are communicated via
  // |message_consumer|; such facts are otherwise ignored.
  void AddFacts(const MessageConsumer& message_consumer,
                const protobufs::FactSequence& facts, opt::IRContext* context);

  // Checks the fact for validity with respect to |context|.  Returns false,
  // with no side effects, if the fact is invalid.  Otherwise adds |fact| to the
  // fact manager.
  bool AddFact(const protobufs::Fact& fact, opt::IRContext* context);

  // Record the fact that |data1| and |data2| are synonymous.
  void AddFactDataSynonym(const protobufs::DataDescriptor& data1,
                          const protobufs::DataDescriptor& data2,
                          opt::IRContext* context);

  // Records the fact that |block_id| is dead.
  void AddFactBlockIsDead(uint32_t block_id);

  // Records the fact that |function_id| is livesafe.
  void AddFactFunctionIsLivesafe(uint32_t function_id);

  // Records the fact that the value of the pointee associated with |pointer_id|
  // is irrelevant: it does not affect the observable behaviour of the module.
  void AddFactValueOfPointeeIsIrrelevant(uint32_t pointer_id);

  // Records the fact that |lhs_id| is defined by the equation:
  //
  //   |lhs_id| = |opcode| |rhs_id[0]| ... |rhs_id[N-1]|
  //
  void AddFactIdEquation(uint32_t lhs_id, SpvOp opcode,
                         const std::vector<uint32_t>& rhs_id,
                         opt::IRContext* context);

  // Inspects all known facts and adds corollary facts; e.g. if we know that
  // a.x == b.x and a.y == b.y, where a and b have vec2 type, we can record
  // that a == b holds.
  //
  // This method is expensive, and should only be called (by applying a
  // transformation) at the start of a fuzzer pass that depends on data
  // synonym facts, rather than calling it every time a new data synonym fact
  // is added.
  //
  // The parameter |maximum_equivalence_class_size| specifies the size beyond
  // which equivalence classes should not be mined for new facts, to avoid
  // excessively-long closure computations.
  void ComputeClosureOfFacts(opt::IRContext* ir_context,
                             uint32_t maximum_equivalence_class_size);

  // The fact manager is responsible for managing a few distinct categories of
  // facts. In principle there could be different fact managers for each kind
  // of fact, but in practice providing one 'go to' place for facts is
  // colwenient.  To keep some separation, the public methods of the fact
  // manager should be grouped according to the kind of fact to which they
  // relate.

  //==============================
  // Querying facts about uniform constants

  // Provides the distinct type ids for which at least one  "constant ==
  // uniform element" fact is known.
  std::vector<uint32_t> GetTypesForWhichUniformValuesAreKnown() const;

  // Provides distinct constant ids with type |type_id| for which at least one
  // "constant == uniform element" fact is known.  If multiple identically-
  // valued constants are relevant, only one will appear in the sequence.
  std::vector<uint32_t> GetConstantsAvailableFromUniformsForType(
      opt::IRContext* ir_context, uint32_t type_id) const;

  // Provides details of all uniform elements that are known to be equal to the
  // constant associated with |constant_id| in |ir_context|.
  const std::vector<protobufs::UniformBufferElementDescriptor>
  GetUniformDescriptorsForConstant(opt::IRContext* ir_context,
                                   uint32_t constant_id) const;

  // Returns the id of a constant whose value is known to match that of
  // |uniform_descriptor|, and whose type matches the type of the uniform
  // element.  If multiple such constant is exist, the one that is returned
  // is arbitrary.  Returns 0 if no such constant id exists.
  uint32_t GetConstantFromUniformDescriptor(
      opt::IRContext* context,
      const protobufs::UniformBufferElementDescriptor& uniform_descriptor)
      const;

  // Returns all "constant == uniform element" facts known to the fact
  // manager, pairing each fact with id of the type that is associated with
  // both the constant and the uniform element.
  const std::vector<std::pair<protobufs::FactConstantUniform, uint32_t>>&
  GetConstantUniformFactsAndTypes() const;

  // End of uniform constant facts
  //==============================

  //==============================
  // Querying facts about id synonyms

  // Returns every id for which a fact of the form "this id is synonymous with
  // this piece of data" is known.
  std::vector<uint32_t> GetIdsForWhichSynonymsAreKnown() const;

  // Returns the equivalence class of all known synonyms of |id|, or an empty
  // set if no synonyms are known.
  std::vector<const protobufs::DataDescriptor*> GetSynonymsForId(
      uint32_t id) const;

  // Returns the equivalence class of all known synonyms of |data_descriptor|,
  // or empty if no synonyms are known.
  std::vector<const protobufs::DataDescriptor*> GetSynonymsForDataDescriptor(
      const protobufs::DataDescriptor& data_descriptor) const;

  // Returns true if and ony if |data_descriptor1| and |data_descriptor2| are
  // known to be synonymous.
  bool IsSynonymous(const protobufs::DataDescriptor& data_descriptor1,
                    const protobufs::DataDescriptor& data_descriptor2) const;

  // End of id synonym facts
  //==============================

  //==============================
  // Querying facts about dead blocks

  // Returns true if and ony if |block_id| is the id of a block known to be
  // dynamically unreachable.
  bool BlockIsDead(uint32_t block_id) const;

  // End of dead block facts
  //==============================

  //==============================
  // Querying facts about livesafe function

  // Returns true if and ony if |function_id| is the id of a function known
  // to be livesafe.
  bool FunctionIsLivesafe(uint32_t function_id) const;

  // End of dead livesafe function facts
  //==============================

  //==============================
  // Querying facts about pointers with irrelevant pointee values

  // Returns true if and ony if the value of the pointee associated with
  // |pointer_id| is irrelevant.
  bool PointeeValueIsIrrelevant(uint32_t pointer_id) const;

  // End of irrelevant pointee value facts
  //==============================

 private:
  // For each distinct kind of fact to be managed, we use a separate opaque
  // class type.

  class ConstantUniformFacts;  // Opaque class for management of
                               // constant uniform facts.
  std::unique_ptr<ConstantUniformFacts>
      uniform_constant_facts_;  // Unique pointer to internal data.

  class DataSynonymAndIdEquationFacts;  // Opaque class for management of data
                                        // synonym and id equation facts.
  std::unique_ptr<DataSynonymAndIdEquationFacts>
      data_synonym_and_id_equation_facts_;  // Unique pointer to internal data.

  class DeadBlockFacts;  // Opaque class for management of dead block facts.
  std::unique_ptr<DeadBlockFacts>
      dead_block_facts_;  // Unique pointer to internal data.

  class LivesafeFunctionFacts;  // Opaque class for management of livesafe
                                // function facts.
  std::unique_ptr<LivesafeFunctionFacts>
      livesafe_function_facts_;  // Unique pointer to internal data.

  class IrrelevantPointeeValueFacts;  // Opaque class for management of
  // facts about pointers whose pointee values do not matter.
  std::unique_ptr<IrrelevantPointeeValueFacts>
      irrelevant_pointee_value_facts_;  // Unique pointer to internal data.
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FACT_MANAGER_H_
