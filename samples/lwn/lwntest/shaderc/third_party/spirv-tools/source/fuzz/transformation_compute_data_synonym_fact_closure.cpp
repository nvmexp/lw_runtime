// Copyright (c) 2020 Google LLC
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

#include "source/fuzz/transformation_compute_data_synonym_fact_closure.h"

namespace spvtools {
namespace fuzz {

TransformationComputeDataSynonymFactClosure::
    TransformationComputeDataSynonymFactClosure(
        const spvtools::fuzz::protobufs::
            TransformationComputeDataSynonymFactClosure& message)
    : message_(message) {}

TransformationComputeDataSynonymFactClosure::
    TransformationComputeDataSynonymFactClosure(
        uint32_t maximum_equivalence_class_size) {
  message_.set_maximum_equivalence_class_size(maximum_equivalence_class_size);
}

bool TransformationComputeDataSynonymFactClosure::IsApplicable(
    opt::IRContext* /*unused*/, const TransformationContext& /*unused*/) const {
  return true;
}

void TransformationComputeDataSynonymFactClosure::Apply(
    opt::IRContext* ir_context,
    TransformationContext* transformation_context) const {
  transformation_context->GetFactManager()->ComputeClosureOfFacts(
      ir_context, message_.maximum_equivalence_class_size());
}

protobufs::Transformation
TransformationComputeDataSynonymFactClosure::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_compute_data_synonym_fact_closure() = message_;
  return result;
}

}  // namespace fuzz
}  // namespace spvtools
