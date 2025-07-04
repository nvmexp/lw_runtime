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

#ifndef SOURCE_FUZZ_TRANSFORMATION_SET_FUNCTION_CONTROL_H_
#define SOURCE_FUZZ_TRANSFORMATION_SET_FUNCTION_CONTROL_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationSetFunctionControl : public Transformation {
 public:
  explicit TransformationSetFunctionControl(
      const protobufs::TransformationSetFunctionControl& message);

  TransformationSetFunctionControl(uint32_t function_id,
                                   uint32_t function_control);

  // - |message_.function_id| must be the result id of an OpFunction
  //   instruction.
  // - |message_.function_control| must be a function control mask that sets
  //   at most one of 'Inline' or 'DontInline', and that may not contain 'Pure'
  //   (respectively 'Const') unless the existing function control mask contains
  //   'Pure' (respectively 'Const').
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // The function control operand of instruction |message_.function_id| is
  // over-written with |message_.function_control|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  opt::Instruction* FindFunctionDefInstruction(
      opt::IRContext* ir_context) const;

  protobufs::TransformationSetFunctionControl message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_SET_FUNCTION_CONTROL_H_
