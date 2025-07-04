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

#ifndef SOURCE_FUZZ_TRANSFORMATION_COPY_OBJECT_H_
#define SOURCE_FUZZ_TRANSFORMATION_COPY_OBJECT_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationCopyObject : public Transformation {
 public:
  explicit TransformationCopyObject(
      const protobufs::TransformationCopyObject& message);

  TransformationCopyObject(
      uint32_t object,
      const protobufs::InstructionDescriptor& instruction_to_insert_before,
      uint32_t fresh_id);

  // - |message_.fresh_id| must not be used by the module.
  // - |message_.object| must be a result id that is a legitimate operand for
  //   OpCopyObject.  In particular, it must be the id of an instruction that
  //   has a result type
  // - |message_.object| must not be the target of any decoration.
  //   TODO(afd): consider copying decorations along with objects.
  // - |message_.base_instruction_id| must be the result id of an instruction
  //   'base' in some block 'blk'.
  // - 'blk' must contain an instruction 'inst' located |message_.offset|
  //   instructions after 'base' (if |message_.offset| = 0 then 'inst' =
  //   'base').
  // - It must be legal to insert an OpCopyObject instruction directly
  //   before 'inst'.
  // - |message_.object| must be available directly before 'inst'.
  // - |message_.object| must not be a null pointer or undefined pointer (so as
  //   to make it legal to load from copied pointers).
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - A new instruction,
  //     %|message_.fresh_id| = OpCopyObject %ty %|message_.object|
  //   is added directly before the instruction at |message_.insert_after_id| +
  //   |message_|.offset, where %ty is the type of |message_.object|.
  // - The fact that |message_.fresh_id| and |message_.object| are synonyms
  //   is added to the fact manager in |transformation_context|.
  // - If |message_.object| is a pointer whose pointee value is known to be
  //   irrelevant, the analogous fact is added to the fact manager in
  //   |transformation_context| about |message_.fresh_id|.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationCopyObject message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_COPY_OBJECT_H_
