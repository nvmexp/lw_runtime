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

#ifndef SOURCE_FUZZ_FUZZER_UTIL_H_
#define SOURCE_FUZZ_FUZZER_UTIL_H_

#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/opt/basic_block.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Provides types and global utility methods for use by the fuzzer
namespace fuzzerutil {

// Function type that produces a SPIR-V module.
using ModuleSupplier = std::function<std::unique_ptr<opt::IRContext>()>;

// Returns true if and only if the module does not define the given id.
bool IsFreshId(opt::IRContext* context, uint32_t id);

// Updates the module's id bound if needed so that it is large enough to
// account for the given id.
void UpdateModuleIdBound(opt::IRContext* context, uint32_t id);

// Return the block with id |maybe_block_id| if it exists, and nullptr
// otherwise.
opt::BasicBlock* MaybeFindBlock(opt::IRContext* context,
                                uint32_t maybe_block_id);

// When adding an edge from |bb_from| to |bb_to| (which are assumed to be blocks
// in the same function), it is important to supply |bb_to| with ids that can be
// used to augment OpPhi instructions in the case that there is not already such
// an edge.  This function returns true if and only if the ids provided in
// |phi_ids| suffice for this purpose,
bool PhiIdsOkForNewEdge(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& phi_ids);

// Returns the id of a boolean constant with value |value| if it exists in the
// module, or 0 otherwise.
uint32_t MaybeGetBoolConstantId(opt::IRContext* context, bool value);

// Requires that a boolean constant with value |condition_value| is available,
// that PhiIdsOkForNewEdge(context, bb_from, bb_to, phi_ids) holds, and that
// bb_from ends with "OpBranch %some_block".  Turns OpBranch into
// "OpBranchConditional |condition_value| ...", such that control will branch
// to %some_block, with |bb_to| being the unreachable alternative.  Updates
// OpPhi instructions in |bb_to| using |phi_ids| so that the new edge is valid.
void AddUnreachableEdgeAndUpdateOpPhis(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    bool condition_value,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& phi_ids);

// Returns true if and only if |maybe_loop_header_id| is a loop header and
// |block_id| is in the continue construct of the associated loop.
bool BlockIsInLoopContinueConstruct(opt::IRContext* context, uint32_t block_id,
                                    uint32_t maybe_loop_header_id);

// If |block| contains |inst|, an iterator for |inst| is returned.
// Otherwise |block|->end() is returned.
opt::BasicBlock::iterator GetIteratorForInstruction(
    opt::BasicBlock* block, const opt::Instruction* inst);

// Returns true if and only if there is a path to |bb| from the entry block of
// the function that contains |bb|.
bool BlockIsReachableInItsFunction(opt::IRContext* context,
                                   opt::BasicBlock* bb);

// Determines whether it is OK to insert an instruction with opcode |opcode|
// before |instruction_in_block|.
bool CanInsertOpcodeBeforeInstruction(
    SpvOp opcode, const opt::BasicBlock::iterator& instruction_in_block);

// Determines whether it is OK to make a synonym of |inst|.
bool CanMakeSynonymOf(opt::IRContext* ir_context, opt::Instruction* inst);

// Determines whether the given type is a composite; that is: an array, matrix,
// struct or vector.
bool IsCompositeType(const opt::analysis::Type* type);

// Returns a vector containing the same elements as |repeated_field|.
std::vector<uint32_t> RepeatedFieldToVector(
    const google::protobuf::RepeatedField<uint32_t>& repeated_field);

// Given a type id, |base_object_type_id|, returns 0 if the type is not a
// composite type or if |index| is too large to be used as an index into the
// composite.  Otherwise returns the type id of the type associated with the
// composite's index.
//
// Example: if |base_object_type_id| is 10, and we have:
//
// %10 = OpTypeStruct %3 %4 %5
//
// then 3 will be returned if |index| is 0, 5 if |index| is 2, and 0 if index
// is 3 or larger.
uint32_t WalkOneCompositeTypeIndex(opt::IRContext* context,
                                   uint32_t base_object_type_id,
                                   uint32_t index);

// Given a type id, |base_object_type_id|, checks that the given sequence of
// |indices| is suitable for indexing into this type.  Returns the id of the
// type of the final sub-object reached via the indices if they are valid, and
// 0 otherwise.
uint32_t WalkCompositeTypeIndices(
    opt::IRContext* context, uint32_t base_object_type_id,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& indices);

// Returns the number of members associated with |struct_type_instruction|,
// which must be an OpStructType instruction.
uint32_t GetNumberOfStructMembers(
    const opt::Instruction& struct_type_instruction);

// Returns the constant size of the array associated with
// |array_type_instruction|, which must be an OpArrayType instruction. Returns
// 0 if there is not a static size.
uint32_t GetArraySize(const opt::Instruction& array_type_instruction,
                      opt::IRContext* context);

// Returns true if and only if |context| is valid, according to the validator
// instantiated with |validator_options|.
bool IsValid(opt::IRContext* context, spv_validator_options validator_options);

// Returns a clone of |context|, by writing |context| to a binary and then
// parsing it again.
std::unique_ptr<opt::IRContext> CloneIRContext(opt::IRContext* context);

// Returns true if and only if |id| is the id of a type that is not a function
// type.
bool IsNonFunctionTypeId(opt::IRContext* ir_context, uint32_t id);

// Returns true if and only if |block_id| is a merge block or continue target
bool IsMergeOrContinue(opt::IRContext* ir_context, uint32_t block_id);

// Returns the result id of an instruction of the form:
//  %id = OpTypeFunction |type_ids|
// or 0 if no such instruction exists.
uint32_t FindFunctionType(opt::IRContext* ir_context,
                          const std::vector<uint32_t>& type_ids);

// Returns a type instruction (OpTypeFunction) for |function|.
// Returns |nullptr| if type is not found.
opt::Instruction* GetFunctionType(opt::IRContext* context,
                                  const opt::Function* function);

// Returns the function with result id |function_id|, or |nullptr| if no such
// function exists.
opt::Function* FindFunction(opt::IRContext* ir_context, uint32_t function_id);

// Returns |true| if one of entry points has function id |function_id|.
bool FunctionIsEntryPoint(opt::IRContext* context, uint32_t function_id);

// Checks whether |id| is available (according to dominance rules) at the use
// point defined by input operand |use_input_operand_index| of
// |use_instruction|.
bool IdIsAvailableAtUse(opt::IRContext* context,
                        opt::Instruction* use_instruction,
                        uint32_t use_input_operand_index, uint32_t id);

// Checks whether |id| is available (according to dominance rules) at the
// program point directly before |instruction|.
bool IdIsAvailableBeforeInstruction(opt::IRContext* context,
                                    opt::Instruction* instruction, uint32_t id);

// Returns true if and only if |instruction| is an OpFunctionParameter
// associated with |function|.
bool InstructionIsFunctionParameter(opt::Instruction* instruction,
                                    opt::Function* function);

// Returns the type id of the instruction defined by |result_id|, or 0 if there
// is no such result id.
uint32_t GetTypeId(opt::IRContext* context, uint32_t result_id);

// Given |pointer_type_inst|, which must be an OpTypePointer instruction,
// returns the id of the associated pointee type.
uint32_t GetPointeeTypeIdFromPointerType(opt::Instruction* pointer_type_inst);

// Given |pointer_type_id|, which must be the id of a pointer type, returns the
// id of the associated pointee type.
uint32_t GetPointeeTypeIdFromPointerType(opt::IRContext* context,
                                         uint32_t pointer_type_id);

// Given |pointer_type_inst|, which must be an OpTypePointer instruction,
// returns the associated storage class.
SpvStorageClass GetStorageClassFromPointerType(
    opt::Instruction* pointer_type_inst);

// Given |pointer_type_id|, which must be the id of a pointer type, returns the
// associated storage class.
SpvStorageClass GetStorageClassFromPointerType(opt::IRContext* context,
                                               uint32_t pointer_type_id);

// Returns the id of a pointer with pointee type |pointee_type_id| and storage
// class |storage_class|, if it exists, and 0 otherwise.
uint32_t MaybeGetPointerType(opt::IRContext* context, uint32_t pointee_type_id,
                             SpvStorageClass storage_class);

// Returns true if and only if |type| is one of the types for which it is legal
// to have an OpConstantNull value.
bool IsNullConstantSupported(const opt::analysis::Type& type);

}  // namespace fuzzerutil

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_UTIL_H_
