// Copyright (c) 2016 Google Inc.
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

#include "source/opt/ir_loader.h"

#include <utility>

#include "DebugInfo.h"
#include "OpenCLDebugInfo100.h"
#include "source/ext_inst.h"
#include "source/opt/log.h"
#include "source/opt/reflect.h"
#include "source/util/make_unique.h"

static const uint32_t kExtInstSetIndex = 4;
static const uint32_t kLexicalScopeIndex = 5;
static const uint32_t kInlinedAtIndex = 6;

namespace spvtools {
namespace opt {

IrLoader::IrLoader(const MessageConsumer& consumer, Module* m)
    : consumer_(consumer),
      module_(m),
      source_("<instruction>"),
      inst_index_(0),
      last_dbg_scope_(kNoDebugScope, kNoInlinedAt) {}

bool IrLoader::AddInstruction(const spv_parsed_instruction_t* inst) {
  ++inst_index_;
  const auto opcode = static_cast<SpvOp>(inst->opcode);
  if (IsDebugLineInst(opcode)) {
    dbg_line_info_.push_back(
        Instruction(module()->context(), *inst, last_dbg_scope_));
    return true;
  }

  // If it is a DebugScope or DebugNoScope of debug extension, we do not
  // create a new instruction, but simply keep the information in
  // struct DebugScope.
  if (opcode == SpvOpExtInst && spvExtInstIsDebugInfo(inst->ext_inst_type)) {
    const uint32_t ext_inst_index = inst->words[kExtInstSetIndex];
    if (inst->ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100) {
      const OpenCLDebugInfo100Instructions ext_inst_key =
          OpenCLDebugInfo100Instructions(ext_inst_index);
      if (ext_inst_key == OpenCLDebugInfo100DebugScope) {
        uint32_t inlined_at = 0;
        if (inst->num_words > kInlinedAtIndex)
          inlined_at = inst->words[kInlinedAtIndex];
        last_dbg_scope_ =
            DebugScope(inst->words[kLexicalScopeIndex], inlined_at);
        module()->SetContainsDebugScope();
        return true;
      }
      if (ext_inst_key == OpenCLDebugInfo100DebugNoScope) {
        last_dbg_scope_ = DebugScope(kNoDebugScope, kNoInlinedAt);
        module()->SetContainsDebugScope();
        return true;
      }
    } else {
      const DebugInfoInstructions ext_inst_key =
          DebugInfoInstructions(ext_inst_index);
      if (ext_inst_key == DebugInfoDebugScope) {
        uint32_t inlined_at = 0;
        if (inst->num_words > kInlinedAtIndex)
          inlined_at = inst->words[kInlinedAtIndex];
        last_dbg_scope_ =
            DebugScope(inst->words[kLexicalScopeIndex], inlined_at);
        module()->SetContainsDebugScope();
        return true;
      }
      if (ext_inst_key == DebugInfoDebugNoScope) {
        last_dbg_scope_ = DebugScope(kNoDebugScope, kNoInlinedAt);
        module()->SetContainsDebugScope();
        return true;
      }
    }
  }

  std::unique_ptr<Instruction> spv_inst(
      new Instruction(module()->context(), *inst, std::move(dbg_line_info_)));
  dbg_line_info_.clear();

  const char* src = source_.c_str();
  spv_position_t loc = {inst_index_, 0, 0};

  // Handle function and basic block boundaries first, then normal
  // instructions.
  if (opcode == SpvOpFunction) {
    if (function_ != nullptr) {
      Error(consumer_, src, loc, "function inside function");
      return false;
    }
    function_ = MakeUnique<Function>(std::move(spv_inst));
  } else if (opcode == SpvOpFunctionEnd) {
    if (function_ == nullptr) {
      Error(consumer_, src, loc,
            "OpFunctionEnd without corresponding OpFunction");
      return false;
    }
    if (block_ != nullptr) {
      Error(consumer_, src, loc, "OpFunctionEnd inside basic block");
      return false;
    }
    function_->SetFunctionEnd(std::move(spv_inst));
    module_->AddFunction(std::move(function_));
    function_ = nullptr;
  } else if (opcode == SpvOpLabel) {
    if (function_ == nullptr) {
      Error(consumer_, src, loc, "OpLabel outside function");
      return false;
    }
    if (block_ != nullptr) {
      Error(consumer_, src, loc, "OpLabel inside basic block");
      return false;
    }
    block_ = MakeUnique<BasicBlock>(std::move(spv_inst));
  } else if (IsTerminatorInst(opcode)) {
    if (function_ == nullptr) {
      Error(consumer_, src, loc, "terminator instruction outside function");
      return false;
    }
    if (block_ == nullptr) {
      Error(consumer_, src, loc, "terminator instruction outside basic block");
      return false;
    }
    if (last_dbg_scope_.GetLexicalScope() != kNoDebugScope)
      spv_inst->SetDebugScope(last_dbg_scope_);
    block_->AddInstruction(std::move(spv_inst));
    function_->AddBasicBlock(std::move(block_));
    block_ = nullptr;
    last_dbg_scope_ = DebugScope(kNoDebugScope, kNoInlinedAt);
  } else {
    if (function_ == nullptr) {  // Outside function definition
      SPIRV_ASSERT(consumer_, block_ == nullptr);
      if (opcode == SpvOpCapability) {
        module_->AddCapability(std::move(spv_inst));
      } else if (opcode == SpvOpExtension) {
        module_->AddExtension(std::move(spv_inst));
      } else if (opcode == SpvOpExtInstImport) {
        module_->AddExtInstImport(std::move(spv_inst));
      } else if (opcode == SpvOpMemoryModel) {
        module_->SetMemoryModel(std::move(spv_inst));
      } else if (opcode == SpvOpEntryPoint) {
        module_->AddEntryPoint(std::move(spv_inst));
      } else if (opcode == SpvOpExelwtionMode) {
        module_->AddExelwtionMode(std::move(spv_inst));
      } else if (IsDebug1Inst(opcode)) {
        module_->AddDebug1Inst(std::move(spv_inst));
      } else if (IsDebug2Inst(opcode)) {
        module_->AddDebug2Inst(std::move(spv_inst));
      } else if (IsDebug3Inst(opcode)) {
        module_->AddDebug3Inst(std::move(spv_inst));
      } else if (IsAnnotationInst(opcode)) {
        module_->AddAnnotationInst(std::move(spv_inst));
      } else if (IsTypeInst(opcode)) {
        module_->AddType(std::move(spv_inst));
      } else if (IsConstantInst(opcode) || opcode == SpvOpVariable ||
                 opcode == SpvOpUndef ||
                 (opcode == SpvOpExtInst &&
                  spvExtInstIsNonSemantic(inst->ext_inst_type))) {
        module_->AddGlobalValue(std::move(spv_inst));
      } else if (opcode == SpvOpExtInst &&
                 spvExtInstIsDebugInfo(inst->ext_inst_type)) {
        module_->AddExtInstDebugInfo(std::move(spv_inst));
      } else {
        Errorf(consumer_, src, loc,
               "Unhandled inst type (opcode: %d) found outside function "
               "definition.",
               opcode);
        return false;
      }
    } else {
      if (opcode == SpvOpLoopMerge || opcode == SpvOpSelectionMerge)
        last_dbg_scope_ = DebugScope(kNoDebugScope, kNoInlinedAt);
      if (last_dbg_scope_.GetLexicalScope() != kNoDebugScope)
        spv_inst->SetDebugScope(last_dbg_scope_);
      if (opcode == SpvOpExtInst &&
          spvExtInstIsDebugInfo(inst->ext_inst_type)) {
        const uint32_t ext_inst_index = inst->words[kExtInstSetIndex];
        if (inst->ext_inst_type == SPV_EXT_INST_TYPE_OPENCL_DEBUGINFO_100) {
          const OpenCLDebugInfo100Instructions ext_inst_key =
              OpenCLDebugInfo100Instructions(ext_inst_index);
          switch (ext_inst_key) {
            case OpenCLDebugInfo100DebugDeclare: {
              if (block_ == nullptr)  // Inside function but outside blocks
                function_->AddDebugInstructionInHeader(std::move(spv_inst));
              else
                block_->AddInstruction(std::move(spv_inst));
              break;
            }
            case OpenCLDebugInfo100DebugValue: {
              if (block_ == nullptr)  // Inside function but outside blocks
                function_->AddDebugInstructionInHeader(std::move(spv_inst));
              else
                block_->AddInstruction(std::move(spv_inst));
              break;
            }
            default: {
              Errorf(consumer_, src, loc,
                     "Debug info extension instruction other than DebugScope, "
                     "DebugNoScope, DebugDeclare, and DebugValue found inside "
                     "function",
                     opcode);
              return false;
            }
          }
        } else {
          const DebugInfoInstructions ext_inst_key =
              DebugInfoInstructions(ext_inst_index);
          switch (ext_inst_key) {
            case DebugInfoDebugDeclare: {
              if (block_ == nullptr)  // Inside function but outside blocks
                function_->AddDebugInstructionInHeader(std::move(spv_inst));
              else
                block_->AddInstruction(std::move(spv_inst));
              break;
            }
            case DebugInfoDebugValue: {
              if (block_ == nullptr)  // Inside function but outside blocks
                function_->AddDebugInstructionInHeader(std::move(spv_inst));
              else
                block_->AddInstruction(std::move(spv_inst));
              break;
            }
            default: {
              Errorf(consumer_, src, loc,
                     "Debug info extension instruction other than DebugScope, "
                     "DebugNoScope, DebugDeclare, and DebugValue found inside "
                     "function",
                     opcode);
              return false;
            }
          }
        }
      } else {
        if (block_ == nullptr) {  // Inside function but outside blocks
          if (opcode != SpvOpFunctionParameter) {
            Errorf(consumer_, src, loc,
                   "Non-OpFunctionParameter (opcode: %d) found inside "
                   "function but outside basic block",
                   opcode);
            return false;
          }
          function_->AddParameter(std::move(spv_inst));
        } else {
          block_->AddInstruction(std::move(spv_inst));
        }
      }
    }
  }
  return true;
}

// Resolves internal references among the module, functions, basic blocks, etc.
// This function should be called after adding all instructions.
void IrLoader::EndModule() {
  if (block_ && function_) {
    // We're in the middle of a basic block, but the terminator is missing.
    // Register the block anyway.  This lets us write tests with less
    // boilerplate.
    function_->AddBasicBlock(std::move(block_));
    block_ = nullptr;
  }
  if (function_) {
    // We're in the middle of a function, but the OpFunctionEnd is missing.
    // Register the function anyway.  This lets us write tests with less
    // boilerplate.
    module_->AddFunction(std::move(function_));
    function_ = nullptr;
  }
  for (auto& function : *module_) {
    for (auto& bb : function) bb.SetParent(&function);
  }

  // Copy any trailing Op*Line instruction into the module
  module_->SetTrailingDbgLineInfo(std::move(dbg_line_info_));
}

}  // namespace opt
}  // namespace spvtools
