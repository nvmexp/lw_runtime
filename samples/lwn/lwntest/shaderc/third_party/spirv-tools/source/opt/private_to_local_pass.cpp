// Copyright (c) 2017 Google Inc.
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

#include "source/opt/private_to_local_pass.h"

#include <memory>
#include <utility>
#include <vector>

#include "source/opt/ir_context.h"
#include "source/spirv_constant.h"

namespace spvtools {
namespace opt {
namespace {

const uint32_t kVariableStorageClassInIdx = 0;
const uint32_t kSpvTypePointerTypeIdInIdx = 1;

}  // namespace

Pass::Status PrivateToLocalPass::Process() {
  bool modified = false;

  // Private variables require the shader capability.  If this is not a shader,
  // there is no work to do.
  if (context()->get_feature_mgr()->HasCapability(SpvCapabilityAddresses))
    return Status::SuccessWithoutChange;

  std::vector<std::pair<Instruction*, Function*>> variables_to_move;
  std::unordered_set<uint32_t> localized_variables;
  for (auto& inst : context()->types_values()) {
    if (inst.opcode() != SpvOpVariable) {
      continue;
    }

    if (inst.GetSingleWordInOperand(kVariableStorageClassInIdx) !=
        SpvStorageClassPrivate) {
      continue;
    }

    Function* target_function = FindLocalFunction(inst);
    if (target_function != nullptr) {
      variables_to_move.push_back({&inst, target_function});
    }
  }

  modified = !variables_to_move.empty();
  for (auto p : variables_to_move) {
    if (!MoveVariable(p.first, p.second)) {
      return Status::Failure;
    }
    localized_variables.insert(p.first->result_id());
  }

  if (get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 4)) {
    // In SPIR-V 1.4 and later entry points must list private storage class
    // variables that are statically used by the entry point. Go through the
    // entry points and remove any references to variables that were localized.
    for (auto& entry : get_module()->entry_points()) {
      std::vector<Operand> new_operands;
      for (uint32_t i = 0; i < entry.NumInOperands(); ++i) {
        // Exelwtion model, function id and name are always kept.
        if (i < 3 ||
            !localized_variables.count(entry.GetSingleWordInOperand(i))) {
          new_operands.push_back(entry.GetInOperand(i));
        }
      }
      if (new_operands.size() != entry.NumInOperands()) {
        entry.SetInOperands(std::move(new_operands));
        context()->AnalyzeUses(&entry);
      }
    }
  }

  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

Function* PrivateToLocalPass::FindLocalFunction(const Instruction& inst) const {
  bool found_first_use = false;
  Function* target_function = nullptr;
  context()->get_def_use_mgr()->ForEachUser(
      inst.result_id(),
      [&target_function, &found_first_use, this](Instruction* use) {
        BasicBlock* lwrrent_block = context()->get_instr_block(use);
        if (lwrrent_block == nullptr) {
          return;
        }

        if (!IsValidUse(use)) {
          found_first_use = true;
          target_function = nullptr;
          return;
        }
        Function* lwrrent_function = lwrrent_block->GetParent();
        if (!found_first_use) {
          found_first_use = true;
          target_function = lwrrent_function;
        } else if (target_function != lwrrent_function) {
          target_function = nullptr;
        }
      });
  return target_function;
}  // namespace opt

bool PrivateToLocalPass::MoveVariable(Instruction* variable,
                                      Function* function) {
  // The variable needs to be removed from the global section, and placed in the
  // header of the function.  First step remove from the global list.
  variable->RemoveFromList();
  std::unique_ptr<Instruction> var(variable);  // Take ownership.
  context()->ForgetUses(variable);

  // Update the storage class of the variable.
  variable->SetInOperand(kVariableStorageClassInIdx, {SpvStorageClassFunction});

  // Update the type as well.
  uint32_t new_type_id = GetNewType(variable->type_id());
  if (new_type_id == 0) {
    return false;
  }
  variable->SetResultType(new_type_id);

  // Place the variable at the start of the first basic block.
  context()->AnalyzeUses(variable);
  context()->set_instr_block(variable, &*function->begin());
  function->begin()->begin()->InsertBefore(move(var));

  // Update uses where the type may have changed.
  return UpdateUses(variable->result_id());
}

uint32_t PrivateToLocalPass::GetNewType(uint32_t old_type_id) {
  auto type_mgr = context()->get_type_mgr();
  Instruction* old_type_inst = get_def_use_mgr()->GetDef(old_type_id);
  uint32_t pointee_type_id =
      old_type_inst->GetSingleWordInOperand(kSpvTypePointerTypeIdInIdx);
  uint32_t new_type_id =
      type_mgr->FindPointerToType(pointee_type_id, SpvStorageClassFunction);
  if (new_type_id != 0) {
    context()->UpdateDefUse(context()->get_def_use_mgr()->GetDef(new_type_id));
  }
  return new_type_id;
}

bool PrivateToLocalPass::IsValidUse(const Instruction* inst) const {
  // The cases in this switch have to match the cases in |UpdateUse|.
  // If we don't know how to update it, it is not valid.
  switch (inst->opcode()) {
    case SpvOpLoad:
    case SpvOpStore:
    case SpvOpImageTexelPointer:  // Treat like a load
      return true;
    case SpvOpAccessChain:
      return context()->get_def_use_mgr()->WhileEachUser(
          inst, [this](const Instruction* user) {
            if (!IsValidUse(user)) return false;
            return true;
          });
    case SpvOpName:
      return true;
    default:
      return spvOpcodeIsDecoration(inst->opcode());
  }
}

bool PrivateToLocalPass::UpdateUse(Instruction* inst) {
  // The cases in this switch have to match the cases in |IsValidUse|.  If we
  // don't think it is valid, the optimization will not view the variable as a
  // candidate, and therefore the use will not be updated.
  switch (inst->opcode()) {
    case SpvOpLoad:
    case SpvOpStore:
    case SpvOpImageTexelPointer:  // Treat like a load
      // The type is fine because it is the type pointed to, and that does not
      // change.
      break;
    case SpvOpAccessChain: {
      context()->ForgetUses(inst);
      uint32_t new_type_id = GetNewType(inst->type_id());
      if (new_type_id == 0) {
        return false;
      }
      inst->SetResultType(new_type_id);
      context()->AnalyzeUses(inst);

      // Update uses where the type may have changed.
      if (!UpdateUses(inst->result_id())) {
        return false;
      }
    } break;
    case SpvOpName:
    case SpvOpEntryPoint:  // entry points will be updated separately.
      break;
    default:
      assert(spvOpcodeIsDecoration(inst->opcode()) &&
             "Do not know how to update the type for this instruction.");
      break;
  }
  return true;
}

bool PrivateToLocalPass::UpdateUses(uint32_t id) {
  std::vector<Instruction*> uses;
  context()->get_def_use_mgr()->ForEachUser(
      id, [&uses](Instruction* use) { uses.push_back(use); });

  for (Instruction* use : uses) {
    if (!UpdateUse(use)) {
      return false;
    }
  }
  return true;
}

}  // namespace opt
}  // namespace spvtools
