// Copyright (c) 2018 Google LLC.
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

#include "source/opt/vector_dce.h"

#include <utility>

namespace spvtools {
namespace opt {
namespace {

const uint32_t kExtractCompositeIdInIdx = 0;
const uint32_t kInsertObjectIdInIdx = 0;
const uint32_t kInsertCompositeIdInIdx = 1;

}  // namespace

Pass::Status VectorDCE::Process() {
  bool modified = false;
  for (Function& function : *get_module()) {
    modified |= VectorDCEFunction(&function);
  }
  return (modified ? Status::SuccessWithChange : Status::SuccessWithoutChange);
}

bool VectorDCE::VectorDCEFunction(Function* function) {
  LiveComponentMap live_components;
  FindLiveComponents(function, &live_components);
  return RewriteInstructions(function, live_components);
}

void VectorDCE::FindLiveComponents(Function* function,
                                   LiveComponentMap* live_components) {
  std::vector<WorkListItem> work_list;

  // Prime the work list.  We will assume that any instruction that does
  // not result in a vector is live.
  //
  // Extending to structures and matrices is not as straight forward because of
  // the nesting.  We cannot simply us a bit vector to keep track of which
  // components are live because of arbitrary nesting of structs.
  function->ForEachInst(
      [&work_list, this, live_components](Instruction* lwrrent_inst) {
        if (!HasVectorOrScalarResult(lwrrent_inst) ||
            !context()->IsCombinatorInstruction(lwrrent_inst)) {
          MarkUsesAsLive(lwrrent_inst, all_components_live_, live_components,
                         &work_list);
        }
      });

  // Process the work list propagating liveness.
  for (uint32_t i = 0; i < work_list.size(); i++) {
    WorkListItem lwrrent_item = work_list[i];
    Instruction* lwrrent_inst = lwrrent_item.instruction;

    switch (lwrrent_inst->opcode()) {
      case SpvOpCompositeExtract:
        MarkExtractUseAsLive(lwrrent_inst, lwrrent_item.components,
                             live_components, &work_list);
        break;
      case SpvOpCompositeInsert:
        MarkInsertUsesAsLive(lwrrent_item, live_components, &work_list);
        break;
      case SpvOpVectorShuffle:
        MarkVectorShuffleUsesAsLive(lwrrent_item, live_components, &work_list);
        break;
      case SpvOpCompositeConstruct:
        MarkCompositeContructUsesAsLive(lwrrent_item, live_components,
                                        &work_list);
        break;
      default:
        if (lwrrent_inst->IsScalarizable()) {
          MarkUsesAsLive(lwrrent_inst, lwrrent_item.components, live_components,
                         &work_list);
        } else {
          MarkUsesAsLive(lwrrent_inst, all_components_live_, live_components,
                         &work_list);
        }
        break;
    }
  }
}

void VectorDCE::MarkExtractUseAsLive(const Instruction* lwrrent_inst,
                                     const utils::BitVector& live_elements,
                                     LiveComponentMap* live_components,
                                     std::vector<WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  uint32_t operand_id =
      lwrrent_inst->GetSingleWordInOperand(kExtractCompositeIdInIdx);
  Instruction* operand_inst = def_use_mgr->GetDef(operand_id);

  if (HasVectorOrScalarResult(operand_inst)) {
    WorkListItem new_item;
    new_item.instruction = operand_inst;
    if (lwrrent_inst->NumInOperands() < 2) {
      new_item.components = live_elements;
    } else {
      new_item.components.Set(lwrrent_inst->GetSingleWordInOperand(1));
    }
    AddItemToWorkListIfNeeded(new_item, live_components, work_list);
  }
}

void VectorDCE::MarkInsertUsesAsLive(
    const VectorDCE::WorkListItem& lwrrent_item,
    LiveComponentMap* live_components,
    std::vector<VectorDCE::WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  if (lwrrent_item.instruction->NumInOperands() > 2) {
    uint32_t insert_position =
        lwrrent_item.instruction->GetSingleWordInOperand(2);

    // Add the elements of the composite object that are used.
    uint32_t operand_id = lwrrent_item.instruction->GetSingleWordInOperand(
        kInsertCompositeIdInIdx);
    Instruction* operand_inst = def_use_mgr->GetDef(operand_id);

    WorkListItem new_item;
    new_item.instruction = operand_inst;
    new_item.components = lwrrent_item.components;
    new_item.components.Clear(insert_position);

    AddItemToWorkListIfNeeded(new_item, live_components, work_list);

    // Add the element being inserted if it is used.
    if (lwrrent_item.components.Get(insert_position)) {
      uint32_t obj_operand_id =
          lwrrent_item.instruction->GetSingleWordInOperand(
              kInsertObjectIdInIdx);
      Instruction* obj_operand_inst = def_use_mgr->GetDef(obj_operand_id);
      WorkListItem new_item_for_obj;
      new_item_for_obj.instruction = obj_operand_inst;
      new_item_for_obj.components.Set(0);
      AddItemToWorkListIfNeeded(new_item_for_obj, live_components, work_list);
    }
  } else {
    // If there are no indices, then this is a copy of the object being
    // inserted.
    uint32_t object_id =
        lwrrent_item.instruction->GetSingleWordInOperand(kInsertObjectIdInIdx);
    Instruction* object_inst = def_use_mgr->GetDef(object_id);

    WorkListItem new_item;
    new_item.instruction = object_inst;
    new_item.components = lwrrent_item.components;
    AddItemToWorkListIfNeeded(new_item, live_components, work_list);
  }
}

void VectorDCE::MarkVectorShuffleUsesAsLive(
    const WorkListItem& lwrrent_item,
    VectorDCE::LiveComponentMap* live_components,
    std::vector<WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  WorkListItem first_operand;
  first_operand.instruction =
      def_use_mgr->GetDef(lwrrent_item.instruction->GetSingleWordInOperand(0));
  WorkListItem second_operand;
  second_operand.instruction =
      def_use_mgr->GetDef(lwrrent_item.instruction->GetSingleWordInOperand(1));

  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Vector* first_type =
      type_mgr->GetType(first_operand.instruction->type_id())->AsVector();
  uint32_t size_of_first_operand = first_type->element_count();

  for (uint32_t in_op = 2; in_op < lwrrent_item.instruction->NumInOperands();
       ++in_op) {
    uint32_t index = lwrrent_item.instruction->GetSingleWordInOperand(in_op);
    if (lwrrent_item.components.Get(in_op - 2)) {
      if (index < size_of_first_operand) {
        first_operand.components.Set(index);
      } else {
        second_operand.components.Set(index - size_of_first_operand);
      }
    }
  }

  AddItemToWorkListIfNeeded(first_operand, live_components, work_list);
  AddItemToWorkListIfNeeded(second_operand, live_components, work_list);
}

void VectorDCE::MarkCompositeContructUsesAsLive(
    VectorDCE::WorkListItem work_item,
    VectorDCE::LiveComponentMap* live_components,
    std::vector<VectorDCE::WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();

  uint32_t lwrrent_component = 0;
  Instruction* lwrrent_inst = work_item.instruction;
  uint32_t num_in_operands = lwrrent_inst->NumInOperands();
  for (uint32_t i = 0; i < num_in_operands; ++i) {
    uint32_t id = lwrrent_inst->GetSingleWordInOperand(i);
    Instruction* op_inst = def_use_mgr->GetDef(id);

    if (HasScalarResult(op_inst)) {
      WorkListItem new_work_item;
      new_work_item.instruction = op_inst;
      if (work_item.components.Get(lwrrent_component)) {
        new_work_item.components.Set(0);
      }
      AddItemToWorkListIfNeeded(new_work_item, live_components, work_list);
      lwrrent_component++;
    } else {
      assert(HasVectorResult(op_inst));
      WorkListItem new_work_item;
      new_work_item.instruction = op_inst;
      uint32_t op_vector_size =
          type_mgr->GetType(op_inst->type_id())->AsVector()->element_count();

      for (uint32_t op_vector_idx = 0; op_vector_idx < op_vector_size;
           op_vector_idx++, lwrrent_component++) {
        if (work_item.components.Get(lwrrent_component)) {
          new_work_item.components.Set(op_vector_idx);
        }
      }
      AddItemToWorkListIfNeeded(new_work_item, live_components, work_list);
    }
  }
}

void VectorDCE::MarkUsesAsLive(
    Instruction* lwrrent_inst, const utils::BitVector& live_elements,
    LiveComponentMap* live_components,
    std::vector<VectorDCE::WorkListItem>* work_list) {
  analysis::DefUseManager* def_use_mgr = context()->get_def_use_mgr();

  lwrrent_inst->ForEachInId([&work_list, &live_elements, this, live_components,
                             def_use_mgr](uint32_t* operand_id) {
    Instruction* operand_inst = def_use_mgr->GetDef(*operand_id);

    if (HasVectorResult(operand_inst)) {
      WorkListItem new_item;
      new_item.instruction = operand_inst;
      new_item.components = live_elements;
      AddItemToWorkListIfNeeded(new_item, live_components, work_list);
    } else if (HasScalarResult(operand_inst)) {
      WorkListItem new_item;
      new_item.instruction = operand_inst;
      new_item.components.Set(0);
      AddItemToWorkListIfNeeded(new_item, live_components, work_list);
    }
  });
}

bool VectorDCE::HasVectorOrScalarResult(const Instruction* inst) const {
  return HasScalarResult(inst) || HasVectorResult(inst);
}

bool VectorDCE::HasVectorResult(const Instruction* inst) const {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  if (inst->type_id() == 0) {
    return false;
  }

  const analysis::Type* lwrrent_type = type_mgr->GetType(inst->type_id());
  switch (lwrrent_type->kind()) {
    case analysis::Type::kVector:
      return true;
    default:
      return false;
  }
}

bool VectorDCE::HasScalarResult(const Instruction* inst) const {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  if (inst->type_id() == 0) {
    return false;
  }

  const analysis::Type* lwrrent_type = type_mgr->GetType(inst->type_id());
  switch (lwrrent_type->kind()) {
    case analysis::Type::kBool:
    case analysis::Type::kInteger:
    case analysis::Type::kFloat:
      return true;
    default:
      return false;
  }
}

bool VectorDCE::RewriteInstructions(
    Function* function, const VectorDCE::LiveComponentMap& live_components) {
  bool modified = false;
  function->ForEachInst(
      [&modified, this, live_components](Instruction* lwrrent_inst) {
        if (!context()->IsCombinatorInstruction(lwrrent_inst)) {
          return;
        }

        auto live_component = live_components.find(lwrrent_inst->result_id());
        if (live_component == live_components.end()) {
          // If this instruction is not in live_components then it does not
          // produce a vector, or it is never referenced and ADCE will remove
          // it.  No point in trying to differentiate.
          return;
        }

        // If no element in the current instruction is used replace it with an
        // OpUndef.
        if (live_component->second.Empty()) {
          modified = true;
          uint32_t undef_id = this->Type2Undef(lwrrent_inst->type_id());
          context()->KillNamesAndDecorates(lwrrent_inst);
          context()->ReplaceAllUsesWith(lwrrent_inst->result_id(), undef_id);
          context()->KillInst(lwrrent_inst);
          return;
        }

        switch (lwrrent_inst->opcode()) {
          case SpvOpCompositeInsert:
            modified |=
                RewriteInsertInstruction(lwrrent_inst, live_component->second);
            break;
          case SpvOpCompositeConstruct:
            // TODO: The members that are not live can be replaced by an undef
            // or constant. This will remove uses of those values, and possibly
            // create opportunities for ADCE.
            break;
          default:
            // Do nothing.
            break;
        }
      });
  return modified;
}

bool VectorDCE::RewriteInsertInstruction(
    Instruction* lwrrent_inst, const utils::BitVector& live_components) {
  // If the value being inserted is not live, then we can skip the insert.

  if (lwrrent_inst->NumInOperands() == 2) {
    // If there are no indices, then this is the same as a copy.
    context()->KillNamesAndDecorates(lwrrent_inst->result_id());
    uint32_t object_id =
        lwrrent_inst->GetSingleWordInOperand(kInsertObjectIdInIdx);
    context()->ReplaceAllUsesWith(lwrrent_inst->result_id(), object_id);
    return true;
  }

  uint32_t insert_index = lwrrent_inst->GetSingleWordInOperand(2);
  if (!live_components.Get(insert_index)) {
    context()->KillNamesAndDecorates(lwrrent_inst->result_id());
    uint32_t composite_id =
        lwrrent_inst->GetSingleWordInOperand(kInsertCompositeIdInIdx);
    context()->ReplaceAllUsesWith(lwrrent_inst->result_id(), composite_id);
    return true;
  }

  // If the values already in the composite are not used, then replace it with
  // an undef.
  utils::BitVector temp = live_components;
  temp.Clear(insert_index);
  if (temp.Empty()) {
    context()->ForgetUses(lwrrent_inst);
    uint32_t undef_id = Type2Undef(lwrrent_inst->type_id());
    lwrrent_inst->SetInOperand(kInsertCompositeIdInIdx, {undef_id});
    context()->AnalyzeUses(lwrrent_inst);
    return true;
  }

  return false;
}

void VectorDCE::AddItemToWorkListIfNeeded(
    WorkListItem work_item, VectorDCE::LiveComponentMap* live_components,
    std::vector<WorkListItem>* work_list) {
  Instruction* lwrrent_inst = work_item.instruction;
  auto it = live_components->find(lwrrent_inst->result_id());
  if (it == live_components->end()) {
    live_components->emplace(
        std::make_pair(lwrrent_inst->result_id(), work_item.components));
    work_list->emplace_back(work_item);
  } else {
    if (it->second.Or(work_item.components)) {
      work_list->emplace_back(work_item);
    }
  }
}

}  // namespace opt
}  // namespace spvtools
