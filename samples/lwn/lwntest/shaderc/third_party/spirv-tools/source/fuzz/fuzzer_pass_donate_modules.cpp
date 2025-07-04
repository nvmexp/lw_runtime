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

#include "source/fuzz/fuzzer_pass_donate_modules.h"

#include <map>
#include <queue>
#include <set>

#include "source/fuzz/call_graph.h"
#include "source/fuzz/instruction_message.h"
#include "source/fuzz/transformation_add_constant_boolean.h"
#include "source/fuzz/transformation_add_constant_composite.h"
#include "source/fuzz/transformation_add_constant_null.h"
#include "source/fuzz/transformation_add_constant_scalar.h"
#include "source/fuzz/transformation_add_function.h"
#include "source/fuzz/transformation_add_global_undef.h"
#include "source/fuzz/transformation_add_global_variable.h"
#include "source/fuzz/transformation_add_type_array.h"
#include "source/fuzz/transformation_add_type_boolean.h"
#include "source/fuzz/transformation_add_type_float.h"
#include "source/fuzz/transformation_add_type_function.h"
#include "source/fuzz/transformation_add_type_int.h"
#include "source/fuzz/transformation_add_type_matrix.h"
#include "source/fuzz/transformation_add_type_pointer.h"
#include "source/fuzz/transformation_add_type_struct.h"
#include "source/fuzz/transformation_add_type_vector.h"

namespace spvtools {
namespace fuzz {

FuzzerPassDonateModules::FuzzerPassDonateModules(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    const std::vector<fuzzerutil::ModuleSupplier>& donor_suppliers)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations),
      donor_suppliers_(donor_suppliers) {}

FuzzerPassDonateModules::~FuzzerPassDonateModules() = default;

void FuzzerPassDonateModules::Apply() {
  // If there are no donor suppliers, this fuzzer pass is a no-op.
  if (donor_suppliers_.empty()) {
    return;
  }

  // Donate at least one module, and probabilistically decide when to stop
  // donating modules.
  do {
    // Choose a donor supplier at random, and get the module that it provides.
    std::unique_ptr<opt::IRContext> donor_ir_context = donor_suppliers_.at(
        GetFuzzerContext()->RandomIndex(donor_suppliers_))();
    assert(donor_ir_context != nullptr && "Supplying of donor failed");
    assert(fuzzerutil::IsValid(
               donor_ir_context.get(),
               GetTransformationContext()->GetValidatorOptions()) &&
           "The donor module must be valid");
    // Donate the supplied module.
    //
    // Randomly decide whether to make the module livesafe (see
    // FactFunctionIsLivesafe); doing so allows it to be used for live code
    // injection but restricts its behaviour to allow this, and means that its
    // functions cannot be transformed as if they were arbitrary dead code.
    bool make_livesafe = GetFuzzerContext()->ChoosePercentage(
        GetFuzzerContext()->ChanceOfMakingDonorLivesafe());
    DonateSingleModule(donor_ir_context.get(), make_livesafe);
  } while (GetFuzzerContext()->ChoosePercentage(
      GetFuzzerContext()->GetChanceOfDonatingAdditionalModule()));
}

void FuzzerPassDonateModules::DonateSingleModule(
    opt::IRContext* donor_ir_context, bool make_livesafe) {
  // The ids used by the donor module may very well clash with ids defined in
  // the recipient module.  Furthermore, some instructions defined in the donor
  // module will be equivalent to instructions defined in the recipient module,
  // and it is not always legal to re-declare equivalent instructions.  For
  // example, OpTypeVoid cannot be declared twice.
  //
  // To handle this, we maintain a mapping from an id used in the donor module
  // to the corresponding id that will be used by the donated code when it
  // appears in the recipient module.
  //
  // This mapping is populated in two ways:
  // (1) by mapping a donor instruction's result id to the id of some equivalent
  //     existing instruction in the recipient (e.g. this has to be done for
  //     OpTypeVoid)
  // (2) by mapping a donor instruction's result id to a freshly chosen id that
  //     is guaranteed to be different from any id already used by the recipient
  //     (or from any id already chosen to handle a previous donor id)
  std::map<uint32_t, uint32_t> original_id_to_donated_id;

  HandleExternalInstructionImports(donor_ir_context,
                                   &original_id_to_donated_id);
  HandleTypesAndValues(donor_ir_context, &original_id_to_donated_id);
  HandleFunctions(donor_ir_context, &original_id_to_donated_id, make_livesafe);

  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3115) Handle some
  //  kinds of decoration.
}

SpvStorageClass FuzzerPassDonateModules::AdaptStorageClass(
    SpvStorageClass donor_storage_class) {
  switch (donor_storage_class) {
    case SpvStorageClassFunction:
    case SpvStorageClassPrivate:
    case SpvStorageClassWorkgroup:
      // We leave these alone
      return donor_storage_class;
    case SpvStorageClassInput:
    case SpvStorageClassOutput:
    case SpvStorageClassUniform:
    case SpvStorageClassUniformConstant:
    case SpvStorageClassPushConstant:
    case SpvStorageClassImage:
      // We change these to Private
      return SpvStorageClassPrivate;
    default:
      // Handle other cases on demand.
      assert(false && "Lwrrently unsupported storage class.");
      return SpvStorageClassMax;
  }
}

void FuzzerPassDonateModules::HandleExternalInstructionImports(
    opt::IRContext* donor_ir_context,
    std::map<uint32_t, uint32_t>* original_id_to_donated_id) {
  // Consider every external instruction set import in the donor module.
  for (auto& donor_import : donor_ir_context->module()->ext_inst_imports()) {
    const auto& donor_import_name_words = donor_import.GetInOperand(0).words;
    // Look for an identical import in the recipient module.
    for (auto& existing_import : GetIRContext()->module()->ext_inst_imports()) {
      const auto& existing_import_name_words =
          existing_import.GetInOperand(0).words;
      if (donor_import_name_words == existing_import_name_words) {
        // A matching import has found.  Map the result id for the donor import
        // to the id of the existing import, so that when donor instructions
        // rely on the import they will be rewritten to use the existing import.
        original_id_to_donated_id->insert(
            {donor_import.result_id(), existing_import.result_id()});
        break;
      }
    }
    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3116): At present
    //  we do not handle donation of instruction imports, i.e. we do not allow
    //  the donor to import instruction sets that the recipient did not already
    //  import.  It might be a good idea to allow this, but it requires some
    //  thought.
    assert(original_id_to_donated_id->count(donor_import.result_id()) &&
           "Donation of imports is not yet supported.");
  }
}

void FuzzerPassDonateModules::HandleTypesAndValues(
    opt::IRContext* donor_ir_context,
    std::map<uint32_t, uint32_t>* original_id_to_donated_id) {
  // Consider every type/global/constant/undef in the module.
  for (auto& type_or_value : donor_ir_context->module()->types_values()) {
    HandleTypeOrValue(type_or_value, original_id_to_donated_id);
  }
}

void FuzzerPassDonateModules::HandleTypeOrValue(
    const opt::Instruction& type_or_value,
    std::map<uint32_t, uint32_t>* original_id_to_donated_id) {
  // The type/value instruction generates a result id, and we need to associate
  // the donor's result id with a new result id.  That new result id will either
  // be the id of some existing instruction, or a fresh id.  This variable
  // captures it.
  uint32_t new_result_id;

  // Decide how to handle each kind of instruction on a case-by-case basis.
  //
  // Because the donor module is required to be valid, when we encounter a
  // type comprised of component types (e.g. an aggregate or pointer), we know
  // that its component types will have been considered previously, and that
  // |original_id_to_donated_id| will already contain an entry for them.
  switch (type_or_value.opcode()) {
    case SpvOpTypeImage:
    case SpvOpTypeSampledImage:
    case SpvOpTypeSampler:
      // We do not donate types and variables that relate to images and
      // samplers, so we skip these types and subsequently skip anything that
      // depends on them.
      return;
    case SpvOpTypeVoid: {
      // Void has to exist already in order for us to have an entry point.
      // Get the existing id of void.
      opt::analysis::Void void_type;
      new_result_id = GetIRContext()->get_type_mgr()->GetId(&void_type);
      assert(new_result_id &&
             "The module being transformed will always have 'void' type "
             "declared.");
    } break;
    case SpvOpTypeBool: {
      // Bool cannot be declared multiple times, so use its existing id if
      // present, or add a declaration of Bool with a fresh id if not.
      opt::analysis::Bool bool_type;
      auto bool_type_id = GetIRContext()->get_type_mgr()->GetId(&bool_type);
      if (bool_type_id) {
        new_result_id = bool_type_id;
      } else {
        new_result_id = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationAddTypeBoolean(new_result_id));
      }
    } break;
    case SpvOpTypeInt: {
      // Int cannot be declared multiple times with the same width and
      // signedness, so check whether an existing identical Int type is
      // present and use its id if so.  Otherwise add a declaration of the
      // Int type used by the donor, with a fresh id.
      const uint32_t width = type_or_value.GetSingleWordInOperand(0);
      const bool is_signed =
          static_cast<bool>(type_or_value.GetSingleWordInOperand(1));
      opt::analysis::Integer int_type(width, is_signed);
      auto int_type_id = GetIRContext()->get_type_mgr()->GetId(&int_type);
      if (int_type_id) {
        new_result_id = int_type_id;
      } else {
        new_result_id = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(
            TransformationAddTypeInt(new_result_id, width, is_signed));
      }
    } break;
    case SpvOpTypeFloat: {
      // Similar to SpvOpTypeInt.
      const uint32_t width = type_or_value.GetSingleWordInOperand(0);
      opt::analysis::Float float_type(width);
      auto float_type_id = GetIRContext()->get_type_mgr()->GetId(&float_type);
      if (float_type_id) {
        new_result_id = float_type_id;
      } else {
        new_result_id = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationAddTypeFloat(new_result_id, width));
      }
    } break;
    case SpvOpTypeVector: {
      // It is not legal to have two Vector type declarations with identical
      // element types and element counts, so check whether an existing
      // identical Vector type is present and use its id if so.  Otherwise add
      // a declaration of the Vector type used by the donor, with a fresh id.

      // When considering the vector's component type id, we look up the id
      // use in the donor to find the id to which this has been remapped.
      uint32_t component_type_id = original_id_to_donated_id->at(
          type_or_value.GetSingleWordInOperand(0));
      auto component_type =
          GetIRContext()->get_type_mgr()->GetType(component_type_id);
      assert(component_type && "The base type should be registered.");
      auto component_count = type_or_value.GetSingleWordInOperand(1);
      opt::analysis::Vector vector_type(component_type, component_count);
      auto vector_type_id = GetIRContext()->get_type_mgr()->GetId(&vector_type);
      if (vector_type_id) {
        new_result_id = vector_type_id;
      } else {
        new_result_id = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationAddTypeVector(
            new_result_id, component_type_id, component_count));
      }
    } break;
    case SpvOpTypeMatrix: {
      // Similar to SpvOpTypeVector.
      uint32_t column_type_id = original_id_to_donated_id->at(
          type_or_value.GetSingleWordInOperand(0));
      auto column_type =
          GetIRContext()->get_type_mgr()->GetType(column_type_id);
      assert(column_type && column_type->AsVector() &&
             "The column type should be a registered vector type.");
      auto column_count = type_or_value.GetSingleWordInOperand(1);
      opt::analysis::Matrix matrix_type(column_type, column_count);
      auto matrix_type_id = GetIRContext()->get_type_mgr()->GetId(&matrix_type);
      if (matrix_type_id) {
        new_result_id = matrix_type_id;
      } else {
        new_result_id = GetFuzzerContext()->GetFreshId();
        ApplyTransformation(TransformationAddTypeMatrix(
            new_result_id, column_type_id, column_count));
      }

    } break;
    case SpvOpTypeArray: {
      // It is OK to have multiple structurally identical array types, so
      // we go ahead and add a remapped version of the type declared by the
      // donor.
      uint32_t component_type_id = type_or_value.GetSingleWordInOperand(0);
      if (!original_id_to_donated_id->count(component_type_id)) {
        // We did not donate the component type of this array type, so we
        // cannot donate the array type.
        return;
      }
      new_result_id = GetFuzzerContext()->GetFreshId();
      ApplyTransformation(TransformationAddTypeArray(
          new_result_id, original_id_to_donated_id->at(component_type_id),
          original_id_to_donated_id->at(
              type_or_value.GetSingleWordInOperand(1))));
    } break;
    case SpvOpTypeRuntimeArray: {
      // A runtime array is allowed as the final member of an SSBO.  During
      // donation we turn runtime arrays into fixed-size arrays.  For dead
      // code donations this is OK because the array is never indexed into at
      // runtime, so it does not matter what its size is.  For live-safe code,
      // all accesses are made in-bounds, so this is also OK.
      //
      // The special OpArrayLength instruction, which works on runtime arrays,
      // is rewritten to yield the fixed length that is used for the array.

      uint32_t component_type_id = type_or_value.GetSingleWordInOperand(0);
      if (!original_id_to_donated_id->count(component_type_id)) {
        // We did not donate the component type of this runtime array type, so
        // we cannot donate it as a fixed-size array.
        return;
      }
      new_result_id = GetFuzzerContext()->GetFreshId();
      ApplyTransformation(TransformationAddTypeArray(
          new_result_id, original_id_to_donated_id->at(component_type_id),
          FindOrCreate32BitIntegerConstant(
              GetFuzzerContext()->GetRandomSizeForNewArray(), false)));
    } break;
    case SpvOpTypeStruct: {
      // Similar to SpvOpTypeArray.
      std::vector<uint32_t> member_type_ids;
      for (uint32_t i = 0; i < type_or_value.NumInOperands(); i++) {
        auto component_type_id = type_or_value.GetSingleWordInOperand(i);
        if (!original_id_to_donated_id->count(component_type_id)) {
          // We did not donate every member type for this struct type, so we
          // cannot donate the struct type.
          return;
        }
        member_type_ids.push_back(
            original_id_to_donated_id->at(component_type_id));
      }
      new_result_id = GetFuzzerContext()->GetFreshId();
      ApplyTransformation(
          TransformationAddTypeStruct(new_result_id, member_type_ids));
    } break;
    case SpvOpTypePointer: {
      // Similar to SpvOpTypeArray.
      uint32_t pointee_type_id = type_or_value.GetSingleWordInOperand(1);
      if (!original_id_to_donated_id->count(pointee_type_id)) {
        // We did not donate the pointee type for this pointer type, so we
        // cannot donate the pointer type.
        return;
      }
      new_result_id = GetFuzzerContext()->GetFreshId();
      ApplyTransformation(TransformationAddTypePointer(
          new_result_id,
          AdaptStorageClass(static_cast<SpvStorageClass>(
              type_or_value.GetSingleWordInOperand(0))),
          original_id_to_donated_id->at(pointee_type_id)));
    } break;
    case SpvOpTypeFunction: {
      // It is not OK to have multiple function types that use identical ids
      // for their return and parameter types.  We thus go through all
      // existing function types to look for a match.  We do not use the
      // type manager here because we want to regard two function types that
      // are structurally identical but that differ with respect to the
      // actual ids used for pointer types as different.
      //
      // Example:
      //
      // %1 = OpTypeVoid
      // %2 = OpTypeInt 32 0
      // %3 = OpTypePointer Function %2
      // %4 = OpTypePointer Function %2
      // %5 = OpTypeFunction %1 %3
      // %6 = OpTypeFunction %1 %4
      //
      // We regard %5 and %6 as distinct function types here, even though
      // they both have the form "uint32* -> void"

      std::vector<uint32_t> return_and_parameter_types;
      for (uint32_t i = 0; i < type_or_value.NumInOperands(); i++) {
        uint32_t return_or_parameter_type =
            type_or_value.GetSingleWordInOperand(i);
        if (!original_id_to_donated_id->count(return_or_parameter_type)) {
          // We did not donate every return/parameter type for this function
          // type, so we cannot donate the function type.
          return;
        }
        return_and_parameter_types.push_back(
            original_id_to_donated_id->at(return_or_parameter_type));
      }
      uint32_t existing_function_id = fuzzerutil::FindFunctionType(
          GetIRContext(), return_and_parameter_types);
      if (existing_function_id) {
        new_result_id = existing_function_id;
      } else {
        // No match was found, so add a remapped version of the function type
        // to the module, with a fresh id.
        new_result_id = GetFuzzerContext()->GetFreshId();
        std::vector<uint32_t> argument_type_ids;
        for (uint32_t i = 1; i < type_or_value.NumInOperands(); i++) {
          argument_type_ids.push_back(original_id_to_donated_id->at(
              type_or_value.GetSingleWordInOperand(i)));
        }
        ApplyTransformation(TransformationAddTypeFunction(
            new_result_id,
            original_id_to_donated_id->at(
                type_or_value.GetSingleWordInOperand(0)),
            argument_type_ids));
      }
    } break;
    case SpvOpConstantTrue:
    case SpvOpConstantFalse: {
      // It is OK to have duplicate definitions of True and False, so add
      // these to the module, using a remapped Bool type.
      new_result_id = GetFuzzerContext()->GetFreshId();
      ApplyTransformation(TransformationAddConstantBoolean(
          new_result_id, type_or_value.opcode() == SpvOpConstantTrue));
    } break;
    case SpvOpConstant: {
      // It is OK to have duplicate constant definitions, so add this to the
      // module using a remapped result type.
      new_result_id = GetFuzzerContext()->GetFreshId();
      std::vector<uint32_t> data_words;
      type_or_value.ForEachInOperand([&data_words](const uint32_t* in_operand) {
        data_words.push_back(*in_operand);
      });
      ApplyTransformation(TransformationAddConstantScalar(
          new_result_id, original_id_to_donated_id->at(type_or_value.type_id()),
          data_words));
    } break;
    case SpvOpConstantComposite: {
      assert(original_id_to_donated_id->count(type_or_value.type_id()) &&
             "Composite types for which it is possible to create a constant "
             "should have been donated.");

      // It is OK to have duplicate constant composite definitions, so add
      // this to the module using remapped versions of all consituent ids and
      // the result type.
      new_result_id = GetFuzzerContext()->GetFreshId();
      std::vector<uint32_t> constituent_ids;
      type_or_value.ForEachInId([&constituent_ids, &original_id_to_donated_id](
                                    const uint32_t* constituent_id) {
        assert(original_id_to_donated_id->count(*constituent_id) &&
               "The constants used to construct this composite should "
               "have been donated.");
        constituent_ids.push_back(
            original_id_to_donated_id->at(*constituent_id));
      });
      ApplyTransformation(TransformationAddConstantComposite(
          new_result_id, original_id_to_donated_id->at(type_or_value.type_id()),
          constituent_ids));
    } break;
    case SpvOpConstantNull: {
      if (!original_id_to_donated_id->count(type_or_value.type_id())) {
        // We did not donate the type associated with this null constant, so
        // we cannot donate the null constant.
        return;
      }

      // It is fine to have multiple OpConstantNull instructions of the same
      // type, so we just add this to the recipient module.
      new_result_id = GetFuzzerContext()->GetFreshId();
      ApplyTransformation(TransformationAddConstantNull(
          new_result_id,
          original_id_to_donated_id->at(type_or_value.type_id())));
    } break;
    case SpvOpVariable: {
      if (!original_id_to_donated_id->count(type_or_value.type_id())) {
        // We did not donate the pointer type associated with this variable,
        // so we cannot donate the variable.
        return;
      }

      // This is a global variable that could have one of various storage
      // classes.  However, we change all global variable pointer storage
      // classes (such as Uniform, Input and Output) to private when donating
      // pointer types, with the exception of the Workgroup storage class.
      //
      // Thus this variable's pointer type is guaranteed to have storage class
      // Private or Workgroup.
      //
      // We add a global variable with either Private or Workgroup storage
      // class, using remapped versions of the result type and initializer ids
      // for the global variable in the donor.
      //
      // We regard the added variable as having an irrelevant value.  This
      // means that future passes can add stores to the variable in any
      // way they wish, and pass them as pointer parameters to functions
      // without worrying about whether their data might get modified.
      new_result_id = GetFuzzerContext()->GetFreshId();
      uint32_t remapped_pointer_type =
          original_id_to_donated_id->at(type_or_value.type_id());
      uint32_t initializer_id;
      SpvStorageClass storage_class =
          static_cast<SpvStorageClass>(type_or_value.GetSingleWordInOperand(
              0)) == SpvStorageClassWorkgroup
              ? SpvStorageClassWorkgroup
              : SpvStorageClassPrivate;
      if (type_or_value.NumInOperands() == 1) {
        // The variable did not have an initializer.  Initialize it to zero
        // if it has Private storage class (to limit problems associated with
        // uninitialized data), and leave it uninitialized if it has Workgroup
        // storage class (as Workgroup variables cannot have initializers).

        // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3275): we
        //  could initialize Workgroup variables at the start of an entry
        //  point, and should do so if their uninitialized nature proves
        //  problematic.
        initializer_id = storage_class == SpvStorageClassWorkgroup
                             ? 0
                             : FindOrCreateZeroConstant(
                                   fuzzerutil::GetPointeeTypeIdFromPointerType(
                                       GetIRContext(), remapped_pointer_type));
      } else {
        // The variable already had an initializer; use its remapped id.
        initializer_id = original_id_to_donated_id->at(
            type_or_value.GetSingleWordInOperand(1));
      }
      ApplyTransformation(
          TransformationAddGlobalVariable(new_result_id, remapped_pointer_type,
                                          storage_class, initializer_id, true));
    } break;
    case SpvOpUndef: {
      if (!original_id_to_donated_id->count(type_or_value.type_id())) {
        // We did not donate the type associated with this undef, so we cannot
        // donate the undef.
        return;
      }

      // It is fine to have multiple Undef instructions of the same type, so
      // we just add this to the recipient module.
      new_result_id = GetFuzzerContext()->GetFreshId();
      ApplyTransformation(TransformationAddGlobalUndef(
          new_result_id,
          original_id_to_donated_id->at(type_or_value.type_id())));
    } break;
    default: {
      assert(0 && "Unknown type/value.");
      new_result_id = 0;
    } break;
  }

  // Update the id mapping to associate the instruction's result id with its
  // corresponding id in the recipient.
  original_id_to_donated_id->insert({type_or_value.result_id(), new_result_id});
}

void FuzzerPassDonateModules::HandleFunctions(
    opt::IRContext* donor_ir_context,
    std::map<uint32_t, uint32_t>* original_id_to_donated_id,
    bool make_livesafe) {
  // Get the ids of functions in the donor module, topologically sorted
  // according to the donor's call graph.
  auto topological_order =
      GetFunctionsInCallGraphTopologicalOrder(donor_ir_context);

  // Donate the functions in reverse topological order.  This ensures that a
  // function gets donated before any function that depends on it.  This allows
  // donation of the functions to be separated into a number of transformations,
  // each adding one function, such that every prefix of transformations leaves
  // the module valid.
  for (auto function_id = topological_order.rbegin();
       function_id != topological_order.rend(); ++function_id) {
    // Find the function to be donated.
    opt::Function* function_to_donate = nullptr;
    for (auto& function : *donor_ir_context->module()) {
      if (function.result_id() == *function_id) {
        function_to_donate = &function;
        break;
      }
    }
    assert(function_to_donate && "Function to be donated was not found.");

    if (!original_id_to_donated_id->count(
            function_to_donate->DefInst().GetSingleWordInOperand(1))) {
      // We were not able to donate this function's type, so we cannot donate
      // the function.
      continue;
    }

    // We will collect up protobuf messages representing the donor function's
    // instructions here, and use them to create an AddFunction transformation.
    std::vector<protobufs::Instruction> donated_instructions;

    // This set tracks the ids of those instructions for which donation was
    // completely skipped: neither the instruction nor a substitute for it was
    // donated.
    std::set<uint32_t> skipped_instructions;

    // Consider every instruction of the donor function.
    function_to_donate->ForEachInst(
        [this, &donated_instructions, donor_ir_context,
         &original_id_to_donated_id,
         &skipped_instructions](const opt::Instruction* instruction) {
          if (instruction->opcode() == SpvOpArrayLength) {
            // We treat OpArrayLength specially.
            HandleOpArrayLength(*instruction, original_id_to_donated_id,
                                &donated_instructions);
          } else if (!CanDonateInstruction(donor_ir_context, *instruction,
                                           *original_id_to_donated_id,
                                           skipped_instructions)) {
            // This is an instruction that we cannot directly donate.
            HandleDiffilwltInstruction(*instruction, original_id_to_donated_id,
                                       &donated_instructions,
                                       &skipped_instructions);
          } else {
            PrepareInstructionForDonation(*instruction, donor_ir_context,
                                          original_id_to_donated_id,
                                          &donated_instructions);
          }
        });

    if (make_livesafe) {
      // Make the function livesafe and then add it.
      AddLivesafeFunction(*function_to_donate, donor_ir_context,
                          *original_id_to_donated_id, donated_instructions);
    } else {
      // Add the function in a non-livesafe manner.
      ApplyTransformation(TransformationAddFunction(donated_instructions));
    }
  }
}

bool FuzzerPassDonateModules::CanDonateInstruction(
    opt::IRContext* donor_ir_context, const opt::Instruction& instruction,
    const std::map<uint32_t, uint32_t>& original_id_to_donated_id,
    const std::set<uint32_t>& skipped_instructions) const {
  if (instruction.type_id() &&
      !original_id_to_donated_id.count(instruction.type_id())) {
    // We could not donate the result type of this instruction, so we cannot
    // donate the instruction.
    return false;
  }

  // Now consider instructions we specifically want to skip because we do not
  // yet support them.
  switch (instruction.opcode()) {
    case SpvOpAtomicLoad:
    case SpvOpAtomicStore:
    case SpvOpAtomicExchange:
    case SpvOpAtomicCompareExchange:
    case SpvOpAtomicCompareExchangeWeak:
    case SpvOpAtomicIIncrement:
    case SpvOpAtomicIDecrement:
    case SpvOpAtomicIAdd:
    case SpvOpAtomicISub:
    case SpvOpAtomicSMin:
    case SpvOpAtomilwMin:
    case SpvOpAtomicSMax:
    case SpvOpAtomilwMax:
    case SpvOpAtomicAnd:
    case SpvOpAtomicOr:
    case SpvOpAtomicXor:
      // We conservatively ignore all atomic instructions at present.
      // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3276): Consider
      //  being less conservative here.
    case SpvOpImageSampleImplicitLod:
    case SpvOpImageSampleExplicitLod:
    case SpvOpImageSampleDrefImplicitLod:
    case SpvOpImageSampleDrefExplicitLod:
    case SpvOpImageSampleProjImplicitLod:
    case SpvOpImageSampleProjExplicitLod:
    case SpvOpImageSampleProjDrefImplicitLod:
    case SpvOpImageSampleProjDrefExplicitLod:
    case SpvOpImageFetch:
    case SpvOpImageGather:
    case SpvOpImageDrefGather:
    case SpvOpImageRead:
    case SpvOpImageWrite:
    case SpvOpImageSparseSampleImplicitLod:
    case SpvOpImageSparseSampleExplicitLod:
    case SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOpImageSparseSampleDrefExplicitLod:
    case SpvOpImageSparseSampleProjImplicitLod:
    case SpvOpImageSparseSampleProjExplicitLod:
    case SpvOpImageSparseSampleProjDrefImplicitLod:
    case SpvOpImageSparseSampleProjDrefExplicitLod:
    case SpvOpImageSparseFetch:
    case SpvOpImageSparseGather:
    case SpvOpImageSparseDrefGather:
    case SpvOpImageSparseRead:
    case SpvOpImageSampleFootprintLW:
    case SpvOpImage:
    case SpvOpImageQueryFormat:
    case SpvOpImageQueryLevels:
    case SpvOpImageQueryLod:
    case SpvOpImageQueryOrder:
    case SpvOpImageQuerySamples:
    case SpvOpImageQuerySize:
    case SpvOpImageQuerySizeLod:
    case SpvOpSampledImage:
      // We ignore all instructions related to accessing images, since we do not
      // donate images.
      return false;
    case SpvOpLoad:
      switch (donor_ir_context->get_def_use_mgr()
                  ->GetDef(instruction.type_id())
                  ->opcode()) {
        case SpvOpTypeImage:
        case SpvOpTypeSampledImage:
        case SpvOpTypeSampler:
          // Again, we ignore instructions that relate to accessing images.
          return false;
        default:
          break;
      }
    default:
      break;
  }

  // Examine each id input operand to the instruction.  If it turns out that we
  // have skipped any of these operands then we cannot donate the instruction.
  bool result = true;
  instruction.WhileEachInId(
      [donor_ir_context, &original_id_to_donated_id, &result,
       &skipped_instructions](const uint32_t* in_id) -> bool {
        if (!original_id_to_donated_id.count(*in_id)) {
          // We do not have a mapped result id for this id operand.  That either
          // means that it is a forward reference (which is OK), that we skipped
          // the instruction that generated it (which is not OK), or that it is
          // the id of a function or global value that we did not donate (which
          // is not OK).  We check for the latter two cases.
          if (skipped_instructions.count(*in_id) ||
              // A function or global value does not have an associated basic
              // block.
              !donor_ir_context->get_instr_block(*in_id)) {
            result = false;
            return false;
          }
        }
        return true;
      });
  return result;
}

bool FuzzerPassDonateModules::IsBasicType(
    const opt::Instruction& instruction) const {
  switch (instruction.opcode()) {
    case SpvOpTypeArray:
    case SpvOpTypeFloat:
    case SpvOpTypeInt:
    case SpvOpTypeMatrix:
    case SpvOpTypeStruct:
    case SpvOpTypeVector:
      return true;
    default:
      return false;
  }
}

std::vector<uint32_t>
FuzzerPassDonateModules::GetFunctionsInCallGraphTopologicalOrder(
    opt::IRContext* context) {
  CallGraph call_graph(context);

  // This is an implementation of Kahn’s algorithm for topological sorting.

  // This is the sorted order of function ids that we will eventually return.
  std::vector<uint32_t> result;

  // Get a copy of the initial in-degrees of all functions.  The algorithm
  // ilwolves decrementing these values, hence why we work on a copy.
  std::map<uint32_t, uint32_t> function_in_degree =
      call_graph.GetFunctionInDegree();

  // Populate a queue with all those function ids with in-degree zero.
  std::queue<uint32_t> queue;
  for (auto& entry : function_in_degree) {
    if (entry.second == 0) {
      queue.push(entry.first);
    }
  }

  // Pop ids from the queue, adding them to the sorted order and decreasing the
  // in-degrees of their successors.  A successor who's in-degree becomes zero
  // gets added to the queue.
  while (!queue.empty()) {
    auto next = queue.front();
    queue.pop();
    result.push_back(next);
    for (auto successor : call_graph.GetDirectCallees(next)) {
      assert(function_in_degree.at(successor) > 0 &&
             "The in-degree cannot be zero if the function is a successor.");
      function_in_degree[successor] = function_in_degree.at(successor) - 1;
      if (function_in_degree.at(successor) == 0) {
        queue.push(successor);
      }
    }
  }

  assert(result.size() == function_in_degree.size() &&
         "Every function should appear in the sort.");

  return result;
}

void FuzzerPassDonateModules::HandleOpArrayLength(
    const opt::Instruction& instruction,
    std::map<uint32_t, uint32_t>* original_id_to_donated_id,
    std::vector<protobufs::Instruction>* donated_instructions) const {
  assert(instruction.opcode() == SpvOpArrayLength &&
         "Precondition: instruction must be OpArrayLength.");
  uint32_t donated_variable_id =
      original_id_to_donated_id->at(instruction.GetSingleWordInOperand(0));
  auto donated_variable_instruction =
      GetIRContext()->get_def_use_mgr()->GetDef(donated_variable_id);
  auto pointer_to_struct_instruction =
      GetIRContext()->get_def_use_mgr()->GetDef(
          donated_variable_instruction->type_id());
  assert(pointer_to_struct_instruction->opcode() == SpvOpTypePointer &&
         "Type of variable must be pointer.");
  auto donated_struct_type_instruction =
      GetIRContext()->get_def_use_mgr()->GetDef(
          pointer_to_struct_instruction->GetSingleWordInOperand(1));
  assert(donated_struct_type_instruction->opcode() == SpvOpTypeStruct &&
         "Pointee type of pointer used by OpArrayLength must be struct.");
  assert(donated_struct_type_instruction->NumInOperands() ==
             instruction.GetSingleWordInOperand(1) + 1 &&
         "OpArrayLength must refer to the final member of the given "
         "struct.");
  uint32_t fixed_size_array_type_id =
      donated_struct_type_instruction->GetSingleWordInOperand(
          donated_struct_type_instruction->NumInOperands() - 1);
  auto fixed_size_array_type_instruction =
      GetIRContext()->get_def_use_mgr()->GetDef(fixed_size_array_type_id);
  assert(fixed_size_array_type_instruction->opcode() == SpvOpTypeArray &&
         "The donated array type must be fixed-size.");
  auto array_size_id =
      fixed_size_array_type_instruction->GetSingleWordInOperand(1);

  if (instruction.result_id() &&
      !original_id_to_donated_id->count(instruction.result_id())) {
    original_id_to_donated_id->insert(
        {instruction.result_id(), GetFuzzerContext()->GetFreshId()});
  }

  donated_instructions->push_back(MakeInstructionMessage(
      SpvOpCopyObject, original_id_to_donated_id->at(instruction.type_id()),
      original_id_to_donated_id->at(instruction.result_id()),
      opt::Instruction::OperandList({{SPV_OPERAND_TYPE_ID, {array_size_id}}})));
}

void FuzzerPassDonateModules::HandleDiffilwltInstruction(
    const opt::Instruction& instruction,
    std::map<uint32_t, uint32_t>* original_id_to_donated_id,
    std::vector<protobufs::Instruction>* donated_instructions,
    std::set<uint32_t>* skipped_instructions) {
  if (!instruction.result_id()) {
    // It does not generate a result id, so it can be ignored.
    return;
  }
  if (!original_id_to_donated_id->count(instruction.type_id())) {
    // We cannot handle this instruction's result type, so we need to skip it
    // all together.
    skipped_instructions->insert(instruction.result_id());
    return;
  }

  // We now attempt to replace the instruction with an OpCopyObject.
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3278): We could do
  //  something more refined here - we could check which operands to the
  //  instruction could not be donated and replace those operands with
  //  references to other ids (such as constants), so that we still get an
  //  instruction with the opcode and easy-to-handle operands of the donor
  //  instruction.
  auto remapped_type_id = original_id_to_donated_id->at(instruction.type_id());
  if (!IsBasicType(
          *GetIRContext()->get_def_use_mgr()->GetDef(remapped_type_id))) {
    // The instruction has a non-basic result type, so we cannot replace it with
    // an object copy of a constant.  We thus skip it completely.
    // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3279): We could
    //  instead look for an available id of the right type and generate an
    //  OpCopyObject of that id.
    skipped_instructions->insert(instruction.result_id());
    return;
  }

  // We are going to add an OpCopyObject instruction.  Add a mapping for the
  // result id of the original instruction if does not already exist (it may
  // exist in the case that it has been forward-referenced).
  if (!original_id_to_donated_id->count(instruction.result_id())) {
    original_id_to_donated_id->insert(
        {instruction.result_id(), GetFuzzerContext()->GetFreshId()});
  }

  // We find or add a zero constant to the receiving module for the type in
  // question, and add an OpCopyObject instruction that copies this zero.
  // TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3177):
  //  Using this particular constant is arbitrary, so if we have a
  //  mechanism for noting that an id use is arbitrary and could be
  //  fuzzed we should use it here.
  auto zero_constant = FindOrCreateZeroConstant(remapped_type_id);
  donated_instructions->push_back(MakeInstructionMessage(
      SpvOpCopyObject, remapped_type_id,
      original_id_to_donated_id->at(instruction.result_id()),
      opt::Instruction::OperandList({{SPV_OPERAND_TYPE_ID, {zero_constant}}})));
}

void FuzzerPassDonateModules::PrepareInstructionForDonation(
    const opt::Instruction& instruction, opt::IRContext* donor_ir_context,
    std::map<uint32_t, uint32_t>* original_id_to_donated_id,
    std::vector<protobufs::Instruction>* donated_instructions) {
  // Get the instruction's input operands into donation-ready form,
  // remapping any id uses in the process.
  opt::Instruction::OperandList input_operands;

  // Consider each input operand in turn.
  for (uint32_t in_operand_index = 0;
       in_operand_index < instruction.NumInOperands(); in_operand_index++) {
    std::vector<uint32_t> operand_data;
    const opt::Operand& in_operand = instruction.GetInOperand(in_operand_index);
    // Check whether this operand is an id.
    if (spvIsIdType(in_operand.type)) {
      // This is an id operand - it consists of a single word of data,
      // which needs to be remapped so that it is replaced with the
      // donated form of the id.
      auto operand_id = in_operand.words[0];
      if (!original_id_to_donated_id->count(operand_id)) {
        // This is a forward reference.  We will choose a corresponding
        // donor id for the referenced id and update the mapping to
        // reflect it.

        // Keep release compilers happy because |donor_ir_context| is only used
        // in this assertion.
        (void)(donor_ir_context);
        assert((donor_ir_context->get_def_use_mgr()
                        ->GetDef(operand_id)
                        ->opcode() == SpvOpLabel ||
                instruction.opcode() == SpvOpPhi) &&
               "Unsupported forward reference.");
        original_id_to_donated_id->insert(
            {operand_id, GetFuzzerContext()->GetFreshId()});
      }
      operand_data.push_back(original_id_to_donated_id->at(operand_id));
    } else {
      // For non-id operands, we just add each of the data words.
      for (auto word : in_operand.words) {
        operand_data.push_back(word);
      }
    }
    input_operands.push_back({in_operand.type, operand_data});
  }

  if (instruction.opcode() == SpvOpVariable &&
      instruction.NumInOperands() == 1) {
    // This is an uninitialized local variable.  Initialize it to zero.
    input_operands.push_back(
        {SPV_OPERAND_TYPE_ID,
         {FindOrCreateZeroConstant(fuzzerutil::GetPointeeTypeIdFromPointerType(
             GetIRContext(),
             original_id_to_donated_id->at(instruction.type_id())))}});
  }

  if (instruction.result_id() &&
      !original_id_to_donated_id->count(instruction.result_id())) {
    original_id_to_donated_id->insert(
        {instruction.result_id(), GetFuzzerContext()->GetFreshId()});
  }

  // Remap the result type and result id (if present) of the
  // instruction, and turn it into a protobuf message.
  donated_instructions->push_back(MakeInstructionMessage(
      instruction.opcode(),
      instruction.type_id()
          ? original_id_to_donated_id->at(instruction.type_id())
          : 0,
      instruction.result_id()
          ? original_id_to_donated_id->at(instruction.result_id())
          : 0,
      input_operands));
}

void FuzzerPassDonateModules::AddLivesafeFunction(
    const opt::Function& function_to_donate, opt::IRContext* donor_ir_context,
    const std::map<uint32_t, uint32_t>& original_id_to_donated_id,
    const std::vector<protobufs::Instruction>& donated_instructions) {
  // Various types and constants must be in place for a function to be made
  // live-safe.  Add them if not already present.
  FindOrCreateBoolType();  // Needed for comparisons
  FindOrCreatePointerTo32BitIntegerType(
      false, SpvStorageClassFunction);  // Needed for adding loop limiters
  FindOrCreate32BitIntegerConstant(
      0, false);  // Needed for initializing loop limiters
  FindOrCreate32BitIntegerConstant(
      1, false);  // Needed for incrementing loop limiters

  // Get a fresh id for the variable that will be used as a loop limiter.
  const uint32_t loop_limiter_variable_id = GetFuzzerContext()->GetFreshId();
  // Choose a random loop limit, and add the required constant to the
  // module if not already there.
  const uint32_t loop_limit = FindOrCreate32BitIntegerConstant(
      GetFuzzerContext()->GetRandomLoopLimit(), false);

  // Consider every loop header in the function to donate, and create a
  // structure capturing the ids to be used for manipulating the loop
  // limiter each time the loop is iterated.
  std::vector<protobufs::LoopLimiterInfo> loop_limiters;
  for (auto& block : function_to_donate) {
    if (block.IsLoopHeader()) {
      protobufs::LoopLimiterInfo loop_limiter;
      // Grab the loop header's id, mapped to its donated value.
      loop_limiter.set_loop_header_id(original_id_to_donated_id.at(block.id()));
      // Get fresh ids that will be used to load the loop limiter, increment
      // it, compare it with the loop limit, and an id for a new block that
      // will contain the loop's original terminator.
      loop_limiter.set_load_id(GetFuzzerContext()->GetFreshId());
      loop_limiter.set_increment_id(GetFuzzerContext()->GetFreshId());
      loop_limiter.set_compare_id(GetFuzzerContext()->GetFreshId());
      loop_limiter.set_logical_op_id(GetFuzzerContext()->GetFreshId());
      loop_limiters.emplace_back(loop_limiter);
    }
  }

  // Consider every access chain in the function to donate, and create a
  // structure containing the ids necessary to clamp the access chain
  // indices to be in-bounds.
  std::vector<protobufs::AccessChainClampingInfo> access_chain_clamping_info;
  for (auto& block : function_to_donate) {
    for (auto& inst : block) {
      switch (inst.opcode()) {
        case SpvOpAccessChain:
        case SpvOpInBoundsAccessChain: {
          protobufs::AccessChainClampingInfo clamping_info;
          clamping_info.set_access_chain_id(
              original_id_to_donated_id.at(inst.result_id()));

          auto base_object = donor_ir_context->get_def_use_mgr()->GetDef(
              inst.GetSingleWordInOperand(0));
          assert(base_object && "The base object must exist.");
          auto pointer_type = donor_ir_context->get_def_use_mgr()->GetDef(
              base_object->type_id());
          assert(pointer_type && pointer_type->opcode() == SpvOpTypePointer &&
                 "The base object must have pointer type.");

          auto should_be_composite_type =
              donor_ir_context->get_def_use_mgr()->GetDef(
                  pointer_type->GetSingleWordInOperand(1));

          // Walk the access chain, creating fresh ids to facilitate
          // clamping each index.  For simplicity we do this for every
          // index, even though constant indices will not end up being
          // clamped.
          for (uint32_t index = 1; index < inst.NumInOperands(); index++) {
            auto compare_and_select_ids =
                clamping_info.add_compare_and_select_ids();
            compare_and_select_ids->set_first(GetFuzzerContext()->GetFreshId());
            compare_and_select_ids->set_second(
                GetFuzzerContext()->GetFreshId());

            // Get the bound for the component being indexed into.
            uint32_t bound;
            if (should_be_composite_type->opcode() == SpvOpTypeRuntimeArray) {
              // The donor is indexing into a runtime array.  We do not
              // donate runtime arrays.  Instead, we donate a corresponding
              // fixed-size array for every runtime array.  We should thus
              // find that donor composite type's result id maps to a fixed-
              // size array.
              auto fixed_size_array_type =
                  GetIRContext()->get_def_use_mgr()->GetDef(
                      original_id_to_donated_id.at(
                          should_be_composite_type->result_id()));
              assert(fixed_size_array_type->opcode() == SpvOpTypeArray &&
                     "A runtime array type in the donor should have been "
                     "replaced by a fixed-sized array in the recipient.");
              // The size of this fixed-size array is a suitable bound.
              bound = TransformationAddFunction::GetBoundForCompositeIndex(
                  GetIRContext(), *fixed_size_array_type);
            } else {
              bound = TransformationAddFunction::GetBoundForCompositeIndex(
                  donor_ir_context, *should_be_composite_type);
            }
            const uint32_t index_id = inst.GetSingleWordInOperand(index);
            auto index_inst =
                donor_ir_context->get_def_use_mgr()->GetDef(index_id);
            auto index_type_inst = donor_ir_context->get_def_use_mgr()->GetDef(
                index_inst->type_id());
            assert(index_type_inst->opcode() == SpvOpTypeInt);
            assert(index_type_inst->GetSingleWordInOperand(0) == 32);
            opt::analysis::Integer* index_int_type =
                donor_ir_context->get_type_mgr()
                    ->GetType(index_type_inst->result_id())
                    ->AsInteger();
            if (index_inst->opcode() != SpvOpConstant) {
              // We will have to clamp this index, so we need a constant
              // whose value is one less than the bound, to compare
              // against and to use as the clamped value.
              FindOrCreate32BitIntegerConstant(bound - 1,
                                               index_int_type->IsSigned());
            }
            should_be_composite_type =
                TransformationAddFunction::FollowCompositeIndex(
                    donor_ir_context, *should_be_composite_type, index_id);
          }
          access_chain_clamping_info.push_back(clamping_info);
          break;
        }
        default:
          break;
      }
    }
  }

  // If the function contains OpKill or OpUnreachable instructions, and has
  // non-void return type, then we need a value %v to use in order to turn
  // these into instructions of the form OpReturn %v.
  uint32_t kill_unreachable_return_value_id;
  auto function_return_type_inst =
      donor_ir_context->get_def_use_mgr()->GetDef(function_to_donate.type_id());
  if (function_return_type_inst->opcode() == SpvOpTypeVoid) {
    // The return type is void, so we don't need a return value.
    kill_unreachable_return_value_id = 0;
  } else {
    // We do need a return value; we use zero.
    assert(function_return_type_inst->opcode() != SpvOpTypePointer &&
           "Function return type must not be a pointer.");
    kill_unreachable_return_value_id = FindOrCreateZeroConstant(
        original_id_to_donated_id.at(function_return_type_inst->result_id()));
  }
  // Add the function in a livesafe manner.
  ApplyTransformation(TransformationAddFunction(
      donated_instructions, loop_limiter_variable_id, loop_limit, loop_limiters,
      kill_unreachable_return_value_id, access_chain_clamping_info));
}

}  // namespace fuzz
}  // namespace spvtools
