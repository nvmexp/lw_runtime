// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <Objects/SemanticType.h>
#include <corelib/compiler/LLVMSupportTypes.h>

#define __OPTIX_INCLUDE_INTERNAL_HEADERS__
#include <optix_7_types.h>
#undef __OPTIX_INCLUDE_INTERNAL_HEADERS__

#include <llvm/Transforms/Utils/ValueMapper.h>

#include <string>

namespace llvm {
class Function;
class FunctionType;
class GlobalVariable;
class Module;
class Type;
}

namespace optix_exp {
class ErrorDetails;
}

namespace optix {
class LLVMManager;
class VariableType;

// Colwert exit instructions.
void replaceExitWithReturn( llvm::Function* function );

// Replace function pointer initializers of global variables by undef values since C14n::addStateParameter()
// can not handle them.
void removeFunctionPointerInitializersOfGlobalVariables( llvm::Module* module );

// Add a parameter for the canonical state to the function and all other functions in the module.
// This ilwalidates all pointers to functions.
llvm::Function* addStateParameter( llvm::Function* func, LLVMManager* llvmManager );

// Replace calls to optixi_getState by a parameter access after the state parameter has been added.
void replaceGetState( llvm::Module* module, llvm::Type* statePtrTy );

// Get a string representation of the SemanticType that can be used to mangle a function name.
const char* annotationForSemanticType( SemanticType ST );

// Ensure a weak first-order form by requiring that all functions
// that call one of the continuation-generating functions in the
// list below are inlined into the main function.
void ensureFirstOrderForm( llvm::Module* module );

// Insert a dummy function to help the type mapping. The canonical program has only
// non-dotted typenames. These will only get mapped to destination types by matching a
// global. So we place a function in the common runtime called "optixi_linkTypes" that
// takes all the types that need to be matched as parameters and a corresponding
// function declaration in the source module for the link.
void addTypeLinkageFunction( llvm::Module* module, LLVMManager* llvmManager );

// Remove all colwersions to global address space that were introduced by lwcc,
// and colwert all operations to use pointers in the generic address space.
// This is required since the AABB is passed as a pointer to generic
// memory, but the AABB program is declared as __global__, which makes
// lwcc generate code that expects the pointer argument to be in global
// address space, too.
void removeGlobalAddressSpaceModifiers( llvm::Function* function );

// Remove all addrspacecast instructions that operate on memory allocated with
// an alloca, i.e., local memory. This allows SROA to remove such alloca
// instructions more easily. If the alloca remains, the back end has no problems
// to re-insert such addrspacecasts if deemed beneficial for performance.
void removeRedundantAddrSpaceCasts( llvm::Function* function );

// Run quick optimization stack to exploit optimization potential uncovered
// after get/set optimization.
void optimizeModuleAfterGetSetOpt( llvm::Module* module );

// Check if the global variable is written anywyere in the module.
bool globalVariableIsWritten( const llvm::GlobalVariable* value );

// Perform some early checks of intersection program ilwariants.
void earlyCheckIntersectionProgram( llvm::Function* function );

// Trigger an exception if certain conditions are violated in intersection programs:
// 1. rtPotentialIntersection and rtReportIntersection must come in pairs, we must have the same number of rtPI and rtRI in the intersection function.
// 2. Access to attributes must be only within the regions of code between rtPotentialIntersectiona and rtReportIntersection.
//    This means that each attribute accessor function must be dominated by a call to rtPI and postdominated by a call to rtRI.
// Possible improvements to the current checks.
// 1. Verify that rtRI is called if rtPI returns true.
void checkIntersectionProgram( llvm::Function* intersectionFunction );

// Return true if a bounds/aabb function has motion index in its signature.
bool hasMotionIndexArg( llvm::Function* boundsFunction );

// Promote user allocations to have 16B alignment.
void alignUserAllocas( llvm::Function* function );

// Detect parameters from kernel functions for AABB and intersection
// programs. TODO: consider making these use normal function
// parameters to leverage the callable program ABI.
llvm::Function* handleKernelParameters( llvm::Function* function, llvm::Type* statePtrType );

// Process callable program function parameters.
OptixResult handleCallableProgramParameters( llvm::Function*& function, optix_exp::ErrorDetails& errDetails );

// Create a VariableType from a type string.
VariableType parseTypename( const std::string& typename_string );

// Canonicalize calls to rtTerminateRay.
void canonicalizeTerminateRay( llvm::Function* function, llvm::Type* statePtrTy, corelib::ValueVector& toDelete );

// Canonicalize calls to rtIgnoreIntersection.
void canonicalizeIgnoreIntersection( llvm::Function* function, corelib::ValueVector& toDelete );

// Clone a function and any dependencies from one module to
// another. Simply mark all global values and functions with internal
// linkage and perform dead code elimination.  For large modules, it
// may be faster to build this directly, but since canonicalization is
// an initialization step we leave that for the future.
// Uses the given value map if non-null.
llvm::Function* makePartialClone( llvm::Function* oldFunc, LLVMManager* llvmManager, llvm::ValueToValueMapTy* vMap = nullptr );

// Mark all non-optix functions as NoInline
void makeUserFunctionsNoInline( llvm::Module* module );

// Attempt to derive "cleaner" types from lwcc-generated byte-array types.
llvm::Type* getCleanType( llvm::Type* type );
llvm::Type* getCleanType( const VariableType& vtype, llvm::Type* defaultType, llvm::Type* optixRayType );
llvm::Type* getCleanTypeForArg( llvm::Type* defaultType );
llvm::FunctionType* getCleanFunctionType( llvm::FunctionType* oldType );

// Link in module using PreserveSource, throwing an exception on error.
void linkOrThrow( llvm::Linker& linker, llvm::Module* module, bool preserveModule, const std::string& msg = "" );
void linkOrThrow( llvm::Linker& linker, const llvm::Module* module, bool preserveModule, const std::string& msg = "" );

// Get an insertion point in the entry block of the given function, after any alloca instructions.
// Inserting branches after alloca instructions is important, otherwise the LLVM mem2reg
// optimization won't promote stack-allocated variables to registers.
llvm::BasicBlock::iterator getSafeInsertionPoint( llvm::Function* function );

// Casts the given value using an alloca. Necessary for casting aggregate types.
llvm::Value* castThroughAlloca( llvm::Value* value, llvm::Type* newType, llvm::Instruction* insertBefore );

}  // namespace optix
