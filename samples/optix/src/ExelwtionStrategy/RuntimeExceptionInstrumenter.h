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

//----------------------------------------------------------------------------//
// MegakernelES:
// If the exception oclwrs in a state function, we set code/detail/next state
// id/return state id and immediately give control back to the state machine
// (= insert a continuation call). The next state id is the id of the exception
// of the current entry point, the return state id is -1 to indicate that
// exelwtion should stop after the exception.
// If the exception oclwrs in some arbitrary helper function that was not
// inlined, then we can't use a continuation call. Instead, we set code/detail
// and return from that function (if necessary with a dummy result if the
// function returns something). Such a function is marked as "may throw". Every
// use of these "may throw" functions is then instrumented such that after it
// returned, we first check if the code was set. If no code is set (code == 0),
// exelwtion continues. Otherwise, the same scheme applies: If we are in a state
// function, do a continuation call, otherwise return one more level and mark
// the function as "may throw".
//----------------------------------------------------------------------------//

#pragma once

#include <corelib/compiler/CoreIRBuilder.h>
#include <internal/optix_declarations.h>  // RTexception
#include <llvm/ADT/SetVector.h>
#include <llvm/IR/IRBuilder.h>

#include <set>

namespace llvm {
class Value;
class ConstantInt;
class Instruction;
class CallInst;
class Function;
class Module;
}

namespace optix {
class Context;
class ProgramManager;

class RuntimeExceptionInstrumenter
{
  public:
    typedef std::set<llvm::Function*> FunctionSet;

    RuntimeExceptionInstrumenter( llvm::Module*         module,
                                  uint64_t              exceptionFlags,
                                  const ProgramManager& programManager,
                                  const FunctionSet&    stateFunctions,
                                  const FunctionSet&    exceptionFunctions,
                                  bool                  isMegakernel );

    // Create all exception handling code.
    void run();

    // Check if the given function may throw an exception.
    // Does not incorporate state functions since they all may throw.
    bool mayThrow( llvm::Function* function );

  private:
    const bool                       m_isMegakernel;
    llvm::Module*                    m_module;
    const uint64_t                   m_exceptionFlags;
    const ProgramManager&            m_programManager;
    std::set<llvm::Function*>        m_stateFunctions;
    std::set<llvm::Function*>        m_exceptionFunctions;
    llvm::SetVector<llvm::Function*> m_throwingFunctions;
    llvm::Type*                      m_canonicalStateType;
    llvm::Type*                      m_canonicalStateMKType;

    // Global variable for "n/a" string.
    llvm::Value* m_gv;


    // Create exception handling code for a particular exception type.

    void insertStackOverflowHandler();
    void insertBufferIdIlwalidHandler();
    void insertTextureIdIlwalidHandler();
    void insertProgramIdIlwalidHandler();
    void insertBufferIndexOutOfBoundsHandler();
    void insertIndexOutOfBoundsHandler();
    void insertIlwalidRayHandler();
    void insertInternalErrorHandler();
    void insertUserExceptionHandler();

    // Create code that resets the exception code to 0 inside the exception
    // programs. This prevents accidental throwing of exceptions inside
    // exception programs.

    void insertClearCodeInExceptionPrograms();

    // Helper functions.

    bool isCanonicalState( llvm::Value* value ) const;
    bool isStateFunction( llvm::Function* func ) const;
    bool isExceptionFunction( llvm::Function* func ) const;

    void createThrow( llvm::Value* state, corelib::CoreIRBuilder& builder );

    template <class BufferAccessFunction>
    void instrumentBufferAccessFunction( llvm::Function* function, llvm::Function* getBufferSizeFunc, llvm::Function* getBufferAddressFunc );

    void createException( llvm::Value*               state,
                          llvm::Value*               condition,
                          RTexception                exception,
                          llvm::Instruction*         insertBefore,
                          std::vector<llvm::Value*>* detail   = nullptr,
                          std::vector<llvm::Value*>* detail64 = nullptr );

    void createStackOverflow( llvm::Value* state, llvm::ConstantInt* frameSize, llvm::Instruction* insertBefore );

    void createIdIlwalid( llvm::Function* checkIdFunc, RTexception exception, llvm::Value* state, llvm::Value* id, llvm::Instruction* insertBefore );

    void insertIdIlwalidHandler( RTexception exception, const char* funcName );

    void insertMaterialIndexOutOfBoundsHandler();
    void insertNodeIndexOutOfBoundsHandler();

    void markAsThrowing( const char* namePrefix );

    // Insert a try-catch block around every call that may throw.
    void insertTryCatch();

    // Insert a try-catch block around the given call.
    void insertTryCatch( llvm::CallInst* call );
};
}  // namespace optix
