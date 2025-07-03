//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#pragma once

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>

#include <lwvm/ClientInterface/LWVM.h>

#include <FrontEnd/PTX/Intrinsics/InlinePtxParser.h>
#include <FrontEnd/PTX/Intrinsics/IntrinsicInfo.h>

#include <ptxIR.h>
#include <ptxInstructions.h>

namespace optix {
namespace PTXIntrinsics {

class D2IRIntrinsicBuilder
{
  public:
    D2IRIntrinsicBuilder( llvm::LLVMContext& context, llvm::Module* module );

    // Add a body for the specified OptiX PTX intrinsic function. Return true
    // if we were able to add the intrinsic, false otherwise.
    bool addIntrinsic( llvm::Function* optixPtxIntrinsic );

    void addStandardIntrinsic( const PTXIntrinsicInfo& instruction );
    void addMathIntrinsic( const PTXIntrinsicInfo& instruction );
    void addBfeIntrinsic( const PTXIntrinsicInfo& instruction );
    void addBfiIntrinsic( const PTXIntrinsicInfo& instruction );
    void addCvtIntrinsic( const PTXIntrinsicInfo& instruction );
    void addTexIntrinsic( const PTXIntrinsicInfo& instruction );
    void addTld4Intrinsic( const PTXIntrinsicInfo& instruction );
    void addTxqIntrinsic( const PTXIntrinsicInfo& instruction );
    void addSurfaceIntrinsic( const PTXIntrinsicInfo& instruction );
    void addAtomOrRedIntrinsic( const PTXIntrinsicInfo& instruction );
    void addSetOrSetpIntrinsic( const PTXIntrinsicInfo& instruction );
    void addPopcIntrinsic( const PTXIntrinsicInfo& instruction );
    void addSelpIntrinsic( const PTXIntrinsicInfo& instruction );
    void addLdStIntrinsic( const PTXIntrinsicInfo& instruction );
    void addShfIntrinsic( const PTXIntrinsicInfo& instruction );
    void addShlIntrinsic( const PTXIntrinsicInfo& instruction );
    void addMovIntrinsic( const PTXIntrinsicInfo& instruction );
    void addDp2aIntrinsic( const PTXIntrinsicInfo& instruction );
    void addDp4aIntrinsic( const PTXIntrinsicInfo& instruction );

    // Validate the generated LLVM module.
    void validate();

  private:
    llvm::LLVMContext& m_context;
    llvm::Module*      m_module;

    // Get the argument types for the optix intrinsic with the given signature
    std::vector<llvm::Type*> getPtxIntrinsicArgTypes( const InstructionSignature& signature );
    // Add a definition for the given intrinsic to the module
    llvm::Function* getOrInsertPtxIntrinsicDeclaration( const std::string&              intrinsicName,
                                                        llvm::Type*                     returnType,
                                                        const std::vector<llvm::Type*>& argTypes );

    // Get the declaration for the given intrinsic. Use the types in the given signature to get intrinsic definitions that are overloaded
    llvm::Function* getIntrinsicDeclaration( llvm::Intrinsic::ID lwvmIntrinsic, const InstructionSignature& signature );
    llvm::Function* getTexIntrinsicDeclaration( llvm::Intrinsic::ID lwvmIntrinsic, const InstructionSignature& signature, const bool isSparse );
    llvm::Function* getTld4IntrinsicDeclaration( llvm::Intrinsic::ID lwvmIntrinsic, const InstructionSignature& signature );
    llvm::Function* getSurfaceIntrinsicDeclaration( llvm::Intrinsic::ID lwvmIntrinsic, const InstructionSignature& signature );
    llvm::Function* getAtomIntrinsicDeclaration( llvm::Intrinsic::ID lwvmIntrinsic, const InstructionSignature& signature );

    // Get the arguments for the given intrinsic.
    std::vector<llvm::Value*> getIntrinsicArguments( llvm::IRBuilder<>& builder, llvm::Function* ptxIntrinsicFunction );
    std::vector<llvm::Value*> getMathIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                         llvm::IRBuilder<>&      builder,
                                                         llvm::Function*         ptxIntrinsicFunction,
                                                         llvm::Function*         intrinsicFunction,
                                                         bool                    hasMathFlag );
    std::vector<llvm::Value*> getCvtIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                        llvm::IRBuilder<>&      builder,
                                                        llvm::Function*         ptxIntrinsicFunction,
                                                        llvm::Function*         intrinsicFunction );

    std::vector<llvm::Value*> getTexIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                        llvm::IRBuilder<>&      builder,
                                                        llvm::Function*         ptxIntrinsicFunction,
                                                        llvm::Function*         intrinsicFunction,
                                                        bool                    hasLOD,
                                                        bool                    hasGradients,
                                                        bool                    isTexLoad );
    std::vector<llvm::Value*> getTld4IntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                         llvm::IRBuilder<>&      builder,
                                                         llvm::Function*         ptxIntrinsicFunction,
                                                         llvm::Function*         intrinsicFunction );
    std::vector<llvm::Value*> getTxqIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                        llvm::IRBuilder<>&      builder,
                                                        llvm::Function*         ptxIntrinsicFunction,
                                                        llvm::Function*         intrinsicFunction );
    std::vector<llvm::Value*> getSurfaceIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                            llvm::IRBuilder<>&      builder,
                                                            llvm::Function*         ptxIntrinsicFunction,
                                                            llvm::Function*         intrinsicFunction );
    std::vector<llvm::Value*> getAtomIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                         llvm::IRBuilder<>&      builder,
                                                         llvm::Function*         ptxIntrinsicFunction,
                                                         llvm::Function*         intrinsicFunction );
    std::vector<llvm::Value*> getLdStIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                         llvm::IRBuilder<>&      builder,
                                                         llvm::Function*         ptxIntrinsicFunction,
                                                         llvm::Function*         intrinsicFunction );
    std::vector<llvm::Value*> getShfIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                        llvm::IRBuilder<>&      builder,
                                                        llvm::Function*         ptxIntrinsicFunction,
                                                        llvm::Function*         intrinsicFunction );
    std::vector<llvm::Value*> getDp2aIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                         llvm::IRBuilder<>&      builder,
                                                         llvm::Function*         ptxIntrinsicFunction,
                                                         llvm::Function*         intrinsicFunction );
    std::vector<llvm::Value*> getDp4aIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                         llvm::IRBuilder<>&      builder,
                                                         llvm::Function*         ptxIntrinsicFunction,
                                                         llvm::Function*         intrinsicFunction );

    // Add an instruction performing the given boolean operation
    llvm::Value* createBoolOp( const BooleanOperator boolOp, llvm::IRBuilder<>& builder, llvm::Value* firstValue, llvm::Value* secondValue );
};
}  // namespace PTXIntrinsics
}  // namespace optix
