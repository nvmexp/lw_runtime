// Copyright (c) 2021, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
// LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
// CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
// LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
// INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <FrontEnd/PTX/Intrinsics/IntrinsicInfo.h>

/**
 * This class implements a limited PTX parser. It determines the op code,
 * signature, and flags for a given inline PTX string.
 */
namespace optix {
namespace PTXIntrinsics {
namespace InlinePtxParser {

struct OpCodeAndName
{
    ptxInstructionCode    opCode;
    const llvm::StringRef name;
};

OpCodeAndName getOpCodeAndNameFromOptixIntrinsic( llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getLdStInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getMathInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getAtomOrRedInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getSetOrSetpInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getShfInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getMovInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getTxqInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getTexInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getTld4InstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getSurfInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getCvtInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getDp2aInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
PTXIntrinsicInfo getDp4aInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic );
}  // InlinePtxParser
}  // PTXIntrinsics
}  // optix
