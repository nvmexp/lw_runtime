//
//  Copyright (c) 2021 LWPU Corporation.  All rights reserved.
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

/**
 * A bunch of miscellaneous utilities for PTX intrinsics. If they're in here,
 * they should probably be moved to a proper home.
 */

#include <FrontEnd/PTX/Intrinsics/IntrinsicInfo.h>

#include <lwvm/ClientInterface/LWVM.h>

namespace optix {
namespace PTXIntrinsics {

InstructionSignature expandTexInstructionSignature( llvm::LLVMContext&           context,
                                                    const InstructionSignature&  signature,
                                                    const PTXIntrinsicModifiers& modifiers,
                                                    const ptxInstructionCode     opCode,
                                                    const bool                   isSparse );
InstructionSignature expandTld4InstructionSignature( llvm::LLVMContext&           context,
                                                     const InstructionSignature&  signature,
                                                     const PTXIntrinsicModifiers& modifiers,
                                                     const ptxInstructionCode     opCode,
                                                     const bool                   isSparse );
InstructionSignature expandSurfaceInstructionSignature( llvm::LLVMContext&           context,
                                                        const InstructionSignature&  signature,
                                                        const PTXIntrinsicModifiers& modifiers );
InstructionSignature expandLdStInstructionSignature( llvm::LLVMContext&           context,
                                                     const ptxInstructionTemplate& instTemplate,
                                                     const InstructionSignature&  signature,
                                                     const PTXIntrinsicModifiers& modifiers );
InstructionSignature expandMathInstructionSignature( llvm::LLVMContext& context, const InstructionSignature& signature, bool hasDoubleRes );
InstructionSignature expandSetOrSetpInstructionSignature( llvm::LLVMContext&          context,
                                                          const InstructionSignature& signature,
                                                          const bool                  hasPredOutput );

}  // namespace PTXIntrinsics
}  // namespace optix
