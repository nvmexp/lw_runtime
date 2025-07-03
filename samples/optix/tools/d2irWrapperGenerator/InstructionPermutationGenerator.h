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

/*
 * Functions used to generate different PTX instruction permutations so we can
 * build intrinsics for them ahead of time.
 */

#include <FrontEnd/PTX/Intrinsics/IntrinsicInfo.h>

#include <InstructionTemplates.h>

#include <utility>
#include <vector>

// TODO(Kincaid): If/when we unify the instruction permutation struct, we probably want to pare down these functions

// Functions to generate instruction permutations from PTX instruction templates.
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getMathInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getCvtInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getTexInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getTld4InstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getTxqInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getSurfaceInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getAtomOrRedInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getSetOrSetpInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getLdStInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getShfInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getMovInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getDp2aInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getDp4aInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
std::vector<optix::PTXIntrinsics::PTXIntrinsicInfo> getInstructionsFromTemplate(
    llvm::LLVMContext& context,
    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
