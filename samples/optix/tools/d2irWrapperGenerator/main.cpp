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

#include <FrontEnd/PTX/Intrinsics/D2IRIntrinsicBuilder.h>

#include <InstructionPermutationGenerator.h>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>

#include <InstructionTemplates.h>

using namespace optix::PTXIntrinsics;

void addIntrinsicsForTemplate( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData );

int main( int argc, char** argv )
{
    try
    {
        ptxParseData parseData{};

        std::vector<std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>> stdTemplates;
        std::vector<std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>> extTemplates;
        getInstructionTemplates( parseData, stdTemplates, extTemplates );

        llvm::LLVMContext llvmContext;
        llvm::Module*     ptxIntrinsicModule = new llvm::Module( "D2IRInstructions", llvmContext );

        D2IRIntrinsicBuilder wrapperBuilder( llvmContext, ptxIntrinsicModule );

        for( const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate : stdTemplates )
            addIntrinsicsForTemplate( llvmContext, wrapperBuilder, instTemplate );

        wrapperBuilder.validate();

        if( argc < 2 )
        {
            std::cout << "usage: " << argv[0] << " [filename|-]\n";
            return 1;
        }

        std::error_code      errorInfo;
        llvm::raw_fd_ostream fileStream( argv[1], errorInfo );
        if( errorInfo )
        {
            printf( "Error opening file for writing:\n%s", errorInfo.message().c_str() );
            return 1;
        }
        llvm::WriteBitcodeToFile( *ptxIntrinsicModule, fileStream );
        return 0;
    }
    catch( std::exception& e )
    {
        std::cerr << "Failed with exception: " << e.what() << "\n";
        return 1;
    }
}

void addStandardIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addMathIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addBfeIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addBfiIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addCvtIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addTexIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addTexLevelIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addTexGradIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addTld4Intrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addTxqIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addSurfaceIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addAtomOrRedIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addSetOrSetpIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addSelpIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addPopcIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addLdStIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addShfIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addShlIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addMovIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addDp2aIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );
void addDp4aIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate );

void addIntrinsicsForTemplate( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    switch( instTemplate.first->code )
    {
        case ptx_abs_Instr:
        case ptx_max_Instr:
        case ptx_min_Instr:
        case ptx_add_Instr:
        case ptx_sub_Instr:
        case ptx_mul_Instr:
        case ptx_mul_hi_Instr:
        case ptx_div_Instr:
        case ptx_div_full_Instr:
        case ptx_fma_Instr:
        case ptx_mad_Instr:
        case ptx_mad_lo_Instr:
        case ptx_mad_hi_Instr:
        case ptx_mad_wide_Instr:
        case ptx_sqrt_Instr:
        case ptx_rsqrt_Instr:
        case ptx_rcp_Instr:
        case ptx_lg2_Instr:
        case ptx_ex2_Instr:
        case ptx_sin_Instr:
        case ptx_cos_Instr:
            addMathIntrinsics( context, builder, instTemplate );
            break;
        case ptx_bfe_Instr:
            addBfeIntrinsics( context, builder, instTemplate );
            break;
        case ptx_bfi_Instr:
            addBfiIntrinsics( context, builder, instTemplate );
            break;
        case ptx_cvt_Instr:
            addCvtIntrinsics( context, builder, instTemplate );
            break;
        case ptx_brev_Instr:
        case ptx_prmt_Instr:
            addStandardIntrinsics( context, builder, instTemplate );
            break;
        case ptx_shf_l_Instr:
        case ptx_shf_r_Instr:
            addShfIntrinsics( context, builder, instTemplate );
            break;
        case ptx_shl_Instr:
            addShlIntrinsics( context, builder, instTemplate );
            break;
        case ptx_tex_Instr:
            addTexIntrinsics( context, builder, instTemplate );
            break;
        case ptx_tex_level_Instr:
            addTexLevelIntrinsics( context, builder, instTemplate );
            break;
        case ptx_tex_grad_Instr:
            addTexGradIntrinsics( context, builder, instTemplate );
            break;
        case ptx_tld4_Instr:
            addTld4Intrinsics( context, builder, instTemplate );
            break;
        case ptx_txq_Instr:
        case ptx_txq_level_Instr:
            addTxqIntrinsics( context, builder, instTemplate );
            break;
        case ptx_suld_b_Instr:
        case ptx_sust_b_Instr:
        case ptx_sust_p_Instr:
            addSurfaceIntrinsics( context, builder, instTemplate );
            break;
        case ptx_atom_Instr:
        case ptx_red_Instr:
            addAtomOrRedIntrinsics( context, builder, instTemplate );
            break;
        case ptx_set_Instr:
        case ptx_setp_Instr:
            addSetOrSetpIntrinsics( context, builder, instTemplate );
            break;
        case ptx_selp_Instr:
            addSelpIntrinsics( context, builder, instTemplate );
            break;
        case ptx_popc_Instr:
            addPopcIntrinsics( context, builder, instTemplate );
            break;
        case ptx_ld_Instr:
        case ptx_st_Instr:
            addLdStIntrinsics( context, builder, instTemplate );
            break;
        case ptx_mov_Instr:
            addMovIntrinsics( context, builder, instTemplate );
            break;
        case ptx_dp2a_hi_Instr:
        case ptx_dp2a_lo_Instr:
            addDp2aIntrinsics( context, builder, instTemplate );
            break;
        case ptx_dp4a_Instr:
            addDp4aIntrinsics( context, builder, instTemplate );
            break;
        default:
            // TODO: Add option to log missing op codes
            return;
    }
}

void addStandardIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addStandardIntrinsic( instruction );
}

void addMathIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getMathInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addMathIntrinsic( instruction );
}

void addBfeIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addBfeIntrinsic( instruction );
}

void addBfiIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addBfiIntrinsic( instruction );
}

void addCvtIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getCvtInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
    {
        // Skip all instructions that have more than two arguments. These
        // include instructions that pack two floats into a single packed
        // float.
        if( instruction.signature.size() > 2 )
            continue;
        builder.addCvtIntrinsic( instruction );
    }
}

void addTexIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getTexInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
    {
        // Skip all instructions that return the packed float type
        if( instruction.signature[0].ptxType.compare( "f16x2" ) == 0 )
            continue;
        // Skip all texture instructions with separate texture/sampler arguments (e.g. the first two arguments are "image")
        if( instruction.signature[2].ptxType.compare( "image" ) == 0 )
            continue;
        // Skip all texture instructions that take optional offset arguments. (e.g. they have more than two arguments + return value)
        if( instruction.signature.size() > 3 )
            continue;

        builder.addTexIntrinsic( instruction );
    }
}

void addTexLevelIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getTexInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
    {
        if( instruction.signature[0].ptxType.compare( "f16x2" ) == 0 )
            continue;
        // Skip all texture instructions with separate texture/sampler arguments (e.g. the first two arguments are "image")
        if( instruction.signature[2].ptxType.compare( "image" ) == 0 )
            continue;
        // Skip all texture instructions that take optional offset arguments.
        if( instruction.signature.size() > 4 )
            continue;

        builder.addTexIntrinsic( instruction);
    }
}

void addTexGradIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getTexInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
    {
        if( instruction.signature[0].ptxType.compare( "f16x2" ) == 0 )
            continue;
        // Skip all texture instructions with separate texture/sampler arguments (e.g. the first two arguments are "image")
        if( instruction.signature[2].ptxType.compare( "image" ) == 0 )
            continue;
        // Skip all texture instructions that take optional offset arguments.
        if( instruction.signature.size() > 5 )
            continue;

        builder.addTexIntrinsic( instruction );
    }
}

void addTld4Intrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getTld4InstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addTld4Intrinsic( instruction );
}

void addTxqIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getTxqInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addTxqIntrinsic( instruction );
}

void addSurfaceIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getSurfaceInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addSurfaceIntrinsic( instruction );
}

void addAtomOrRedIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getAtomOrRedInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
    {
        // Skip all non-CAS instructions that have more than two arguments
        if( instruction.modifiers.atomicOp != AtomicOperation::cas && instruction.signature.size() != 3 )
            continue;
        builder.addAtomOrRedIntrinsic( instruction );
    }
}

void addSetOrSetpIntrinsics( llvm::LLVMContext&    context,
                             D2IRIntrinsicBuilder& builder,
                             const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getSetOrSetpInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
    {
        // Skip instructions with packed float arguments
        if( instruction.signature[1].ptxType.compare( "f16x2" ) == 0 )
            continue;
        builder.addSetOrSetpIntrinsic( instruction );
    }
}

void addSelpIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
    {
        // Skip instructions with packed float arguments
        if( instruction.signature[1].ptxType.compare( "f16x2" ) == 0 )
            continue;
        builder.addSelpIntrinsic( instruction );
    }
}

void addPopcIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addPopcIntrinsic( instruction );
}

void addLdStIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getLdStInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addLdStIntrinsic( instruction );
}

void addShfIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getShfInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addShfIntrinsic( instruction );
}

void addShlIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addShlIntrinsic( instruction );
}

void addMovIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getMovInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addMovIntrinsic( instruction );
}

void addDp2aIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getDp2aInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addDp2aIntrinsic( instruction );
}

void addDp4aIntrinsics( llvm::LLVMContext& context, D2IRIntrinsicBuilder& builder, const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplate )
{
    std::vector<PTXIntrinsicInfo> possibleInstructions = getDp4aInstructionsFromTemplate( context, instTemplate );

    for( const PTXIntrinsicInfo& instruction : possibleInstructions )
        builder.addDp4aIntrinsic( instruction );
}
