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

#include <FrontEnd/PTX/Intrinsics/D2IRIntrinsicBuilder.h>

#include <FrontEnd/PTX/Intrinsics/IntrinsicHelpers.h>
#include <FrontEnd/PTX/Intrinsics/IntrinsicInfo.h>
#include <FrontEnd/PTX/Intrinsics/OpCodeMappings.h>
#include <FrontEnd/PTX/Intrinsics/PTXToLLVMHelpers.h>

#include <prodlib/exceptions/Assert.h>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/raw_ostream.h>

#include <cstddef>
#include <vector>

namespace optix {
namespace PTXIntrinsics {

static bool startswith( const std::string& someString, const std::string& prefix )
{
    return someString.substr( 0, prefix.size() ).compare( prefix ) == 0;
}

static std::string getPtxIntrinsicName( const std::string& instructionName )
{
    return "optix.ptx." + instructionName;
}

D2IRIntrinsicBuilder::D2IRIntrinsicBuilder( llvm::LLVMContext& context, llvm::Module* module )
    : m_context( context )
    , m_module( module ){};

bool D2IRIntrinsicBuilder::addIntrinsic( llvm::Function* optixPtxIntrinsic )
{
    InlinePtxParser::OpCodeAndName opCodeAndName = InlinePtxParser::getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    switch( opCodeAndName.opCode )
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
        {
            const PTXIntrinsicInfo mathInst = InlinePtxParser::getMathInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addMathIntrinsic( mathInst );
            return true;
        }
        case ptx_bfe_Instr:
        {
            const PTXIntrinsicInfo inst = InlinePtxParser::getInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addBfeIntrinsic( inst );
            return true;
        }
        case ptx_bfi_Instr:
        {
            const PTXIntrinsicInfo inst = InlinePtxParser::getInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addBfiIntrinsic( inst );
            return true;
        }
        case ptx_cvt_Instr:
        {
            const PTXIntrinsicInfo cvtInst = InlinePtxParser::getCvtInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addCvtIntrinsic( cvtInst );
            return true;
        }
        case ptx_brev_Instr:
        case ptx_prmt_Instr:
        {
            const PTXIntrinsicInfo inst = InlinePtxParser::getInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addStandardIntrinsic( inst );
            return true;
        }
        case ptx_shf_l_Instr:
        case ptx_shf_r_Instr:
        {
            const PTXIntrinsicInfo shfInst = InlinePtxParser::getShfInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addShfIntrinsic( shfInst );
            return true;
        }
        case ptx_shl_Instr:
        {
            const PTXIntrinsicInfo inst = InlinePtxParser::getInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addShlIntrinsic( inst );
            return true;
        }
        case ptx_tex_Instr:
        case ptx_tex_level_Instr:
        case ptx_tex_grad_Instr:
        {
            const PTXIntrinsicInfo texInst = InlinePtxParser::getTexInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addTexIntrinsic( texInst );
            return true;
        }
        case ptx_tld4_Instr:
        {
            const PTXIntrinsicInfo tld4Inst = InlinePtxParser::getTld4InstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addTld4Intrinsic( tld4Inst );
            return true;
        }
        case ptx_txq_Instr:
        case ptx_txq_level_Instr:
        {
            const PTXIntrinsicInfo txqInst = InlinePtxParser::getTxqInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addTxqIntrinsic( txqInst );
            return true;
        }

        case ptx_atom_Instr:
        case ptx_red_Instr:
        {
            const PTXIntrinsicInfo atomOrRedInst =
                InlinePtxParser::getAtomOrRedInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addAtomOrRedIntrinsic( atomOrRedInst );
            return true;
        }
        case ptx_set_Instr:
        case ptx_setp_Instr:
        {
            const PTXIntrinsicInfo setOrSetpInst =
                InlinePtxParser::getSetOrSetpInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addSetOrSetpIntrinsic( setOrSetpInst );
            return true;
        }
        case ptx_selp_Instr:
        {
            const PTXIntrinsicInfo inst = InlinePtxParser::getInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addSelpIntrinsic( inst );
            return true;
        }
        case ptx_popc_Instr:
        {
            const PTXIntrinsicInfo inst = InlinePtxParser::getInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addPopcIntrinsic( inst );
            return true;
        }
        case ptx_ld_Instr:
        case ptx_st_Instr:
        {
            const PTXIntrinsicInfo inst = InlinePtxParser::getLdStInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addLdStIntrinsic( inst );
            return true;
        }
        case ptx_suld_b_Instr:
        case ptx_sust_b_Instr:
        case ptx_sust_p_Instr:
        {
            const PTXIntrinsicInfo inst = InlinePtxParser::getSurfInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addSurfaceIntrinsic( inst );
            return true;
        }
        case ptx_mov_Instr:
        {
            const PTXIntrinsicInfo movInst = InlinePtxParser::getMovInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addMovIntrinsic( movInst );
            return true;
        }
        case ptx_dp2a_hi_Instr:
        case ptx_dp2a_lo_Instr:
        {
            const PTXIntrinsicInfo dp2aInst = InlinePtxParser::getDp2aInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addDp2aIntrinsic( dp2aInst );
            return true;
        }
        case ptx_dp4a_Instr:
        {
            const PTXIntrinsicInfo dp4aInst = InlinePtxParser::getDp4aInstructionFromOptixIntrinsic( m_context, optixPtxIntrinsic );
            addDp4aIntrinsic( dp4aInst );
            return true;
        }
        default:
            // TODO: Add option to log missing op codes
            return false;
    }

    return false;
}

void D2IRIntrinsicBuilder::validate()
{
    llvm::legacy::PassManager PM;
    PM.add( llvm::createVerifierPass() );
    PM.run( *m_module );
}

void D2IRIntrinsicBuilder::addStandardIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Intrinsic::ID lwvmIntrinsic;
    bool                intrinsicHasMapping = lookupIntrinsic( instruction, lwvmIntrinsic );
    if( !intrinsicHasMapping )
        return;

    llvm::Function* intrinsicFunc = getIntrinsicDeclaration( lwvmIntrinsic, instruction.signature );
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs = getIntrinsicArguments( builder, ptxIntrinsicFunc );
    llvm::Value*              returlwalue   = builder.CreateCall( intrinsicFunc, intrinsicArgs );
    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addMathIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Intrinsic::ID lwvmIntrinsic;
    bool                hasMathFlag;
    bool                intrinsicHasMapping = lookupMathIntrinsic( instruction, lwvmIntrinsic, hasMathFlag );
    if( !intrinsicHasMapping )
        return;

    llvm::Function* intrinsicFunc = getIntrinsicDeclaration( lwvmIntrinsic, instruction.signature );
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs =
        getMathIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc, hasMathFlag );

    llvm::Value* returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    // We represent f16x2s as i32s. Cast from <2 x half> to i32 if we're returning halves
    if( instruction.signature[0].ptxType.compare( "f16x2" ) == 0 )
        returlwalue = builder.CreateBitCast( returlwalue, llvm::Type::getInt32Ty( m_context ) );

    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addBfeIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Intrinsic::ID lwvmIntrinsic;
    bool                intrinsicHasMapping = lookupIntrinsic( instruction, lwvmIntrinsic );
    if( !intrinsicHasMapping )
        return;

    llvm::Function* intrinsicFunc = getIntrinsicDeclaration( lwvmIntrinsic, instruction.signature );
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs = getIntrinsicArguments( builder, ptxIntrinsicFunc );
    // LWVM's BFE intrinsic takes a single i32 argument, reverse, that will always be false when mapped from PTX
    llvm::Type*  i1Ty       = llvm::IntegerType::get( m_context, 1 );
    llvm::Value* reverseVal = llvm::ConstantInt::get( i1Ty, 0 );
    intrinsicArgs.push_back( reverseVal );

    llvm::Value* returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );
    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addBfiIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Intrinsic::ID lwvmIntrinsic;
    bool                intrinsicHasMapping = lookupIntrinsic( instruction, lwvmIntrinsic );
    if( !intrinsicHasMapping )
        return;

    llvm::Function* intrinsicFunc = getIntrinsicDeclaration( lwvmIntrinsic, instruction.signature );
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    llvm::SmallVector<llvm::Value*, 8> ptxIntrinsicArgs;
    for( auto it = ptxIntrinsicFunc->arg_begin(), end = ptxIntrinsicFunc->arg_end(); it != end; ++it )
        ptxIntrinsicArgs.push_back( it );

    // LWVM's BFI has transposed arguments from PTX. LWVM's BFI inserts operand
    // 1 into operand 0, while PTX's BFI inserts operand 0 into operand 1
    std::vector<llvm::Value*> intrinsicArgs;
    intrinsicArgs.push_back( ptxIntrinsicArgs[1] );
    intrinsicArgs.push_back( ptxIntrinsicArgs[0] );
    for( size_t i = 2; i < ptxIntrinsicArgs.size(); ++i )
        intrinsicArgs.push_back( ptxIntrinsicArgs[i] );

    llvm::Value* returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );
    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addCvtIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Intrinsic::ID lwvmIntrinsic = llvm::Intrinsic::lwvm_cvt;
    llvm::Function*     intrinsicFunc = getIntrinsicDeclaration( lwvmIntrinsic, instruction.signature );
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs = getCvtIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );
    llvm::Value*              returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );
    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addTexIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    const bool hasLOD       = instruction.ptxOpCode == ptx_tex_level_Instr;
    const bool hasGradients = instruction.ptxOpCode == ptx_tex_grad_Instr;

    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    const bool usesF32Coordinates = instruction.signature[2].ptxType[0] == 'f';

    // We use tex_load instead of tex_fetch when the instruction uses integer
    // coordinates and doesn't supply gradients
    const bool isTexLoad = !usesF32Coordinates && !hasGradients;

    llvm::Intrinsic::ID lwvmIntrinsic;
    if( isTexLoad )
    {
        lwvmIntrinsic = instruction.hasPredicateOutput ? llvm::Intrinsic::lwvm_sparse_tex_load : llvm::Intrinsic::lwvm_tex_load;
    }
    else
    {
        if( hasGradients )
            lwvmIntrinsic = instruction.hasPredicateOutput ? llvm::Intrinsic::lwvm_sparse_tex_fetch_grad :
                                                             llvm::Intrinsic::lwvm_tex_fetch_grad;
        else
            lwvmIntrinsic = instruction.hasPredicateOutput ? llvm::Intrinsic::lwvm_sparse_tex_fetch : llvm::Intrinsic::lwvm_tex_fetch;
    }

    llvm::Function* intrinsicFunc =
        getTexIntrinsicDeclaration( lwvmIntrinsic, instruction.signature, instruction.hasPredicateOutput );

    llvm::Type* returnType           = instruction.signature[0].llvmType;
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs =
        getTexIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc, hasLOD, hasGradients, isTexLoad );

    llvm::Value* intrinsicCallResult = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    llvm::Value* texData;
    llvm::Value* isResidentPredicate = nullptr;
    if( instruction.hasPredicateOutput )
    {
        texData = builder.CreateExtractValue( intrinsicCallResult, 0 );

        // The LWVM IR intrinsic's predicate result is the opposite of PTX's
        // equivalent. In PTX, true means the texture is resident, while in
        // LWVM IR true means it is non-resident. We must ilwert the LWVM
        // intrinsic's result to match PTX's semantics.
        llvm::Value* sparseIsResidentResult = builder.CreateExtractValue( intrinsicCallResult, 1 );
        isResidentPredicate                 = builder.CreateNot( sparseIsResidentResult );
    }
    else
        texData = intrinsicCallResult;

    // Our PTX wrappers don't return texture fetch results as vectors if they
    // have a single value (i.e. we never return a v1, we just return the
    // value). The LWVM IR intrinsic always returns vectors, so we need to
    // unpack the texture data result if it's in a v1.
    RT_ASSERT( texData->getType()->isVectorTy() );
    llvm::VectorType* texDataTy = llvm::dyn_cast<llvm::VectorType>( texData->getType() );
    if( texDataTy->getNumElements() == 1 )
    {
        // Extract the texture element from the data.
        llvm::Value* index = llvm::ConstantInt::get( llvm::Type::getInt32Ty( m_context ), 0 );
        texData            = builder.CreateExtractElement( texData, index );
    }

    llvm::Value* returlwalue = nullptr;

    // If this wrapper has a result predicate, re-pack the texture data and
    // ilwerted residence predicate into a struct.
    if( instruction.hasPredicateOutput )
    {
        llvm::SmallVector<llvm::Type*, 2> resultStructMembers = {texData->getType(), isResidentPredicate->getType()};
        llvm::StructType* resultStructTy = llvm::StructType::get( m_context, resultStructMembers );

        returlwalue = llvm::UndefValue::get( resultStructTy );
        returlwalue = builder.CreateInsertValue( returlwalue, texData, 0 );
        returlwalue = builder.CreateInsertValue( returlwalue, isResidentPredicate, 1 );
    }
    // Otherwise, just return the texture data.
    else
        returlwalue = texData;

    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addTld4Intrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Intrinsic::ID lwvmIntrinsic = llvm::Intrinsic::lwvm_sparse_tex_gather;
    llvm::Function*     intrinsicFunc = getTld4IntrinsicDeclaration( lwvmIntrinsic, instruction.signature );

    llvm::Type* returnType           = instruction.signature[0].llvmType;
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs = getTld4IntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );

    llvm::Value* intrinsicCallResult = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    llvm::Value* texData = builder.CreateExtractValue( intrinsicCallResult, 0 );

    // The LWVM IR intrinsic's predicate result is the opposite of PTX's
    // equivalent. In PTX, true means the texture is resident, while in
    // LWVM IR true means it is non-resident. We must ilwert the LWVM
    // intrinsic's result to match PTX's semantics.
    llvm::Value* sparseIsResidentResult = builder.CreateExtractValue( intrinsicCallResult, 1 );
    llvm::Value* isResidentPredicate    = builder.CreateNot( sparseIsResidentResult );

    // Re-pack the texture data and ilwerted residence predicate into a struct.
    llvm::SmallVector<llvm::Type*, 2> resultStructMembers = {texData->getType(), isResidentPredicate->getType()};
    llvm::StructType* resultStructTy = llvm::StructType::get( m_context, resultStructMembers );

    llvm::Value* returlwalue = llvm::UndefValue::get( resultStructTy );
    returlwalue              = builder.CreateInsertValue( returlwalue, texData, 0 );
    returlwalue              = builder.CreateInsertValue( returlwalue, isResidentPredicate, 1 );

    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addTxqIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Function* intrinsicFunc = llvm::Intrinsic::getDeclaration( m_module, llvm::Intrinsic::lwvm_tex_query );

    llvm::Type* returnType           = instruction.signature[0].llvmType;
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs = getTxqIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );

    llvm::Value* intrinsicCallResult = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    builder.CreateRet( intrinsicCallResult );
}

void D2IRIntrinsicBuilder::addSurfaceIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type* returnType = instruction.signature[0].llvmType;

    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Intrinsic::ID lwvmIntrinsic;
    if( instruction.ptxOpCode == ptx_suld_b_Instr )
        lwvmIntrinsic = llvm::Intrinsic::lwvm_surface_load;
    else if( instruction.ptxOpCode == ptx_sust_b_Instr || instruction.ptxOpCode == ptx_sust_p_Instr )
        lwvmIntrinsic = llvm::Intrinsic::lwvm_surface_store;
    else
        RT_ASSERT_MSG( false, "Unable to get lwvm intrinsic for surface instruction" );

    llvm::Function* intrinsicFunc = getSurfaceIntrinsicDeclaration( lwvmIntrinsic, instruction.signature );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs = getSurfaceIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );

    llvm::Value* returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    if( returnType != llvm::Type::getVoidTy( m_context ) )
        builder.CreateRet( returlwalue );
    else
        builder.CreateRetVoid();
}

void D2IRIntrinsicBuilder::addAtomOrRedIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Intrinsic::ID lwvmIntrinsic;
    bool                hasMapping = lookupAtomOrRedIntrinsic( instruction, lwvmIntrinsic );
    if( !hasMapping )
        return;

    llvm::Function* intrinsicFunc = getAtomIntrinsicDeclaration( lwvmIntrinsic, instruction.signature );
    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs = getAtomIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );
    llvm::Value*              returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    if( returnType != llvm::Type::getVoidTy( m_context ) )
        builder.CreateRet( returlwalue );
    else
        builder.CreateRetVoid();
}

llvm::Value* D2IRIntrinsicBuilder::createBoolOp( const BooleanOperator boolOp,
                                                 llvm::IRBuilder<>&    builder,
                                                 llvm::Value*          firstValue,
                                                 llvm::Value*          secondValue )
{
    switch( boolOp )
    {
        case BooleanOperator::andOp:
            return builder.CreateAnd( firstValue, secondValue );
        case BooleanOperator::orOp:
            return builder.CreateOr( firstValue, secondValue );
        case BooleanOperator::xorOp:
            return builder.CreateXor( firstValue, secondValue );
        default:
            RT_ASSERT_FAIL_MSG( "invalid boolean operator" );
    }
    return nullptr;
}

void D2IRIntrinsicBuilder::addSetOrSetpIntrinsic( const PTXIntrinsicInfo& instruction )
{
    // TODO: We don't handle the ftz flag. See if the "denormal-fp-math"
    // attribute can be used to get the same behavior as FTZ. If not, open a
    // bug requesting a dedicated intrinsic.
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type* returnType = instruction.signature[0].llvmType;

    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    llvm::SmallVector<llvm::Value*, 8> ptxIntrinsicArgs;
    for( auto it = ptxIntrinsicFunc->arg_begin(), end = ptxIntrinsicFunc->arg_end(); it != end; ++it )
        ptxIntrinsicArgs.push_back( it );

    const bool               isSigned = instruction.signature[1].ptxType[0] != 'u';
    const bool               isFloat  = instruction.signature[1].ptxType[0] == 'f';
    llvm::CmpInst::Predicate cmpPred  = ptxToLwvmCompareOperator( instruction.modifiers.cmpOp, isSigned, isFloat );

    // There isn't an lwvm intrinsic that maps directly to setp; instead we generate cmp and select instructions
    llvm::Value* cmpResult;
    if( llvm::CmpInst::isFPPredicate( cmpPred ) )
        cmpResult = builder.CreateFCmp( cmpPred, ptxIntrinsicArgs[0], ptxIntrinsicArgs[1] );
    else
        cmpResult = builder.CreateICmp( cmpPred, ptxIntrinsicArgs[0], ptxIntrinsicArgs[1] );

    if( returnType->isStructTy() )
    {
        llvm::Type*  i1Ty             = llvm::IntegerType::get( m_context, 1 );
        llvm::Value* trueValue        = llvm::ConstantInt::get( i1Ty, 1 );
        llvm::Value* ilwerseCmpResult = builder.CreateXor( cmpResult, trueValue );

        // setp returns a struct of two results: one with the boolean operation
        // applied to the result, and one with the boolean operation applied to the
        // ilwerse of the result.
        if( instruction.modifiers.boolOp != BooleanOperator::unspecified )
        {
            RT_ASSERT( ptxIntrinsicArgs.size() >= 3 );
            cmpResult        = createBoolOp( instruction.modifiers.boolOp, builder, cmpResult, ptxIntrinsicArgs[2] );
            ilwerseCmpResult = createBoolOp( instruction.modifiers.boolOp, builder, ilwerseCmpResult, ptxIntrinsicArgs[2] );
        }

        llvm::Value* returlwalue = llvm::UndefValue::get( returnType );
        returlwalue              = builder.CreateInsertValue( returlwalue, cmpResult, 0 );
        returlwalue              = builder.CreateInsertValue( returlwalue, ilwerseCmpResult, 1 );

        builder.CreateRet( returlwalue );
    }
    else
    {
        llvm::Value* ifTrueRetValue;   // We write 0xffffffff if we're comparing ints, 1.0f if floats.
        llvm::Value* ifFalseRetValue;  // We return 0x00000000 on false
        if( returnType->isFloatingPointTy() )
        {
            ifTrueRetValue  = llvm::ConstantFP::get( returnType, 1.0f );
            ifFalseRetValue = llvm::ConstantFP::get( returnType, 0 );
        }
        else
        {
            ifTrueRetValue  = llvm::ConstantInt::get( returnType, 0xffffffff );
            ifFalseRetValue = llvm::ConstantInt::get( returnType, 0 );
        }


        if( instruction.modifiers.boolOp != BooleanOperator::unspecified )
            cmpResult = createBoolOp( instruction.modifiers.boolOp, builder, cmpResult, ptxIntrinsicArgs[2] );

        llvm::Value* returlwalue = builder.CreateSelect( cmpResult, ifTrueRetValue, ifFalseRetValue );
        builder.CreateRet( returlwalue );
    }
}

void D2IRIntrinsicBuilder::addSelpIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    // selp is basically just a select instruction
    auto         it      = ptxIntrinsicFunc->arg_begin();
    llvm::Value* ifTrue  = it++;
    llvm::Value* ifFalse = it++;
    llvm::Value* pred    = it;

    llvm::Value* returlwalue = builder.CreateSelect( pred, ifTrue, ifFalse );

    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addPopcIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    // llvm.ctpop differs from popc in PTX in that it returns an integer that
    // is the same width as its argument, while popc always returns a u32.
    // Because of that, we need to use the first argument type from the intrinsic
    // when resolving the overloaded intrinsic.
    llvm::Intrinsic::ID      lwvmIntrinsic  = llvm::Intrinsic::ctpop;
    std::vector<llvm::Type*> overloadedArgs = {intrinsicArgTypes[0]};
    llvm::Function*          intrinsicFunc = llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic, overloadedArgs );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> intrinsicArgs = getIntrinsicArguments( builder, ptxIntrinsicFunc );
    llvm::Value*              returlwalue   = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    // popc always returns a u32
    llvm::Type* i32Ty = llvm::Type::getInt32Ty( m_context );
    returlwalue       = builder.CreateTrunc( returlwalue, i32Ty );

    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addLdStIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    // Load instructions have a return value, store instructions do not.
    llvm::Type* returnType = instruction.signature[0].llvmType;

    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    // We only need to use the lwvm.load intrinsic if we can't express the load
    // as a standard LLVM instruction. This only happens if we're using a
    // non-default cache modifier. Based on observation, non-default cache
    // modifiers are only applied to global and generic memory loads, and are
    // ignored with all other address spaces.
    //
    // See https://docs.lwpu.com/lwca/parallel-thread-exelwtion/index.html#cache-operators for more information.
    const bool needLoadStoreIntrinsic = ( instruction.modifiers.addressSpace == AddressSpace::global
                                          || instruction.modifiers.addressSpace == AddressSpace::unspecified )
                                        && instruction.modifiers.cacheOp != CacheOp::ca;

    llvm::Value* returlwalue = nullptr;
    if( needLoadStoreIntrinsic )
    {
        // Load and store instructions use different arguments and their ptr and value args are reversed
        llvm::Intrinsic::ID lwvmIntrinsic;
        int                 valIdx;
        const int           ptrIdx = 1;
        if( instruction.ptxOpCode == ptx_ld_Instr )
        {
            lwvmIntrinsic = llvm::Intrinsic::lwvm_ld;
            valIdx        = 0;
        }
        else if( instruction.ptxOpCode == ptx_st_Instr )
        {
            lwvmIntrinsic = llvm::Intrinsic::lwvm_st;
            valIdx        = 2;
        }
        else
            RT_ASSERT_FAIL_MSG( "unrecognized PTX op code" );

        // TODO: Handle ld.global.nc instructions (these result in a .CONSTANT modifier for loads, which we aren't lwrrently emitting)
        std::vector<llvm::Type*> overLoadedArgTypes = {instruction.signature[valIdx].llvmType,
                                                       instruction.signature[ptrIdx].llvmType};
        llvm::Function* intrinsicFunc = llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic, overLoadedArgTypes );

        std::vector<llvm::Value*> intrinsicArgs = getLdStIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );
        returlwalue                             = builder.CreateCall( intrinsicFunc, intrinsicArgs );
    }
    // The compiler ignores cache modifiers for local, const, shared, and param
    // instructions, so we can emit standard LLVM load instructions for those.
    else
    {
        // TODO: The compiler fails with the PARAM address space. Investigate how we should handle those.
        // - The lwvm::AddressSpace enum has a comment stating that the PARAM is "LWVM internal" and "use only for LWPTX"
        // - The LWPTX backend has a bunch of special handling for that address space.
        // - The PTX ISA states that the param address space has some extra restrictions (they're read only,
        //   always load parameters, etc). Should we attempt to enforce those restrictions? Is there an existing pass
        //   that will do so?

        if( instruction.ptxOpCode == ptx_ld_Instr )
        {
            RT_ASSERT_MSG( ptxIntrinsicFunc->arg_size() == 1, "Wrong number of LD arguments" );
            llvm::Value* valToLoad = ptxIntrinsicFunc->arg_begin();

            returlwalue = builder.CreateLoad( valToLoad );
        }
        else
        {
            RT_ASSERT_MSG( ptxIntrinsicFunc->arg_size() == 2, "Wrong number of ST arguments" );
            auto         arg_it     = ptxIntrinsicFunc->arg_begin();
            llvm::Value* valToStore = arg_it++;
            llvm::Value* storeLoc   = arg_it;

            builder.CreateStore( storeLoc, valToStore );
        }
    }

    if( instruction.ptxOpCode == ptx_ld_Instr )
        builder.CreateRet( returlwalue );
    else
        builder.CreateRetVoid();
}

void D2IRIntrinsicBuilder::addShfIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    llvm::Intrinsic::ID      lwvmIntrinsic  = llvm::Intrinsic::lwvm_shf;
    std::vector<llvm::Type*> overloadedArgs = {intrinsicArgTypes[0]};
    llvm::Function*          intrinsicFunc = llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic, overloadedArgs );

    std::vector<llvm::Value*> intrinsicArgs = getShfIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );
    llvm::Value*              returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );
    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addShlIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    std::vector<llvm::Value*> instructionArgs = getIntrinsicArguments( builder, ptxIntrinsicFunc );

    llvm::Value* valToShift = instructionArgs[0];

    // PTX shl instructions always take a 32 bit int as their shift value,
    // while LLVM shl takes an int the same width as the bit register, so we
    // need to truncate/extend arg 2 to i<N>, where N is the return value. This
    // shouldn't matter in practice, as PTX clamps the shift value to N anyway.
    llvm::Value* shiftVal = instructionArgs[1];
    if( shiftVal->getType()->getIntegerBitWidth() < returnType->getIntegerBitWidth() )
        shiftVal = builder.CreateZExt( shiftVal, returnType );
    else if( shiftVal->getType()->getIntegerBitWidth() > returnType->getIntegerBitWidth() )
        shiftVal = builder.CreateTrunc( shiftVal, returnType );

    // TODO: The PTX ISA states that inputs are clamped to the register size
    // (e.g. shifting a b16 will get clamped to 16). The LLVM instruction
    // reference states that it produces a poison value if a value greater than
    // register width is used with its shl. Should we add instructions to
    // handle this case?

    llvm::Value* returlwalue = builder.CreateShl( valToShift, shiftVal );

    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addMovIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    // All mov instructions should have a return value plus single argument
    RT_ASSERT( instruction.signature.size() == 2 );

    // Make sure the return type and arg type are the same.
    RT_ASSERT( instruction.signature[0].llvmType == instruction.signature[1].llvmType );

    // For mov instructions, we simply return the given argument.
    llvm::Type*              returnType        = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes = {instruction.signature[1].llvmType};

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    builder.CreateRet( ptxIntrinsicFunc->arg_begin() );
}

void D2IRIntrinsicBuilder::addDp2aIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    llvm::Intrinsic::ID llvmIntrinsic = llvm::Intrinsic::lwvm_idp2a;
    llvm::Function*     intrinsicFunc = llvm::Intrinsic::getDeclaration( m_module, llvmIntrinsic );

    std::vector<llvm::Value*> intrinsicArgs = getDp2aIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );

    llvm::Value* returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    builder.CreateRet( returlwalue );
}

void D2IRIntrinsicBuilder::addDp4aIntrinsic( const PTXIntrinsicInfo& instruction )
{
    std::string ptxIntrinsicName = getPtxIntrinsicName( instruction.name );

    llvm::Type*              returnType = instruction.signature[0].llvmType;
    std::vector<llvm::Type*> intrinsicArgTypes;
    for( size_t i = 1; i < instruction.signature.size(); ++i )
        intrinsicArgTypes.push_back( instruction.signature[i].llvmType );

    llvm::Function* ptxIntrinsicFunc = getOrInsertPtxIntrinsicDeclaration( ptxIntrinsicName, returnType, intrinsicArgTypes );

    llvm::BasicBlock* entryBlock = llvm::BasicBlock::Create( m_context, "", ptxIntrinsicFunc );
    llvm::IRBuilder<> builder( entryBlock );

    llvm::Intrinsic::ID llvmIntrinsic = llvm::Intrinsic::lwvm_idp4a;
    llvm::Function*     intrinsicFunc = llvm::Intrinsic::getDeclaration( m_module, llvmIntrinsic );

    std::vector<llvm::Value*> intrinsicArgs = getDp4aIntrinsicArguments( instruction, builder, ptxIntrinsicFunc, intrinsicFunc );

    llvm::Value* returlwalue = builder.CreateCall( intrinsicFunc, intrinsicArgs );

    builder.CreateRet( returlwalue );
}

std::vector<llvm::Type*> D2IRIntrinsicBuilder::getPtxIntrinsicArgTypes( const InstructionSignature& signature )
{
    std::vector<llvm::Type*> result;

    for( size_t i = 1; i < signature.size(); ++i )
        result.push_back( signature[i].llvmType );

    return result;
}

llvm::Function* D2IRIntrinsicBuilder::getOrInsertPtxIntrinsicDeclaration( const std::string& ptxIntrinsicName,
                                                                          llvm::Type*        returnType,
                                                                          const std::vector<llvm::Type*>& argTypes )
{
    llvm::FunctionType* ptxIntrinsicFuncType = llvm::FunctionType::get( returnType, argTypes, false );

    llvm::Function* oldPtxIntrinsicFunc = m_module->getFunction( ptxIntrinsicName );
    llvm::Function* ptxIntrinsicFunc =
        llvm::Function::Create( ptxIntrinsicFuncType, llvm::GlobalValue::LinkOnceAnyLinkage, ptxIntrinsicName, m_module );

    if( oldPtxIntrinsicFunc != nullptr )
    {
        ptxIntrinsicFunc->takeName( oldPtxIntrinsicFunc );
        oldPtxIntrinsicFunc->replaceAllUsesWith( ptxIntrinsicFunc );
    }

    // Intrinsic function should have "alwaysinline" and "nounwind"
    ptxIntrinsicFunc->addFnAttr( llvm::Attribute::AlwaysInline );
    ptxIntrinsicFunc->addFnAttr( llvm::Attribute::NoUnwind );

    return ptxIntrinsicFunc;
}

llvm::Function* D2IRIntrinsicBuilder::getTexIntrinsicDeclaration( llvm::Intrinsic::ID         lwvmIntrinsic,
                                                                  const InstructionSignature& signature,
                                                                  const bool                  isSparse )
{
    // Texture intrinsics are overloaded on the type of the returned texture data.
    llvm::Type* texDataType = signature[0].llvmType;

    // Sparse intrinsics return an additional argument in a struct (e.g. { <4 x T>, i1} )
    if( isSparse )
    {
        llvm::StructType* returnTypeStruct = llvm::cast<llvm::StructType>( texDataType );

        // The texture data is always returned as the first element of the struct.
        if( returnTypeStruct->element_begin() == returnTypeStruct->element_end() )
            RT_ASSERT_FAIL_MSG( "Texture instruction has empty return struct" );
        texDataType = *returnTypeStruct->element_begin();
    }

    // LWVM texture intrinsics return texture data as vectors, even if they
    // contain a single elements. Our OptiX PTX intrinsics don't do that, so we
    // need to adjust the type here.
    if( !texDataType->isVectorTy() )
        texDataType = llvm::VectorType::get( texDataType, 1 );

    llvm::SmallVector<llvm::Type*, 1> args = {texDataType};
    return llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic, args );
}

llvm::Function* D2IRIntrinsicBuilder::getTld4IntrinsicDeclaration( llvm::Intrinsic::ID         lwvmIntrinsic,
                                                                   const InstructionSignature& signature )
{
    // Texture intrinsics are overloaded on the type of the returned texture data.

    llvm::StructType* returnTypeStruct = llvm::cast<llvm::StructType>( signature[0].llvmType );

    // The texture data is always returned as the first element of the struct.
    if( returnTypeStruct->element_begin() == returnTypeStruct->element_end() )
        RT_ASSERT_FAIL_MSG( "Texture instruction has empty return struct" );
    llvm::Type* texDataType = *returnTypeStruct->element_begin();

    llvm::SmallVector<llvm::Type*, 1> args = {texDataType};
    return llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic, args );
}

llvm::Function* D2IRIntrinsicBuilder::getSurfaceIntrinsicDeclaration( llvm::Intrinsic::ID         lwvmIntrinsic,
                                                                      const InstructionSignature& signature )
{
    llvm::Type* texDataType = signature[0].llvmType;
    if( lwvmIntrinsic == llvm::Intrinsic::lwvm_surface_store )
        texDataType = signature[signature.size() - 1].llvmType;

    llvm::SmallVector<llvm::Type*, 1> args = {texDataType};
    return llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic, args );
}

llvm::Function* D2IRIntrinsicBuilder::getIntrinsicDeclaration( llvm::Intrinsic::ID lwvmIntrinsic, const InstructionSignature& signature )
{
    if( llvm::Intrinsic::isOverloaded( lwvmIntrinsic ) )
    {
        std::vector<llvm::Type*> args = {};

        llvm::SmallVector<llvm::Intrinsic::IITDescriptor, 128> someDesc;
        llvm::Intrinsic::getIntrinsicInfoTableEntries( lwvmIntrinsic, someDesc );

        // For every overloaded type in the IITDescriptor (those types
        // show up as "Argument"), add a corresponding LLVM type to the list of
        // arguments we use to get the declaration.
        // Example:
        //     ToTy llvm.lwvm.cvt.ToTy.FromTy( i32 flags, FromTy x, ToTy y )
        // ends up as:
        //     [ FromTy, ToTy ]
        int                    numOverloadedTypes = 0;
        std::set<unsigned int> overloadedArgNumbers;
        for( size_t i = 0; i < someDesc.size(); i++ )
        {
            const llvm::Intrinsic::IITDescriptor& lwrrDesc = someDesc[i];
            if( lwrrDesc.Kind == llvm::Intrinsic::IITDescriptor::Argument )
            {
                // If we haven't already added this type to the overloaded arguments, do so.
                unsigned int lwrrArgNumber = lwrrDesc.getArgumentNumber();
                if( overloadedArgNumbers.find( lwrrArgNumber ) == overloadedArgNumbers.end() )
                {
                    llvm::Type* llvmType = signature[numOverloadedTypes].llvmType;

                    // LWVM IR intrinsics use <2 x half> for f16x2, but OptiX's PTX intrinsics use i32.
                    if( signature[numOverloadedTypes].ptxType.compare( "f16x2" ) == 0 )
                        llvmType = llvm::VectorType::get( llvm::Type::getHalfTy( m_context ), 2 );

                    numOverloadedTypes++;

                    args.push_back( llvmType );
                    overloadedArgNumbers.insert( lwrrArgNumber );
                }
            }
        }

        return llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic, args );
    }
    else
        return llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic );
}

llvm::Function* D2IRIntrinsicBuilder::getAtomIntrinsicDeclaration( llvm::Intrinsic::ID         lwvmIntrinsic,
                                                                   const InstructionSignature& signature )
{
    // The atomic intrinsics are overloaded based on their pointer argument
    std::vector<llvm::Type*> args = {signature[1].llvmType};
    return llvm::Intrinsic::getDeclaration( m_module, lwvmIntrinsic, args );
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getIntrinsicArguments( llvm::IRBuilder<>& builder, llvm::Function* ptxIntrinsicFunction )
{
    std::vector<llvm::Value*> args;

    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        args.push_back( it );

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getMathIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                           llvm::IRBuilder<>&      builder,
                                                                           llvm::Function*         ptxIntrinsicFunction,
                                                                           llvm::Function*         intrinsicFunction,
                                                                           bool                    hasMathFlag )
{
    std::vector<llvm::Value*> args;

    if( hasMathFlag )
    {
        lwvm::MathFlag mathFlags{};
        mathFlags.U.Rounding = ptxToLwvmRoundMode( instruction.modifiers.roundMode );
        mathFlags.U.Saturate = ptxToLwvmSaturate( instruction.modifiers.sat );

        llvm::Type*  i32Ty           = llvm::IntegerType::get( m_context, 32 );
        llvm::Value* packedMathFlags = llvm::ConstantInt::get( i32Ty, mathFlags.V );
        args.push_back( packedMathFlags );
    }

    int lwrrArg = 1;
    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
    {
        if( instruction.signature[lwrrArg].ptxType.compare( "f16x2" ) == 0 )
            args.push_back( builder.CreateBitCast( it, llvm::VectorType::get( llvm::Type::getHalfTy( m_context ), 2 ) ) );
        else
            args.push_back( it );
        lwrrArg++;
    }

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getCvtIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                          llvm::IRBuilder<>&      builder,
                                                                          llvm::Function*         ptxIntrinsicFunction,
                                                                          llvm::Function*         intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    lwvm::CvtFlag cvtFlags{};
    cvtFlags.U.Ftz            = ptxToLwvmFtz( instruction.modifiers.ftz );
    cvtFlags.U.Sat            = ptxToLwvmSaturate( instruction.modifiers.sat );
    cvtFlags.U.Rounding       = ptxToLwvmRoundMode( instruction.modifiers.roundMode );
    cvtFlags.U.SrcIsUnsigned  = startswith( instruction.signature[1].ptxType, "u" );
    cvtFlags.U.DestIsUnsigned = startswith( instruction.signature[0].ptxType, "u" );

    llvm::Type*  i32Ty          = llvm::IntegerType::get( m_context, 32 );
    llvm::Value* packedCvtFlags = llvm::ConstantInt::get( i32Ty, cvtFlags.V );
    args.push_back( packedCvtFlags );

    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        args.push_back( it );

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getTexIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                          llvm::IRBuilder<>&      builder,
                                                                          llvm::Function*         ptxIntrinsicFunction,
                                                                          llvm::Function*         intrinsicFunction,
                                                                          bool                    hasLOD,
                                                                          bool                    hasGradients,
                                                                          bool                    isLoad )
{
    std::vector<llvm::Value*> args;

    // Build the mode flags.
    lwvm::TexMode texFlags{};

    texFlags.U.Dim = ptxToLwvmTextureDimensionality( instruction.modifiers.texDim );

    // The PTX compiler lowers texture fetches in unified mode (i.e. no sampler) with float coordinates to 2D texture
    // calls as a workaround for a driver bug. We must do the same.
    if( instruction.modifiers.texDim == TextureDimensionality::dim1D && instruction.signature[2].ptxType[0] == 'f' )
        texFlags.U.Dim = lwvm::TexSurfDim::DIM_2D;

    if( hasLOD )
        texFlags.U.LodAdjust = lwvm::LOD_ABSOLUTE;

    texFlags.U.IsCombinedTexSamp = true;

    llvm::Type*  i32Ty          = llvm::IntegerType::get( m_context, 32 );
    llvm::Value* packedTexFlags = llvm::ConstantInt::get( i32Ty, texFlags.V );
    args.push_back( packedTexFlags );

    // Put intrinsic func args into vector, for colwenience.
    llvm::SmallVector<llvm::Value*, 8> ptxIntrinsicArgs;
    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        ptxIntrinsicArgs.push_back( it );

    // Texture object
    args.push_back( ptxIntrinsicArgs[0] );

    // Undef for sampler (OptiX doesn't seem to support separate textures and samplers, maybe 7 does?)
    llvm::Type*  i64Ty    = llvm::IntegerType::get( m_context, 64 );
    llvm::Value* undefI64 = llvm::UndefValue::get( i64Ty );
    args.push_back( undefI64 );

    llvm::Type* floatTy = llvm::Type::getFloatTy( m_context );

    // Unpack the x, y, and z coordinates from the input vector and add
    // them to the intrinsic arguments.

    llvm::Value* coordAggregate = ptxIntrinsicArgs[1];
    for( int i = 0; i < getDimensionalitySize( instruction.modifiers.texDim ); ++i )
    {
        llvm::Value* lwrrCoord;

        // If this is an array instruction, the first argument is a struct
        // whose first argument is the array index, otherwise, it's either
        // a vector or scalar.
        if( isArrayDimensionality( instruction.modifiers.texDim ) )
            lwrrCoord = builder.CreateExtractValue( coordAggregate, i + 1 );
        else if( coordAggregate->getType()->isVectorTy() )
        {
            llvm::Value* index = llvm::ConstantInt::get( i32Ty, i );
            lwrrCoord          = builder.CreateExtractElement( coordAggregate, index );
        }
        else
            lwrrCoord = coordAggregate;

        // TODO: Confirm that this is the same thing the LWPTX path does in SASS
        // If the current coordinate is not a float value, colwert it to one.
        if( lwrrCoord->getType() != floatTy && hasGradients )
        {
            if( instruction.signature[2].ptxType == "s32" )
                lwrrCoord = builder.CreateSIToFP( lwrrCoord, floatTy );
            else
                lwrrCoord = builder.CreateUIToFP( lwrrCoord, floatTy );
        }

        args.push_back( lwrrCoord );
    }

    // Push back zeroes for unused coord arguments.
    const bool   usesF32Coordinates = instruction.signature[2].ptxType[0] == 'f';
    llvm::Type*  coordType          = usesF32Coordinates || hasGradients ? floatTy : i32Ty;
    llvm::Value* zeroVal            = llvm::Constant::getNullValue( coordType );

    for( int i = getDimensionalitySize( instruction.modifiers.texDim ); i < 3; ++i )
        args.push_back( zeroVal );

    // Push back array idx arg (undef if not an array dimensionality, the first
    // argument of the coordinate aggregate otherwise)
    if( !isArrayDimensionality( instruction.modifiers.texDim ) )
    {
        llvm::Value* undefI32 = llvm::UndefValue::get( i32Ty );
        args.push_back( undefI32 );
    }
    else
    {
        // Array index
        llvm::Value* arrayIdx = builder.CreateExtractValue( coordAggregate, 0 );
        args.push_back( arrayIdx );
    }

    // If the signature contains a texture level,
    if( hasLOD )
    {
        llvm::Value* lod = ptxIntrinsicArgs[2];
        args.push_back( lod );
    }
    else if( hasGradients )
    {
        // Grad instructions use derivatives instead of an lod.
        for( unsigned int i = 0; i < 2; ++i )
        {
            llvm::Value* lwrrDerivativeVector = ptxIntrinsicArgs[2 + i];
            for( int j = 0; j < getDimensionalitySize( instruction.modifiers.texDim ); ++j )
            {
                llvm::Value* index = llvm::ConstantInt::get( i32Ty, j );
                llvm::Value* deriv = builder.CreateExtractElement( lwrrDerivativeVector, index );
                args.push_back( deriv );
            }

            llvm::Value* zeroFloat = llvm::Constant::getNullValue( floatTy );
            for( int j = getDimensionalitySize( instruction.modifiers.texDim ); j < 3; ++j )
            {
                args.push_back( zeroFloat );
            }
        }
    }
    else
    {
        llvm::Value* undefLod = llvm::UndefValue::get( coordType );
        args.push_back( undefLod );
    }

    // Last argument is lodClamp for sparse textures, multisampling pos for texture loads. We don't lwrrently use it in
    // either case
    if( instruction.hasPredicateOutput || isLoad )
    {
        llvm::Value* lodClamp = llvm::UndefValue::get( coordType );
        args.push_back( lodClamp );
    }

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getTld4IntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                           llvm::IRBuilder<>&      builder,
                                                                           llvm::Function*         ptxIntrinsicFunction,
                                                                           llvm::Function*         intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    // Build the mode flags.
    lwvm::TexMode texFlags{};
    texFlags.U.Dim = ptxToLwvmTextureDimensionality( instruction.modifiers.texDim );

    llvm::Type*  i32Ty           = llvm::IntegerType::get( m_context, 32 );
    llvm::Value* packedTld4Flags = llvm::ConstantInt::get( i32Ty, texFlags.V );
    args.push_back( packedTld4Flags );

    // Put intrinsic func args into vector, for colwenience.
    llvm::SmallVector<llvm::Value*, 8> ptxIntrinsicArgs;
    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        ptxIntrinsicArgs.push_back( it );

    // Texture object
    args.push_back( ptxIntrinsicArgs[0] );

    // Undef for sampler (OptiX doesn't seem to support separate textures and samplers, maybe 7 does?)
    llvm::Type*  i64Ty    = llvm::IntegerType::get( m_context, 64 );
    llvm::Value* undefI64 = llvm::UndefValue::get( i64Ty );
    args.push_back( undefI64 );

    llvm::Type* floatTy = llvm::Type::getFloatTy( m_context );

    // Unpack the x, y, and z coordinates from the input vector and add
    // them to the intrinsic arguments.

    llvm::Value* coordAggregate = ptxIntrinsicArgs[1];
    for( int i = 0; i < getDimensionalitySize( instruction.modifiers.texDim ); ++i )
    {
        llvm::Value* lwrrCoord;

        // If this is an array instruction, the first argument is a struct
        // whose first argument is the array index, otherwise, it's either
        // a vector or scalar.
        if( isArrayDimensionality( instruction.modifiers.texDim ) )
            lwrrCoord = builder.CreateExtractValue( coordAggregate, i + 1 );
        else if( coordAggregate->getType()->isVectorTy() )
        {
            llvm::Value* index = llvm::ConstantInt::get( i32Ty, i );
            lwrrCoord          = builder.CreateExtractElement( coordAggregate, index );
        }
        else
            lwrrCoord = coordAggregate;

        // TODO: Confirm that this is the same thing the LWPTX path does in SASS
        // If the current coordinate is not a float value, colwert it to one.
        if( lwrrCoord->getType() != floatTy )
        {
            if( instruction.signature[2].ptxType == "s32" )
                lwrrCoord = builder.CreateSIToFP( lwrrCoord, floatTy );
            else
                lwrrCoord = builder.CreateUIToFP( lwrrCoord, floatTy );
        }

        args.push_back( lwrrCoord );
    }

    // Push back undefs for unused coord arguments.
    llvm::Value* undefFloat = llvm::UndefValue::get( floatTy );
    for( int i = getDimensionalitySize( instruction.modifiers.texDim ); i < 3; ++i )
        args.push_back( undefFloat );

    // Push back array idx arg (undef if not an array dimensionality, the first
    // argument of the coordinate aggregate otherwise)
    if( !isArrayDimensionality( instruction.modifiers.texDim ) )
    {
        llvm::Value* undefI32 = llvm::UndefValue::get( i32Ty );
        args.push_back( undefI32 );
    }
    else
    {
        // Array index
        llvm::Value* arrayIdx = builder.CreateExtractValue( coordAggregate, 0 );
        args.push_back( arrayIdx );
    }

    // Push back component
    llvm::Value* component = llvm::ConstantInt::get( i32Ty, ptxToLwvmRgbaComponent( instruction.modifiers.rgbaComponent ) );
    args.push_back( component );

    // Undef for lodclamp
    args.push_back( undefFloat );

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getTxqIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                          llvm::IRBuilder<>&      builder,
                                                                          llvm::Function*         ptxIntrinsicFunction,
                                                                          llvm::Function*         intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    // Build the mode flags.
    lwvm::TexSurfQueryMode queryFlags{};

    queryFlags.U.Query = ptxToLwvmTextureQuery( instruction.modifiers.texQuery );

    // TODO(Kincaid): I'm pretty sure dimension is ignored, unless we're querying for array size. But, is that correct? The PTX instruction never takes a dimension argument.

    if( instruction.ptxOpCode == ptx_txq_level_Instr )
        queryFlags.U.IsExplicitLod = 1;

    // TODO(Kincaid): LWVM IR docs state that sample query requires index, but the PTX instruction doesn't have an argument. Should it infer this?

    llvm::Type*  i32Ty          = llvm::IntegerType::get( m_context, 32 );
    llvm::Value* packedTxqFlags = llvm::ConstantInt::get( i32Ty, queryFlags.V );
    args.push_back( packedTxqFlags );

    // Pass PTX intrinsic args to call
    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        args.push_back( it );

    // If we aren't doing an LOD query, pass undef for idx argument
    if( instruction.ptxOpCode != ptx_txq_level_Instr )
    {
        llvm::Value* undefI32 = llvm::UndefValue::get( i32Ty );
        args.push_back( undefI32 );
    }

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getSurfaceIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                              llvm::IRBuilder<>&      builder,
                                                                              llvm::Function* ptxIntrinsicFunction,
                                                                              llvm::Function* intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    // Build the mode flags.
    lwvm::SurfMode surfFlags{};

    surfFlags.U.Dim    = ptxToLwvmTextureDimensionality( instruction.modifiers.texDim );
    surfFlags.U.Border = ptxToLwvmClampMode( instruction.modifiers.clampMode );
    surfFlags.U.Opr    = ptxToLwvmCacheOp( instruction.modifiers.cacheOp );

    if( instruction.ptxOpCode == ptx_sust_p_Instr )
    {
        surfFlags.U.FMode = lwvm::FMT_MODE_FORMATTED;
        surfFlags.U.AMode = lwvm::ADDR_MODE_BYTE;
    }
    else if( instruction.ptxOpCode == ptx_sust_b_Instr || instruction.ptxOpCode == ptx_suld_b_Instr )
        surfFlags.U.FMode = lwvm::FMT_MODE_RAW;
    else
        RT_ASSERT_FAIL_MSG( "unrecognized PTX op code" );

    llvm::Type*  i64Ty           = llvm::IntegerType::get( m_context, 64 );
    llvm::Value* packedSurfFlags = llvm::ConstantInt::get( i64Ty, surfFlags.V );
    args.push_back( packedSurfFlags );

    // Put intrinsic func args into vector, for colwenience.
    llvm::SmallVector<llvm::Value*, 8> ptxIntrinsicArgs;
    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        ptxIntrinsicArgs.push_back( it );

    // Surface
    args.push_back( ptxIntrinsicArgs[0] );

    // Unpack the x, y, and z coordinates from the input vector and add
    // them to the intrinsic arguments.
    llvm::Type* i32Ty = llvm::IntegerType::get( m_context, 32 );

    llvm::Value* coordArg = ptxIntrinsicArgs[1];

    // 1D coordinates aren't packed in vectors
    if( coordArg->getType()->isVectorTy() )
    {
        // When reading from and writing to arrays, the first coordinate in the
        // vector is the array index.
        int arrayDimOffset = isArrayDimensionality( instruction.modifiers.texDim ) ? 1 : 0;
        for( int i = arrayDimOffset; i < getDimensionalitySize( instruction.modifiers.texDim ) + arrayDimOffset; ++i )
        {
            llvm::Value* index     = llvm::ConstantInt::get( i32Ty, i );
            llvm::Value* lwrrCoord = builder.CreateExtractElement( coordArg, index );

            args.push_back( lwrrCoord );
        }
    }
    else
        args.push_back( coordArg );

    // Push back undefs for unused coord arguments.
    llvm::Value* undefI32 = llvm::UndefValue::get( i32Ty );
    for( int i = getDimensionalitySize( instruction.modifiers.texDim ); i < 3; ++i )
        args.push_back( undefI32 );

    // Push back array idx arg (undef if not an array dimensionality, the first
    // argument of the coordinate aggregate otherwise)
    if( !isArrayDimensionality( instruction.modifiers.texDim ) )
        args.push_back( undefI32 );
    else
    {
        // Array index
        llvm::Value* coordVecIdx = llvm::ConstantInt::get( i32Ty, 0 );
        llvm::Value* arrayIdx    = builder.CreateExtractElement( coordArg, coordVecIdx );
        args.push_back( arrayIdx );
    }

    // If performing a store, push back value we'd like to store
    if( instruction.ptxOpCode == ptx_sust_b_Instr || instruction.ptxOpCode == ptx_sust_p_Instr )
        args.push_back( ptxIntrinsicArgs[2] );

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getAtomIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                           llvm::IRBuilder<>&      builder,
                                                                           llvm::Function*         ptxIntrinsicFunction,
                                                                           llvm::Function*         intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    lwvm::AtomicFlag flags{};
    flags.U.Ordering = ptxToLwvmMemOrdering( instruction.modifiers.memOrdering );
    flags.U.Scope    = ptxToLwvmMemScope( instruction.modifiers.memScope );

    const bool isFloat  = instruction.signature[0].ptxType[0] == 'f';
    const bool isSigned = instruction.signature[0].ptxType[0] == 's';
    flags.U.Opc         = ptxToLwvmAtomicOperation( instruction.modifiers.atomicOp, isSigned, isFloat );

    llvm::Type*  i32Ty       = llvm::IntegerType::get( m_context, 32 );
    llvm::Value* packedFlags = llvm::ConstantInt::get( i32Ty, flags.V );
    args.push_back( packedFlags );

    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        args.push_back( it );

    return args;
}

bool isStrongOrdering( MemOrdering ordering )
{
    // See table 17 at https://docs.lwpu.com/lwca/parallel-thread-exelwtion/index.html#operation-types
    return ordering == MemOrdering::relaxed || ordering == MemOrdering::acq || ordering == MemOrdering::rel
           || ordering == MemOrdering::acq_rel;
}

bool getCacheScopeOrder( const PTXIntrinsicInfo& instruction, lwvm::LoadStoreExtMode::CSO& outCacheScopeOrdering )
{
    // Default memory ordering is weak.
    MemOrdering memOrdering =
        instruction.modifiers.memOrdering != MemOrdering::unspecified ? instruction.modifiers.memOrdering : MemOrdering::weak;

    // If this instruction uses the texture domain (e.g. ld.global.nc), we want to sue the CONSTANT ordering.
    if( instruction.modifiers.texDomain == TexDomain::nc )
    {
        outCacheScopeOrdering = lwvm::LoadStoreExtMode::CSO::CONSTANT;
        return true;
    }
    else if( memOrdering == MemOrdering::weak )
    {
        if( instruction.modifiers.cacheOp == CacheOp::cg )
            outCacheScopeOrdering = lwvm::LoadStoreExtMode::CSO::WEAK_CG;  // .CG
        else
            outCacheScopeOrdering = lwvm::LoadStoreExtMode::CSO::WEAK;  // .CA
        return true;
    }
    else if( isStrongOrdering( memOrdering ) )
    {
        // Default memory scope is system.
        MemScope memScope =
            instruction.modifiers.memScope != MemScope::unspecified ? instruction.modifiers.memScope : MemScope::system;

        switch( memScope )
        {
            case MemScope::gpu:
                outCacheScopeOrdering = lwvm::LoadStoreExtMode::CSO::STRONG_GPU;  // .CG
                return true;
            case MemScope::cta:
                outCacheScopeOrdering = lwvm::LoadStoreExtMode::CSO::STRONG_CTA;  // .CA
                return true;
            case MemScope::system:
                outCacheScopeOrdering = lwvm::LoadStoreExtMode::CSO::STRONG_SYS;  // .CG
                return true;
            default:
                return false;
        }
    }

    return false;
}

uint64_t getCopHint( const PTXIntrinsicInfo& instruction )
{
    switch( instruction.modifiers.cacheOp )
    {
        case CacheOp::cs:
            return 6;  // NO_ALLOC (streaming)
        case CacheOp::lu:
            return 5;  // LAST_USE
        // TODO: Figure out how these last three map to the hints in table 39 at
        // https://p4viewer.lwpu.com/get/sw/compiler/docs/UnifiedLWVMIR/asciidoc/html/lwvmIR-1x.html#cop_encoding
        case CacheOp::ca:  // cache at all levels
        case CacheOp::cg:  // cache at global level
        case CacheOp::cv:  // don't cache and fetch again
        default:
            return 0;  // DEFAULT
    }
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getLdStIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                           llvm::IRBuilder<>&      builder,
                                                                           llvm::Function*         ptxIntrinsicFunction,
                                                                           llvm::Function*         intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    lwvm::LoadStoreExtMode ldStFlags{};

    lwvm::LoadStoreExtMode::CSO cso;
    bool                        validCacheScopeOrdering = getCacheScopeOrder( instruction, cso );
    RT_ASSERT_MSG( validCacheScopeOrdering, "Invalid cache scope ordering when generating llvm.lwvm.{ld,st} call" );
    ldStFlags.CacheScopeOrder = cso;

    ldStFlags.AllowConstantCache = 1;
    ldStFlags.IsVolatile         = ptxToLwvmIsVolatile( instruction.modifiers.vol );

    ldStFlags.COPHints = getCopHint( instruction );

    llvm::Type*  i64Ty         = llvm::IntegerType::get( m_context, 64 );
    llvm::Value* packedLdFlags = llvm::ConstantInt::get( i64Ty, ldStFlags.to() );
    args.push_back( packedLdFlags );

    // The ST PTX instruction arguments are [dstPtr, valToStore], while the LWVM intrinsic uses [valToStore, dstPtr]
    if( instruction.ptxOpCode == ptx_ld_Instr )
    {
        for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
            args.push_back( it );
    }
    else
    {
        RT_ASSERT_MSG( ptxIntrinsicFunction->arg_size() == 2, "Unexpected number of arguments for ST intrinsic" );
        auto         arg_it     = ptxIntrinsicFunction->arg_begin();
        llvm::Value* dstPtr     = arg_it++;
        llvm::Value* valToStore = arg_it;
        args.push_back( valToStore );
        args.push_back( dstPtr );
    }

    // lwvm.ld takes an optional i64 argument for a type descriptor. Use undef for it.
    llvm::Value* i64Undef = llvm::UndefValue::get( i64Ty );
    args.push_back( i64Undef );

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getShfIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                          llvm::IRBuilder<>&      builder,
                                                                          llvm::Function*         ptxIntrinsicFunction,
                                                                          llvm::Function*         intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    lwvm::ShfMode shfFlags{};

    if( instruction.ptxOpCode == ptx_shf_l_Instr )
        shfFlags.U.Direction = lwvm::SHIFT_LEFT;
    else if( instruction.ptxOpCode == ptx_shf_r_Instr )
        shfFlags.U.Direction = lwvm::SHIFT_RIGHT;
    else
        RT_ASSERT_FAIL_MSG( "unrecognized PTX op code" );

    shfFlags.U.ShiftType = ptxToLwvmShiftMode( instruction.modifiers.funnelShiftWrapMode );

    llvm::Type*  i32Ty          = llvm::IntegerType::get( m_context, 32 );
    llvm::Value* packedShfFlags = llvm::ConstantInt::get( i32Ty, shfFlags.V );
    args.push_back( packedShfFlags );

    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        args.push_back( it );

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getDp2aIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                           llvm::IRBuilder<>&      builder,
                                                                           llvm::Function*         ptxIntrinsicFunction,
                                                                           llvm::Function*         intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    // Put intrinsic func args into vector, for colwenience.
    llvm::SmallVector<llvm::Value*, 8> ptxIntrinsicArgs;
    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        ptxIntrinsicArgs.push_back( it );

    args.push_back( ptxIntrinsicArgs[0] );

    llvm::Type* i1Ty = llvm::IntegerType::get( m_context, 1 );

    const bool      isAUnsigned    = instruction.signature[1].ptxType[0] == 'u';
    llvm::Constant* isAUnsignedVal = llvm::ConstantInt::get( i1Ty, isAUnsigned ? 1 : 0 );
    args.push_back( isAUnsignedVal );

    args.push_back( ptxIntrinsicArgs[1] );

    const bool      isBUnsigned    = instruction.signature[2].ptxType[0] == 'u';
    llvm::Constant* isBUnsignedVal = llvm::ConstantInt::get( i1Ty, isBUnsigned ? 1 : 0 );
    args.push_back( isBUnsignedVal );

    int hiLoVal;
    if( instruction.ptxOpCode == ptx_dp2a_hi_Instr )
        hiLoVal = 1;
    else if( instruction.ptxOpCode == ptx_dp2a_lo_Instr )
        hiLoVal = 0;
    else
        RT_ASSERT_FAIL_MSG( "unrecognized PTX op code" );

    llvm::Constant* flag = llvm::ConstantInt::get( i1Ty, hiLoVal );
    args.push_back( flag );

    args.push_back( ptxIntrinsicArgs[2] );

    return args;
}

std::vector<llvm::Value*> D2IRIntrinsicBuilder::getDp4aIntrinsicArguments( const PTXIntrinsicInfo& instruction,
                                                                           llvm::IRBuilder<>&      builder,
                                                                           llvm::Function*         ptxIntrinsicFunction,
                                                                           llvm::Function*         intrinsicFunction )
{
    std::vector<llvm::Value*> args;

    // Put intrinsic func args into vector, for colwenience.
    llvm::SmallVector<llvm::Value*, 8> ptxIntrinsicArgs;
    for( auto it = ptxIntrinsicFunction->arg_begin(), end = ptxIntrinsicFunction->arg_end(); it != end; ++it )
        ptxIntrinsicArgs.push_back( it );

    args.push_back( ptxIntrinsicArgs[0] );

    llvm::Type* i1Ty = llvm::IntegerType::get( m_context, 1 );

    const bool      isAUnsigned    = instruction.signature[1].ptxType[0] == 'u';
    llvm::Constant* isAUnsignedVal = llvm::ConstantInt::get( i1Ty, isAUnsigned ? 1 : 0 );
    args.push_back( isAUnsignedVal );

    args.push_back( ptxIntrinsicArgs[1] );

    const bool      isBUnsigned    = instruction.signature[2].ptxType[0] == 'u';
    llvm::Constant* isBUnsignedVal = llvm::ConstantInt::get( i1Ty, isBUnsigned ? 1 : 0 );
    args.push_back( isBUnsignedVal );

    args.push_back( ptxIntrinsicArgs[2] );

    return args;
}

}  // namespace PTXIntrinsics
}  // namespace optix
