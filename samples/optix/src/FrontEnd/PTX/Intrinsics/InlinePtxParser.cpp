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

#include <FrontEnd/PTX/Intrinsics/InlinePtxParser.h>

#include <prodlib/exceptions/Assert.h>

#include <llvm/IR/Module.h>

namespace {
inline const llvm::StringRef removePrefix( const llvm::StringRef& someString, const llvm::StringRef& prefix )
{
    return someString.slice( prefix.size(), someString.size() );
}
}

namespace optix {
namespace PTXIntrinsics {
namespace InlinePtxParser {

OpCodeAndName getOpCodeAndNameFromOptixIntrinsic( llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef optixPtxPrefix = "optix.ptx.";
    if( !optixPtxIntrinsic->getName().startswith( optixPtxPrefix ) )
        return {ptx_unknown_Instr, llvm::StringRef( "" )};

    llvm::StringRef instructionString =
        optixPtxIntrinsic->getName().slice( optixPtxPrefix.size(), optixPtxIntrinsic->getName().size() );

    // We add a non-opcode prefix to some tex instruction
    const llvm::StringRef nonsparsePrefix = "nonsparse.";
    if( instructionString.startswith( nonsparsePrefix ) )
        instructionString = instructionString.slice( nonsparsePrefix.size(), instructionString.size() );

    if( instructionString.startswith( "abs." ) )
        return {ptx_abs_Instr, "abs"};
    if( instructionString.startswith( "max." ) )
        return {ptx_max_Instr, "max"};
    if( instructionString.startswith( "min." ) )
        return {ptx_min_Instr, "min"};
    if( instructionString.startswith( "add." ) )
        return {ptx_add_Instr, "add"};
    if( instructionString.startswith( "sub." ) )
        return {ptx_sub_Instr, "sub"};
    if( instructionString.startswith( "mul.hi." ) )
        return {ptx_mul_hi_Instr, "mul.hi"};
    if( instructionString.startswith( "mul." ) )
        return {ptx_mul_Instr, "mul"};
    if( instructionString.startswith( "div.full." ) )
        return {ptx_div_full_Instr, "div.full"};
    if( instructionString.startswith( "div." ) )
        return {ptx_div_Instr, "div"};
    if( instructionString.startswith( "fma." ) )
        return {ptx_fma_Instr, "fma"};
    if( instructionString.startswith( "mad.lo." ) )
        return {ptx_mad_lo_Instr, "mad.lo"};
    if( instructionString.startswith( "mad.hi." ) )
        return {ptx_mad_hi_Instr, "mad.hi"};
    if( instructionString.startswith( "mad.wide." ) )
        return {ptx_mad_wide_Instr, "mad.wide"};
    if( instructionString.startswith( "mad." ) )
        return {ptx_mad_Instr, "mad"};
    if( instructionString.startswith( "sqrt." ) )
        return {ptx_sqrt_Instr, "sqrt"};
    if( instructionString.startswith( "rsqrt." ) )
        return {ptx_rsqrt_Instr, "rsqrt"};
    if( instructionString.startswith( "rcp." ) )
        return {ptx_rcp_Instr, "rcp"};
    if( instructionString.startswith( "lg2." ) )
        return {ptx_lg2_Instr, "lg2"};
    if( instructionString.startswith( "ex2." ) )
        return {ptx_ex2_Instr, "ex2"};
    if( instructionString.startswith( "sin." ) )
        return {ptx_sin_Instr, "sin"};
    if( instructionString.startswith( "cos." ) )
        return {ptx_cos_Instr, "cos"};
    if( instructionString.startswith( "bfe." ) )
        return {ptx_bfe_Instr, "bfe"};
    if( instructionString.startswith( "bfi." ) )
        return {ptx_bfi_Instr, "bfi"};
    if( instructionString.startswith( "cvt." ) )
        return {ptx_cvt_Instr, "cvt"};
    if( instructionString.startswith( "brev." ) )
        return {ptx_brev_Instr, "brev"};
    if( instructionString.startswith( "prmt." ) )
        return {ptx_prmt_Instr, "prmt"};
    if( instructionString.startswith( "shf.l." ) )
        return {ptx_shf_l_Instr, "shf.l"};
    if( instructionString.startswith( "shf.r." ) )
        return {ptx_shf_r_Instr, "shf.r"};
    if( instructionString.startswith( "shl." ) )
        return {ptx_shl_Instr, "shl"};
    if( instructionString.startswith( "tex.grad." ) )
        return {ptx_tex_grad_Instr, "tex.grad"};
    if( instructionString.startswith( "tex.level." ) )
        return {ptx_tex_level_Instr, "tex.level"};
    if( instructionString.startswith( "tex." ) )
        return {ptx_tex_Instr, "tex"};
    if( instructionString.startswith( "tld4." ) )
        return {ptx_tld4_Instr, "tld4"};
    if( instructionString.startswith( "txq.level." ) )
        return {ptx_txq_level_Instr, "txq.level"};
    if( instructionString.startswith( "txq." ) )
        return {ptx_txq_Instr, "txq"};
    if( instructionString.startswith( "suld.b." ) )
        return {ptx_suld_b_Instr, "suld.b"};
    if( instructionString.startswith( "sust.b." ) )
        return {ptx_sust_b_Instr, "sust.b"};
    if( instructionString.startswith( "sust.p." ) )
        return {ptx_sust_p_Instr, "sust.p"};
    if( instructionString.startswith( "atom." ) )
        return {ptx_atom_Instr, "atom"};
    if( instructionString.startswith( "red." ) )
        return {ptx_red_Instr, "red"};
    if( instructionString.startswith( "set." ) )
        return {ptx_set_Instr, "set"};
    if( instructionString.startswith( "setp." ) )
        return {ptx_setp_Instr, "setp"};
    if( instructionString.startswith( "selp." ) )
        return {ptx_selp_Instr, "selp"};
    if( instructionString.startswith( "popc." ) )
        return {ptx_popc_Instr, "popc"};
    if( instructionString.startswith( "ld." ) )
        return {ptx_ld_Instr, "ld"};
    if( instructionString.startswith( "st." ) )
        return {ptx_st_Instr, "st"};
    if( instructionString.startswith( "mov." ) )
        return {ptx_mov_Instr, "mov"};
    if( instructionString.startswith( "dp2a.hi" ) )
        return {ptx_dp2a_hi_Instr, "dp2a.hi"};
    if( instructionString.startswith( "dp2a.lo" ) )
        return {ptx_dp2a_lo_Instr, "dp2a.lo"};
    if( instructionString.startswith( "dp4a." ) )
        return {ptx_dp4a_Instr, "dp4a"};

    return {ptx_unknown_Instr, ""};
}

static llvm::SmallVector<llvm::StringRef, 8> tokenizePtxString( const llvm::StringRef& ptxString )
{
    llvm::SmallVector<llvm::StringRef, 8> components;

    size_t lwrrComponentStart = 0;

    while( lwrrComponentStart < ptxString.size() )
    {
        const size_t lwrrComponentEnd = ptxString.find_first_of( '.', lwrrComponentStart );
        if( lwrrComponentEnd == llvm::StringRef::npos )
        {
            components.push_back( ptxString.slice( lwrrComponentStart, ptxString.size() ) );
            break;
        }

        components.push_back( ptxString.slice( lwrrComponentStart, lwrrComponentEnd ) );
        lwrrComponentStart = lwrrComponentEnd + 1;
    }

    return components;
}

static OperandType getOperandType( llvm::LLVMContext& context, llvm::Type* llvmType, const llvm::StringRef& disambiguateType )
{
    if( llvmType->isVoidTy() )
        return OperandType( "void", llvmType, false );

    llvm::Type* elementType = llvmType;
    if( elementType->isPointerTy() )
        elementType = elementType->getPointerElementType();

    if( llvm::StructType* typeAsStruct = llvm::dyn_cast<llvm::StructType>( elementType ) )
    {
        RT_ASSERT( typeAsStruct->getNumElements() >= 1 );
        // This handles two cases:
        // - Structs containing optional predicate outputs (e.g. {i32, 1}). In
        //   this case, the first element has the instruction's return type.
        // - Structs containing coordinates or input data. In this case, the
        //   elements are always the same width (because these types represent
        //   PTX vectors), so we can use the first element and disambiguation
        //   type to determine the PTX type.
        elementType = typeAsStruct->getElementType( 0 );
    }
    if( llvm::VectorType* typeAsVec = llvm::dyn_cast<llvm::VectorType>( elementType ) )
        elementType = typeAsVec->getElementType();

    // If the element is an integer type, use the first character of the
    // disambiguation type to determine its PTX type
    if( elementType->isIntegerTy() )
    {
        // We represent packed halfs as i32s
        if( disambiguateType.equals( "f16x2" ) )
            return OperandType( "f16x2", llvmType, llvmType->isSingleValueType() );

        char        typeChar      = disambiguateType[0];
        std::string ptxTypeString = std::string( 1, typeChar ) + std::to_string( elementType->getIntegerBitWidth() );
        return OperandType( ptxTypeString, llvmType, llvmType->isSingleValueType() );
    }
    else if( elementType->isDoubleTy() )
        return OperandType( "f64", llvmType, llvmType->isSingleValueType() );
    else if( elementType->isFloatTy() )
        return OperandType( "f32", llvmType, llvmType->isSingleValueType() );
    else if( elementType->isHalfTy() )
        return OperandType( "f16", llvmType, llvmType->isSingleValueType() );

    throw std::runtime_error( "Encountered unknown LLVM type" );
}

static InstructionSignature getSignatureFromOptixIntrinsic( llvm::LLVMContext&     context,
                                                            llvm::Function*        optixPtxIntrinsic,
                                                            const llvm::StringRef& retType,
                                                            const llvm::StringRef* inputType )
{
    // Get the InstructionSignature for the given OptiX intrinsic. Use retType
    // and inputType to disambiguate when necessary (for example, to get "u32",
    // "s32", or "b32" for an LLVM "i32")

    // If we have both retType and inputType, use retType to disambiguate the
    // return value, and inputType to disambiguate other values. Otherwise, use
    // retType to disambiguate all values.

    InstructionSignature signature;

    signature.push_back( getOperandType( context, optixPtxIntrinsic->getReturnType(), retType ) );

    const llvm::StringRef& disambiguateType = inputType ? *inputType : retType;
    for( auto arg_it = optixPtxIntrinsic->arg_begin(), arg_end = optixPtxIntrinsic->arg_end(); arg_it != arg_end; ++arg_it )
        signature.push_back( getOperandType( context, arg_it->getType(), disambiguateType ) );

    return signature;
}

PTXIntrinsicInfo getInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    OpCodeAndName         opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( opCodeAndName.name.size(), intrinsicString.size() ) );

    // Standard instructions indicate the type in their last component (e.g. ".u32" in "mov.u32")
    RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
    const llvm::StringRef retType = tokenizedIntrinsicString[tokenizedIntrinsicString.size() - 1];

    InstructionSignature signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, retType, nullptr );

    PTXIntrinsicInfo intrinsicInfo{};
    intrinsicInfo.name      = intrinsicString;
    intrinsicInfo.ptxOpCode = opCodeAndName.opCode;
    intrinsicInfo.signature = signature;
    return intrinsicInfo;
}

PTXIntrinsicInfo getLdStInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
    const llvm::StringRef disambiguateType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, disambiguateType, nullptr );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        // Memory ordering
        if( lwrrFlag.equals( "weak" ) )
            m.memOrdering = MemOrdering::weak;
        else if( lwrrFlag.equals( "relaxed" ) )
            m.memOrdering = MemOrdering::relaxed;
        else if( lwrrFlag.equals( "release" ) )
            m.memOrdering = MemOrdering::rel;
        else if( lwrrFlag.equals( "acquire" ) )
            m.memOrdering = MemOrdering::acq;
        else if( lwrrFlag.equals( "volatile" ) )
        {
            m.vol         = Volatile::vol;
            m.memOrdering = MemOrdering::relaxed;
        }
        // Scope
        else if( lwrrFlag.equals( "cta" ) )
            m.memScope = MemScope::cta;
        else if( lwrrFlag.equals( "gpu" ) )
            m.memScope = MemScope::gpu;
        else if( lwrrFlag.equals( "sys" ) )
            m.memScope = MemScope::system;
        // Cache modifiers
        else if( lwrrFlag.equals( "ca" ) )
            m.cacheOp = CacheOp::ca;
        else if( lwrrFlag.equals( "cg" ) )
            m.cacheOp = CacheOp::ca;
        else if( lwrrFlag.equals( "cs" ) )
            m.cacheOp = CacheOp::cs;
        else if( lwrrFlag.equals( "lu" ) )
            m.cacheOp = CacheOp::lu;
        else if( lwrrFlag.equals( "cv" ) )
            m.cacheOp = CacheOp::cv;
        else if( lwrrFlag.equals( "wb" ) )
            m.cacheOp = CacheOp::wb;
        else if( lwrrFlag.equals( "wt" ) )
            m.cacheOp = CacheOp::wt;
        // Address space
        else if( lwrrFlag.equals( "const" ) )
            m.addressSpace = AddressSpace::constant;
        else if( lwrrFlag.equals( "global" ) )
            m.addressSpace = AddressSpace::global;
        else if( lwrrFlag.equals( "local" ) )
            m.addressSpace = AddressSpace::local;
        else if( lwrrFlag.equals( "param" ) )
            m.addressSpace = AddressSpace::param;
        else if( lwrrFlag.equals( "shared" ) )
            m.addressSpace = AddressSpace::shared;
        else if( lwrrFlag.equals( "v2" ) )
            m.vectorSize = VectorSize::v2;
        else if( lwrrFlag.equals( "v4" ) )
            m.vectorSize = VectorSize::v4;
        else if( lwrrFlag.equals( "nc" ) )
            m.texDomain = TexDomain::nc;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getMathInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
    const llvm::StringRef disambiguateType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, disambiguateType, nullptr );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        if( lwrrFlag.equals( "approx" ) )
            m.approx = Approx::approx;
        else if( lwrrFlag.equals( "sat" ) )
            m.sat = Sat::sat;
        else if( lwrrFlag.equals( "ftz" ) )
            m.ftz = Ftz::ftz;
        else if( lwrrFlag.equals( "rn" ) )
            m.roundMode = RoundMode::rn;
        else if( lwrrFlag.equals( "rm" ) )
            m.roundMode = RoundMode::rm;
        else if( lwrrFlag.equals( "rp" ) )
            m.roundMode = RoundMode::rp;
        else if( lwrrFlag.equals( "rz" ) )
            m.roundMode = RoundMode::rz;
        else if( lwrrFlag.equals( "rni" ) )
            m.roundMode = RoundMode::rni;
        else if( lwrrFlag.equals( "rmi" ) )
            m.roundMode = RoundMode::rmi;
        else if( lwrrFlag.equals( "rpi" ) )
            m.roundMode = RoundMode::rpi;
        else if( lwrrFlag.equals( "rzi" ) )
            m.roundMode = RoundMode::rzi;
        else if( lwrrFlag.equals( "v2" ) )
            m.vectorSize = VectorSize::v2;
        else if( lwrrFlag.equals( "v4" ) )
            m.vectorSize = VectorSize::v4;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getAtomOrRedInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
    const llvm::StringRef disambiguateType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, disambiguateType, nullptr );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        // instructions
        if( lwrrFlag.equals( "add" ) )
            m.atomicOp = AtomicOperation::add;
        else if( lwrrFlag.equals( "inc" ) )
            m.atomicOp = AtomicOperation::inc;
        else if( lwrrFlag.equals( "dec" ) )
            m.atomicOp = AtomicOperation::dec;
        else if( lwrrFlag.equals( "max" ) )
            m.atomicOp = AtomicOperation::max;
        else if( lwrrFlag.equals( "min" ) )
            m.atomicOp = AtomicOperation::min;
        else if( lwrrFlag.equals( "exch" ) )
            m.atomicOp = AtomicOperation::exch;
        else if( lwrrFlag.equals( "and" ) )
            m.atomicOp = AtomicOperation::andOp;
        else if( lwrrFlag.equals( "or" ) )
            m.atomicOp = AtomicOperation::orOp;
        else if( lwrrFlag.equals( "xor" ) )
            m.atomicOp = AtomicOperation::xorOp;
        else if( lwrrFlag.equals( "cas" ) )
            m.atomicOp = AtomicOperation::cas;
        // memspace
        else if( lwrrFlag.equals( "global" ) )
            m.addressSpace = AddressSpace::global;
        else if( lwrrFlag.equals( "shared" ) )
            // TODO Isn't that unsupported in OptiX?
            m.addressSpace = AddressSpace::shared;
        // semantics
        else if( lwrrFlag.equals( "relaxed" ) )
            m.memOrdering = MemOrdering::relaxed;
        else if( lwrrFlag.equals( "acquire" ) )
            m.memOrdering = MemOrdering::acq;
        else if( lwrrFlag.equals( "release" ) )
            m.memOrdering = MemOrdering::rel;
        else if( lwrrFlag.equals( "acq_rel" ) )
            m.memOrdering = MemOrdering::acq_rel;
            // scope
        else if( lwrrFlag.equals( "cta" ) )
            m.memScope = MemScope::cta;
        else if( lwrrFlag.equals( "gpu" ) )
            m.memScope = MemScope::gpu;
        else if( lwrrFlag.equals( "sys" ) )
            m.memScope = MemScope::system;
        else if( lwrrFlag.equals( "noftz" ) )
            m.noftz = Noftz::noftz;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getSetOrSetpInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    InstructionSignature signature;

    if( opCodeAndName.opCode == ptx_set_Instr )
    {
        RT_ASSERT( tokenizedIntrinsicString.size() >= 2 );
        const llvm::StringRef compareDisambiguationType = tokenizedIntrinsicString.pop_back_val();
        const llvm::StringRef returnDisambiguationType  = tokenizedIntrinsicString.pop_back_val();
        signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, returnDisambiguationType, &compareDisambiguationType );
    }
    else
    {
        RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
        const llvm::StringRef disambiguateType = tokenizedIntrinsicString.pop_back_val();
        signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, disambiguateType, nullptr );
    }

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        if( lwrrFlag.equals( "ftz" ) )
            m.ftz = Ftz::ftz;
        else if( lwrrFlag.equals( "eq" ) )
            m.cmpOp = CompareOperator::eq;
        else if( lwrrFlag.equals( "ne" ) )
            m.cmpOp = CompareOperator::ne;
        else if( lwrrFlag.equals( "lt" ) )
            m.cmpOp = CompareOperator::lt;
        else if( lwrrFlag.equals( "le" ) )
            m.cmpOp = CompareOperator::le;
        else if( lwrrFlag.equals( "gt" ) )
            m.cmpOp = CompareOperator::gt;
        else if( lwrrFlag.equals( "ge" ) )
            m.cmpOp = CompareOperator::ge;
        else if( lwrrFlag.equals( "lo" ) )
            m.cmpOp = CompareOperator::lo;
        else if( lwrrFlag.equals( "ls" ) )
            m.cmpOp = CompareOperator::ls;
        else if( lwrrFlag.equals( "hi" ) )
            m.cmpOp = CompareOperator::hi;
        else if( lwrrFlag.equals( "hs" ) )
            m.cmpOp = CompareOperator::hs;
        else if( lwrrFlag.equals( "num" ) )
            m.cmpOp = CompareOperator::num;
        else if( lwrrFlag.equals( "nan" ) )
            m.cmpOp = CompareOperator::nan;
        else if( lwrrFlag.equals( "equ" ) )
            m.cmpOp = CompareOperator::equ;
        else if( lwrrFlag.equals( "neu" ) )
            m.cmpOp = CompareOperator::neu;
        else if( lwrrFlag.equals( "ltu" ) )
            m.cmpOp = CompareOperator::ltu;
        else if( lwrrFlag.equals( "leu" ) )
            m.cmpOp = CompareOperator::leu;
        else if( lwrrFlag.equals( "gtu" ) )
            m.cmpOp = CompareOperator::gtu;
        else if( lwrrFlag.equals( "geu" ) )
            m.cmpOp = CompareOperator::geu;
        else if( lwrrFlag.equals( "and" ) )
            m.boolOp = BooleanOperator::andOp;
        else if( lwrrFlag.equals( "or" ) )
            m.boolOp = BooleanOperator::orOp;
        else if( lwrrFlag.equals( "xor" ) )
            m.boolOp = BooleanOperator::xorOp;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getShfInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
    const llvm::StringRef disambiguateType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, disambiguateType, nullptr );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        if( lwrrFlag.equals( "wrap" ) )
            m.funnelShiftWrapMode = FunnelShiftWrapMode::wrap;
        else if( lwrrFlag.equals( "clamp" ) )
            m.funnelShiftWrapMode = FunnelShiftWrapMode::clamp;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getMovInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
    const llvm::StringRef disambiguateType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, disambiguateType, nullptr );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        if( lwrrFlag.equals( "v2" ) )
            m.vectorSize = VectorSize::v2;
        else if( lwrrFlag.equals( "v4" ) )
            m.vectorSize = VectorSize::v4;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getTxqInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
    const llvm::StringRef disambiguateType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, disambiguateType, nullptr );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        if( lwrrFlag.equals( "width" ) )
            m.texQuery = TextureQuery::width;
        else if( lwrrFlag.equals( "height" ) )
            m.texQuery = TextureQuery::height;
        else if( lwrrFlag.equals( "depth" ) )
            m.texQuery = TextureQuery::depth;
        else if( lwrrFlag.equals( "channel_data_type" ) )
            m.texQuery = TextureQuery::channelDataType;
        else if( lwrrFlag.equals( "channel_order" ) )
            m.texQuery = TextureQuery::channelOrder;
        else if( lwrrFlag.equals( "array_size" ) )
            m.texQuery = TextureQuery::arraySize;
        else if( lwrrFlag.equals( "num_mipmap_levels" ) )
            m.texQuery = TextureQuery::numMipmapLevels;
        else if( lwrrFlag.equals( "num_samples" ) )
            m.texQuery = TextureQuery::numSamples;
        else if( lwrrFlag.equals( "normalized_coords" ) )
            m.texQuery = TextureQuery::normalizedCoords;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getTexInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicStringWithSparsityPrefix =
        removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );

    bool            isSparse        = true;
    llvm::StringRef intrinsicString = intrinsicStringWithSparsityPrefix;
    if( intrinsicString.startswith( "nonsparse." ) )
    {
        isSparse        = false;
        intrinsicString = removePrefix( intrinsicString, "nonsparse." );
    }

    const OpCodeAndName opCodeAndName = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 2 );
    const llvm::StringRef coordDisambiguationType  = tokenizedIntrinsicString.pop_back_val();
    const llvm::StringRef returnDisambiguationType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature  signature =
        getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, returnDisambiguationType, &coordDisambiguationType );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        // Dimensionality
        if( lwrrFlag.equals( "1d" ) )
            m.texDim = TextureDimensionality::dim1D;
        else if( lwrrFlag.equals( "2d" ) )
            m.texDim = TextureDimensionality::dim2D;
        else if( lwrrFlag.equals( "3d" ) )
            m.texDim = TextureDimensionality::dim3D;
        else if( lwrrFlag.equals( "a1d" ) )
            m.texDim = TextureDimensionality::dim1DArray;
        else if( lwrrFlag.equals( "a2d" ) )
            m.texDim = TextureDimensionality::dim2DArray;
        else if( lwrrFlag.equals( "lwbe" ) )
            m.texDim = TextureDimensionality::dimLwbe;
        else if( lwrrFlag.equals( "alwbe" ) )
            m.texDim = TextureDimensionality::dimLwbeArray;
        else if( lwrrFlag.equals( "2dms" ) )
            m.texDim = TextureDimensionality::dim2D;
        else if( lwrrFlag.equals( "a2dms" ) )
            m.texDim = TextureDimensionality::dim2DArray;
        // Vector size
        else if( lwrrFlag.equals( "v2" ) )
            m.vectorSize = VectorSize::v2;
        else if( lwrrFlag.equals( "v4" ) )
            m.vectorSize = VectorSize::v4;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicStringWithSparsityPrefix, opCodeAndName.opCode, m, signature, isSparse};
}

PTXIntrinsicInfo getTld4InstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 2 );
    const llvm::StringRef coordDisambiguationType  = tokenizedIntrinsicString.pop_back_val();
    const llvm::StringRef returnDisambiguationType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature  signature =
        getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, returnDisambiguationType, &coordDisambiguationType );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        // Dimensionality
        if( lwrrFlag.equals( "2d" ) )
            m.texDim = TextureDimensionality::dim2D;
        else if( lwrrFlag.equals( "a1d" ) )
            m.texDim = TextureDimensionality::dim1DArray;
        else if( lwrrFlag.equals( "a2d" ) )
            m.texDim = TextureDimensionality::dim2DArray;
        else if( lwrrFlag.equals( "lwbe" ) )
            m.texDim = TextureDimensionality::dimLwbe;
        else if( lwrrFlag.equals( "alwbe" ) )
            m.texDim = TextureDimensionality::dimLwbeArray;
        // Vector size
        else if( lwrrFlag.equals( "v4" ) )
            m.vectorSize = VectorSize::v4;
        else if( lwrrFlag.equals( "r" ) )
            m.rgbaComponent = RgbaComponent::r;
        else if( lwrrFlag.equals( "g" ) )
            m.rgbaComponent = RgbaComponent::g;
        else if( lwrrFlag.equals( "b" ) )
            m.rgbaComponent = RgbaComponent::b;
        else if( lwrrFlag.equals( "a" ) )
            m.rgbaComponent = RgbaComponent::a;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getSurfInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 1 );
    const llvm::StringRef disambiguateType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature signature = getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, disambiguateType, nullptr );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        // Dimensionality
        if( lwrrFlag.equals( "1d" ) )
            m.texDim = TextureDimensionality::dim1D;
        else if( lwrrFlag.equals( "2d" ) )
            m.texDim = TextureDimensionality::dim2D;
        else if( lwrrFlag.equals( "3d" ) )
            m.texDim = TextureDimensionality::dim3D;
        else if( lwrrFlag.equals( "a1d" ) )
            m.texDim = TextureDimensionality::dim1DArray;
        else if( lwrrFlag.equals( "a2d" ) )
            m.texDim = TextureDimensionality::dim2DArray;
        // Vector size
        else if( lwrrFlag.equals( "v2" ) )
            m.vectorSize = VectorSize::v2;
        else if( lwrrFlag.equals( "v4" ) )
            m.vectorSize = VectorSize::v4;
        // Cache modifiers
        else if( lwrrFlag.equals( "ca" ) )
            m.cacheOp = CacheOp::ca;
        else if( lwrrFlag.equals( "cg" ) )
            m.cacheOp = CacheOp::cg;
        else if( lwrrFlag.equals( "cs" ) )
            m.cacheOp = CacheOp::cs;
        else if( lwrrFlag.equals( "lu" ) )
            m.cacheOp = CacheOp::lu;
        else if( lwrrFlag.equals( "cv" ) )
            m.cacheOp = CacheOp::cv;
        else if( lwrrFlag.equals( "wb" ) )
            m.cacheOp = CacheOp::wb;
        else if( lwrrFlag.equals( "wt" ) )
            m.cacheOp = CacheOp::wt;
        // Clamp mode
        else if( lwrrFlag.equals( "clamp" ) )
            m.clampMode = ClampMode::clamp;
        else if( lwrrFlag.equals( "trap" ) )
            m.clampMode = ClampMode::trap;
        else if( lwrrFlag.equals( "zero" ) )
            m.clampMode = ClampMode::zero;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

PTXIntrinsicInfo getCvtInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    // The instruction's type is always the last component
    RT_ASSERT( tokenizedIntrinsicString.size() >= 2 );
    const llvm::StringRef coordDisambiguationType  = tokenizedIntrinsicString.pop_back_val();
    const llvm::StringRef returnDisambiguationType = tokenizedIntrinsicString.pop_back_val();
    InstructionSignature  signature =
        getSignatureFromOptixIntrinsic( context, optixPtxIntrinsic, returnDisambiguationType, &coordDisambiguationType );

    PTXIntrinsicModifiers m{};

    for( const llvm::StringRef& lwrrFlag : tokenizedIntrinsicString )
    {
        if( lwrrFlag.equals( "rn" ) )
            m.roundMode = RoundMode::rn;
        else if( lwrrFlag.equals( "rm" ) )
            m.roundMode = RoundMode::rm;
        else if( lwrrFlag.equals( "rp" ) )
            m.roundMode = RoundMode::rp;
        else if( lwrrFlag.equals( "rz" ) )
            m.roundMode = RoundMode::rz;
        else if( lwrrFlag.equals( "rni" ) )
            m.roundMode = RoundMode::rni;
        else if( lwrrFlag.equals( "rmi" ) )
            m.roundMode = RoundMode::rmi;
        else if( lwrrFlag.equals( "rpi" ) )
            m.roundMode = RoundMode::rpi;
        else if( lwrrFlag.equals( "rzi" ) )
            m.roundMode = RoundMode::rzi;
        else if( lwrrFlag.equals( "sat" ) )
            m.sat = Sat::sat;
        else if( lwrrFlag.equals( "ftz" ) )
            m.ftz = Ftz::ftz;
        else
            RT_ASSERT_FAIL_MSG( "Unrecognized flag" );
    }

    return {intrinsicString, opCodeAndName.opCode, m, signature};
}

static InstructionSignature getSignatureFromOptixDotProductIntrinsic( llvm::LLVMContext&     context,
                                                                      llvm::Function*        optixPtxIntrinsic,
                                                                      const llvm::StringRef& aPtxType,
                                                                      const llvm::StringRef& bPtxType )
{
    // Dot product instructions are a little unusual, in that their type
    // specifiers don't determine the input and return types, but instead
    // indicate whether or their packed arguments are signed, and the return
    // type is dependent on the signed-ness of those types.

    // c is signed if either a or b is signed
    const bool            aOrBIsSigned = aPtxType[0] == 's' || bPtxType[0] == 's';
    const llvm::StringRef cPtxType     = aOrBIsSigned ? "s32" : "u32";

    InstructionSignature signature;

    // Argument c's type determines dest type
    signature.push_back( getOperandType( context, optixPtxIntrinsic->getReturnType(), cPtxType ) );

    RT_ASSERT( optixPtxIntrinsic->getFunctionType()->getNumParams() == 3 );
    auto arg_it = optixPtxIntrinsic->arg_begin();
    signature.push_back( getOperandType( context, arg_it->getType(), aPtxType ) );
    arg_it++;
    signature.push_back( getOperandType( context, arg_it->getType(), bPtxType ) );
    arg_it++;
    signature.push_back( getOperandType( context, arg_it->getType(), cPtxType ) );

    return signature;
}

PTXIntrinsicInfo getDp2aInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    RT_ASSERT( tokenizedIntrinsicString.size() >= 2 );
    const llvm::StringRef bPtxType = tokenizedIntrinsicString.pop_back_val();
    const llvm::StringRef aPtxType = tokenizedIntrinsicString.pop_back_val();

    InstructionSignature signature = getSignatureFromOptixDotProductIntrinsic( context, optixPtxIntrinsic, aPtxType, bPtxType );

    return {intrinsicString, opCodeAndName.opCode, {}, signature};
}

PTXIntrinsicInfo getDp4aInstructionFromOptixIntrinsic( llvm::LLVMContext& context, llvm::Function* optixPtxIntrinsic )
{
    const llvm::StringRef intrinsicString = removePrefix( optixPtxIntrinsic->getName(), "optix.ptx." );
    const OpCodeAndName   opCodeAndName   = getOpCodeAndNameFromOptixIntrinsic( optixPtxIntrinsic );

    const size_t flagStart = opCodeAndName.name.size() + 1;
    llvm::SmallVector<llvm::StringRef, 8> tokenizedIntrinsicString =
        tokenizePtxString( intrinsicString.slice( flagStart, intrinsicString.size() ) );

    RT_ASSERT( tokenizedIntrinsicString.size() >= 2 );
    const llvm::StringRef bPtxType = tokenizedIntrinsicString.pop_back_val();
    const llvm::StringRef aPtxType = tokenizedIntrinsicString.pop_back_val();

    InstructionSignature signature = getSignatureFromOptixDotProductIntrinsic( context, optixPtxIntrinsic, aPtxType, bPtxType );

    return {intrinsicString, opCodeAndName.opCode, {}, signature};
}

}  // InlinePtxParser
}  // PTXIntrinsics
}  //  optix
