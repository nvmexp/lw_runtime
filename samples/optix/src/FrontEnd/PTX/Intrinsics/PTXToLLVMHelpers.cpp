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


#include <FrontEnd/PTX/Intrinsics/PTXToLLVMHelpers.h>

#include <prodlib/exceptions/Assert.h>

namespace optix {
namespace PTXIntrinsics {

OperandType ptxArgToOperandTy( llvm::LLVMContext& context, ptxArgumentType type )
{
    switch( type )
    {
        case ptxB32AType:
        {
            llvm::Type* llvmType = llvm::IntegerType::get( context, 32 );
            return OperandType( "b32", llvmType, false );
        }
        case ptxU16AType:
        {
            llvm::Type* llvmType = llvm::IntegerType::get( context, 16 );
            return OperandType( "u16", llvmType, false );
        }
        case ptxU32AType:
        {
            llvm::Type* llvmType = llvm::IntegerType::get( context, 32 );
            return OperandType( "u32", llvmType, false );
        }
        case ptxS32AType:
        {
            llvm::Type* llvmType = llvm::IntegerType::get( context, 32 );
            return OperandType( "s32", llvmType, false );
        }
        case ptxScalarF32AType:
        {
            llvm::Type* llvmType = llvm::Type::getFloatTy( context );
            return OperandType( "f32", llvmType, true );
        }
        case ptxF32AType:
        {
            llvm::Type* llvmType = llvm::Type::getFloatTy( context );
            return OperandType( "f32", llvmType, false );
        }
        case ptxU64AType:
        {
            llvm::Type* llvmType = llvm::IntegerType::get( context, 64 );
            return OperandType( "u64", llvmType, false );
        }
        case ptxImageAType:
        {
            llvm::Type* llvmType = llvm::IntegerType::get( context, 64 );
            return OperandType( "image", llvmType, true );
        }
        case ptxF16x2AType:
        {
            llvm::Type* llvmType = llvm::Type::getFloatTy( context );
            return OperandType( "f16x2", llvmType, false );
        }
        case ptxVoidAType:
        {
            llvm::Type* llvmType = llvm::Type::getVoidTy( context );
            return OperandType( "void", llvmType, false );
        }
        case ptxMemoryAType:
        {
            // This is filled in later, based on the instruction modifiers
            return OperandType( "memory", nullptr, false );
        }
        case ptxPredicateAType:
        {
            llvm::Type* llvmType = llvm::IntegerType::get( context, 1 );
            return OperandType( "pred", llvmType, false );
        }
        default:
            RT_ASSERT_FAIL_MSG( "Encountered unknown PTX operand type: " + std::to_string( type ) );
    }

    RT_ASSERT_FAIL_MSG( "Encountered unknown PTX operand type: " + std::to_string( type ) );
}

bool isArrayDimensionality( TextureDimensionality dim )
{
    switch( dim )
    {
        case TextureDimensionality::unspecified:
            RT_ASSERT_FAIL_MSG( "unspecified texture dimensionality" );
        case TextureDimensionality::dim1DArray:
        case TextureDimensionality::dim2DArray:
        case TextureDimensionality::dim3DArray:
        case TextureDimensionality::dimLwbeArray:
            return true;
        case TextureDimensionality::dim1D:
        case TextureDimensionality::dim1DBuffer:
        case TextureDimensionality::dim2D:
        case TextureDimensionality::dim3D:
        case TextureDimensionality::dimLwbe:
            return false;
    }

    return false;
}

int getVectorSizeForDimensionality( TextureDimensionality dim )
{
    switch( dim )
    {
        case TextureDimensionality::unspecified:
            RT_ASSERT_FAIL_MSG( "unspecified texture dimensionality" );
        case TextureDimensionality::dim1D:
        case TextureDimensionality::dim1DArray:
        case TextureDimensionality::dim1DBuffer:
            return 1;
        case TextureDimensionality::dim2D:
        case TextureDimensionality::dim2DArray:
            return 2;
        case TextureDimensionality::dim3D:
        case TextureDimensionality::dim3DArray:
        case TextureDimensionality::dimLwbe:
        case TextureDimensionality::dimLwbeArray:
            return 4;
    }
    RT_ASSERT_FAIL_MSG( "Unrecognized texture dimensionality" );
}

int getDimensionalitySize( TextureDimensionality dim )
{
    switch( dim )
    {
        case TextureDimensionality::unspecified:
            // There is no default dimensionality
            RT_ASSERT_FAIL_MSG( "unspecified texture dimensionality" );
        case TextureDimensionality::dim1D:
        case TextureDimensionality::dim1DArray:
        case TextureDimensionality::dim1DBuffer:
            return 1;
        case TextureDimensionality::dim2D:
        case TextureDimensionality::dim2DArray:
            return 2;
        case TextureDimensionality::dim3D:
        case TextureDimensionality::dim3DArray:
        case TextureDimensionality::dimLwbe:
        case TextureDimensionality::dimLwbeArray:
            return 3;
    }
    RT_ASSERT_FAIL_MSG( "Unrecognized texture dimensionality" );
}

lwvm::TexSurfDim ptxToLwvmTextureDimensionality( TextureDimensionality dim )
{
    switch( dim )
    {
        case TextureDimensionality::unspecified:
            // There is no default dimensionality
            RT_ASSERT_FAIL_MSG( "unspecified texture dimensionality" );
        case TextureDimensionality::dim1D:
            return lwvm::DIM_1D;
        case TextureDimensionality::dim1DArray:
            return lwvm::DIM_1D_ARRAY;
        case TextureDimensionality::dim1DBuffer:
            return lwvm::DIM_1D_BUFFER;
        case TextureDimensionality::dim2D:
            return lwvm::DIM_2D;
        case TextureDimensionality::dim2DArray:
            return lwvm::DIM_2D_ARRAY;
        case TextureDimensionality::dim3D:
            return lwvm::DIM_3D;
        case TextureDimensionality::dim3DArray:
            return lwvm::DIM_3D_ARRAY;
        case TextureDimensionality::dimLwbe:
            return lwvm::DIM_LWBE;
        case TextureDimensionality::dimLwbeArray:
            return lwvm::DIM_LWBE_ARRAY;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized dimensionality" );
}

lwvm::RoundingMode ptxToLwvmRoundMode( RoundMode mode )
{
    // lwvm doesn't distinguish between integer and floating point round modes
    switch( mode )
    {
        // default round mode is rn
        case RoundMode::unspecified:
        case RoundMode::rn:
        case RoundMode::rni:
            return lwvm::RoundingMode::ROUND_RN;
        case RoundMode::rm:
        case RoundMode::rmi:
            return lwvm::RoundingMode::ROUND_RM;
        case RoundMode::rp:
        case RoundMode::rpi:
            return lwvm::RoundingMode::ROUND_RP;
        case RoundMode::rz:
        case RoundMode::rzi:
            return lwvm::RoundingMode::ROUND_RZ;
    }
    RT_ASSERT_FAIL_MSG( "Unrecognized round mode" );
}

unsigned int ptxToLwvmSaturate( Sat sat )
{
    if( sat == Sat::sat )
        return 1;
    else if( sat == Sat::unspecified )
        return 0;
    else
        RT_ASSERT_FAIL_MSG( "Unrecognized saturate mode" );
}

unsigned int ptxToLwvmFtz( Ftz ftz )
{
    if( ftz == Ftz::ftz )
        return 1;
    else if( ftz == Ftz::unspecified )
        return 0;
    else
        RT_ASSERT_FAIL_MSG( "Unrecognized ftz mode" );
}

unsigned int ptxToLwvmRgbaComponent( RgbaComponent comp )
{
    switch( comp )
    {
        case RgbaComponent::unspecified:
            // no default component
            RT_ASSERT_FAIL_MSG( "unspecified rbga component" );
        case RgbaComponent::r:
            return 0;
        case RgbaComponent::g:
            return 1;
        case RgbaComponent::b:
            return 2;
        case RgbaComponent::a:
            return 3;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized component" );
}

lwvm::CacheOp ptxToLwvmCacheOp( CacheOp cacheOp )
{
    switch( cacheOp )
    {
        case CacheOp::unspecified:
            return lwvm::CacheOp::CG;
        case CacheOp::cg:
            return lwvm::CacheOp::CG;
        case CacheOp::cs:
            return lwvm::CacheOp::CS;
        case CacheOp::ca:
            return lwvm::CacheOp::CA;
        case CacheOp::lu:
            return lwvm::CacheOp::LU;
        case CacheOp::cv:
            return lwvm::CacheOp::CV;
        case CacheOp::ci:
            return lwvm::CacheOp::CI;
        case CacheOp::wb:
            return lwvm::CacheOp::WB;
        case CacheOp::wt:
            return lwvm::CacheOp::WT;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized cache op" );
}

lwvm::BorderBehavior ptxToLwvmClampMode( ClampMode clampMode )
{
    switch( clampMode )
    {
        case ClampMode::unspecified:
            // no default clamp operator
            RT_ASSERT_FAIL_MSG( "unspecified texture dimensionality" );
        case ClampMode::clamp:
            return lwvm::BorderBehavior::OOB_CLAMP_NEAR;
        case ClampMode::trap:
            return lwvm::BorderBehavior::OOB_TRAP;
        case ClampMode::zero:
            return lwvm::BorderBehavior::OOB_IGNORE;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized clamp mode" );
}

lwvm::TexSurfQuery ptxToLwvmTextureQuery( TextureQuery query )
{
    switch( query )
    {
        case TextureQuery::unspecified:
            // no default texture query
            RT_ASSERT_FAIL_MSG( "unspecified texture query" );
        case TextureQuery::width:
            return lwvm::TexSurfQuery::QUERY_WIDTH;
        case TextureQuery::height:
            return lwvm::TexSurfQuery::QUERY_HEIGHT;
        case TextureQuery::depth:
            return lwvm::TexSurfQuery::QUERY_DEPTH;
        case TextureQuery::numMipmapLevels:
            return lwvm::TexSurfQuery::QUERY_MIPMAPS;
        case TextureQuery::numSamples:
            return lwvm::TexSurfQuery::QUERY_SAMPLES;
        case TextureQuery::arraySize:
            return lwvm::TexSurfQuery::QUERY_ARRAY_SIZE;
        case TextureQuery::normalizedCoords:
            return lwvm::TexSurfQuery::QUERY_SAMPLE_POS;
        case TextureQuery::channelOrder:
            return lwvm::TexSurfQuery::QUERY_CHANNEL_ORD;
        case TextureQuery::channelDataType:
            return lwvm::TexSurfQuery::QUERY_CHANNEL_DT;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized query mode" );
}

lwvm::MMOrdering ptxToLwvmMemOrdering( MemOrdering ordering )
{
    switch( ordering )
    {
        case MemOrdering::unspecified:
            // default ordering
            return lwvm::MMOrdering::RELAXED;
        case MemOrdering::weak:
            return lwvm::MMOrdering::WEAK;
        case MemOrdering::relaxed:
            return lwvm::MMOrdering::RELAXED;
        case MemOrdering::acq:
            return lwvm::MMOrdering::ACQ;
        case MemOrdering::rel:
            return lwvm::MMOrdering::REL;
        case MemOrdering::acq_rel:
            return lwvm::MMOrdering::ACQ_REL;
        case MemOrdering::sc:
            return lwvm::MMOrdering::SC;
        case MemOrdering::mmio:
            return lwvm::MMOrdering::MMIO;
        case MemOrdering::constant:
            return lwvm::MMOrdering::CONSTANT;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized memory ordering" );
}

lwvm::MMScope ptxToLwvmMemScope( MemScope scope )
{
    switch( scope )
    {
        case MemScope::unspecified:
            // default scope
            return lwvm::MMScope::GPU;
        case MemScope::gpu:
            return lwvm::MMScope::GPU;
        case MemScope::cta:
            return lwvm::MMScope::CTA;
        case MemScope::system:
            return lwvm::MMScope::SYSTEM;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized memory scope" );
}

lwvm::AtomicOpc ptxToLwvmAtomicOperation( AtomicOperation op, bool isSigned, bool isFloat )
{
    switch( op )
    {
        case AtomicOperation::unspecified:
            // no default
            RT_ASSERT_FAIL_MSG( "unspecified atomic operation" );
        case AtomicOperation::exch:
            return lwvm::AtomicOpc::EXCH;
        case AtomicOperation::add:
            return isFloat ? lwvm::AtomicOpc::FADD : lwvm::AtomicOpc::ADD;
        case AtomicOperation::sub:
            return lwvm::AtomicOpc::SUB;
        case AtomicOperation::andOp:
            return lwvm::AtomicOpc::AND;
        case AtomicOperation::orOp:
            return lwvm::AtomicOpc::OR;
        case AtomicOperation::xorOp:
            return lwvm::AtomicOpc::XOR;
        case AtomicOperation::max:
            return isSigned ? lwvm::AtomicOpc::MAX : lwvm::AtomicOpc::UMAX;
        case AtomicOperation::min:
            return isSigned ? lwvm::AtomicOpc::MIN : lwvm::AtomicOpc::UMIN;
        case AtomicOperation::inc:
            return lwvm::AtomicOpc::INC;
        case AtomicOperation::dec:
            return lwvm::AtomicOpc::DEC;
        case AtomicOperation::cas:
            return lwvm::AtomicOpc::CAS;
        case AtomicOperation::cast:
            return lwvm::AtomicOpc::CAST;
        case AtomicOperation::cast_spin:
            return lwvm::AtomicOpc::CAST_SPIN;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized atomic operation" );
}

lwvm::AddressSpace ptxToLwvmAddressSpace( AddressSpace addressSpace )
{
    switch( addressSpace )
    {
        case AddressSpace::unspecified:
            return lwvm::ADDRESS_SPACE_GENERIC;
        case AddressSpace::local:
            return lwvm::ADDRESS_SPACE_LOCAL;
        case AddressSpace::global:
            return lwvm::ADDRESS_SPACE_GLOBAL;
        case AddressSpace::shared:
            return lwvm::ADDRESS_SPACE_SHARED;
        case AddressSpace::constant:
            return lwvm::ADDRESS_SPACE_CONST;
        case AddressSpace::param:
            return lwvm::ADDRESS_SPACE_PARAM;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized address space" );
}

llvm::CmpInst::Predicate ptxToLwvmCompareOperator( CompareOperator op, bool isSigned, bool isFloat )
{
    switch( op )
    {
        case CompareOperator::unspecified:
            // no default
            RT_ASSERT_FAIL_MSG( "unspecified comparison operator" );
        case CompareOperator::eq:
            return isFloat ? llvm::CmpInst::FCMP_OEQ : llvm::CmpInst::ICMP_EQ;
        case CompareOperator::ne:
            return isFloat ? llvm::CmpInst::FCMP_ONE : llvm::CmpInst::ICMP_NE;
        case CompareOperator::lt:
        {
            if( isFloat )
                return llvm::CmpInst::FCMP_OLT;
            else
                return isSigned ? llvm::CmpInst::ICMP_SLT : llvm::CmpInst::ICMP_ULT;
        }
        case CompareOperator::le:
        {
            if( isFloat )
                return llvm::CmpInst::FCMP_OLE;
            else
                return isSigned ? llvm::CmpInst::ICMP_SLE : llvm::CmpInst::ICMP_ULE;
        }
        case CompareOperator::gt:
        {
            if( isFloat )
                return llvm::CmpInst::FCMP_OGT;
            else
                return isSigned ? llvm::CmpInst::ICMP_SGT : llvm::CmpInst::ICMP_UGT;
        }
        case CompareOperator::ge:
        {
            if( isFloat )
                return llvm::CmpInst::FCMP_OGE;
            else
                return isSigned ? llvm::CmpInst::ICMP_SGE : llvm::CmpInst::ICMP_UGE;
        }
        case CompareOperator::lo:
            return llvm::CmpInst::ICMP_ULT;
        case CompareOperator::ls:
            return llvm::CmpInst::ICMP_ULE;
        case CompareOperator::hi:
            return llvm::CmpInst::ICMP_UGT;
        case CompareOperator::hs:
            return llvm::CmpInst::ICMP_UGE;
        case CompareOperator::equ:
            return llvm::CmpInst::FCMP_UEQ;
        case CompareOperator::neu:
            return llvm::CmpInst::FCMP_UNE;
        case CompareOperator::ltu:
            return llvm::CmpInst::FCMP_ULT;
        case CompareOperator::leu:
            return llvm::CmpInst::FCMP_ULE;
        case CompareOperator::gtu:
            return llvm::CmpInst::FCMP_UGT;
        case CompareOperator::geu:
            return llvm::CmpInst::FCMP_UGE;
        case CompareOperator::num:
            return llvm::CmpInst::FCMP_ORD;
        case CompareOperator::nan:
            return llvm::CmpInst::FCMP_UNO;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized comparison operator" );
}

unsigned int ptxToLwvmIsVolatile( Volatile vol )
{
    if( vol == Volatile::unspecified )
        return 0;
    else if( vol == Volatile::vol )
        return 1;
    RT_ASSERT_FAIL_MSG( "unrecognized volatility" );
}

lwvm::ShiftType ptxToLwvmShiftMode( FunnelShiftWrapMode mode )
{
    switch( mode )
    {
        case FunnelShiftWrapMode::unspecified:
            // no default
            RT_ASSERT_FAIL_MSG( "unspecified funnel shift wrap mode" );
        case FunnelShiftWrapMode::clamp:
            return lwvm::SHIFT_CLAMP;
        case FunnelShiftWrapMode::wrap:
            return lwvm::SHIFT_WRAP;
    }
    RT_ASSERT_FAIL_MSG( "unrecognized funnel shift mode" );
}

int vectorSizeToInt( VectorSize vecSize )
{
    switch( vecSize )
    {
        case VectorSize::unspecified:
            return 1;
        case VectorSize::v2:
            return 2;
        case VectorSize::v4:
            return 4;
    }
    RT_ASSERT_FAIL_MSG( "Unrecognized vector size" );
}

}  // namespace PTXIntrinsics
}  // namespace optix
