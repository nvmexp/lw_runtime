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
 * Helpers that translate from PTX to LWVM types.
 */

#include <FrontEnd/PTX/Intrinsics/IntrinsicInfo.h>

namespace optix {
namespace PTXIntrinsics {

OperandType ptxArgToOperandTy( llvm::LLVMContext& context, ptxArgumentType type );

bool isArrayDimensionality( TextureDimensionality dim );
int getVectorSizeForDimensionality( TextureDimensionality dim );
int getDimensionalitySize( TextureDimensionality dim );
lwvm::TexSurfDim ptxToLwvmTextureDimensionality( TextureDimensionality dim );

lwvm::RoundingMode ptxToLwvmRoundMode( RoundMode mode );
unsigned int ptxToLwvmSaturate( Sat sat );
unsigned int ptxToLwvmFtz( Ftz sat );
unsigned int ptxToLwvmRgbaComponent( RgbaComponent comp );
lwvm::CacheOp ptxToLwvmCacheOp( CacheOp cacheOp );
lwvm::BorderBehavior ptxToLwvmClampMode( ClampMode clampMode );
lwvm::TexSurfQuery ptxToLwvmTextureQuery( TextureQuery query );
lwvm::MMOrdering ptxToLwvmMemOrdering( MemOrdering ordering );
lwvm::MMScope ptxToLwvmMemScope( MemScope scope );
lwvm::AtomicOpc ptxToLwvmAtomicOperation( AtomicOperation op, bool isSigned, bool isFloat );
lwvm::AddressSpace ptxToLwvmAddressSpace( AddressSpace addressSpace );
llvm::CmpInst::Predicate ptxToLwvmCompareOperator( CompareOperator op, bool isSigned, bool isFloat );
unsigned int ptxToLwvmIsVolatile( Volatile vol );
lwvm::ShiftType ptxToLwvmShiftMode( FunnelShiftWrapMode mode );

int vectorSizeToInt( VectorSize vecSize );

}  // namespace PTXIntrinsics
}  // namespace optix

