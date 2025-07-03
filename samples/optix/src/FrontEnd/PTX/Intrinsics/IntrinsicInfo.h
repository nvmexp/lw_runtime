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
 * Information necessary to generate OptiX PTX intrinsics.
 */

#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <lwvm/ClientInterface/LWVM.h>

#include <ptxIR.h>
#include <ptxInstructions.h>

#include <string>

namespace optix {

namespace PTXIntrinsics {

// TODO(Kincaid): We probably don't have to construct instruction name strings
// dynamically, now that we're able to store them in metadata.
// i.e. we should be able to delete ptxType

// Contains the LLVM type and corresponding PTX instruction string.
struct OperandType
{
    // The type of the PTX operand
    std::string ptxType;
    // The corresponding LLVM type
    llvm::Type* llvmType;
    // True if this type is explicitly marked as scalar
    bool isScalar;

    OperandType() {}

    OperandType( std::string ptxType, llvm::Type* llvmType, bool isScalar )
        : ptxType( ptxType )
        , llvmType( llvmType )
        , isScalar( isScalar )
    {
    }

    bool operator==( const OperandType& rhs )
    {
        if( llvmType != rhs.llvmType )
            return false;
        if( ptxType.compare( rhs.ptxType ) != 0 )
            return false;

        return true;
    };

    bool operator!=( const OperandType& rhs ) { return !( *this == rhs ); };
};

using InstructionSignature = std::vector<OperandType>;

enum class RoundMode
{
    unspecified,
    rn,
    rm,
    rp,
    rz,
    rni,
    rmi,
    rpi,
    rzi
};

enum class Approx {
    unspecified,
    approx
};

enum class Sat {
    unspecified,
    sat
};

enum class Ftz {
    unspecified,
    ftz
};

enum class Noftz {
    unspecified,
    noftz
};

enum class VectorSize {
    unspecified,
    v2,
    v4
};

enum class TextureDimensionality {
    unspecified,
    dim1D,
    dim1DArray,
    dim1DBuffer,
    dim2D,
    dim2DArray,
    dim3D,
    dim3DArray,
    dimLwbe,
    dimLwbeArray
};

enum class RgbaComponent {
    unspecified,
    r,
    g,
    b,
    a
};

enum class TextureQuery {
    unspecified,
    width,
    height,
    depth,
    numMipmapLevels,
    numSamples,
    arraySize,
    normalizedCoords,
    channelOrder,
    channelDataType
};

enum class CacheOp {
    unspecified,
    cg,
    cs,
    ca,
    lu,
    cv,
    ci,
    wb,
    wt
};

enum class ClampMode {
    unspecified,
    clamp,
    trap,
    zero
};

enum class FunnelShiftWrapMode {
    unspecified,
    clamp,
    wrap
};

enum class FormatMode {
    unspecified,
    bytes,
    pixel
};

enum class MemOrdering {
    unspecified,
    weak,
    relaxed,
    acq,
    rel,
    acq_rel,
    sc,
    mmio,
    constant
};

enum class MemScope {
    unspecified,
    gpu,
    cta,
    system
};

enum class AddressSpace {
    unspecified,
    local,
    global,
    shared,
    constant,
    param
};

enum class AtomicOperation {
    unspecified,
    exch,
    add,
    sub,
    andOp,
    orOp,
    xorOp,
    max,
    min,
    inc,
    dec,
    cas,
    cast,
    cast_spin
};

enum class CompareOperator {
    unspecified,
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    lo,
    ls,
    hi,
    hs,
    equ,
    neu,
    ltu,
    leu,
    gtu,
    geu,
    num,
    nan
};

enum class BooleanOperator {
    unspecified,
    andOp,
    orOp,
    xorOp
};

enum class Volatile {
    unspecified,
    vol
};

enum class TexDomain {
    unspecified,
    nc
};

struct PTXIntrinsicModifiers
{
    RoundMode             roundMode;
    Approx                approx;
    Sat                   sat;
    Ftz                   ftz;
    Noftz                 noftz;
    VectorSize            vectorSize;
    TextureDimensionality texDim;
    RgbaComponent         rgbaComponent;
    TextureQuery          texQuery;
    CacheOp               cacheOp;
    ClampMode             clampMode;
    FunnelShiftWrapMode   funnelShiftWrapMode;
    FormatMode            formatMode;
    MemOrdering           memOrdering;
    MemScope              memScope;
    AddressSpace          addressSpace;
    AtomicOperation       atomicOp;
    CompareOperator       cmpOp;
    BooleanOperator       boolOp;
    Volatile              vol;
    TexDomain             texDomain;
};

struct PTXIntrinsicInfo {
    std::string           name;
    ptxInstructionCode    ptxOpCode;
    PTXIntrinsicModifiers modifiers;
    InstructionSignature  signature;
    bool                  hasPredicateOutput;
};

}  // namespace optix
}  // namespace PTXIntrinsics
