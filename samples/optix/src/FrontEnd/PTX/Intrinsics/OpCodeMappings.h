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

#include <prodlib/exceptions/Assert.h>

#include <cstddef>

// PTX instruction types, as fields in a bit vector
#define B8 0x2
#define U8 0x4
#define S8 0x8

#define B16 0x10
#define U16 0x20
#define S16 0x40
#define F16 0x80

#define B32 0x100
#define U32 0x200
#define S32 0x400
#define F32 0x800

#define B64 0x1000
#define U64 0x2000
#define S64 0x4000
#define F64 0x8000

#define F16x2 0x10000
#define BF16x2 0x20000

#define BF16 0x40000
#define TF32 0x80000

#define ANY_FLOAT F16 | F16x2 | F32 | F64
#define ANY_INT U16 | U32 | U64 | S16 | S32 | S64
#define ANY_TYPE B8 | U8 | S8 | B16 | U16 | S16 | F16 | B32 | U32 | S32 | F32 | B64 | U64 | S64 | F64

namespace optix {
namespace PTXIntrinsics {

// Colwert the given PTX type from a string to one of the integers above
inline unsigned int ptxTypeToUnsignedInt( const std::string& ptxType );

struct OpCodeMappingKey
{
    ptxInstructionCode ptxOpCode;
    unsigned int       returnType;
};

// Mapping from a given PTX instruction to an LWVM intrinsic.
struct OpCodeMapping
{
    OpCodeMappingKey    key;
    llvm::Intrinsic::ID lwvmIntrinsic;
};

// Lookup the LWVM intrinsic for the specified instruction. Return true if a
// mapping exists, false otherwise.
inline bool lookupIntrinsic( const PTXIntrinsicInfo& instruction, llvm::Intrinsic::ID& outIntrinsic );

// Mappings for simple PTX instructions (doesn't include instructions that have
// intrinsics based on different modifiers, like ftz)
// clang-format off
std::vector<OpCodeMapping> opCodeMappings = {
    //     ptxOpCode              ptxReturnType    lwvmIntrinsic
    {    { ptx_bfi_Instr,         B32 },           llvm::Intrinsic::lwvm_bfi_i32 },
    {    { ptx_bfi_Instr,         B64 },           llvm::Intrinsic::lwvm_bfi_i64 },
    {    { ptx_bfe_Instr,         U32 },           llvm::Intrinsic::lwvm_bfe_u_i32 },
    {    { ptx_bfe_Instr,         S32 },           llvm::Intrinsic::lwvm_bfe_s_i32 },
    {    { ptx_bfe_Instr,         U64 },           llvm::Intrinsic::lwvm_bfe_u_i64 },
    {    { ptx_bfe_Instr,         S64 },           llvm::Intrinsic::lwvm_bfe_s_i64 },

    {    { ptx_prmt_Instr,        B32 },           llvm::Intrinsic::lwvm_prmt },

    {    { ptx_brev_Instr,        B32|B64 },       llvm::Intrinsic::bitreverse },
};
// clang-format on

inline bool lookupIntrinsic( const PTXIntrinsicInfo& instruction, llvm::Intrinsic::ID& outIntrinsic )
{
    for( size_t i = 0; i < opCodeMappings.size(); ++i )
    {
        bool typeMatches = ( ptxTypeToUnsignedInt( instruction.signature[0].ptxType ) & opCodeMappings[i].key.returnType ) != 0;

        if( opCodeMappings[i].key.ptxOpCode == instruction.ptxOpCode && typeMatches )
        {
            outIntrinsic = opCodeMappings[i].lwvmIntrinsic;
            return true;
        }
    }

    return false;
}

// Atomic operators, as bitvector fields
#define OP_EXCH 0x2
#define OP_ADD 0x4
#define OP_SUB 0x8
#define OP_AND 0x10
#define OP_OR 0x20
#define OP_XOR 0x40
#define OP_MAX 0x80
#define OP_MIN 0x100
#define OP_INC 0x200 
#define OP_DEC 0x400 
#define OP_CAS 0x800
#define OP_CAST 0x1000
#define OP_CAST_SPIN 0x2000

#define ANY_OPERATION                                                                                                  \
    OP_EXCH | OP_ADD | OP_SUB | OP_AND | OP_OR | OP_XOR | OP_MAX | OP_MIN | OP_INC | OP_DEC | OP_CAS | OP_CAST | OP_CAST_SPIN

struct AtomOpCodeMappingKey
{
    unsigned int atomicOperations;
    unsigned int returnType;
};

// Mapping from a given atomic PTX instruction to an LWVM intrinsic
struct AtomOpCodeMapping
{
    AtomOpCodeMappingKey key;
    llvm::Intrinsic::ID  lwvmIntrinsic;
};

// Mappings from atomic instructions to their associated LWVM intrinsics
// clang-format off
std::vector<AtomOpCodeMapping> atomOpCodeMappings = {
    //     atomicOperations    ptxReturnType      lwvmIntrinsic
    {    { OP_CAS,             B16         },     llvm::Intrinsic::lwvm_atomic_cas_i16   },
    {    { OP_CAS,             B32         },     llvm::Intrinsic::lwvm_atomic_cas_i32   },
    {    { OP_CAS,             B64         },     llvm::Intrinsic::lwvm_atomic_cas_i64   },

    {    { ANY_OPERATION,      F16x2       },     llvm::Intrinsic::lwvm_atomic_rmw_f16x2 },
    {    { ANY_OPERATION,      F32         },     llvm::Intrinsic::lwvm_atomic_rmw_f32   },
    {    { ANY_OPERATION,      F64         },     llvm::Intrinsic::lwvm_atomic_rmw_f64   },
    {    { ANY_OPERATION,      B32|U32|S32 },     llvm::Intrinsic::lwvm_atomic_rmw_i32   },
    {    { ANY_OPERATION,      B64|U64|S64 },     llvm::Intrinsic::lwvm_atomic_rmw_i64   },
};
// clang-format on


inline unsigned int atomicOpToUnsignedInt( AtomicOperation op );

// Lookup the LWVM intrinsic for the given atomic instruction; return true if a
// mapping exists, false otherwise
inline bool lookupAtomOrRedIntrinsic( const PTXIntrinsicInfo& instruction, llvm::Intrinsic::ID& outIntrinsic )
{
    for( size_t i = 0; i < atomOpCodeMappings.size(); ++i )
    {
        bool typeMatches =
            ( ptxTypeToUnsignedInt( instruction.signature[2].ptxType ) & atomOpCodeMappings[i].key.returnType ) != 0;
        bool operationMatches =
            ( atomicOpToUnsignedInt( instruction.modifiers.atomicOp ) & atomOpCodeMappings[i].key.atomicOperations ) != 0;

        if( operationMatches && typeMatches )
        {
            outIntrinsic = atomOpCodeMappings[i].lwvmIntrinsic;
            return true;
        }
    }

    return false;
}

inline unsigned int atomicOpToUnsignedInt( AtomicOperation operation )
{
    switch( operation )
    {
        case AtomicOperation::exch:
            return OP_EXCH;
        case AtomicOperation::add:
            return OP_ADD;
        case AtomicOperation::sub:
            return OP_SUB;
        case AtomicOperation::andOp:
            return OP_AND;
        case AtomicOperation::orOp:
            return OP_OR;
        case AtomicOperation::xorOp:
            return OP_XOR;
        case AtomicOperation::max:
            return OP_MAX;
        case AtomicOperation::min:
            return OP_MIN;
        case AtomicOperation::inc:
            return OP_INC;
        case AtomicOperation::dec:
            return OP_DEC;
        case AtomicOperation::cas:
            return OP_CAS;
        case AtomicOperation::cast:
            return OP_CAST;
        case AtomicOperation::cast_spin:
            return OP_CAST_SPIN;
        default:
            RT_ASSERT_FAIL_MSG( "Unrecognized atomic operation" );
    }
}

typedef enum class HasFTZ { either, yes, no } HasFTZ;
typedef enum class HasApprox { either, yes, no } HasApprox;

struct MathOpCodeMappingKey
{
    ptxInstructionCode ptxOpCode;
    unsigned int       returnType;
    HasFTZ             hasFTZ;
    HasApprox          hasApprox;
};

struct MathOpCodeMappingValue
{
    bool                hasMathFlag;
    llvm::Intrinsic::ID lwvmIntrinsic;
};

struct MathOpCodeMapping
{
    MathOpCodeMappingKey   key;
    MathOpCodeMappingValue value;
};

// Mappings from PTX math instructions to their associated intrinsics, and whether or not they require the LWVM math flag
// clang-format off
std::vector<MathOpCodeMapping> mathOpCodeMappings = {
    //     ptxOpCode              ptxReturnType      hasFTZ             HasApprox                  hasMathFlag    lwvmIntrinsic
    {    { ptx_abs_Instr,         S16|S32|S64,       HasFTZ::no,        HasApprox::either },     { false,         llvm::Intrinsic::lwvm_abs }                 },
    {    { ptx_abs_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::either },     { false,         llvm::Intrinsic::lwvm_fabs }                },
    {    { ptx_abs_Instr,         ANY_FLOAT,         HasFTZ::yes,       HasApprox::either },     { false,         llvm::Intrinsic::lwvm_fabs_ftz }            },
 
    {    { ptx_max_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::either },     { false,         llvm::Intrinsic::lwvm_fmax }                },
    {    { ptx_max_Instr,         ANY_FLOAT,         HasFTZ::yes,       HasApprox::either },     { false,         llvm::Intrinsic::lwvm_fmax_ftz }            },
 
    {    { ptx_min_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::either },     { false,         llvm::Intrinsic::lwvm_fmin }                },
    {    { ptx_min_Instr,         ANY_FLOAT,         HasFTZ::yes,       HasApprox::either },     { false,         llvm::Intrinsic::lwvm_fmin_ftz }            },
 
    {    { ptx_add_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::either },     { true,          llvm::Intrinsic::lwvm_add }                 },
    {    { ptx_add_Instr,         ANY_FLOAT,         HasFTZ::yes,       HasApprox::either },     { true,          llvm::Intrinsic::lwvm_add_ftz }             },
 
    {    { ptx_sub_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::either },     { true,          llvm::Intrinsic::lwvm_sub }                 },
    {    { ptx_sub_Instr,         ANY_FLOAT,         HasFTZ::yes,       HasApprox::either },     { true,          llvm::Intrinsic::lwvm_sub_ftz }             },
 
    {    { ptx_mul_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_mul }                 },
    {    { ptx_mul_Instr,         ANY_FLOAT,         HasFTZ::yes,       HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_mul_ftz }             },
 
    {    { ptx_mul_hi_Instr,      U32|U64,           HasFTZ::either,    HasApprox::either },     { false,         llvm::Intrinsic::lwvm_mul_hi_u }            },
    {    { ptx_mul_hi_Instr,      S32|S64,           HasFTZ::either,    HasApprox::either },     { false,         llvm::Intrinsic::lwvm_mul_hi_s }            },
 
    {    { ptx_div_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_div }                 },
    {    { ptx_div_Instr,         F32,               HasFTZ::yes,       HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_div_ftz_f32 }         },
    {    { ptx_div_Instr,         F32,               HasFTZ::no,        HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_div_approx_f32 }      },
    {    { ptx_div_Instr,         F32,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_div_approx_ftz_f32 }  },
 
    {    { ptx_div_full_Instr,    F32,               HasFTZ::no,        HasApprox::either },     { false,         llvm::Intrinsic::lwvm_div_full_f32 }        },
    {    { ptx_div_full_Instr,    F32,               HasFTZ::yes,       HasApprox::either },     { false,         llvm::Intrinsic::lwvm_div_full_ftz_f32 }    },
 
    {    { ptx_fma_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_fma }                 },
    {    { ptx_fma_Instr,         ANY_FLOAT,         HasFTZ::yes,       HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_fma_ftz }             },
 
    {    { ptx_mad_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_fma }                 },
    {    { ptx_mad_Instr,         ANY_FLOAT,         HasFTZ::yes,       HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_fma_ftz }             },

    {    { ptx_mad_lo_Instr,      ANY_INT,           HasFTZ::either,    HasApprox::either },     { false,         llvm::Intrinsic::lwvm_mad_lo }              },

    {    { ptx_mad_hi_Instr,      U16|U32|U64,       HasFTZ::either,    HasApprox::either },     { false,         llvm::Intrinsic::lwvm_mad_hi_u }            },
    {    { ptx_mad_hi_Instr,      S16|S32|S64,       HasFTZ::either,    HasApprox::either },     { false,         llvm::Intrinsic::lwvm_mad_hi_s }            },

    {    { ptx_mad_wide_Instr,    U64,               HasFTZ::either,    HasApprox::either },     { false,         llvm::Intrinsic::lwvm_mad_wide_u_i32 }      },
    {    { ptx_mad_wide_Instr,    S64,               HasFTZ::either,    HasApprox::either },     { false,         llvm::Intrinsic::lwvm_mad_wide_s_i32 }      },
 
    {    { ptx_sqrt_Instr,        ANY_FLOAT,         HasFTZ::no,        HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_sqrt }                },
    {    { ptx_sqrt_Instr,        F32,               HasFTZ::yes,       HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_sqrt_ftz_f32 }        },
    {    { ptx_sqrt_Instr,        F32,               HasFTZ::no,        HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_sqrt_approx }         },
    {    { ptx_sqrt_Instr,        F32,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_sqrt_approx_ftz_f32 } },

    {    { ptx_rsqrt_Instr,       ANY_FLOAT,         HasFTZ::no,        HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_rsqrt_approx }         },
    {    { ptx_rsqrt_Instr,       F32,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_rsqrt_approx_ftz_f32 } },
    {    { ptx_rsqrt_Instr,       F64,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_rsqrt_approx_ftz_f64 } },
 
    {    { ptx_rcp_Instr,         F32|F64,           HasFTZ::no,        HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_rcp }                 },
    {    { ptx_rcp_Instr,         F32,               HasFTZ::yes,       HasApprox::no     },     { true,          llvm::Intrinsic::lwvm_rcp_ftz_f32 }         },
    {    { ptx_rcp_Instr,         F32|F64,           HasFTZ::no,        HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_rcp_approx }          },
    {    { ptx_rcp_Instr,         F32,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_rcp_approx_ftz_f32 }  },
    {    { ptx_rcp_Instr,         F64,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_rcp_approx_ftz_f64 }  },
 
    {    { ptx_lg2_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_lg2_approx }          },
    {    { ptx_lg2_Instr,         F32,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_lg2_approx_ftz_f32 }  },

    {    { ptx_ex2_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_ex2_approx }          },
    {    { ptx_ex2_Instr,         F32,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_ex2_approx_ftz_f32 }  },

    {    { ptx_sin_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_sin_approx }          },
    {    { ptx_sin_Instr,         F32,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_sin_approx_ftz_f32 }  },

    {    { ptx_cos_Instr,         ANY_FLOAT,         HasFTZ::no,        HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_cos_approx }          },
    {    { ptx_cos_Instr,         F32,               HasFTZ::yes,       HasApprox::yes    },     { false,         llvm::Intrinsic::lwvm_cos_approx_ftz_f32 }  },
};
// clang-format on

inline bool lookupMathIntrinsic( const PTXIntrinsicInfo& mathInstruction, llvm::Intrinsic::ID& outIntrinsic, bool& outHasMathFlag )
{
    for( size_t i = 0; i < mathOpCodeMappings.size(); ++i )
    {
        bool ftzMatches = ( mathOpCodeMappings[i].key.hasFTZ == HasFTZ::either )
                          || ( mathOpCodeMappings[i].key.hasFTZ == HasFTZ::yes && mathInstruction.modifiers.ftz == Ftz::ftz )
                          || ( mathOpCodeMappings[i].key.hasFTZ == HasFTZ::no && mathInstruction.modifiers.ftz == Ftz::unspecified );
        bool approxMatches =
            ( mathOpCodeMappings[i].key.hasApprox == HasApprox::either )
            || ( mathOpCodeMappings[i].key.hasApprox == HasApprox::yes && mathInstruction.modifiers.approx == Approx::approx )
            || ( mathOpCodeMappings[i].key.hasApprox == HasApprox::no && mathInstruction.modifiers.approx == Approx::unspecified );
        bool typeMatches =
            ( ptxTypeToUnsignedInt( mathInstruction.signature[0].ptxType ) & mathOpCodeMappings[i].key.returnType ) != 0;

        if( mathOpCodeMappings[i].key.ptxOpCode == mathInstruction.ptxOpCode && typeMatches && ftzMatches && approxMatches )
        {
            outIntrinsic   = mathOpCodeMappings[i].value.lwvmIntrinsic;
            outHasMathFlag = mathOpCodeMappings[i].value.hasMathFlag;
            return true;
        }
    }

    return false;
}

inline unsigned int ptxTypeToUnsignedInt( const std::string& ptxType )
{
    if( ptxType.compare( "b8" ) == 0 )
        return B8;
    if( ptxType.compare( "u8" ) == 0 )
        return U8;
    if( ptxType.compare( "s8" ) == 0 )
        return S8;
    if( ptxType.compare( "b16" ) == 0 )
        return B16;
    if( ptxType.compare( "u16" ) == 0 )
        return U16;
    if( ptxType.compare( "s16" ) == 0 )
        return S16;
    if( ptxType.compare( "f16" ) == 0 )
        return F16;
    if( ptxType.compare( "bf16" ) == 0 )
        return BF16;
    if( ptxType.compare( "b32" ) == 0 )
        return B32;
    if( ptxType.compare( "u32" ) == 0 )
        return U32;
    if( ptxType.compare( "s32" ) == 0 )
        return S32;
    if( ptxType.compare( "f32" ) == 0 )
        return F32;
    if( ptxType.compare( "tf32" ) == 0 )
        return TF32;
    if( ptxType.compare( "b64" ) == 0 )
        return B64;
    if( ptxType.compare( "u64" ) == 0 )
        return U64;
    if( ptxType.compare( "s64" ) == 0 )
        return S64;
    if( ptxType.compare( "f64" ) == 0 )
        return F64;
    if( ptxType.compare( "f16x2" ) == 0 )
        return F16x2;
    if( ptxType.compare( "bf16x2" ) == 0 )
        return BF16x2;

    RT_ASSERT_FAIL_MSG( "Couldn't colwert ptx type to unsigned int" );
}
}  // namespace PTXIntrinsics
}  // namespace optix
