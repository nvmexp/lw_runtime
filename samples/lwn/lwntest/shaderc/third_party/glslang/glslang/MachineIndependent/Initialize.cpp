//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2016 LunarG, Inc.
// Copyright (C) 2015-2020 Google, Inc.
// Copyright (C) 2017 ARM Limited.
// Modifications Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

//
// Create strings that declare built-in definitions, add built-ins programmatically
// that cannot be expressed in the strings, and establish mappings between
// built-in functions and operators.
//
// Where to put a built-in:
//   TBuiltIns::initialize(version,profile)       context-independent textual built-ins; add them to the right string
//   TBuiltIns::initialize(resources,...)         context-dependent textual built-ins; add them to the right string
//   TBuiltIns::identifyBuiltIns(...,symbolTable) context-independent programmatic additions/mappings to the symbol table,
//                                                including identifying what extensions are needed if a version does not allow a symbol
//   TBuiltIns::identifyBuiltIns(...,symbolTable, resources) context-dependent programmatic additions/mappings to the symbol table,
//                                                including identifying what extensions are needed if a version does not allow a symbol
//

#include "../Include/intermediate.h"
#include "Initialize.h"

namespace glslang {

// TODO: ARB_Compatability: do full extension support
const bool ARBCompatibility = true;

const bool ForwardCompatibility = false;

// change this back to false if depending on textual spellings of texturing calls when consuming the AST
// Using PureOperatorBuiltins=false is deprecated.
bool PureOperatorBuiltins = true;

namespace {

//
// A set of definitions for tabling of the built-in functions.
//

// Order matters here, as does correlation with the subsequent
// "const int ..." declarations and the ArgType enumerants.
const char* TypeString[] = {
   "bool",  "bvec2", "bvec3", "bvec4",
   "float",  "vec2",  "vec3",  "vec4",
   "int",   "ivec2", "ivec3", "ivec4",
   "uint",  "uvec2", "uvec3", "uvec4",
};
const int TypeStringCount = sizeof(TypeString) / sizeof(char*); // number of entries in 'TypeString'
const int TypeStringRowShift = 2;                               // shift amount to go downe one row in 'TypeString'
const int TypeStringColumnMask = (1 << TypeStringRowShift) - 1; // reduce type to its column number in 'TypeString'
const int TypeStringScalarMask = ~TypeStringColumnMask;         // take type to its scalar column in 'TypeString'

enum ArgType {
    // numbers hardcoded to correspond to 'TypeString'; order and value matter
    TypeB    = 1 << 0,  // Boolean
    TypeF    = 1 << 1,  // float 32
    TypeI    = 1 << 2,  // int 32
    TypeU    = 1 << 3,  // uint 32
    TypeF16  = 1 << 4,  // float 16
    TypeF64  = 1 << 5,  // float 64
    TypeI8   = 1 << 6,  // int 8
    TypeI16  = 1 << 7,  // int 16
    TypeI64  = 1 << 8,  // int 64
    TypeU8   = 1 << 9,  // uint 8
    TypeU16  = 1 << 10, // uint 16
    TypeU64  = 1 << 11, // uint 64
};
// Mixtures of the above, to help the function tables
const ArgType TypeFI  = static_cast<ArgType>(TypeF | TypeI);
const ArgType TypeFIB = static_cast<ArgType>(TypeF | TypeI | TypeB);
const ArgType TypeIU  = static_cast<ArgType>(TypeI | TypeU);

// The relationships between arguments and return type, whether anything is
// output, or other unusual situations.
enum ArgClass {
    ClassRegular     = 0,  // nothing special, just all vector widths with matching return type; traditional arithmetic
    ClassLS     = 1 << 0,  // the last argument is also held fixed as a (type-matched) scalar while the others cycle
    ClassXLS    = 1 << 1,  // the last argument is exclusively a (type-matched) scalar while the others cycle
    ClassLS2    = 1 << 2,  // the last two arguments are held fixed as a (type-matched) scalar while the others cycle
    ClassFS     = 1 << 3,  // the first argument is held fixed as a (type-matched) scalar while the others cycle
    ClassFS2    = 1 << 4,  // the first two arguments are held fixed as a (type-matched) scalar while the others cycle
    ClassLO     = 1 << 5,  // the last argument is an output
    ClassB      = 1 << 6,  // return type cycles through only bool/bvec, matching vector width of args
    ClassLB     = 1 << 7,  // last argument cycles through only bool/bvec, matching vector width of args
    ClassV1     = 1 << 8,  // scalar only
    ClassFIO    = 1 << 9,  // first argument is inout
    ClassRS     = 1 << 10, // the return is held scalar as the arguments cycle
    ClassNS     = 1 << 11, // no scalar prototype
    ClassCV     = 1 << 12, // first argument is 'coherent volatile'
    ClassFO     = 1 << 13, // first argument is output
    ClassV3     = 1 << 14, // vec3 only
};
// Mixtures of the above, to help the function tables
const ArgClass ClassV1FIOCV = (ArgClass)(ClassV1 | ClassFIO | ClassCV);
const ArgClass ClassBNS     = (ArgClass)(ClassB  | ClassNS);
const ArgClass ClassRSNS    = (ArgClass)(ClassRS | ClassNS);

// A descriptor, for a single profile, of when something is available.
// If the current profile does not match 'profile' mask below, the other fields
// do not apply (nor validate).
// profiles == EBadProfile is the end of an array of these
struct Versioning {
    EProfile profiles;       // the profile(s) (mask) that the following fields are valid for
    int minExtendedVersion;  // earliest version when extensions are enabled; ignored if numExtensions is 0
    int minCoreVersion;      // earliest version function is in core; 0 means never
    int numExtensions;       // how many extensions are in the 'extensions' list
    const char** extensions; // list of extension names enabling the function
};

EProfile EDesktopProfile = static_cast<EProfile>(ENoProfile | ECoreProfile | ECompatibilityProfile);

// Declare pointers to put into the table for versioning.
#ifdef GLSLANG_WEB
    const Versioning* Es300Desktop130 = nullptr;
#else
    const Versioning Es300Desktop130Version[] = { { EEsProfile,      0, 300, 0, nullptr },
                                                  { EDesktopProfile, 0, 130, 0, nullptr },
                                                  { EBadProfile } };
    const Versioning* Es300Desktop130 = &Es300Desktop130Version[0];

    const Versioning Es310Desktop420Version[] = { { EEsProfile,      0, 310, 0, nullptr },
                                                  { EDesktopProfile, 0, 420, 0, nullptr },
                                                  { EBadProfile } };
    const Versioning* Es310Desktop420 = &Es310Desktop420Version[0];

    const Versioning Es310Desktop450Version[] = { { EEsProfile,      0, 310, 0, nullptr },
                                                  { EDesktopProfile, 0, 450, 0, nullptr },
                                                  { EBadProfile } };
    const Versioning* Es310Desktop450 = &Es310Desktop450Version[0];
#endif

// The main descriptor of what a set of function prototypes can look like, and
// a pointer to extra versioning information, when needed.
struct BuiltInFunction {
    TOperator op;                 // operator to map the name to
    const char* name;             // function name
    int numArguments;             // number of arguments (overloads with varying arguments need different entries)
    ArgType types;                // ArgType mask
    ArgClass classes;             // the ways this particular function entry manifests
    const Versioning* versioning; // nullptr means always a valid version
};

// The tables can have the same built-in function name more than one time,
// but the exact same prototype must be indicated at most once.
// The prototypes that get declared are the union of all those indicated.
// This is important when different releases add new prototypes for the same name.
// It also also congnitively simpler tiling of the prototype space.
// In practice, most names can be fully represented with one entry.
//
// Table is terminated by an OpNull TOperator.

const BuiltInFunction BaseFunctions[] = {
//    TOperator,           name,       arg-count,   ArgType,   ArgClass,     versioning
//    ---------            ----        ---------    -------    --------      ----------
    { EOpRadians,          "radians",          1,   TypeF,     ClassRegular, nullptr },
    { EOpDegrees,          "degrees",          1,   TypeF,     ClassRegular, nullptr },
    { EOpSin,              "sin",              1,   TypeF,     ClassRegular, nullptr },
    { EOpCos,              "cos",              1,   TypeF,     ClassRegular, nullptr },
    { EOpTan,              "tan",              1,   TypeF,     ClassRegular, nullptr },
    { EOpAsin,             "asin",             1,   TypeF,     ClassRegular, nullptr },
    { EOpAcos,             "acos",             1,   TypeF,     ClassRegular, nullptr },
    { EOpAtan,             "atan",             2,   TypeF,     ClassRegular, nullptr },
    { EOpAtan,             "atan",             1,   TypeF,     ClassRegular, nullptr },
    { EOpPow,              "pow",              2,   TypeF,     ClassRegular, nullptr },
    { EOpExp,              "exp",              1,   TypeF,     ClassRegular, nullptr },
    { EOpLog,              "log",              1,   TypeF,     ClassRegular, nullptr },
    { EOpExp2,             "exp2",             1,   TypeF,     ClassRegular, nullptr },
    { EOpLog2,             "log2",             1,   TypeF,     ClassRegular, nullptr },
    { EOpSqrt,             "sqrt",             1,   TypeF,     ClassRegular, nullptr },
    { EOpIlwerseSqrt,      "ilwersesqrt",      1,   TypeF,     ClassRegular, nullptr },
    { EOpAbs,              "abs",              1,   TypeF,     ClassRegular, nullptr },
    { EOpSign,             "sign",             1,   TypeF,     ClassRegular, nullptr },
    { EOpFloor,            "floor",            1,   TypeF,     ClassRegular, nullptr },
    { EOpCeil,             "ceil",             1,   TypeF,     ClassRegular, nullptr },
    { EOpFract,            "fract",            1,   TypeF,     ClassRegular, nullptr },
    { EOpMod,              "mod",              2,   TypeF,     ClassLS,      nullptr },
    { EOpMin,              "min",              2,   TypeF,     ClassLS,      nullptr },
    { EOpMax,              "max",              2,   TypeF,     ClassLS,      nullptr },
    { EOpClamp,            "clamp",            3,   TypeF,     ClassLS2,     nullptr },
    { EOpMix,              "mix",              3,   TypeF,     ClassLS,      nullptr },
    { EOpStep,             "step",             2,   TypeF,     ClassFS,      nullptr },
    { EOpSmoothStep,       "smoothstep",       3,   TypeF,     ClassFS2,     nullptr },
    { EOpNormalize,        "normalize",        1,   TypeF,     ClassRegular, nullptr },
    { EOpFaceForward,      "faceforward",      3,   TypeF,     ClassRegular, nullptr },
    { EOpReflect,          "reflect",          2,   TypeF,     ClassRegular, nullptr },
    { EOpRefract,          "refract",          3,   TypeF,     ClassXLS,     nullptr },
    { EOpLength,           "length",           1,   TypeF,     ClassRS,      nullptr },
    { EOpDistance,         "distance",         2,   TypeF,     ClassRS,      nullptr },
    { EOpDot,              "dot",              2,   TypeF,     ClassRS,      nullptr },
    { EOpCross,            "cross",            2,   TypeF,     ClassV3,      nullptr },
    { EOpLessThan,         "lessThan",         2,   TypeFI,    ClassBNS,     nullptr },
    { EOpLessThanEqual,    "lessThanEqual",    2,   TypeFI,    ClassBNS,     nullptr },
    { EOpGreaterThan,      "greaterThan",      2,   TypeFI,    ClassBNS,     nullptr },
    { EOpGreaterThanEqual, "greaterThanEqual", 2,   TypeFI,    ClassBNS,     nullptr },
    { EOpVectorEqual,      "equal",            2,   TypeFIB,   ClassBNS,     nullptr },
    { EOpVectorNotEqual,   "notEqual",         2,   TypeFIB,   ClassBNS,     nullptr },
    { EOpAny,              "any",              1,   TypeB,     ClassRSNS,    nullptr },
    { EOpAll,              "all",              1,   TypeB,     ClassRSNS,    nullptr },
    { EOpVectorLogicalNot, "not",              1,   TypeB,     ClassNS,      nullptr },
    { EOpSinh,             "sinh",             1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpCosh,             "cosh",             1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpTanh,             "tanh",             1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpAsinh,            "asinh",            1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpAcosh,            "acosh",            1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpAtanh,            "atanh",            1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpAbs,              "abs",              1,   TypeI,     ClassRegular, Es300Desktop130 },
    { EOpSign,             "sign",             1,   TypeI,     ClassRegular, Es300Desktop130 },
    { EOpTrunc,            "trunc",            1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpRound,            "round",            1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpRoundEven,        "roundEven",        1,   TypeF,     ClassRegular, Es300Desktop130 },
    { EOpModf,             "modf",             2,   TypeF,     ClassLO,      Es300Desktop130 },
    { EOpMin,              "min",              2,   TypeIU,    ClassLS,      Es300Desktop130 },
    { EOpMax,              "max",              2,   TypeIU,    ClassLS,      Es300Desktop130 },
    { EOpClamp,            "clamp",            3,   TypeIU,    ClassLS2,     Es300Desktop130 },
    { EOpMix,              "mix",              3,   TypeF,     ClassLB,      Es300Desktop130 },
    { EOpIsInf,            "isinf",            1,   TypeF,     ClassB,       Es300Desktop130 },
    { EOpIsNan,            "isnan",            1,   TypeF,     ClassB,       Es300Desktop130 },
    { EOpLessThan,         "lessThan",         2,   TypeU,     ClassBNS,     Es300Desktop130 },
    { EOpLessThanEqual,    "lessThanEqual",    2,   TypeU,     ClassBNS,     Es300Desktop130 },
    { EOpGreaterThan,      "greaterThan",      2,   TypeU,     ClassBNS,     Es300Desktop130 },
    { EOpGreaterThanEqual, "greaterThanEqual", 2,   TypeU,     ClassBNS,     Es300Desktop130 },
    { EOpVectorEqual,      "equal",            2,   TypeU,     ClassBNS,     Es300Desktop130 },
    { EOpVectorNotEqual,   "notEqual",         2,   TypeU,     ClassBNS,     Es300Desktop130 },
    { EOpAtomicAdd,        "atomicAdd",        2,   TypeIU,    ClassV1FIOCV, Es310Desktop420 },
    { EOpAtomicMin,        "atomicMin",        2,   TypeIU,    ClassV1FIOCV, Es310Desktop420 },
    { EOpAtomicMax,        "atomicMax",        2,   TypeIU,    ClassV1FIOCV, Es310Desktop420 },
    { EOpAtomicAnd,        "atomicAnd",        2,   TypeIU,    ClassV1FIOCV, Es310Desktop420 },
    { EOpAtomicOr,         "atomicOr",         2,   TypeIU,    ClassV1FIOCV, Es310Desktop420 },
    { EOpAtomicXor,        "atomicXor",        2,   TypeIU,    ClassV1FIOCV, Es310Desktop420 },
    { EOpAtomicExchange,   "atomicExchange",   2,   TypeIU,    ClassV1FIOCV, Es310Desktop420 },
    { EOpAtomicCompSwap,   "atomicCompSwap",   3,   TypeIU,    ClassV1FIOCV, Es310Desktop420 },
#ifndef GLSLANG_WEB
    { EOpMix,              "mix",              3,   TypeB,     ClassRegular, Es310Desktop450 },
    { EOpMix,              "mix",              3,   TypeIU,    ClassLB,      Es310Desktop450 },
#endif

    { EOpNull }
};

const BuiltInFunction DerivativeFunctions[] = {
    { EOpDPdx,             "dFdx",             1,   TypeF,     ClassRegular, nullptr },
    { EOpDPdy,             "dFdy",             1,   TypeF,     ClassRegular, nullptr },
    { EOpFwidth,           "fwidth",           1,   TypeF,     ClassRegular, nullptr },
    { EOpNull }
};

// For functions declared some other way, but still use the table to relate to operator.
struct LwstomFunction {
    TOperator op;                 // operator to map the name to
    const char* name;             // function name
    const Versioning* versioning; // nullptr means always a valid version
};

const LwstomFunction LwstomFunctions[] = {
    { EOpBarrier,             "barrier",             nullptr },
    { EOpMemoryBarrierShared, "memoryBarrierShared", nullptr },
    { EOpGroupMemoryBarrier,  "groupMemoryBarrier",  nullptr },
    { EOpMemoryBarrier,       "memoryBarrier",       nullptr },
    { EOpMemoryBarrierBuffer, "memoryBarrierBuffer", nullptr },

    { EOpPackSnorm2x16,       "packSnorm2x16",       nullptr },
    { EOpUnpackSnorm2x16,     "unpackSnorm2x16",     nullptr },
    { EOpPackUnorm2x16,       "packUnorm2x16",       nullptr },
    { EOpUnpackUnorm2x16,     "unpackUnorm2x16",     nullptr },
    { EOpPackHalf2x16,        "packHalf2x16",        nullptr },
    { EOpUnpackHalf2x16,      "unpackHalf2x16",      nullptr },

    { EOpMul,                 "matrixCompMult",      nullptr },
    { EOpOuterProduct,        "outerProduct",        nullptr },
    { EOpTranspose,           "transpose",           nullptr },
    { EOpDeterminant,         "determinant",         nullptr },
    { EOpMatrixIlwerse,       "ilwerse",             nullptr },
    { EOpFloatBitsToInt,      "floatBitsToInt",      nullptr },
    { EOpFloatBitsToUint,     "floatBitsToUint",     nullptr },
    { EOpIntBitsToFloat,      "intBitsToFloat",      nullptr },
    { EOpUintBitsToFloat,     "uintBitsToFloat",     nullptr },

    { EOpTextureQuerySize,      "textureSize",           nullptr },
    { EOpTextureQueryLod,       "textureQueryLod",       nullptr },
    { EOpTextureQueryLevels,    "textureQueryLevels",    nullptr },
    { EOpTextureQuerySamples,   "textureSamples",        nullptr },
    { EOpTexture,               "texture",               nullptr },
    { EOpTextureProj,           "textureProj",           nullptr },
    { EOpTextureLod,            "textureLod",            nullptr },
    { EOpTextureOffset,         "textureOffset",         nullptr },
    { EOpTextureFetch,          "texelFetch",            nullptr },
    { EOpTextureFetchOffset,    "texelFetchOffset",      nullptr },
    { EOpTextureProjOffset,     "textureProjOffset",     nullptr },
    { EOpTextureLodOffset,      "textureLodOffset",      nullptr },
    { EOpTextureProjLod,        "textureProjLod",        nullptr },
    { EOpTextureProjLodOffset,  "textureProjLodOffset",  nullptr },
    { EOpTextureGrad,           "textureGrad",           nullptr },
    { EOpTextureGradOffset,     "textureGradOffset",     nullptr },
    { EOpTextureProjGrad,       "textureProjGrad",       nullptr },
    { EOpTextureProjGradOffset, "textureProjGradOffset", nullptr },

    { EOpNull }
};

// For the given table of functions, add all the indicated prototypes for each
// one, to be returned in the passed in decls.
void AddTabledBuiltin(TString& decls, const BuiltInFunction& function)
{
    const auto isScalarType = [](int type) { return (type & TypeStringColumnMask) == 0; };

    // loop across these two:
    //  0: the varying arg set, and
    //  1: the fixed scalar args
    const ArgClass ClassFixed = (ArgClass)(ClassLS | ClassXLS | ClassLS2 | ClassFS | ClassFS2);
    for (int fixed = 0; fixed < ((function.classes & ClassFixed) > 0 ? 2 : 1); ++fixed) {

        if (fixed == 0 && (function.classes & ClassXLS))
            continue;

        // walk the type strings in TypeString[]
        for (int type = 0; type < TypeStringCount; ++type) {
            // skip types not selected: go from type to row number to type bit
            if ((function.types & (1 << (type >> TypeStringRowShift))) == 0)
                continue;

            // if we aren't on a scalar, and should be, skip
            if ((function.classes & ClassV1) && !isScalarType(type))
                continue;

            // if we aren't on a 3-vector, and should be, skip
            if ((function.classes & ClassV3) && (type & TypeStringColumnMask) != 2)
                continue;

            // skip replication of all arg scalars between the varying arg set and the fixed args
            if (fixed == 1 && type == (type & TypeStringScalarMask) && (function.classes & ClassXLS) == 0)
                continue;

            // skip scalars when we are told to
            if ((function.classes & ClassNS) && isScalarType(type))
                continue;

            // return type
            if (function.classes & ClassB)
                decls.append(TypeString[type & TypeStringColumnMask]);
            else if (function.classes & ClassRS)
                decls.append(TypeString[type & TypeStringScalarMask]);
            else
                decls.append(TypeString[type]);
            decls.append(" ");
            decls.append(function.name);
            decls.append("(");

            // arguments
            for (int arg = 0; arg < function.numArguments; ++arg) {
                if (arg == function.numArguments - 1 && (function.classes & ClassLO))
                    decls.append("out ");
                if (arg == 0) {
#ifndef GLSLANG_WEB
                    if (function.classes & ClassCV)
                        decls.append("coherent volatile ");
#endif
                    if (function.classes & ClassFIO)
                        decls.append("inout ");
                    if (function.classes & ClassFO)
                        decls.append("out ");
                }
                if ((function.classes & ClassLB) && arg == function.numArguments - 1)
                    decls.append(TypeString[type & TypeStringColumnMask]);
                else if (fixed && ((arg == function.numArguments - 1 && (function.classes & (ClassLS | ClassXLS |
                                                                                                       ClassLS2))) ||
                                   (arg == function.numArguments - 2 && (function.classes & ClassLS2))             ||
                                   (arg == 0                         && (function.classes & (ClassFS | ClassFS2))) ||
                                   (arg == 1                         && (function.classes & ClassFS2))))
                    decls.append(TypeString[type & TypeStringScalarMask]);
                else
                    decls.append(TypeString[type]);
                if (arg < function.numArguments - 1)
                    decls.append(",");
            }
            decls.append(");\n");
        }
    }
}

// See if the tabled versioning information allows the current version.
bool ValidVersion(const BuiltInFunction& function, int version, EProfile profile, const SpvVersion& /* spVersion */)
{
#ifdef GLSLANG_WEB
    // all entries in table are valid
    return true;
#endif

    // nullptr means always valid
    if (function.versioning == nullptr)
        return true;

    // check for what is said about our current profile
    for (const Versioning* v = function.versioning; v->profiles != EBadProfile; ++v) {
        if ((v->profiles & profile) != 0) {
            if (v->minCoreVersion <= version || (v->numExtensions > 0 && v->minExtendedVersion <= version))
                return true;
        }
    }

    return false;
}

// Relate a single table of built-ins to their AST operator.
// This can get called redundantly (especially for the common built-ins, when
// called once per stage). This is a performance issue only, not a correctness
// concern.  It is done for quality arising from simplicity, as there are subtleties
// to get correct if instead trying to do it surgically.
template<class FunctionT>
void RelateTabledBuiltins(const FunctionT* functions, TSymbolTable& symbolTable)
{
    while (functions->op != EOpNull) {
        symbolTable.relateToOperator(functions->name, functions->op);
        ++functions;
    }
}

} // end anonymous namespace

// Add declarations for all tables of built-in functions.
void TBuiltIns::addTabledBuiltins(int version, EProfile profile, const SpvVersion& spvVersion)
{
    const auto forEachFunction = [&](TString& decls, const BuiltInFunction* function) {
        while (function->op != EOpNull) {
            if (ValidVersion(*function, version, profile, spvVersion))
                AddTabledBuiltin(decls, *function);
            ++function;
        }
    };

    forEachFunction(commonBuiltins, BaseFunctions);
    forEachFunction(stageBuiltins[EShLangFragment], DerivativeFunctions);

    if ((profile == EEsProfile && version >= 320) || (profile != EEsProfile && version >= 450))
        forEachFunction(stageBuiltins[EShLangCompute], DerivativeFunctions);
}

// Relate all tables of built-ins to the AST operators.
void TBuiltIns::relateTabledBuiltins(int /* version */, EProfile /* profile */, const SpvVersion& /* spvVersion */, EShLanguage /* stage */, TSymbolTable& symbolTable)
{
    RelateTabledBuiltins(BaseFunctions, symbolTable);
    RelateTabledBuiltins(DerivativeFunctions, symbolTable);
    RelateTabledBuiltins(LwstomFunctions, symbolTable);
}

inline bool IncludeLegacy(int version, EProfile profile, const SpvVersion& spvVersion)
{
    return profile != EEsProfile && (version <= 130 || (spvVersion.spv == 0 && ARBCompatibility) || profile == ECompatibilityProfile);
}

// Construct TBuiltInParseables base class.  This can be used for language-common constructs.
TBuiltInParseables::TBuiltInParseables()
{
}

// Destroy TBuiltInParseables.
TBuiltInParseables::~TBuiltInParseables()
{
}

TBuiltIns::TBuiltIns()
{
    // Set up textual representations for making all the permutations
    // of texturing/imaging functions.
    prefixes[EbtFloat] =  "";
    prefixes[EbtInt]   = "i";
    prefixes[EbtUint]  = "u";
#ifndef GLSLANG_WEB
    prefixes[EbtFloat16] = "f16";
    prefixes[EbtInt8]  = "i8";
    prefixes[EbtUint8] = "u8";
    prefixes[EbtInt16]  = "i16";
    prefixes[EbtUint16] = "u16";
#endif

    postfixes[2] = "2";
    postfixes[3] = "3";
    postfixes[4] = "4";

    // Map from symbolic class of texturing dimension to numeric dimensions.
    dimMap[Esd2D] = 2;
    dimMap[Esd3D] = 3;
    dimMap[EsdLwbe] = 3;
#ifndef GLSLANG_WEB
    dimMap[Esd1D] = 1;
    dimMap[EsdRect] = 2;
    dimMap[EsdBuffer] = 1;
    dimMap[EsdSubpass] = 2;  // potentially unused for now
#endif
}

TBuiltIns::~TBuiltIns()
{
}


//
// Add all context-independent built-in functions and variables that are present
// for the given version and profile.  Share common ones across stages, otherwise
// make stage-specific entries.
//
// Most built-ins variables can be added as simple text strings.  Some need to
// be added programmatically, which is done later in IdentifyBuiltIns() below.
//
void TBuiltIns::initialize(int version, EProfile profile, const SpvVersion& spvVersion)
{
#ifdef GLSLANG_WEB
    version = 310;
    profile = EEsProfile;
#endif
    addTabledBuiltins(version, profile, spvVersion);

    //============================================================================
    //
    // Prototypes for built-in functions used repeatly by different shaders
    //
    //============================================================================

#ifndef GLSLANG_WEB
    //
    // Derivatives Functions.
    //
    TString derivativeControls (
        "float dFdxFine(float p);"
        "vec2  dFdxFine(vec2  p);"
        "vec3  dFdxFine(vec3  p);"
        "vec4  dFdxFine(vec4  p);"

        "float dFdyFine(float p);"
        "vec2  dFdyFine(vec2  p);"
        "vec3  dFdyFine(vec3  p);"
        "vec4  dFdyFine(vec4  p);"

        "float fwidthFine(float p);"
        "vec2  fwidthFine(vec2  p);"
        "vec3  fwidthFine(vec3  p);"
        "vec4  fwidthFine(vec4  p);"

        "float dFdxCoarse(float p);"
        "vec2  dFdxCoarse(vec2  p);"
        "vec3  dFdxCoarse(vec3  p);"
        "vec4  dFdxCoarse(vec4  p);"

        "float dFdyCoarse(float p);"
        "vec2  dFdyCoarse(vec2  p);"
        "vec3  dFdyCoarse(vec3  p);"
        "vec4  dFdyCoarse(vec4  p);"

        "float fwidthCoarse(float p);"
        "vec2  fwidthCoarse(vec2  p);"
        "vec3  fwidthCoarse(vec3  p);"
        "vec4  fwidthCoarse(vec4  p);"
    );

    TString derivativesAndControl16bits (
        "float16_t dFdx(float16_t);"
        "f16vec2   dFdx(f16vec2);"
        "f16vec3   dFdx(f16vec3);"
        "f16vec4   dFdx(f16vec4);"

        "float16_t dFdy(float16_t);"
        "f16vec2   dFdy(f16vec2);"
        "f16vec3   dFdy(f16vec3);"
        "f16vec4   dFdy(f16vec4);"

        "float16_t dFdxFine(float16_t);"
        "f16vec2   dFdxFine(f16vec2);"
        "f16vec3   dFdxFine(f16vec3);"
        "f16vec4   dFdxFine(f16vec4);"

        "float16_t dFdyFine(float16_t);"
        "f16vec2   dFdyFine(f16vec2);"
        "f16vec3   dFdyFine(f16vec3);"
        "f16vec4   dFdyFine(f16vec4);"

        "float16_t dFdxCoarse(float16_t);"
        "f16vec2   dFdxCoarse(f16vec2);"
        "f16vec3   dFdxCoarse(f16vec3);"
        "f16vec4   dFdxCoarse(f16vec4);"

        "float16_t dFdyCoarse(float16_t);"
        "f16vec2   dFdyCoarse(f16vec2);"
        "f16vec3   dFdyCoarse(f16vec3);"
        "f16vec4   dFdyCoarse(f16vec4);"

        "float16_t fwidth(float16_t);"
        "f16vec2   fwidth(f16vec2);"
        "f16vec3   fwidth(f16vec3);"
        "f16vec4   fwidth(f16vec4);"

        "float16_t fwidthFine(float16_t);"
        "f16vec2   fwidthFine(f16vec2);"
        "f16vec3   fwidthFine(f16vec3);"
        "f16vec4   fwidthFine(f16vec4);"

        "float16_t fwidthCoarse(float16_t);"
        "f16vec2   fwidthCoarse(f16vec2);"
        "f16vec3   fwidthCoarse(f16vec3);"
        "f16vec4   fwidthCoarse(f16vec4);"
    );

    TString derivativesAndControl64bits (
        "float64_t dFdx(float64_t);"
        "f64vec2   dFdx(f64vec2);"
        "f64vec3   dFdx(f64vec3);"
        "f64vec4   dFdx(f64vec4);"

        "float64_t dFdy(float64_t);"
        "f64vec2   dFdy(f64vec2);"
        "f64vec3   dFdy(f64vec3);"
        "f64vec4   dFdy(f64vec4);"

        "float64_t dFdxFine(float64_t);"
        "f64vec2   dFdxFine(f64vec2);"
        "f64vec3   dFdxFine(f64vec3);"
        "f64vec4   dFdxFine(f64vec4);"

        "float64_t dFdyFine(float64_t);"
        "f64vec2   dFdyFine(f64vec2);"
        "f64vec3   dFdyFine(f64vec3);"
        "f64vec4   dFdyFine(f64vec4);"

        "float64_t dFdxCoarse(float64_t);"
        "f64vec2   dFdxCoarse(f64vec2);"
        "f64vec3   dFdxCoarse(f64vec3);"
        "f64vec4   dFdxCoarse(f64vec4);"

        "float64_t dFdyCoarse(float64_t);"
        "f64vec2   dFdyCoarse(f64vec2);"
        "f64vec3   dFdyCoarse(f64vec3);"
        "f64vec4   dFdyCoarse(f64vec4);"

        "float64_t fwidth(float64_t);"
        "f64vec2   fwidth(f64vec2);"
        "f64vec3   fwidth(f64vec3);"
        "f64vec4   fwidth(f64vec4);"

        "float64_t fwidthFine(float64_t);"
        "f64vec2   fwidthFine(f64vec2);"
        "f64vec3   fwidthFine(f64vec3);"
        "f64vec4   fwidthFine(f64vec4);"

        "float64_t fwidthCoarse(float64_t);"
        "f64vec2   fwidthCoarse(f64vec2);"
        "f64vec3   fwidthCoarse(f64vec3);"
        "f64vec4   fwidthCoarse(f64vec4);"
    );

    //============================================================================
    //
    // Prototypes for built-in functions seen by both vertex and fragment shaders.
    //
    //============================================================================

    //
    // double functions added to desktop 4.00, but not fma, frexp, ldexp, or pack/unpack
    //
    if (profile != EEsProfile && version >= 150) {  // ARB_gpu_shader_fp64
        commonBuiltins.append(

            "double sqrt(double);"
            "dvec2  sqrt(dvec2);"
            "dvec3  sqrt(dvec3);"
            "dvec4  sqrt(dvec4);"

            "double ilwersesqrt(double);"
            "dvec2  ilwersesqrt(dvec2);"
            "dvec3  ilwersesqrt(dvec3);"
            "dvec4  ilwersesqrt(dvec4);"

            "double abs(double);"
            "dvec2  abs(dvec2);"
            "dvec3  abs(dvec3);"
            "dvec4  abs(dvec4);"

            "double sign(double);"
            "dvec2  sign(dvec2);"
            "dvec3  sign(dvec3);"
            "dvec4  sign(dvec4);"

            "double floor(double);"
            "dvec2  floor(dvec2);"
            "dvec3  floor(dvec3);"
            "dvec4  floor(dvec4);"

            "double trunc(double);"
            "dvec2  trunc(dvec2);"
            "dvec3  trunc(dvec3);"
            "dvec4  trunc(dvec4);"

            "double round(double);"
            "dvec2  round(dvec2);"
            "dvec3  round(dvec3);"
            "dvec4  round(dvec4);"

            "double roundEven(double);"
            "dvec2  roundEven(dvec2);"
            "dvec3  roundEven(dvec3);"
            "dvec4  roundEven(dvec4);"

            "double ceil(double);"
            "dvec2  ceil(dvec2);"
            "dvec3  ceil(dvec3);"
            "dvec4  ceil(dvec4);"

            "double fract(double);"
            "dvec2  fract(dvec2);"
            "dvec3  fract(dvec3);"
            "dvec4  fract(dvec4);"

            "double mod(double, double);"
            "dvec2  mod(dvec2 , double);"
            "dvec3  mod(dvec3 , double);"
            "dvec4  mod(dvec4 , double);"
            "dvec2  mod(dvec2 , dvec2);"
            "dvec3  mod(dvec3 , dvec3);"
            "dvec4  mod(dvec4 , dvec4);"

            "double modf(double, out double);"
            "dvec2  modf(dvec2,  out dvec2);"
            "dvec3  modf(dvec3,  out dvec3);"
            "dvec4  modf(dvec4,  out dvec4);"

            "double min(double, double);"
            "dvec2  min(dvec2,  double);"
            "dvec3  min(dvec3,  double);"
            "dvec4  min(dvec4,  double);"
            "dvec2  min(dvec2,  dvec2);"
            "dvec3  min(dvec3,  dvec3);"
            "dvec4  min(dvec4,  dvec4);"

            "double max(double, double);"
            "dvec2  max(dvec2 , double);"
            "dvec3  max(dvec3 , double);"
            "dvec4  max(dvec4 , double);"
            "dvec2  max(dvec2 , dvec2);"
            "dvec3  max(dvec3 , dvec3);"
            "dvec4  max(dvec4 , dvec4);"

            "double clamp(double, double, double);"
            "dvec2  clamp(dvec2 , double, double);"
            "dvec3  clamp(dvec3 , double, double);"
            "dvec4  clamp(dvec4 , double, double);"
            "dvec2  clamp(dvec2 , dvec2 , dvec2);"
            "dvec3  clamp(dvec3 , dvec3 , dvec3);"
            "dvec4  clamp(dvec4 , dvec4 , dvec4);"

            "double mix(double, double, double);"
            "dvec2  mix(dvec2,  dvec2,  double);"
            "dvec3  mix(dvec3,  dvec3,  double);"
            "dvec4  mix(dvec4,  dvec4,  double);"
            "dvec2  mix(dvec2,  dvec2,  dvec2);"
            "dvec3  mix(dvec3,  dvec3,  dvec3);"
            "dvec4  mix(dvec4,  dvec4,  dvec4);"
            "double mix(double, double, bool);"
            "dvec2  mix(dvec2,  dvec2,  bvec2);"
            "dvec3  mix(dvec3,  dvec3,  bvec3);"
            "dvec4  mix(dvec4,  dvec4,  bvec4);"

            "double step(double, double);"
            "dvec2  step(dvec2 , dvec2);"
            "dvec3  step(dvec3 , dvec3);"
            "dvec4  step(dvec4 , dvec4);"
            "dvec2  step(double, dvec2);"
            "dvec3  step(double, dvec3);"
            "dvec4  step(double, dvec4);"

            "double smoothstep(double, double, double);"
            "dvec2  smoothstep(dvec2 , dvec2 , dvec2);"
            "dvec3  smoothstep(dvec3 , dvec3 , dvec3);"
            "dvec4  smoothstep(dvec4 , dvec4 , dvec4);"
            "dvec2  smoothstep(double, double, dvec2);"
            "dvec3  smoothstep(double, double, dvec3);"
            "dvec4  smoothstep(double, double, dvec4);"

            "bool  isnan(double);"
            "bvec2 isnan(dvec2);"
            "bvec3 isnan(dvec3);"
            "bvec4 isnan(dvec4);"

            "bool  isinf(double);"
            "bvec2 isinf(dvec2);"
            "bvec3 isinf(dvec3);"
            "bvec4 isinf(dvec4);"

            "double length(double);"
            "double length(dvec2);"
            "double length(dvec3);"
            "double length(dvec4);"

            "double distance(double, double);"
            "double distance(dvec2 , dvec2);"
            "double distance(dvec3 , dvec3);"
            "double distance(dvec4 , dvec4);"

            "double dot(double, double);"
            "double dot(dvec2 , dvec2);"
            "double dot(dvec3 , dvec3);"
            "double dot(dvec4 , dvec4);"

            "dvec3 cross(dvec3, dvec3);"

            "double normalize(double);"
            "dvec2  normalize(dvec2);"
            "dvec3  normalize(dvec3);"
            "dvec4  normalize(dvec4);"

            "double faceforward(double, double, double);"
            "dvec2  faceforward(dvec2,  dvec2,  dvec2);"
            "dvec3  faceforward(dvec3,  dvec3,  dvec3);"
            "dvec4  faceforward(dvec4,  dvec4,  dvec4);"

            "double reflect(double, double);"
            "dvec2  reflect(dvec2 , dvec2 );"
            "dvec3  reflect(dvec3 , dvec3 );"
            "dvec4  reflect(dvec4 , dvec4 );"

            "double refract(double, double, double);"
            "dvec2  refract(dvec2 , dvec2 , double);"
            "dvec3  refract(dvec3 , dvec3 , double);"
            "dvec4  refract(dvec4 , dvec4 , double);"

            "dmat2 matrixCompMult(dmat2, dmat2);"
            "dmat3 matrixCompMult(dmat3, dmat3);"
            "dmat4 matrixCompMult(dmat4, dmat4);"
            "dmat2x3 matrixCompMult(dmat2x3, dmat2x3);"
            "dmat2x4 matrixCompMult(dmat2x4, dmat2x4);"
            "dmat3x2 matrixCompMult(dmat3x2, dmat3x2);"
            "dmat3x4 matrixCompMult(dmat3x4, dmat3x4);"
            "dmat4x2 matrixCompMult(dmat4x2, dmat4x2);"
            "dmat4x3 matrixCompMult(dmat4x3, dmat4x3);"

            "dmat2   outerProduct(dvec2, dvec2);"
            "dmat3   outerProduct(dvec3, dvec3);"
            "dmat4   outerProduct(dvec4, dvec4);"
            "dmat2x3 outerProduct(dvec3, dvec2);"
            "dmat3x2 outerProduct(dvec2, dvec3);"
            "dmat2x4 outerProduct(dvec4, dvec2);"
            "dmat4x2 outerProduct(dvec2, dvec4);"
            "dmat3x4 outerProduct(dvec4, dvec3);"
            "dmat4x3 outerProduct(dvec3, dvec4);"

            "dmat2   transpose(dmat2);"
            "dmat3   transpose(dmat3);"
            "dmat4   transpose(dmat4);"
            "dmat2x3 transpose(dmat3x2);"
            "dmat3x2 transpose(dmat2x3);"
            "dmat2x4 transpose(dmat4x2);"
            "dmat4x2 transpose(dmat2x4);"
            "dmat3x4 transpose(dmat4x3);"
            "dmat4x3 transpose(dmat3x4);"

            "double determinant(dmat2);"
            "double determinant(dmat3);"
            "double determinant(dmat4);"

            "dmat2 ilwerse(dmat2);"
            "dmat3 ilwerse(dmat3);"
            "dmat4 ilwerse(dmat4);"

            "bvec2 lessThan(dvec2, dvec2);"
            "bvec3 lessThan(dvec3, dvec3);"
            "bvec4 lessThan(dvec4, dvec4);"

            "bvec2 lessThanEqual(dvec2, dvec2);"
            "bvec3 lessThanEqual(dvec3, dvec3);"
            "bvec4 lessThanEqual(dvec4, dvec4);"

            "bvec2 greaterThan(dvec2, dvec2);"
            "bvec3 greaterThan(dvec3, dvec3);"
            "bvec4 greaterThan(dvec4, dvec4);"

            "bvec2 greaterThanEqual(dvec2, dvec2);"
            "bvec3 greaterThanEqual(dvec3, dvec3);"
            "bvec4 greaterThanEqual(dvec4, dvec4);"

            "bvec2 equal(dvec2, dvec2);"
            "bvec3 equal(dvec3, dvec3);"
            "bvec4 equal(dvec4, dvec4);"

            "bvec2 notEqual(dvec2, dvec2);"
            "bvec3 notEqual(dvec3, dvec3);"
            "bvec4 notEqual(dvec4, dvec4);"

            "\n");
    }

    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(

            "int64_t abs(int64_t);"
            "i64vec2 abs(i64vec2);"
            "i64vec3 abs(i64vec3);"
            "i64vec4 abs(i64vec4);"

            "int64_t sign(int64_t);"
            "i64vec2 sign(i64vec2);"
            "i64vec3 sign(i64vec3);"
            "i64vec4 sign(i64vec4);"

            "int64_t  min(int64_t,  int64_t);"
            "i64vec2  min(i64vec2,  int64_t);"
            "i64vec3  min(i64vec3,  int64_t);"
            "i64vec4  min(i64vec4,  int64_t);"
            "i64vec2  min(i64vec2,  i64vec2);"
            "i64vec3  min(i64vec3,  i64vec3);"
            "i64vec4  min(i64vec4,  i64vec4);"
            "uint64_t min(uint64_t, uint64_t);"
            "u64vec2  min(u64vec2,  uint64_t);"
            "u64vec3  min(u64vec3,  uint64_t);"
            "u64vec4  min(u64vec4,  uint64_t);"
            "u64vec2  min(u64vec2,  u64vec2);"
            "u64vec3  min(u64vec3,  u64vec3);"
            "u64vec4  min(u64vec4,  u64vec4);"

            "int64_t  max(int64_t,  int64_t);"
            "i64vec2  max(i64vec2,  int64_t);"
            "i64vec3  max(i64vec3,  int64_t);"
            "i64vec4  max(i64vec4,  int64_t);"
            "i64vec2  max(i64vec2,  i64vec2);"
            "i64vec3  max(i64vec3,  i64vec3);"
            "i64vec4  max(i64vec4,  i64vec4);"
            "uint64_t max(uint64_t, uint64_t);"
            "u64vec2  max(u64vec2,  uint64_t);"
            "u64vec3  max(u64vec3,  uint64_t);"
            "u64vec4  max(u64vec4,  uint64_t);"
            "u64vec2  max(u64vec2,  u64vec2);"
            "u64vec3  max(u64vec3,  u64vec3);"
            "u64vec4  max(u64vec4,  u64vec4);"

            "int64_t  clamp(int64_t,  int64_t,  int64_t);"
            "i64vec2  clamp(i64vec2,  int64_t,  int64_t);"
            "i64vec3  clamp(i64vec3,  int64_t,  int64_t);"
            "i64vec4  clamp(i64vec4,  int64_t,  int64_t);"
            "i64vec2  clamp(i64vec2,  i64vec2,  i64vec2);"
            "i64vec3  clamp(i64vec3,  i64vec3,  i64vec3);"
            "i64vec4  clamp(i64vec4,  i64vec4,  i64vec4);"
            "uint64_t clamp(uint64_t, uint64_t, uint64_t);"
            "u64vec2  clamp(u64vec2,  uint64_t, uint64_t);"
            "u64vec3  clamp(u64vec3,  uint64_t, uint64_t);"
            "u64vec4  clamp(u64vec4,  uint64_t, uint64_t);"
            "u64vec2  clamp(u64vec2,  u64vec2,  u64vec2);"
            "u64vec3  clamp(u64vec3,  u64vec3,  u64vec3);"
            "u64vec4  clamp(u64vec4,  u64vec4,  u64vec4);"

            "int64_t  mix(int64_t,  int64_t,  bool);"
            "i64vec2  mix(i64vec2,  i64vec2,  bvec2);"
            "i64vec3  mix(i64vec3,  i64vec3,  bvec3);"
            "i64vec4  mix(i64vec4,  i64vec4,  bvec4);"
            "uint64_t mix(uint64_t, uint64_t, bool);"
            "u64vec2  mix(u64vec2,  u64vec2,  bvec2);"
            "u64vec3  mix(u64vec3,  u64vec3,  bvec3);"
            "u64vec4  mix(u64vec4,  u64vec4,  bvec4);"

            "int64_t doubleBitsToInt64(double);"
            "i64vec2 doubleBitsToInt64(dvec2);"
            "i64vec3 doubleBitsToInt64(dvec3);"
            "i64vec4 doubleBitsToInt64(dvec4);"

            "uint64_t doubleBitsToUint64(double);"
            "u64vec2  doubleBitsToUint64(dvec2);"
            "u64vec3  doubleBitsToUint64(dvec3);"
            "u64vec4  doubleBitsToUint64(dvec4);"

            "double int64BitsToDouble(int64_t);"
            "dvec2  int64BitsToDouble(i64vec2);"
            "dvec3  int64BitsToDouble(i64vec3);"
            "dvec4  int64BitsToDouble(i64vec4);"

            "double uint64BitsToDouble(uint64_t);"
            "dvec2  uint64BitsToDouble(u64vec2);"
            "dvec3  uint64BitsToDouble(u64vec3);"
            "dvec4  uint64BitsToDouble(u64vec4);"

            "int64_t  packInt2x32(ivec2);"
            "uint64_t packUint2x32(uvec2);"
            "ivec2    unpackInt2x32(int64_t);"
            "uvec2    unpackUint2x32(uint64_t);"

            "bvec2 lessThan(i64vec2, i64vec2);"
            "bvec3 lessThan(i64vec3, i64vec3);"
            "bvec4 lessThan(i64vec4, i64vec4);"
            "bvec2 lessThan(u64vec2, u64vec2);"
            "bvec3 lessThan(u64vec3, u64vec3);"
            "bvec4 lessThan(u64vec4, u64vec4);"

            "bvec2 lessThanEqual(i64vec2, i64vec2);"
            "bvec3 lessThanEqual(i64vec3, i64vec3);"
            "bvec4 lessThanEqual(i64vec4, i64vec4);"
            "bvec2 lessThanEqual(u64vec2, u64vec2);"
            "bvec3 lessThanEqual(u64vec3, u64vec3);"
            "bvec4 lessThanEqual(u64vec4, u64vec4);"

            "bvec2 greaterThan(i64vec2, i64vec2);"
            "bvec3 greaterThan(i64vec3, i64vec3);"
            "bvec4 greaterThan(i64vec4, i64vec4);"
            "bvec2 greaterThan(u64vec2, u64vec2);"
            "bvec3 greaterThan(u64vec3, u64vec3);"
            "bvec4 greaterThan(u64vec4, u64vec4);"

            "bvec2 greaterThanEqual(i64vec2, i64vec2);"
            "bvec3 greaterThanEqual(i64vec3, i64vec3);"
            "bvec4 greaterThanEqual(i64vec4, i64vec4);"
            "bvec2 greaterThanEqual(u64vec2, u64vec2);"
            "bvec3 greaterThanEqual(u64vec3, u64vec3);"
            "bvec4 greaterThanEqual(u64vec4, u64vec4);"

            "bvec2 equal(i64vec2, i64vec2);"
            "bvec3 equal(i64vec3, i64vec3);"
            "bvec4 equal(i64vec4, i64vec4);"
            "bvec2 equal(u64vec2, u64vec2);"
            "bvec3 equal(u64vec3, u64vec3);"
            "bvec4 equal(u64vec4, u64vec4);"

            "bvec2 notEqual(i64vec2, i64vec2);"
            "bvec3 notEqual(i64vec3, i64vec3);"
            "bvec4 notEqual(i64vec4, i64vec4);"
            "bvec2 notEqual(u64vec2, u64vec2);"
            "bvec3 notEqual(u64vec3, u64vec3);"
            "bvec4 notEqual(u64vec4, u64vec4);"

            "int64_t findLSB(int64_t);"
            "i64vec2 findLSB(i64vec2);"
            "i64vec3 findLSB(i64vec3);"
            "i64vec4 findLSB(i64vec4);"

            "int64_t findLSB(uint64_t);"
            "i64vec2 findLSB(u64vec2);"
            "i64vec3 findLSB(u64vec3);"
            "i64vec4 findLSB(u64vec4);"

            "int64_t findMSB(int64_t);"
            "i64vec2 findMSB(i64vec2);"
            "i64vec3 findMSB(i64vec3);"
            "i64vec4 findMSB(i64vec4);"

            "int64_t findMSB(uint64_t);"
            "i64vec2 findMSB(u64vec2);"
            "i64vec3 findMSB(u64vec3);"
            "i64vec4 findMSB(u64vec4);"

            "\n"
        );
    }

    // GL_AMD_shader_trinary_minmax
    if (profile != EEsProfile && version >= 430) {
        commonBuiltins.append(
            "float min3(float, float, float);"
            "vec2  min3(vec2,  vec2,  vec2);"
            "vec3  min3(vec3,  vec3,  vec3);"
            "vec4  min3(vec4,  vec4,  vec4);"

            "int   min3(int,   int,   int);"
            "ivec2 min3(ivec2, ivec2, ivec2);"
            "ivec3 min3(ivec3, ivec3, ivec3);"
            "ivec4 min3(ivec4, ivec4, ivec4);"

            "uint  min3(uint,  uint,  uint);"
            "uvec2 min3(uvec2, uvec2, uvec2);"
            "uvec3 min3(uvec3, uvec3, uvec3);"
            "uvec4 min3(uvec4, uvec4, uvec4);"

            "float max3(float, float, float);"
            "vec2  max3(vec2,  vec2,  vec2);"
            "vec3  max3(vec3,  vec3,  vec3);"
            "vec4  max3(vec4,  vec4,  vec4);"

            "int   max3(int,   int,   int);"
            "ivec2 max3(ivec2, ivec2, ivec2);"
            "ivec3 max3(ivec3, ivec3, ivec3);"
            "ivec4 max3(ivec4, ivec4, ivec4);"

            "uint  max3(uint,  uint,  uint);"
            "uvec2 max3(uvec2, uvec2, uvec2);"
            "uvec3 max3(uvec3, uvec3, uvec3);"
            "uvec4 max3(uvec4, uvec4, uvec4);"

            "float mid3(float, float, float);"
            "vec2  mid3(vec2,  vec2,  vec2);"
            "vec3  mid3(vec3,  vec3,  vec3);"
            "vec4  mid3(vec4,  vec4,  vec4);"

            "int   mid3(int,   int,   int);"
            "ivec2 mid3(ivec2, ivec2, ivec2);"
            "ivec3 mid3(ivec3, ivec3, ivec3);"
            "ivec4 mid3(ivec4, ivec4, ivec4);"

            "uint  mid3(uint,  uint,  uint);"
            "uvec2 mid3(uvec2, uvec2, uvec2);"
            "uvec3 mid3(uvec3, uvec3, uvec3);"
            "uvec4 mid3(uvec4, uvec4, uvec4);"

            "float16_t min3(float16_t, float16_t, float16_t);"
            "f16vec2   min3(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   min3(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   min3(f16vec4,   f16vec4,   f16vec4);"

            "float16_t max3(float16_t, float16_t, float16_t);"
            "f16vec2   max3(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   max3(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   max3(f16vec4,   f16vec4,   f16vec4);"

            "float16_t mid3(float16_t, float16_t, float16_t);"
            "f16vec2   mid3(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   mid3(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   mid3(f16vec4,   f16vec4,   f16vec4);"

            "int16_t   min3(int16_t,   int16_t,   int16_t);"
            "i16vec2   min3(i16vec2,   i16vec2,   i16vec2);"
            "i16vec3   min3(i16vec3,   i16vec3,   i16vec3);"
            "i16vec4   min3(i16vec4,   i16vec4,   i16vec4);"

            "int16_t   max3(int16_t,   int16_t,   int16_t);"
            "i16vec2   max3(i16vec2,   i16vec2,   i16vec2);"
            "i16vec3   max3(i16vec3,   i16vec3,   i16vec3);"
            "i16vec4   max3(i16vec4,   i16vec4,   i16vec4);"

            "int16_t   mid3(int16_t,   int16_t,   int16_t);"
            "i16vec2   mid3(i16vec2,   i16vec2,   i16vec2);"
            "i16vec3   mid3(i16vec3,   i16vec3,   i16vec3);"
            "i16vec4   mid3(i16vec4,   i16vec4,   i16vec4);"

            "uint16_t  min3(uint16_t,  uint16_t,  uint16_t);"
            "u16vec2   min3(u16vec2,   u16vec2,   u16vec2);"
            "u16vec3   min3(u16vec3,   u16vec3,   u16vec3);"
            "u16vec4   min3(u16vec4,   u16vec4,   u16vec4);"

            "uint16_t  max3(uint16_t,  uint16_t,  uint16_t);"
            "u16vec2   max3(u16vec2,   u16vec2,   u16vec2);"
            "u16vec3   max3(u16vec3,   u16vec3,   u16vec3);"
            "u16vec4   max3(u16vec4,   u16vec4,   u16vec4);"

            "uint16_t  mid3(uint16_t,  uint16_t,  uint16_t);"
            "u16vec2   mid3(u16vec2,   u16vec2,   u16vec2);"
            "u16vec3   mid3(u16vec3,   u16vec3,   u16vec3);"
            "u16vec4   mid3(u16vec4,   u16vec4,   u16vec4);"

            "\n"
        );
    }

    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 430)) {
        commonBuiltins.append(
            "uint atomicAdd(coherent volatile inout uint, uint, int, int, int);"
            " int atomicAdd(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicMin(coherent volatile inout uint, uint, int, int, int);"
            " int atomicMin(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicMax(coherent volatile inout uint, uint, int, int, int);"
            " int atomicMax(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicAnd(coherent volatile inout uint, uint, int, int, int);"
            " int atomicAnd(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicOr (coherent volatile inout uint, uint, int, int, int);"
            " int atomicOr (coherent volatile inout  int,  int, int, int, int);"

            "uint atomicXor(coherent volatile inout uint, uint, int, int, int);"
            " int atomicXor(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicExchange(coherent volatile inout uint, uint, int, int, int);"
            " int atomicExchange(coherent volatile inout  int,  int, int, int, int);"

            "uint atomicCompSwap(coherent volatile inout uint, uint, uint, int, int, int, int, int);"
            " int atomicCompSwap(coherent volatile inout  int,  int,  int, int, int, int, int, int);"

            "uint atomicLoad(coherent volatile in uint, int, int, int);"
            " int atomicLoad(coherent volatile in  int, int, int, int);"

            "void atomicStore(coherent volatile out uint, uint, int, int, int);"
            "void atomicStore(coherent volatile out  int,  int, int, int, int);"

            "\n");
    }

    if (profile != EEsProfile && version >= 440) {
        commonBuiltins.append(
            "uint64_t atomicMin(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicMin(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicMin(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicMin(coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicMax(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicMax(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicMax(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicMax(coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicAnd(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicAnd(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicAnd(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicAnd(coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicOr (coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicOr (coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicOr (coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicOr (coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicXor(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicXor(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicXor(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicXor(coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicAdd(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicAdd(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicAdd(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicAdd(coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicExchange(coherent volatile inout uint64_t, uint64_t);"
            " int64_t atomicExchange(coherent volatile inout  int64_t,  int64_t);"
            "uint64_t atomicExchange(coherent volatile inout uint64_t, uint64_t, int, int, int);"
            " int64_t atomicExchange(coherent volatile inout  int64_t,  int64_t, int, int, int);"

            "uint64_t atomicCompSwap(coherent volatile inout uint64_t, uint64_t, uint64_t);"
            " int64_t atomicCompSwap(coherent volatile inout  int64_t,  int64_t,  int64_t);"
            "uint64_t atomicCompSwap(coherent volatile inout uint64_t, uint64_t, uint64_t, int, int, int, int, int);"
            " int64_t atomicCompSwap(coherent volatile inout  int64_t,  int64_t,  int64_t, int, int, int, int, int);"

            "uint64_t atomicLoad(coherent volatile in uint64_t, int, int, int);"
            " int64_t atomicLoad(coherent volatile in  int64_t, int, int, int);"

            "void atomicStore(coherent volatile out uint64_t, uint64_t, int, int, int);"
            "void atomicStore(coherent volatile out  int64_t,  int64_t, int, int, int);"
            "\n");
    }
#endif

    if ((profile == EEsProfile && version >= 300) ||
        (profile != EEsProfile && version >= 150)) { // GL_ARB_shader_bit_encoding
        commonBuiltins.append(
            "int   floatBitsToInt(highp float value);"
            "ivec2 floatBitsToInt(highp vec2  value);"
            "ivec3 floatBitsToInt(highp vec3  value);"
            "ivec4 floatBitsToInt(highp vec4  value);"

            "uint  floatBitsToUint(highp float value);"
            "uvec2 floatBitsToUint(highp vec2  value);"
            "uvec3 floatBitsToUint(highp vec3  value);"
            "uvec4 floatBitsToUint(highp vec4  value);"

            "float intBitsToFloat(highp int   value);"
            "vec2  intBitsToFloat(highp ivec2 value);"
            "vec3  intBitsToFloat(highp ivec3 value);"
            "vec4  intBitsToFloat(highp ivec4 value);"

            "float uintBitsToFloat(highp uint  value);"
            "vec2  uintBitsToFloat(highp uvec2 value);"
            "vec3  uintBitsToFloat(highp uvec3 value);"
            "vec4  uintBitsToFloat(highp uvec4 value);"

            "\n");
    }

#ifndef GLSLANG_WEB
    if ((profile != EEsProfile && version >= 400) ||
        (profile == EEsProfile && version >= 310)) {    // GL_OES_gpu_shader5

        commonBuiltins.append(
            "float  fma(float,  float,  float );"
            "vec2   fma(vec2,   vec2,   vec2  );"
            "vec3   fma(vec3,   vec3,   vec3  );"
            "vec4   fma(vec4,   vec4,   vec4  );"
            "\n");
    }

    if (profile != EEsProfile && version >= 150) {  // ARB_gpu_shader_fp64
            commonBuiltins.append(
                "double fma(double, double, double);"
                "dvec2  fma(dvec2,  dvec2,  dvec2 );"
                "dvec3  fma(dvec3,  dvec3,  dvec3 );"
                "dvec4  fma(dvec4,  dvec4,  dvec4 );"
                "\n");
    }

    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 400)) {
        commonBuiltins.append(
            "float frexp(highp float, out highp int);"
            "vec2  frexp(highp vec2,  out highp ivec2);"
            "vec3  frexp(highp vec3,  out highp ivec3);"
            "vec4  frexp(highp vec4,  out highp ivec4);"

            "float ldexp(highp float, highp int);"
            "vec2  ldexp(highp vec2,  highp ivec2);"
            "vec3  ldexp(highp vec3,  highp ivec3);"
            "vec4  ldexp(highp vec4,  highp ivec4);"

            "\n");
    }

    if (profile != EEsProfile && version >= 150) { // ARB_gpu_shader_fp64
        commonBuiltins.append(
            "double frexp(double, out int);"
            "dvec2  frexp( dvec2, out ivec2);"
            "dvec3  frexp( dvec3, out ivec3);"
            "dvec4  frexp( dvec4, out ivec4);"

            "double ldexp(double, int);"
            "dvec2  ldexp( dvec2, ivec2);"
            "dvec3  ldexp( dvec3, ivec3);"
            "dvec4  ldexp( dvec4, ivec4);"

            "double packDouble2x32(uvec2);"
            "uvec2 unpackDouble2x32(double);"

            "\n");
    }
#endif

    if ((profile == EEsProfile && version >= 300) ||
        (profile != EEsProfile && version >= 150)) {
        commonBuiltins.append(
            "highp uint packUnorm2x16(vec2);"
                  "vec2 unpackUnorm2x16(highp uint);"
            "\n");
    }

    if ((profile == EEsProfile && version >= 300) ||
        (profile != EEsProfile && version >= 150)) {
        commonBuiltins.append(
            "highp uint packSnorm2x16(vec2);"
            "      vec2 unpackSnorm2x16(highp uint);"
            "highp uint packHalf2x16(vec2);"
            "\n");
    }

    if (profile == EEsProfile && version >= 300) {
        commonBuiltins.append(
            "mediump vec2 unpackHalf2x16(highp uint);"
            "\n");
    } else if (profile != EEsProfile && version >= 150) {
        commonBuiltins.append(
            "        vec2 unpackHalf2x16(highp uint);"
            "\n");
    }

#ifndef GLSLANG_WEB
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 150)) {
        commonBuiltins.append(
            "highp uint packSnorm4x8(vec4);"
            "highp uint packUnorm4x8(vec4);"
            "\n");
    }

    if (profile == EEsProfile && version >= 310) {
        commonBuiltins.append(
            "mediump vec4 unpackSnorm4x8(highp uint);"
            "mediump vec4 unpackUnorm4x8(highp uint);"
            "\n");
    } else if (profile != EEsProfile && version >= 150) {
        commonBuiltins.append(
                    "vec4 unpackSnorm4x8(highp uint);"
                    "vec4 unpackUnorm4x8(highp uint);"
            "\n");
    }
#endif

    //
    // Matrix Functions.
    //
    commonBuiltins.append(
        "mat2 matrixCompMult(mat2 x, mat2 y);"
        "mat3 matrixCompMult(mat3 x, mat3 y);"
        "mat4 matrixCompMult(mat4 x, mat4 y);"

        "\n");

    // 120 is correct for both ES and desktop
    if (version >= 120) {
        commonBuiltins.append(
            "mat2   outerProduct(vec2 c, vec2 r);"
            "mat3   outerProduct(vec3 c, vec3 r);"
            "mat4   outerProduct(vec4 c, vec4 r);"
            "mat2x3 outerProduct(vec3 c, vec2 r);"
            "mat3x2 outerProduct(vec2 c, vec3 r);"
            "mat2x4 outerProduct(vec4 c, vec2 r);"
            "mat4x2 outerProduct(vec2 c, vec4 r);"
            "mat3x4 outerProduct(vec4 c, vec3 r);"
            "mat4x3 outerProduct(vec3 c, vec4 r);"

            "mat2   transpose(mat2   m);"
            "mat3   transpose(mat3   m);"
            "mat4   transpose(mat4   m);"
            "mat2x3 transpose(mat3x2 m);"
            "mat3x2 transpose(mat2x3 m);"
            "mat2x4 transpose(mat4x2 m);"
            "mat4x2 transpose(mat2x4 m);"
            "mat3x4 transpose(mat4x3 m);"
            "mat4x3 transpose(mat3x4 m);"

            "mat2x3 matrixCompMult(mat2x3, mat2x3);"
            "mat2x4 matrixCompMult(mat2x4, mat2x4);"
            "mat3x2 matrixCompMult(mat3x2, mat3x2);"
            "mat3x4 matrixCompMult(mat3x4, mat3x4);"
            "mat4x2 matrixCompMult(mat4x2, mat4x2);"
            "mat4x3 matrixCompMult(mat4x3, mat4x3);"

            "\n");

        // 150 is correct for both ES and desktop
        if (version >= 150) {
            commonBuiltins.append(
                "float determinant(mat2 m);"
                "float determinant(mat3 m);"
                "float determinant(mat4 m);"

                "mat2 ilwerse(mat2 m);"
                "mat3 ilwerse(mat3 m);"
                "mat4 ilwerse(mat4 m);"

                "\n");
        }
    }

#ifndef GLSLANG_WEB
    //
    // Original-style texture functions existing in all stages.
    // (Per-stage functions below.)
    //
    if ((profile == EEsProfile && version == 100) ||
         profile == ECompatibilityProfile ||
        (profile == ECoreProfile && version < 420) ||
         profile == ENoProfile) {
        if (spvVersion.spv == 0) {
            commonBuiltins.append(
                "vec4 texture2D(sampler2D, vec2);"

                "vec4 texture2DProj(sampler2D, vec3);"
                "vec4 texture2DProj(sampler2D, vec4);"

                "vec4 texture3D(sampler3D, vec3);"     // OES_texture_3D, but caught by keyword check
                "vec4 texture3DProj(sampler3D, vec4);" // OES_texture_3D, but caught by keyword check

                "vec4 textureLwbe(samplerLwbe, vec3);"

                "\n");
        }
    }

    if ( profile == ECompatibilityProfile ||
        (profile == ECoreProfile && version < 420) ||
         profile == ENoProfile) {
        if (spvVersion.spv == 0) {
            commonBuiltins.append(
                "vec4 texture1D(sampler1D, float);"

                "vec4 texture1DProj(sampler1D, vec2);"
                "vec4 texture1DProj(sampler1D, vec4);"

                "vec4 shadow1D(sampler1DShadow, vec3);"
                "vec4 shadow2D(sampler2DShadow, vec3);"
                "vec4 shadow1DProj(sampler1DShadow, vec4);"
                "vec4 shadow2DProj(sampler2DShadow, vec4);"

                "vec4 texture2DRect(sampler2DRect, vec2);"          // GL_ARB_texture_rectangle, caught by keyword check
                "vec4 texture2DRectProj(sampler2DRect, vec3);"      // GL_ARB_texture_rectangle, caught by keyword check
                "vec4 texture2DRectProj(sampler2DRect, vec4);"      // GL_ARB_texture_rectangle, caught by keyword check
                "vec4 shadow2DRect(sampler2DRectShadow, vec3);"     // GL_ARB_texture_rectangle, caught by keyword check
                "vec4 shadow2DRectProj(sampler2DRectShadow, vec4);" // GL_ARB_texture_rectangle, caught by keyword check

                "\n");
        }
    }

    if (profile == EEsProfile) {
        if (spvVersion.spv == 0) {
            if (version < 300) {
                commonBuiltins.append(
                    "vec4 texture2D(samplerExternalOES, vec2 coord);" // GL_OES_EGL_image_external
                    "vec4 texture2DProj(samplerExternalOES, vec3);"   // GL_OES_EGL_image_external
                    "vec4 texture2DProj(samplerExternalOES, vec4);"   // GL_OES_EGL_image_external
                "\n");
            } else {
                commonBuiltins.append(
                    "highp ivec2 textureSize(samplerExternalOES, int lod);"   // GL_OES_EGL_image_external_essl3
                    "vec4 texture(samplerExternalOES, vec2);"                 // GL_OES_EGL_image_external_essl3
                    "vec4 texture(samplerExternalOES, vec2, float bias);"     // GL_OES_EGL_image_external_essl3
                    "vec4 textureProj(samplerExternalOES, vec3);"             // GL_OES_EGL_image_external_essl3
                    "vec4 textureProj(samplerExternalOES, vec3, float bias);" // GL_OES_EGL_image_external_essl3
                    "vec4 textureProj(samplerExternalOES, vec4);"             // GL_OES_EGL_image_external_essl3
                    "vec4 textureProj(samplerExternalOES, vec4, float bias);" // GL_OES_EGL_image_external_essl3
                    "vec4 texelFetch(samplerExternalOES, ivec2, int lod);"    // GL_OES_EGL_image_external_essl3
                "\n");
            }
            commonBuiltins.append(
                "highp ivec2 textureSize(__samplerExternal2DY2YEXT, int lod);" // GL_EXT_YUV_target
                "vec4 texture(__samplerExternal2DY2YEXT, vec2);"               // GL_EXT_YUV_target
                "vec4 texture(__samplerExternal2DY2YEXT, vec2, float bias);"   // GL_EXT_YUV_target
                "vec4 textureProj(__samplerExternal2DY2YEXT, vec3);"           // GL_EXT_YUV_target
                "vec4 textureProj(__samplerExternal2DY2YEXT, vec3, float bias);" // GL_EXT_YUV_target
                "vec4 textureProj(__samplerExternal2DY2YEXT, vec4);"           // GL_EXT_YUV_target
                "vec4 textureProj(__samplerExternal2DY2YEXT, vec4, float bias);" // GL_EXT_YUV_target
                "vec4 texelFetch(__samplerExternal2DY2YEXT sampler, ivec2, int lod);" // GL_EXT_YUV_target
                "\n");
            commonBuiltins.append(
                "vec4 texture2DGradEXT(sampler2D, vec2, vec2, vec2);"      // GL_EXT_shader_texture_lod
                "vec4 texture2DProjGradEXT(sampler2D, vec3, vec2, vec2);"  // GL_EXT_shader_texture_lod
                "vec4 texture2DProjGradEXT(sampler2D, vec4, vec2, vec2);"  // GL_EXT_shader_texture_lod
                "vec4 textureLwbeGradEXT(samplerLwbe, vec3, vec3, vec3);"  // GL_EXT_shader_texture_lod

                "float shadow2DEXT(sampler2DShadow, vec3);"     // GL_EXT_shadow_samplers
                "float shadow2DProjEXT(sampler2DShadow, vec4);" // GL_EXT_shadow_samplers

                "\n");
        }
    }

    //
    // Noise functions.
    //
    if (spvVersion.spv == 0 && profile != EEsProfile) {
        commonBuiltins.append(
            "float noise1(float x);"
            "float noise1(vec2  x);"
            "float noise1(vec3  x);"
            "float noise1(vec4  x);"

            "vec2 noise2(float x);"
            "vec2 noise2(vec2  x);"
            "vec2 noise2(vec3  x);"
            "vec2 noise2(vec4  x);"

            "vec3 noise3(float x);"
            "vec3 noise3(vec2  x);"
            "vec3 noise3(vec3  x);"
            "vec3 noise3(vec4  x);"

            "vec4 noise4(float x);"
            "vec4 noise4(vec2  x);"
            "vec4 noise4(vec3  x);"
            "vec4 noise4(vec4  x);"

            "\n");
    }

    if (spvVersion.vulkan == 0) {
        //
        // Atomic counter functions.
        //
        if ((profile != EEsProfile && version >= 300) ||
            (profile == EEsProfile && version >= 310)) {
            commonBuiltins.append(
                "uint atomicCounterIncrement(atomic_uint);"
                "uint atomicCounterDecrement(atomic_uint);"
                "uint atomicCounter(atomic_uint);"

                "\n");
        }
        if (profile != EEsProfile && version >= 460) {
            commonBuiltins.append(
                "uint atomicCounterAdd(atomic_uint, uint);"
                "uint atomicCounterSubtract(atomic_uint, uint);"
                "uint atomicCounterMin(atomic_uint, uint);"
                "uint atomicCounterMax(atomic_uint, uint);"
                "uint atomicCounterAnd(atomic_uint, uint);"
                "uint atomicCounterOr(atomic_uint, uint);"
                "uint atomicCounterXor(atomic_uint, uint);"
                "uint atomicCounterExchange(atomic_uint, uint);"
                "uint atomicCounterCompSwap(atomic_uint, uint, uint);"

                "\n");
        }
    }

    // Bitfield
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 400)) {
        commonBuiltins.append(
            "  int bitfieldExtract(  int, int, int);"
            "ivec2 bitfieldExtract(ivec2, int, int);"
            "ivec3 bitfieldExtract(ivec3, int, int);"
            "ivec4 bitfieldExtract(ivec4, int, int);"

            " uint bitfieldExtract( uint, int, int);"
            "uvec2 bitfieldExtract(uvec2, int, int);"
            "uvec3 bitfieldExtract(uvec3, int, int);"
            "uvec4 bitfieldExtract(uvec4, int, int);"

            "  int bitfieldInsert(  int base,   int, int, int);"
            "ivec2 bitfieldInsert(ivec2 base, ivec2, int, int);"
            "ivec3 bitfieldInsert(ivec3 base, ivec3, int, int);"
            "ivec4 bitfieldInsert(ivec4 base, ivec4, int, int);"

            " uint bitfieldInsert( uint base,  uint, int, int);"
            "uvec2 bitfieldInsert(uvec2 base, uvec2, int, int);"
            "uvec3 bitfieldInsert(uvec3 base, uvec3, int, int);"
            "uvec4 bitfieldInsert(uvec4 base, uvec4, int, int);"

            "\n");
    }

    if (profile != EEsProfile && version >= 400) {
        commonBuiltins.append(
            "  int findLSB(  int);"
            "ivec2 findLSB(ivec2);"
            "ivec3 findLSB(ivec3);"
            "ivec4 findLSB(ivec4);"

            "  int findLSB( uint);"
            "ivec2 findLSB(uvec2);"
            "ivec3 findLSB(uvec3);"
            "ivec4 findLSB(uvec4);"

            "\n");
    } else if (profile == EEsProfile && version >= 310) {
        commonBuiltins.append(
            "lowp   int findLSB(  int);"
            "lowp ivec2 findLSB(ivec2);"
            "lowp ivec3 findLSB(ivec3);"
            "lowp ivec4 findLSB(ivec4);"

            "lowp   int findLSB( uint);"
            "lowp ivec2 findLSB(uvec2);"
            "lowp ivec3 findLSB(uvec3);"
            "lowp ivec4 findLSB(uvec4);"

            "\n");
    }

    if (profile != EEsProfile && version >= 400) {
        commonBuiltins.append(
            "  int bitCount(  int);"
            "ivec2 bitCount(ivec2);"
            "ivec3 bitCount(ivec3);"
            "ivec4 bitCount(ivec4);"

            "  int bitCount( uint);"
            "ivec2 bitCount(uvec2);"
            "ivec3 bitCount(uvec3);"
            "ivec4 bitCount(uvec4);"

            "  int findMSB(highp   int);"
            "ivec2 findMSB(highp ivec2);"
            "ivec3 findMSB(highp ivec3);"
            "ivec4 findMSB(highp ivec4);"

            "  int findMSB(highp  uint);"
            "ivec2 findMSB(highp uvec2);"
            "ivec3 findMSB(highp uvec3);"
            "ivec4 findMSB(highp uvec4);"

            "\n");
    }

    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 400)) {
        commonBuiltins.append(
            " uint uaddCarry(highp  uint, highp  uint, out lowp  uint carry);"
            "uvec2 uaddCarry(highp uvec2, highp uvec2, out lowp uvec2 carry);"
            "uvec3 uaddCarry(highp uvec3, highp uvec3, out lowp uvec3 carry);"
            "uvec4 uaddCarry(highp uvec4, highp uvec4, out lowp uvec4 carry);"

            " uint usubBorrow(highp  uint, highp  uint, out lowp  uint borrow);"
            "uvec2 usubBorrow(highp uvec2, highp uvec2, out lowp uvec2 borrow);"
            "uvec3 usubBorrow(highp uvec3, highp uvec3, out lowp uvec3 borrow);"
            "uvec4 usubBorrow(highp uvec4, highp uvec4, out lowp uvec4 borrow);"

            "void umulExtended(highp  uint, highp  uint, out highp  uint, out highp  uint lsb);"
            "void umulExtended(highp uvec2, highp uvec2, out highp uvec2, out highp uvec2 lsb);"
            "void umulExtended(highp uvec3, highp uvec3, out highp uvec3, out highp uvec3 lsb);"
            "void umulExtended(highp uvec4, highp uvec4, out highp uvec4, out highp uvec4 lsb);"

            "void imulExtended(highp   int, highp   int, out highp   int, out highp   int lsb);"
            "void imulExtended(highp ivec2, highp ivec2, out highp ivec2, out highp ivec2 lsb);"
            "void imulExtended(highp ivec3, highp ivec3, out highp ivec3, out highp ivec3 lsb);"
            "void imulExtended(highp ivec4, highp ivec4, out highp ivec4, out highp ivec4 lsb);"

            "  int bitfieldReverse(highp   int);"
            "ivec2 bitfieldReverse(highp ivec2);"
            "ivec3 bitfieldReverse(highp ivec3);"
            "ivec4 bitfieldReverse(highp ivec4);"

            " uint bitfieldReverse(highp  uint);"
            "uvec2 bitfieldReverse(highp uvec2);"
            "uvec3 bitfieldReverse(highp uvec3);"
            "uvec4 bitfieldReverse(highp uvec4);"

            "\n");
    }

    if (profile == EEsProfile && version >= 310) {
        commonBuiltins.append(
            "lowp   int bitCount(  int);"
            "lowp ivec2 bitCount(ivec2);"
            "lowp ivec3 bitCount(ivec3);"
            "lowp ivec4 bitCount(ivec4);"

            "lowp   int bitCount( uint);"
            "lowp ivec2 bitCount(uvec2);"
            "lowp ivec3 bitCount(uvec3);"
            "lowp ivec4 bitCount(uvec4);"

            "lowp   int findMSB(highp   int);"
            "lowp ivec2 findMSB(highp ivec2);"
            "lowp ivec3 findMSB(highp ivec3);"
            "lowp ivec4 findMSB(highp ivec4);"

            "lowp   int findMSB(highp  uint);"
            "lowp ivec2 findMSB(highp uvec2);"
            "lowp ivec3 findMSB(highp uvec3);"
            "lowp ivec4 findMSB(highp uvec4);"

            "\n");
    }

    // GL_ARB_shader_ballot
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "uint64_t ballotARB(bool);"

            "float readIlwocationARB(float, uint);"
            "vec2  readIlwocationARB(vec2,  uint);"
            "vec3  readIlwocationARB(vec3,  uint);"
            "vec4  readIlwocationARB(vec4,  uint);"

            "int   readIlwocationARB(int,   uint);"
            "ivec2 readIlwocationARB(ivec2, uint);"
            "ivec3 readIlwocationARB(ivec3, uint);"
            "ivec4 readIlwocationARB(ivec4, uint);"

            "uint  readIlwocationARB(uint,  uint);"
            "uvec2 readIlwocationARB(uvec2, uint);"
            "uvec3 readIlwocationARB(uvec3, uint);"
            "uvec4 readIlwocationARB(uvec4, uint);"

            "float readFirstIlwocationARB(float);"
            "vec2  readFirstIlwocationARB(vec2);"
            "vec3  readFirstIlwocationARB(vec3);"
            "vec4  readFirstIlwocationARB(vec4);"

            "int   readFirstIlwocationARB(int);"
            "ivec2 readFirstIlwocationARB(ivec2);"
            "ivec3 readFirstIlwocationARB(ivec3);"
            "ivec4 readFirstIlwocationARB(ivec4);"

            "uint  readFirstIlwocationARB(uint);"
            "uvec2 readFirstIlwocationARB(uvec2);"
            "uvec3 readFirstIlwocationARB(uvec3);"
            "uvec4 readFirstIlwocationARB(uvec4);"

            "\n");
    }

    // GL_ARB_shader_group_vote
    if (profile != EEsProfile && version >= 430) {
        commonBuiltins.append(
            "bool anyIlwocationARB(bool);"
            "bool allIlwocationsARB(bool);"
            "bool allIlwocationsEqualARB(bool);"

            "\n");
    }

    // GL_KHR_shader_subgroup
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 140)) {
        commonBuiltins.append(
            "void subgroupBarrier();"
            "void subgroupMemoryBarrier();"
            "void subgroupMemoryBarrierBuffer();"
            "void subgroupMemoryBarrierImage();"
            "bool subgroupElect();"

            "bool   subgroupAll(bool);\n"
            "bool   subgroupAny(bool);\n"
            "uvec4  subgroupBallot(bool);\n"
            "bool   subgroupIlwerseBallot(uvec4);\n"
            "bool   subgroupBallotBitExtract(uvec4, uint);\n"
            "uint   subgroupBallotBitCount(uvec4);\n"
            "uint   subgroupBallotInclusiveBitCount(uvec4);\n"
            "uint   subgroupBallotExclusiveBitCount(uvec4);\n"
            "uint   subgroupBallotFindLSB(uvec4);\n"
            "uint   subgroupBallotFindMSB(uvec4);\n"
            );

        // Generate all flavors of subgroup ops.
        static const char *subgroupOps[] = 
        {
            "bool   subgroupAllEqual(%s);\n",
            "%s     subgroupBroadcast(%s, uint);\n",
            "%s     subgroupBroadcastFirst(%s);\n",
            "%s     subgroupShuffle(%s, uint);\n",
            "%s     subgroupShuffleXor(%s, uint);\n",
            "%s     subgroupShuffleUp(%s, uint delta);\n",
            "%s     subgroupShuffleDown(%s, uint delta);\n",
            "%s     subgroupAdd(%s);\n",
            "%s     subgroupMul(%s);\n",
            "%s     subgroupMin(%s);\n",
            "%s     subgroupMax(%s);\n",
            "%s     subgroupAnd(%s);\n",
            "%s     subgroupOr(%s);\n",
            "%s     subgroupXor(%s);\n",
            "%s     subgroupInclusiveAdd(%s);\n",
            "%s     subgroupInclusiveMul(%s);\n",
            "%s     subgroupInclusiveMin(%s);\n",
            "%s     subgroupInclusiveMax(%s);\n",
            "%s     subgroupInclusiveAnd(%s);\n",
            "%s     subgroupInclusiveOr(%s);\n",
            "%s     subgroupInclusiveXor(%s);\n",
            "%s     subgroupExclusiveAdd(%s);\n",
            "%s     subgroupExclusiveMul(%s);\n",
            "%s     subgroupExclusiveMin(%s);\n",
            "%s     subgroupExclusiveMax(%s);\n",
            "%s     subgroupExclusiveAnd(%s);\n",
            "%s     subgroupExclusiveOr(%s);\n",
            "%s     subgroupExclusiveXor(%s);\n",
            "%s     subgroupClusteredAdd(%s, uint);\n",
            "%s     subgroupClusteredMul(%s, uint);\n",
            "%s     subgroupClusteredMin(%s, uint);\n",
            "%s     subgroupClusteredMax(%s, uint);\n",
            "%s     subgroupClusteredAnd(%s, uint);\n",
            "%s     subgroupClusteredOr(%s, uint);\n",
            "%s     subgroupClusteredXor(%s, uint);\n",
            "%s     subgroupQuadBroadcast(%s, uint);\n",
            "%s     subgroupQuadSwapHorizontal(%s);\n",
            "%s     subgroupQuadSwapVertical(%s);\n",
            "%s     subgroupQuadSwapDiagonal(%s);\n",
            "uvec4  subgroupPartitionLW(%s);\n",
            "%s     subgroupPartitionedAddLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedMulLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedMinLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedMaxLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedAndLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedOrLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedXorLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveAddLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveMulLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveMinLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveMaxLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveAndLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveOrLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedInclusiveXorLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveAddLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveMulLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveMinLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveMaxLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveAndLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveOrLW(%s, uvec4 ballot);\n",
            "%s     subgroupPartitionedExclusiveXorLW(%s, uvec4 ballot);\n",
        };

        static const char *floatTypes[] = { 
            "float", "vec2", "vec3", "vec4", 
            "float16_t", "f16vec2", "f16vec3", "f16vec4", 
        };
        static const char *doubleTypes[] = { 
            "double", "dvec2", "dvec3", "dvec4", 
        };
        static const char *intTypes[] = { 
            "int8_t", "i8vec2", "i8vec3", "i8vec4", 
            "int16_t", "i16vec2", "i16vec3", "i16vec4", 
            "int", "ivec2", "ivec3", "ivec4", 
            "int64_t", "i64vec2", "i64vec3", "i64vec4", 
            "uint8_t", "u8vec2", "u8vec3", "u8vec4", 
            "uint16_t", "u16vec2", "u16vec3", "u16vec4", 
            "uint", "uvec2", "uvec3", "uvec4", 
            "uint64_t", "u64vec2", "u64vec3", "u64vec4", 
        };
        static const char *boolTypes[] = { 
            "bool", "bvec2", "bvec3", "bvec4", 
        };

        for (size_t i = 0; i < sizeof(subgroupOps)/sizeof(subgroupOps[0]); ++i) {
            const char *op = subgroupOps[i];

            // Logical operations don't support float
            bool logicalOp = strstr(op, "Or") || strstr(op, "And") ||
                             (strstr(op, "Xor") && !strstr(op, "ShuffleXor"));
            // Math operations don't support bool
            bool mathOp = strstr(op, "Add") || strstr(op, "Mul") || strstr(op, "Min") || strstr(op, "Max");

            const int bufSize = 256;
            char buf[bufSize];

            if (!logicalOp) {
                for (size_t j = 0; j < sizeof(floatTypes)/sizeof(floatTypes[0]); ++j) {
                    snprintf(buf, bufSize, op, floatTypes[j], floatTypes[j]);
                    commonBuiltins.append(buf);
                }
                if (profile != EEsProfile && version >= 400) {
                    for (size_t j = 0; j < sizeof(doubleTypes)/sizeof(doubleTypes[0]); ++j) {
                        snprintf(buf, bufSize, op, doubleTypes[j], doubleTypes[j]);
                        commonBuiltins.append(buf);
                    }
                }
            }
            if (!mathOp) {
                for (size_t j = 0; j < sizeof(boolTypes)/sizeof(boolTypes[0]); ++j) {
                    snprintf(buf, bufSize, op, boolTypes[j], boolTypes[j]);
                    commonBuiltins.append(buf);
                }
            }
            for (size_t j = 0; j < sizeof(intTypes)/sizeof(intTypes[0]); ++j) {
                snprintf(buf, bufSize, op, intTypes[j], intTypes[j]);
                commonBuiltins.append(buf);
            }
        }

        stageBuiltins[EShLangCompute].append(
            "void subgroupMemoryBarrierShared();"

            "\n"
            );
        stageBuiltins[EShLangMeshLW].append(
            "void subgroupMemoryBarrierShared();"
            "\n"
            );
        stageBuiltins[EShLangTaskLW].append(
            "void subgroupMemoryBarrierShared();"
            "\n"
            );
    }

    if (profile != EEsProfile && version >= 460) {
        commonBuiltins.append(
            "bool anyIlwocation(bool);"
            "bool allIlwocations(bool);"
            "bool allIlwocationsEqual(bool);"

            "\n");
    }

    // GL_AMD_shader_ballot
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "float minIlwocationsAMD(float);"
            "vec2  minIlwocationsAMD(vec2);"
            "vec3  minIlwocationsAMD(vec3);"
            "vec4  minIlwocationsAMD(vec4);"

            "int   minIlwocationsAMD(int);"
            "ivec2 minIlwocationsAMD(ivec2);"
            "ivec3 minIlwocationsAMD(ivec3);"
            "ivec4 minIlwocationsAMD(ivec4);"

            "uint  minIlwocationsAMD(uint);"
            "uvec2 minIlwocationsAMD(uvec2);"
            "uvec3 minIlwocationsAMD(uvec3);"
            "uvec4 minIlwocationsAMD(uvec4);"

            "double minIlwocationsAMD(double);"
            "dvec2  minIlwocationsAMD(dvec2);"
            "dvec3  minIlwocationsAMD(dvec3);"
            "dvec4  minIlwocationsAMD(dvec4);"

            "int64_t minIlwocationsAMD(int64_t);"
            "i64vec2 minIlwocationsAMD(i64vec2);"
            "i64vec3 minIlwocationsAMD(i64vec3);"
            "i64vec4 minIlwocationsAMD(i64vec4);"

            "uint64_t minIlwocationsAMD(uint64_t);"
            "u64vec2  minIlwocationsAMD(u64vec2);"
            "u64vec3  minIlwocationsAMD(u64vec3);"
            "u64vec4  minIlwocationsAMD(u64vec4);"

            "float16_t minIlwocationsAMD(float16_t);"
            "f16vec2   minIlwocationsAMD(f16vec2);"
            "f16vec3   minIlwocationsAMD(f16vec3);"
            "f16vec4   minIlwocationsAMD(f16vec4);"

            "int16_t minIlwocationsAMD(int16_t);"
            "i16vec2 minIlwocationsAMD(i16vec2);"
            "i16vec3 minIlwocationsAMD(i16vec3);"
            "i16vec4 minIlwocationsAMD(i16vec4);"

            "uint16_t minIlwocationsAMD(uint16_t);"
            "u16vec2  minIlwocationsAMD(u16vec2);"
            "u16vec3  minIlwocationsAMD(u16vec3);"
            "u16vec4  minIlwocationsAMD(u16vec4);"

            "float minIlwocationsInclusiveScanAMD(float);"
            "vec2  minIlwocationsInclusiveScanAMD(vec2);"
            "vec3  minIlwocationsInclusiveScanAMD(vec3);"
            "vec4  minIlwocationsInclusiveScanAMD(vec4);"

            "int   minIlwocationsInclusiveScanAMD(int);"
            "ivec2 minIlwocationsInclusiveScanAMD(ivec2);"
            "ivec3 minIlwocationsInclusiveScanAMD(ivec3);"
            "ivec4 minIlwocationsInclusiveScanAMD(ivec4);"

            "uint  minIlwocationsInclusiveScanAMD(uint);"
            "uvec2 minIlwocationsInclusiveScanAMD(uvec2);"
            "uvec3 minIlwocationsInclusiveScanAMD(uvec3);"
            "uvec4 minIlwocationsInclusiveScanAMD(uvec4);"

            "double minIlwocationsInclusiveScanAMD(double);"
            "dvec2  minIlwocationsInclusiveScanAMD(dvec2);"
            "dvec3  minIlwocationsInclusiveScanAMD(dvec3);"
            "dvec4  minIlwocationsInclusiveScanAMD(dvec4);"

            "int64_t minIlwocationsInclusiveScanAMD(int64_t);"
            "i64vec2 minIlwocationsInclusiveScanAMD(i64vec2);"
            "i64vec3 minIlwocationsInclusiveScanAMD(i64vec3);"
            "i64vec4 minIlwocationsInclusiveScanAMD(i64vec4);"

            "uint64_t minIlwocationsInclusiveScanAMD(uint64_t);"
            "u64vec2  minIlwocationsInclusiveScanAMD(u64vec2);"
            "u64vec3  minIlwocationsInclusiveScanAMD(u64vec3);"
            "u64vec4  minIlwocationsInclusiveScanAMD(u64vec4);"

            "float16_t minIlwocationsInclusiveScanAMD(float16_t);"
            "f16vec2   minIlwocationsInclusiveScanAMD(f16vec2);"
            "f16vec3   minIlwocationsInclusiveScanAMD(f16vec3);"
            "f16vec4   minIlwocationsInclusiveScanAMD(f16vec4);"

            "int16_t minIlwocationsInclusiveScanAMD(int16_t);"
            "i16vec2 minIlwocationsInclusiveScanAMD(i16vec2);"
            "i16vec3 minIlwocationsInclusiveScanAMD(i16vec3);"
            "i16vec4 minIlwocationsInclusiveScanAMD(i16vec4);"

            "uint16_t minIlwocationsInclusiveScanAMD(uint16_t);"
            "u16vec2  minIlwocationsInclusiveScanAMD(u16vec2);"
            "u16vec3  minIlwocationsInclusiveScanAMD(u16vec3);"
            "u16vec4  minIlwocationsInclusiveScanAMD(u16vec4);"

            "float minIlwocationsExclusiveScanAMD(float);"
            "vec2  minIlwocationsExclusiveScanAMD(vec2);"
            "vec3  minIlwocationsExclusiveScanAMD(vec3);"
            "vec4  minIlwocationsExclusiveScanAMD(vec4);"

            "int   minIlwocationsExclusiveScanAMD(int);"
            "ivec2 minIlwocationsExclusiveScanAMD(ivec2);"
            "ivec3 minIlwocationsExclusiveScanAMD(ivec3);"
            "ivec4 minIlwocationsExclusiveScanAMD(ivec4);"

            "uint  minIlwocationsExclusiveScanAMD(uint);"
            "uvec2 minIlwocationsExclusiveScanAMD(uvec2);"
            "uvec3 minIlwocationsExclusiveScanAMD(uvec3);"
            "uvec4 minIlwocationsExclusiveScanAMD(uvec4);"

            "double minIlwocationsExclusiveScanAMD(double);"
            "dvec2  minIlwocationsExclusiveScanAMD(dvec2);"
            "dvec3  minIlwocationsExclusiveScanAMD(dvec3);"
            "dvec4  minIlwocationsExclusiveScanAMD(dvec4);"

            "int64_t minIlwocationsExclusiveScanAMD(int64_t);"
            "i64vec2 minIlwocationsExclusiveScanAMD(i64vec2);"
            "i64vec3 minIlwocationsExclusiveScanAMD(i64vec3);"
            "i64vec4 minIlwocationsExclusiveScanAMD(i64vec4);"

            "uint64_t minIlwocationsExclusiveScanAMD(uint64_t);"
            "u64vec2  minIlwocationsExclusiveScanAMD(u64vec2);"
            "u64vec3  minIlwocationsExclusiveScanAMD(u64vec3);"
            "u64vec4  minIlwocationsExclusiveScanAMD(u64vec4);"

            "float16_t minIlwocationsExclusiveScanAMD(float16_t);"
            "f16vec2   minIlwocationsExclusiveScanAMD(f16vec2);"
            "f16vec3   minIlwocationsExclusiveScanAMD(f16vec3);"
            "f16vec4   minIlwocationsExclusiveScanAMD(f16vec4);"

            "int16_t minIlwocationsExclusiveScanAMD(int16_t);"
            "i16vec2 minIlwocationsExclusiveScanAMD(i16vec2);"
            "i16vec3 minIlwocationsExclusiveScanAMD(i16vec3);"
            "i16vec4 minIlwocationsExclusiveScanAMD(i16vec4);"

            "uint16_t minIlwocationsExclusiveScanAMD(uint16_t);"
            "u16vec2  minIlwocationsExclusiveScanAMD(u16vec2);"
            "u16vec3  minIlwocationsExclusiveScanAMD(u16vec3);"
            "u16vec4  minIlwocationsExclusiveScanAMD(u16vec4);"

            "float maxIlwocationsAMD(float);"
            "vec2  maxIlwocationsAMD(vec2);"
            "vec3  maxIlwocationsAMD(vec3);"
            "vec4  maxIlwocationsAMD(vec4);"

            "int   maxIlwocationsAMD(int);"
            "ivec2 maxIlwocationsAMD(ivec2);"
            "ivec3 maxIlwocationsAMD(ivec3);"
            "ivec4 maxIlwocationsAMD(ivec4);"

            "uint  maxIlwocationsAMD(uint);"
            "uvec2 maxIlwocationsAMD(uvec2);"
            "uvec3 maxIlwocationsAMD(uvec3);"
            "uvec4 maxIlwocationsAMD(uvec4);"

            "double maxIlwocationsAMD(double);"
            "dvec2  maxIlwocationsAMD(dvec2);"
            "dvec3  maxIlwocationsAMD(dvec3);"
            "dvec4  maxIlwocationsAMD(dvec4);"

            "int64_t maxIlwocationsAMD(int64_t);"
            "i64vec2 maxIlwocationsAMD(i64vec2);"
            "i64vec3 maxIlwocationsAMD(i64vec3);"
            "i64vec4 maxIlwocationsAMD(i64vec4);"

            "uint64_t maxIlwocationsAMD(uint64_t);"
            "u64vec2  maxIlwocationsAMD(u64vec2);"
            "u64vec3  maxIlwocationsAMD(u64vec3);"
            "u64vec4  maxIlwocationsAMD(u64vec4);"

            "float16_t maxIlwocationsAMD(float16_t);"
            "f16vec2   maxIlwocationsAMD(f16vec2);"
            "f16vec3   maxIlwocationsAMD(f16vec3);"
            "f16vec4   maxIlwocationsAMD(f16vec4);"

            "int16_t maxIlwocationsAMD(int16_t);"
            "i16vec2 maxIlwocationsAMD(i16vec2);"
            "i16vec3 maxIlwocationsAMD(i16vec3);"
            "i16vec4 maxIlwocationsAMD(i16vec4);"

            "uint16_t maxIlwocationsAMD(uint16_t);"
            "u16vec2  maxIlwocationsAMD(u16vec2);"
            "u16vec3  maxIlwocationsAMD(u16vec3);"
            "u16vec4  maxIlwocationsAMD(u16vec4);"

            "float maxIlwocationsInclusiveScanAMD(float);"
            "vec2  maxIlwocationsInclusiveScanAMD(vec2);"
            "vec3  maxIlwocationsInclusiveScanAMD(vec3);"
            "vec4  maxIlwocationsInclusiveScanAMD(vec4);"

            "int   maxIlwocationsInclusiveScanAMD(int);"
            "ivec2 maxIlwocationsInclusiveScanAMD(ivec2);"
            "ivec3 maxIlwocationsInclusiveScanAMD(ivec3);"
            "ivec4 maxIlwocationsInclusiveScanAMD(ivec4);"

            "uint  maxIlwocationsInclusiveScanAMD(uint);"
            "uvec2 maxIlwocationsInclusiveScanAMD(uvec2);"
            "uvec3 maxIlwocationsInclusiveScanAMD(uvec3);"
            "uvec4 maxIlwocationsInclusiveScanAMD(uvec4);"

            "double maxIlwocationsInclusiveScanAMD(double);"
            "dvec2  maxIlwocationsInclusiveScanAMD(dvec2);"
            "dvec3  maxIlwocationsInclusiveScanAMD(dvec3);"
            "dvec4  maxIlwocationsInclusiveScanAMD(dvec4);"

            "int64_t maxIlwocationsInclusiveScanAMD(int64_t);"
            "i64vec2 maxIlwocationsInclusiveScanAMD(i64vec2);"
            "i64vec3 maxIlwocationsInclusiveScanAMD(i64vec3);"
            "i64vec4 maxIlwocationsInclusiveScanAMD(i64vec4);"

            "uint64_t maxIlwocationsInclusiveScanAMD(uint64_t);"
            "u64vec2  maxIlwocationsInclusiveScanAMD(u64vec2);"
            "u64vec3  maxIlwocationsInclusiveScanAMD(u64vec3);"
            "u64vec4  maxIlwocationsInclusiveScanAMD(u64vec4);"

            "float16_t maxIlwocationsInclusiveScanAMD(float16_t);"
            "f16vec2   maxIlwocationsInclusiveScanAMD(f16vec2);"
            "f16vec3   maxIlwocationsInclusiveScanAMD(f16vec3);"
            "f16vec4   maxIlwocationsInclusiveScanAMD(f16vec4);"

            "int16_t maxIlwocationsInclusiveScanAMD(int16_t);"
            "i16vec2 maxIlwocationsInclusiveScanAMD(i16vec2);"
            "i16vec3 maxIlwocationsInclusiveScanAMD(i16vec3);"
            "i16vec4 maxIlwocationsInclusiveScanAMD(i16vec4);"

            "uint16_t maxIlwocationsInclusiveScanAMD(uint16_t);"
            "u16vec2  maxIlwocationsInclusiveScanAMD(u16vec2);"
            "u16vec3  maxIlwocationsInclusiveScanAMD(u16vec3);"
            "u16vec4  maxIlwocationsInclusiveScanAMD(u16vec4);"

            "float maxIlwocationsExclusiveScanAMD(float);"
            "vec2  maxIlwocationsExclusiveScanAMD(vec2);"
            "vec3  maxIlwocationsExclusiveScanAMD(vec3);"
            "vec4  maxIlwocationsExclusiveScanAMD(vec4);"

            "int   maxIlwocationsExclusiveScanAMD(int);"
            "ivec2 maxIlwocationsExclusiveScanAMD(ivec2);"
            "ivec3 maxIlwocationsExclusiveScanAMD(ivec3);"
            "ivec4 maxIlwocationsExclusiveScanAMD(ivec4);"

            "uint  maxIlwocationsExclusiveScanAMD(uint);"
            "uvec2 maxIlwocationsExclusiveScanAMD(uvec2);"
            "uvec3 maxIlwocationsExclusiveScanAMD(uvec3);"
            "uvec4 maxIlwocationsExclusiveScanAMD(uvec4);"

            "double maxIlwocationsExclusiveScanAMD(double);"
            "dvec2  maxIlwocationsExclusiveScanAMD(dvec2);"
            "dvec3  maxIlwocationsExclusiveScanAMD(dvec3);"
            "dvec4  maxIlwocationsExclusiveScanAMD(dvec4);"

            "int64_t maxIlwocationsExclusiveScanAMD(int64_t);"
            "i64vec2 maxIlwocationsExclusiveScanAMD(i64vec2);"
            "i64vec3 maxIlwocationsExclusiveScanAMD(i64vec3);"
            "i64vec4 maxIlwocationsExclusiveScanAMD(i64vec4);"

            "uint64_t maxIlwocationsExclusiveScanAMD(uint64_t);"
            "u64vec2  maxIlwocationsExclusiveScanAMD(u64vec2);"
            "u64vec3  maxIlwocationsExclusiveScanAMD(u64vec3);"
            "u64vec4  maxIlwocationsExclusiveScanAMD(u64vec4);"

            "float16_t maxIlwocationsExclusiveScanAMD(float16_t);"
            "f16vec2   maxIlwocationsExclusiveScanAMD(f16vec2);"
            "f16vec3   maxIlwocationsExclusiveScanAMD(f16vec3);"
            "f16vec4   maxIlwocationsExclusiveScanAMD(f16vec4);"

            "int16_t maxIlwocationsExclusiveScanAMD(int16_t);"
            "i16vec2 maxIlwocationsExclusiveScanAMD(i16vec2);"
            "i16vec3 maxIlwocationsExclusiveScanAMD(i16vec3);"
            "i16vec4 maxIlwocationsExclusiveScanAMD(i16vec4);"

            "uint16_t maxIlwocationsExclusiveScanAMD(uint16_t);"
            "u16vec2  maxIlwocationsExclusiveScanAMD(u16vec2);"
            "u16vec3  maxIlwocationsExclusiveScanAMD(u16vec3);"
            "u16vec4  maxIlwocationsExclusiveScanAMD(u16vec4);"

            "float addIlwocationsAMD(float);"
            "vec2  addIlwocationsAMD(vec2);"
            "vec3  addIlwocationsAMD(vec3);"
            "vec4  addIlwocationsAMD(vec4);"

            "int   addIlwocationsAMD(int);"
            "ivec2 addIlwocationsAMD(ivec2);"
            "ivec3 addIlwocationsAMD(ivec3);"
            "ivec4 addIlwocationsAMD(ivec4);"

            "uint  addIlwocationsAMD(uint);"
            "uvec2 addIlwocationsAMD(uvec2);"
            "uvec3 addIlwocationsAMD(uvec3);"
            "uvec4 addIlwocationsAMD(uvec4);"

            "double  addIlwocationsAMD(double);"
            "dvec2   addIlwocationsAMD(dvec2);"
            "dvec3   addIlwocationsAMD(dvec3);"
            "dvec4   addIlwocationsAMD(dvec4);"

            "int64_t addIlwocationsAMD(int64_t);"
            "i64vec2 addIlwocationsAMD(i64vec2);"
            "i64vec3 addIlwocationsAMD(i64vec3);"
            "i64vec4 addIlwocationsAMD(i64vec4);"

            "uint64_t addIlwocationsAMD(uint64_t);"
            "u64vec2  addIlwocationsAMD(u64vec2);"
            "u64vec3  addIlwocationsAMD(u64vec3);"
            "u64vec4  addIlwocationsAMD(u64vec4);"

            "float16_t addIlwocationsAMD(float16_t);"
            "f16vec2   addIlwocationsAMD(f16vec2);"
            "f16vec3   addIlwocationsAMD(f16vec3);"
            "f16vec4   addIlwocationsAMD(f16vec4);"

            "int16_t addIlwocationsAMD(int16_t);"
            "i16vec2 addIlwocationsAMD(i16vec2);"
            "i16vec3 addIlwocationsAMD(i16vec3);"
            "i16vec4 addIlwocationsAMD(i16vec4);"

            "uint16_t addIlwocationsAMD(uint16_t);"
            "u16vec2  addIlwocationsAMD(u16vec2);"
            "u16vec3  addIlwocationsAMD(u16vec3);"
            "u16vec4  addIlwocationsAMD(u16vec4);"

            "float addIlwocationsInclusiveScanAMD(float);"
            "vec2  addIlwocationsInclusiveScanAMD(vec2);"
            "vec3  addIlwocationsInclusiveScanAMD(vec3);"
            "vec4  addIlwocationsInclusiveScanAMD(vec4);"

            "int   addIlwocationsInclusiveScanAMD(int);"
            "ivec2 addIlwocationsInclusiveScanAMD(ivec2);"
            "ivec3 addIlwocationsInclusiveScanAMD(ivec3);"
            "ivec4 addIlwocationsInclusiveScanAMD(ivec4);"

            "uint  addIlwocationsInclusiveScanAMD(uint);"
            "uvec2 addIlwocationsInclusiveScanAMD(uvec2);"
            "uvec3 addIlwocationsInclusiveScanAMD(uvec3);"
            "uvec4 addIlwocationsInclusiveScanAMD(uvec4);"

            "double  addIlwocationsInclusiveScanAMD(double);"
            "dvec2   addIlwocationsInclusiveScanAMD(dvec2);"
            "dvec3   addIlwocationsInclusiveScanAMD(dvec3);"
            "dvec4   addIlwocationsInclusiveScanAMD(dvec4);"

            "int64_t addIlwocationsInclusiveScanAMD(int64_t);"
            "i64vec2 addIlwocationsInclusiveScanAMD(i64vec2);"
            "i64vec3 addIlwocationsInclusiveScanAMD(i64vec3);"
            "i64vec4 addIlwocationsInclusiveScanAMD(i64vec4);"

            "uint64_t addIlwocationsInclusiveScanAMD(uint64_t);"
            "u64vec2  addIlwocationsInclusiveScanAMD(u64vec2);"
            "u64vec3  addIlwocationsInclusiveScanAMD(u64vec3);"
            "u64vec4  addIlwocationsInclusiveScanAMD(u64vec4);"

            "float16_t addIlwocationsInclusiveScanAMD(float16_t);"
            "f16vec2   addIlwocationsInclusiveScanAMD(f16vec2);"
            "f16vec3   addIlwocationsInclusiveScanAMD(f16vec3);"
            "f16vec4   addIlwocationsInclusiveScanAMD(f16vec4);"

            "int16_t addIlwocationsInclusiveScanAMD(int16_t);"
            "i16vec2 addIlwocationsInclusiveScanAMD(i16vec2);"
            "i16vec3 addIlwocationsInclusiveScanAMD(i16vec3);"
            "i16vec4 addIlwocationsInclusiveScanAMD(i16vec4);"

            "uint16_t addIlwocationsInclusiveScanAMD(uint16_t);"
            "u16vec2  addIlwocationsInclusiveScanAMD(u16vec2);"
            "u16vec3  addIlwocationsInclusiveScanAMD(u16vec3);"
            "u16vec4  addIlwocationsInclusiveScanAMD(u16vec4);"

            "float addIlwocationsExclusiveScanAMD(float);"
            "vec2  addIlwocationsExclusiveScanAMD(vec2);"
            "vec3  addIlwocationsExclusiveScanAMD(vec3);"
            "vec4  addIlwocationsExclusiveScanAMD(vec4);"

            "int   addIlwocationsExclusiveScanAMD(int);"
            "ivec2 addIlwocationsExclusiveScanAMD(ivec2);"
            "ivec3 addIlwocationsExclusiveScanAMD(ivec3);"
            "ivec4 addIlwocationsExclusiveScanAMD(ivec4);"

            "uint  addIlwocationsExclusiveScanAMD(uint);"
            "uvec2 addIlwocationsExclusiveScanAMD(uvec2);"
            "uvec3 addIlwocationsExclusiveScanAMD(uvec3);"
            "uvec4 addIlwocationsExclusiveScanAMD(uvec4);"

            "double  addIlwocationsExclusiveScanAMD(double);"
            "dvec2   addIlwocationsExclusiveScanAMD(dvec2);"
            "dvec3   addIlwocationsExclusiveScanAMD(dvec3);"
            "dvec4   addIlwocationsExclusiveScanAMD(dvec4);"

            "int64_t addIlwocationsExclusiveScanAMD(int64_t);"
            "i64vec2 addIlwocationsExclusiveScanAMD(i64vec2);"
            "i64vec3 addIlwocationsExclusiveScanAMD(i64vec3);"
            "i64vec4 addIlwocationsExclusiveScanAMD(i64vec4);"

            "uint64_t addIlwocationsExclusiveScanAMD(uint64_t);"
            "u64vec2  addIlwocationsExclusiveScanAMD(u64vec2);"
            "u64vec3  addIlwocationsExclusiveScanAMD(u64vec3);"
            "u64vec4  addIlwocationsExclusiveScanAMD(u64vec4);"

            "float16_t addIlwocationsExclusiveScanAMD(float16_t);"
            "f16vec2   addIlwocationsExclusiveScanAMD(f16vec2);"
            "f16vec3   addIlwocationsExclusiveScanAMD(f16vec3);"
            "f16vec4   addIlwocationsExclusiveScanAMD(f16vec4);"

            "int16_t addIlwocationsExclusiveScanAMD(int16_t);"
            "i16vec2 addIlwocationsExclusiveScanAMD(i16vec2);"
            "i16vec3 addIlwocationsExclusiveScanAMD(i16vec3);"
            "i16vec4 addIlwocationsExclusiveScanAMD(i16vec4);"

            "uint16_t addIlwocationsExclusiveScanAMD(uint16_t);"
            "u16vec2  addIlwocationsExclusiveScanAMD(u16vec2);"
            "u16vec3  addIlwocationsExclusiveScanAMD(u16vec3);"
            "u16vec4  addIlwocationsExclusiveScanAMD(u16vec4);"

            "float minIlwocationsNonUniformAMD(float);"
            "vec2  minIlwocationsNonUniformAMD(vec2);"
            "vec3  minIlwocationsNonUniformAMD(vec3);"
            "vec4  minIlwocationsNonUniformAMD(vec4);"

            "int   minIlwocationsNonUniformAMD(int);"
            "ivec2 minIlwocationsNonUniformAMD(ivec2);"
            "ivec3 minIlwocationsNonUniformAMD(ivec3);"
            "ivec4 minIlwocationsNonUniformAMD(ivec4);"

            "uint  minIlwocationsNonUniformAMD(uint);"
            "uvec2 minIlwocationsNonUniformAMD(uvec2);"
            "uvec3 minIlwocationsNonUniformAMD(uvec3);"
            "uvec4 minIlwocationsNonUniformAMD(uvec4);"

            "double minIlwocationsNonUniformAMD(double);"
            "dvec2  minIlwocationsNonUniformAMD(dvec2);"
            "dvec3  minIlwocationsNonUniformAMD(dvec3);"
            "dvec4  minIlwocationsNonUniformAMD(dvec4);"

            "int64_t minIlwocationsNonUniformAMD(int64_t);"
            "i64vec2 minIlwocationsNonUniformAMD(i64vec2);"
            "i64vec3 minIlwocationsNonUniformAMD(i64vec3);"
            "i64vec4 minIlwocationsNonUniformAMD(i64vec4);"

            "uint64_t minIlwocationsNonUniformAMD(uint64_t);"
            "u64vec2  minIlwocationsNonUniformAMD(u64vec2);"
            "u64vec3  minIlwocationsNonUniformAMD(u64vec3);"
            "u64vec4  minIlwocationsNonUniformAMD(u64vec4);"

            "float16_t minIlwocationsNonUniformAMD(float16_t);"
            "f16vec2   minIlwocationsNonUniformAMD(f16vec2);"
            "f16vec3   minIlwocationsNonUniformAMD(f16vec3);"
            "f16vec4   minIlwocationsNonUniformAMD(f16vec4);"

            "int16_t minIlwocationsNonUniformAMD(int16_t);"
            "i16vec2 minIlwocationsNonUniformAMD(i16vec2);"
            "i16vec3 minIlwocationsNonUniformAMD(i16vec3);"
            "i16vec4 minIlwocationsNonUniformAMD(i16vec4);"

            "uint16_t minIlwocationsNonUniformAMD(uint16_t);"
            "u16vec2  minIlwocationsNonUniformAMD(u16vec2);"
            "u16vec3  minIlwocationsNonUniformAMD(u16vec3);"
            "u16vec4  minIlwocationsNonUniformAMD(u16vec4);"

            "float minIlwocationsInclusiveScanNonUniformAMD(float);"
            "vec2  minIlwocationsInclusiveScanNonUniformAMD(vec2);"
            "vec3  minIlwocationsInclusiveScanNonUniformAMD(vec3);"
            "vec4  minIlwocationsInclusiveScanNonUniformAMD(vec4);"

            "int   minIlwocationsInclusiveScanNonUniformAMD(int);"
            "ivec2 minIlwocationsInclusiveScanNonUniformAMD(ivec2);"
            "ivec3 minIlwocationsInclusiveScanNonUniformAMD(ivec3);"
            "ivec4 minIlwocationsInclusiveScanNonUniformAMD(ivec4);"

            "uint  minIlwocationsInclusiveScanNonUniformAMD(uint);"
            "uvec2 minIlwocationsInclusiveScanNonUniformAMD(uvec2);"
            "uvec3 minIlwocationsInclusiveScanNonUniformAMD(uvec3);"
            "uvec4 minIlwocationsInclusiveScanNonUniformAMD(uvec4);"

            "double minIlwocationsInclusiveScanNonUniformAMD(double);"
            "dvec2  minIlwocationsInclusiveScanNonUniformAMD(dvec2);"
            "dvec3  minIlwocationsInclusiveScanNonUniformAMD(dvec3);"
            "dvec4  minIlwocationsInclusiveScanNonUniformAMD(dvec4);"

            "int64_t minIlwocationsInclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 minIlwocationsInclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 minIlwocationsInclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 minIlwocationsInclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t minIlwocationsInclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  minIlwocationsInclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  minIlwocationsInclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  minIlwocationsInclusiveScanNonUniformAMD(u64vec4);"

            "float16_t minIlwocationsInclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   minIlwocationsInclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   minIlwocationsInclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   minIlwocationsInclusiveScanNonUniformAMD(f16vec4);"

            "int16_t minIlwocationsInclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 minIlwocationsInclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 minIlwocationsInclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 minIlwocationsInclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t minIlwocationsInclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  minIlwocationsInclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  minIlwocationsInclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  minIlwocationsInclusiveScanNonUniformAMD(u16vec4);"

            "float minIlwocationsExclusiveScanNonUniformAMD(float);"
            "vec2  minIlwocationsExclusiveScanNonUniformAMD(vec2);"
            "vec3  minIlwocationsExclusiveScanNonUniformAMD(vec3);"
            "vec4  minIlwocationsExclusiveScanNonUniformAMD(vec4);"

            "int   minIlwocationsExclusiveScanNonUniformAMD(int);"
            "ivec2 minIlwocationsExclusiveScanNonUniformAMD(ivec2);"
            "ivec3 minIlwocationsExclusiveScanNonUniformAMD(ivec3);"
            "ivec4 minIlwocationsExclusiveScanNonUniformAMD(ivec4);"

            "uint  minIlwocationsExclusiveScanNonUniformAMD(uint);"
            "uvec2 minIlwocationsExclusiveScanNonUniformAMD(uvec2);"
            "uvec3 minIlwocationsExclusiveScanNonUniformAMD(uvec3);"
            "uvec4 minIlwocationsExclusiveScanNonUniformAMD(uvec4);"

            "double minIlwocationsExclusiveScanNonUniformAMD(double);"
            "dvec2  minIlwocationsExclusiveScanNonUniformAMD(dvec2);"
            "dvec3  minIlwocationsExclusiveScanNonUniformAMD(dvec3);"
            "dvec4  minIlwocationsExclusiveScanNonUniformAMD(dvec4);"

            "int64_t minIlwocationsExclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 minIlwocationsExclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 minIlwocationsExclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 minIlwocationsExclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t minIlwocationsExclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  minIlwocationsExclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  minIlwocationsExclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  minIlwocationsExclusiveScanNonUniformAMD(u64vec4);"

            "float16_t minIlwocationsExclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   minIlwocationsExclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   minIlwocationsExclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   minIlwocationsExclusiveScanNonUniformAMD(f16vec4);"

            "int16_t minIlwocationsExclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 minIlwocationsExclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 minIlwocationsExclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 minIlwocationsExclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t minIlwocationsExclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  minIlwocationsExclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  minIlwocationsExclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  minIlwocationsExclusiveScanNonUniformAMD(u16vec4);"

            "float maxIlwocationsNonUniformAMD(float);"
            "vec2  maxIlwocationsNonUniformAMD(vec2);"
            "vec3  maxIlwocationsNonUniformAMD(vec3);"
            "vec4  maxIlwocationsNonUniformAMD(vec4);"

            "int   maxIlwocationsNonUniformAMD(int);"
            "ivec2 maxIlwocationsNonUniformAMD(ivec2);"
            "ivec3 maxIlwocationsNonUniformAMD(ivec3);"
            "ivec4 maxIlwocationsNonUniformAMD(ivec4);"

            "uint  maxIlwocationsNonUniformAMD(uint);"
            "uvec2 maxIlwocationsNonUniformAMD(uvec2);"
            "uvec3 maxIlwocationsNonUniformAMD(uvec3);"
            "uvec4 maxIlwocationsNonUniformAMD(uvec4);"

            "double maxIlwocationsNonUniformAMD(double);"
            "dvec2  maxIlwocationsNonUniformAMD(dvec2);"
            "dvec3  maxIlwocationsNonUniformAMD(dvec3);"
            "dvec4  maxIlwocationsNonUniformAMD(dvec4);"

            "int64_t maxIlwocationsNonUniformAMD(int64_t);"
            "i64vec2 maxIlwocationsNonUniformAMD(i64vec2);"
            "i64vec3 maxIlwocationsNonUniformAMD(i64vec3);"
            "i64vec4 maxIlwocationsNonUniformAMD(i64vec4);"

            "uint64_t maxIlwocationsNonUniformAMD(uint64_t);"
            "u64vec2  maxIlwocationsNonUniformAMD(u64vec2);"
            "u64vec3  maxIlwocationsNonUniformAMD(u64vec3);"
            "u64vec4  maxIlwocationsNonUniformAMD(u64vec4);"

            "float16_t maxIlwocationsNonUniformAMD(float16_t);"
            "f16vec2   maxIlwocationsNonUniformAMD(f16vec2);"
            "f16vec3   maxIlwocationsNonUniformAMD(f16vec3);"
            "f16vec4   maxIlwocationsNonUniformAMD(f16vec4);"

            "int16_t maxIlwocationsNonUniformAMD(int16_t);"
            "i16vec2 maxIlwocationsNonUniformAMD(i16vec2);"
            "i16vec3 maxIlwocationsNonUniformAMD(i16vec3);"
            "i16vec4 maxIlwocationsNonUniformAMD(i16vec4);"

            "uint16_t maxIlwocationsNonUniformAMD(uint16_t);"
            "u16vec2  maxIlwocationsNonUniformAMD(u16vec2);"
            "u16vec3  maxIlwocationsNonUniformAMD(u16vec3);"
            "u16vec4  maxIlwocationsNonUniformAMD(u16vec4);"

            "float maxIlwocationsInclusiveScanNonUniformAMD(float);"
            "vec2  maxIlwocationsInclusiveScanNonUniformAMD(vec2);"
            "vec3  maxIlwocationsInclusiveScanNonUniformAMD(vec3);"
            "vec4  maxIlwocationsInclusiveScanNonUniformAMD(vec4);"

            "int   maxIlwocationsInclusiveScanNonUniformAMD(int);"
            "ivec2 maxIlwocationsInclusiveScanNonUniformAMD(ivec2);"
            "ivec3 maxIlwocationsInclusiveScanNonUniformAMD(ivec3);"
            "ivec4 maxIlwocationsInclusiveScanNonUniformAMD(ivec4);"

            "uint  maxIlwocationsInclusiveScanNonUniformAMD(uint);"
            "uvec2 maxIlwocationsInclusiveScanNonUniformAMD(uvec2);"
            "uvec3 maxIlwocationsInclusiveScanNonUniformAMD(uvec3);"
            "uvec4 maxIlwocationsInclusiveScanNonUniformAMD(uvec4);"

            "double maxIlwocationsInclusiveScanNonUniformAMD(double);"
            "dvec2  maxIlwocationsInclusiveScanNonUniformAMD(dvec2);"
            "dvec3  maxIlwocationsInclusiveScanNonUniformAMD(dvec3);"
            "dvec4  maxIlwocationsInclusiveScanNonUniformAMD(dvec4);"

            "int64_t maxIlwocationsInclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 maxIlwocationsInclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 maxIlwocationsInclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 maxIlwocationsInclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t maxIlwocationsInclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  maxIlwocationsInclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  maxIlwocationsInclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  maxIlwocationsInclusiveScanNonUniformAMD(u64vec4);"

            "float16_t maxIlwocationsInclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   maxIlwocationsInclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   maxIlwocationsInclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   maxIlwocationsInclusiveScanNonUniformAMD(f16vec4);"

            "int16_t maxIlwocationsInclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 maxIlwocationsInclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 maxIlwocationsInclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 maxIlwocationsInclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t maxIlwocationsInclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  maxIlwocationsInclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  maxIlwocationsInclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  maxIlwocationsInclusiveScanNonUniformAMD(u16vec4);"

            "float maxIlwocationsExclusiveScanNonUniformAMD(float);"
            "vec2  maxIlwocationsExclusiveScanNonUniformAMD(vec2);"
            "vec3  maxIlwocationsExclusiveScanNonUniformAMD(vec3);"
            "vec4  maxIlwocationsExclusiveScanNonUniformAMD(vec4);"

            "int   maxIlwocationsExclusiveScanNonUniformAMD(int);"
            "ivec2 maxIlwocationsExclusiveScanNonUniformAMD(ivec2);"
            "ivec3 maxIlwocationsExclusiveScanNonUniformAMD(ivec3);"
            "ivec4 maxIlwocationsExclusiveScanNonUniformAMD(ivec4);"

            "uint  maxIlwocationsExclusiveScanNonUniformAMD(uint);"
            "uvec2 maxIlwocationsExclusiveScanNonUniformAMD(uvec2);"
            "uvec3 maxIlwocationsExclusiveScanNonUniformAMD(uvec3);"
            "uvec4 maxIlwocationsExclusiveScanNonUniformAMD(uvec4);"

            "double maxIlwocationsExclusiveScanNonUniformAMD(double);"
            "dvec2  maxIlwocationsExclusiveScanNonUniformAMD(dvec2);"
            "dvec3  maxIlwocationsExclusiveScanNonUniformAMD(dvec3);"
            "dvec4  maxIlwocationsExclusiveScanNonUniformAMD(dvec4);"

            "int64_t maxIlwocationsExclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 maxIlwocationsExclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 maxIlwocationsExclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 maxIlwocationsExclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t maxIlwocationsExclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  maxIlwocationsExclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  maxIlwocationsExclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  maxIlwocationsExclusiveScanNonUniformAMD(u64vec4);"

            "float16_t maxIlwocationsExclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   maxIlwocationsExclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   maxIlwocationsExclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   maxIlwocationsExclusiveScanNonUniformAMD(f16vec4);"

            "int16_t maxIlwocationsExclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 maxIlwocationsExclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 maxIlwocationsExclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 maxIlwocationsExclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t maxIlwocationsExclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  maxIlwocationsExclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  maxIlwocationsExclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  maxIlwocationsExclusiveScanNonUniformAMD(u16vec4);"

            "float addIlwocationsNonUniformAMD(float);"
            "vec2  addIlwocationsNonUniformAMD(vec2);"
            "vec3  addIlwocationsNonUniformAMD(vec3);"
            "vec4  addIlwocationsNonUniformAMD(vec4);"

            "int   addIlwocationsNonUniformAMD(int);"
            "ivec2 addIlwocationsNonUniformAMD(ivec2);"
            "ivec3 addIlwocationsNonUniformAMD(ivec3);"
            "ivec4 addIlwocationsNonUniformAMD(ivec4);"

            "uint  addIlwocationsNonUniformAMD(uint);"
            "uvec2 addIlwocationsNonUniformAMD(uvec2);"
            "uvec3 addIlwocationsNonUniformAMD(uvec3);"
            "uvec4 addIlwocationsNonUniformAMD(uvec4);"

            "double addIlwocationsNonUniformAMD(double);"
            "dvec2  addIlwocationsNonUniformAMD(dvec2);"
            "dvec3  addIlwocationsNonUniformAMD(dvec3);"
            "dvec4  addIlwocationsNonUniformAMD(dvec4);"

            "int64_t addIlwocationsNonUniformAMD(int64_t);"
            "i64vec2 addIlwocationsNonUniformAMD(i64vec2);"
            "i64vec3 addIlwocationsNonUniformAMD(i64vec3);"
            "i64vec4 addIlwocationsNonUniformAMD(i64vec4);"

            "uint64_t addIlwocationsNonUniformAMD(uint64_t);"
            "u64vec2  addIlwocationsNonUniformAMD(u64vec2);"
            "u64vec3  addIlwocationsNonUniformAMD(u64vec3);"
            "u64vec4  addIlwocationsNonUniformAMD(u64vec4);"

            "float16_t addIlwocationsNonUniformAMD(float16_t);"
            "f16vec2   addIlwocationsNonUniformAMD(f16vec2);"
            "f16vec3   addIlwocationsNonUniformAMD(f16vec3);"
            "f16vec4   addIlwocationsNonUniformAMD(f16vec4);"

            "int16_t addIlwocationsNonUniformAMD(int16_t);"
            "i16vec2 addIlwocationsNonUniformAMD(i16vec2);"
            "i16vec3 addIlwocationsNonUniformAMD(i16vec3);"
            "i16vec4 addIlwocationsNonUniformAMD(i16vec4);"

            "uint16_t addIlwocationsNonUniformAMD(uint16_t);"
            "u16vec2  addIlwocationsNonUniformAMD(u16vec2);"
            "u16vec3  addIlwocationsNonUniformAMD(u16vec3);"
            "u16vec4  addIlwocationsNonUniformAMD(u16vec4);"

            "float addIlwocationsInclusiveScanNonUniformAMD(float);"
            "vec2  addIlwocationsInclusiveScanNonUniformAMD(vec2);"
            "vec3  addIlwocationsInclusiveScanNonUniformAMD(vec3);"
            "vec4  addIlwocationsInclusiveScanNonUniformAMD(vec4);"

            "int   addIlwocationsInclusiveScanNonUniformAMD(int);"
            "ivec2 addIlwocationsInclusiveScanNonUniformAMD(ivec2);"
            "ivec3 addIlwocationsInclusiveScanNonUniformAMD(ivec3);"
            "ivec4 addIlwocationsInclusiveScanNonUniformAMD(ivec4);"

            "uint  addIlwocationsInclusiveScanNonUniformAMD(uint);"
            "uvec2 addIlwocationsInclusiveScanNonUniformAMD(uvec2);"
            "uvec3 addIlwocationsInclusiveScanNonUniformAMD(uvec3);"
            "uvec4 addIlwocationsInclusiveScanNonUniformAMD(uvec4);"

            "double addIlwocationsInclusiveScanNonUniformAMD(double);"
            "dvec2  addIlwocationsInclusiveScanNonUniformAMD(dvec2);"
            "dvec3  addIlwocationsInclusiveScanNonUniformAMD(dvec3);"
            "dvec4  addIlwocationsInclusiveScanNonUniformAMD(dvec4);"

            "int64_t addIlwocationsInclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 addIlwocationsInclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 addIlwocationsInclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 addIlwocationsInclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t addIlwocationsInclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  addIlwocationsInclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  addIlwocationsInclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  addIlwocationsInclusiveScanNonUniformAMD(u64vec4);"

            "float16_t addIlwocationsInclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   addIlwocationsInclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   addIlwocationsInclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   addIlwocationsInclusiveScanNonUniformAMD(f16vec4);"

            "int16_t addIlwocationsInclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 addIlwocationsInclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 addIlwocationsInclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 addIlwocationsInclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t addIlwocationsInclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  addIlwocationsInclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  addIlwocationsInclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  addIlwocationsInclusiveScanNonUniformAMD(u16vec4);"

            "float addIlwocationsExclusiveScanNonUniformAMD(float);"
            "vec2  addIlwocationsExclusiveScanNonUniformAMD(vec2);"
            "vec3  addIlwocationsExclusiveScanNonUniformAMD(vec3);"
            "vec4  addIlwocationsExclusiveScanNonUniformAMD(vec4);"

            "int   addIlwocationsExclusiveScanNonUniformAMD(int);"
            "ivec2 addIlwocationsExclusiveScanNonUniformAMD(ivec2);"
            "ivec3 addIlwocationsExclusiveScanNonUniformAMD(ivec3);"
            "ivec4 addIlwocationsExclusiveScanNonUniformAMD(ivec4);"

            "uint  addIlwocationsExclusiveScanNonUniformAMD(uint);"
            "uvec2 addIlwocationsExclusiveScanNonUniformAMD(uvec2);"
            "uvec3 addIlwocationsExclusiveScanNonUniformAMD(uvec3);"
            "uvec4 addIlwocationsExclusiveScanNonUniformAMD(uvec4);"

            "double addIlwocationsExclusiveScanNonUniformAMD(double);"
            "dvec2  addIlwocationsExclusiveScanNonUniformAMD(dvec2);"
            "dvec3  addIlwocationsExclusiveScanNonUniformAMD(dvec3);"
            "dvec4  addIlwocationsExclusiveScanNonUniformAMD(dvec4);"

            "int64_t addIlwocationsExclusiveScanNonUniformAMD(int64_t);"
            "i64vec2 addIlwocationsExclusiveScanNonUniformAMD(i64vec2);"
            "i64vec3 addIlwocationsExclusiveScanNonUniformAMD(i64vec3);"
            "i64vec4 addIlwocationsExclusiveScanNonUniformAMD(i64vec4);"

            "uint64_t addIlwocationsExclusiveScanNonUniformAMD(uint64_t);"
            "u64vec2  addIlwocationsExclusiveScanNonUniformAMD(u64vec2);"
            "u64vec3  addIlwocationsExclusiveScanNonUniformAMD(u64vec3);"
            "u64vec4  addIlwocationsExclusiveScanNonUniformAMD(u64vec4);"

            "float16_t addIlwocationsExclusiveScanNonUniformAMD(float16_t);"
            "f16vec2   addIlwocationsExclusiveScanNonUniformAMD(f16vec2);"
            "f16vec3   addIlwocationsExclusiveScanNonUniformAMD(f16vec3);"
            "f16vec4   addIlwocationsExclusiveScanNonUniformAMD(f16vec4);"

            "int16_t addIlwocationsExclusiveScanNonUniformAMD(int16_t);"
            "i16vec2 addIlwocationsExclusiveScanNonUniformAMD(i16vec2);"
            "i16vec3 addIlwocationsExclusiveScanNonUniformAMD(i16vec3);"
            "i16vec4 addIlwocationsExclusiveScanNonUniformAMD(i16vec4);"

            "uint16_t addIlwocationsExclusiveScanNonUniformAMD(uint16_t);"
            "u16vec2  addIlwocationsExclusiveScanNonUniformAMD(u16vec2);"
            "u16vec3  addIlwocationsExclusiveScanNonUniformAMD(u16vec3);"
            "u16vec4  addIlwocationsExclusiveScanNonUniformAMD(u16vec4);"

            "float swizzleIlwocationsAMD(float, uvec4);"
            "vec2  swizzleIlwocationsAMD(vec2,  uvec4);"
            "vec3  swizzleIlwocationsAMD(vec3,  uvec4);"
            "vec4  swizzleIlwocationsAMD(vec4,  uvec4);"

            "int   swizzleIlwocationsAMD(int,   uvec4);"
            "ivec2 swizzleIlwocationsAMD(ivec2, uvec4);"
            "ivec3 swizzleIlwocationsAMD(ivec3, uvec4);"
            "ivec4 swizzleIlwocationsAMD(ivec4, uvec4);"

            "uint  swizzleIlwocationsAMD(uint,  uvec4);"
            "uvec2 swizzleIlwocationsAMD(uvec2, uvec4);"
            "uvec3 swizzleIlwocationsAMD(uvec3, uvec4);"
            "uvec4 swizzleIlwocationsAMD(uvec4, uvec4);"

            "float swizzleIlwocationsMaskedAMD(float, uvec3);"
            "vec2  swizzleIlwocationsMaskedAMD(vec2,  uvec3);"
            "vec3  swizzleIlwocationsMaskedAMD(vec3,  uvec3);"
            "vec4  swizzleIlwocationsMaskedAMD(vec4,  uvec3);"

            "int   swizzleIlwocationsMaskedAMD(int,   uvec3);"
            "ivec2 swizzleIlwocationsMaskedAMD(ivec2, uvec3);"
            "ivec3 swizzleIlwocationsMaskedAMD(ivec3, uvec3);"
            "ivec4 swizzleIlwocationsMaskedAMD(ivec4, uvec3);"

            "uint  swizzleIlwocationsMaskedAMD(uint,  uvec3);"
            "uvec2 swizzleIlwocationsMaskedAMD(uvec2, uvec3);"
            "uvec3 swizzleIlwocationsMaskedAMD(uvec3, uvec3);"
            "uvec4 swizzleIlwocationsMaskedAMD(uvec4, uvec3);"

            "float writeIlwocationAMD(float, float, uint);"
            "vec2  writeIlwocationAMD(vec2,  vec2,  uint);"
            "vec3  writeIlwocationAMD(vec3,  vec3,  uint);"
            "vec4  writeIlwocationAMD(vec4,  vec4,  uint);"

            "int   writeIlwocationAMD(int,   int,   uint);"
            "ivec2 writeIlwocationAMD(ivec2, ivec2, uint);"
            "ivec3 writeIlwocationAMD(ivec3, ivec3, uint);"
            "ivec4 writeIlwocationAMD(ivec4, ivec4, uint);"

            "uint  writeIlwocationAMD(uint,  uint,  uint);"
            "uvec2 writeIlwocationAMD(uvec2, uvec2, uint);"
            "uvec3 writeIlwocationAMD(uvec3, uvec3, uint);"
            "uvec4 writeIlwocationAMD(uvec4, uvec4, uint);"

            "uint mbcntAMD(uint64_t);"

            "\n");
    }

    // GL_AMD_gcn_shader
    if (profile != EEsProfile && version >= 440) {
        commonBuiltins.append(
            "float lwbeFaceIndexAMD(vec3);"
            "vec2  lwbeFaceCoordAMD(vec3);"
            "uint64_t timeAMD();"

            "in int gl_SIMDGroupSizeAMD;"
            "\n");
    }

    // GL_AMD_shader_fragment_mask
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "uint fragmentMaskFetchAMD(sampler2DMS,       ivec2);"
            "uint fragmentMaskFetchAMD(isampler2DMS,      ivec2);"
            "uint fragmentMaskFetchAMD(usampler2DMS,      ivec2);"

            "uint fragmentMaskFetchAMD(sampler2DMSArray,  ivec3);"
            "uint fragmentMaskFetchAMD(isampler2DMSArray, ivec3);"
            "uint fragmentMaskFetchAMD(usampler2DMSArray, ivec3);"

            "vec4  fragmentFetchAMD(sampler2DMS,       ivec2, uint);"
            "ivec4 fragmentFetchAMD(isampler2DMS,      ivec2, uint);"
            "uvec4 fragmentFetchAMD(usampler2DMS,      ivec2, uint);"

            "vec4  fragmentFetchAMD(sampler2DMSArray,  ivec3, uint);"
            "ivec4 fragmentFetchAMD(isampler2DMSArray, ivec3, uint);"
            "uvec4 fragmentFetchAMD(usampler2DMSArray, ivec3, uint);"

            "\n");
    }

    if ((profile != EEsProfile && version >= 130) ||
        (profile == EEsProfile && version >= 300)) {
        commonBuiltins.append(
            "uint countLeadingZeros(uint);"
            "uvec2 countLeadingZeros(uvec2);"
            "uvec3 countLeadingZeros(uvec3);"
            "uvec4 countLeadingZeros(uvec4);"

            "uint countTrailingZeros(uint);"
            "uvec2 countTrailingZeros(uvec2);"
            "uvec3 countTrailingZeros(uvec3);"
            "uvec4 countTrailingZeros(uvec4);"

            "uint absoluteDifference(int, int);"
            "uvec2 absoluteDifference(ivec2, ivec2);"
            "uvec3 absoluteDifference(ivec3, ivec3);"
            "uvec4 absoluteDifference(ivec4, ivec4);"

            "uint16_t absoluteDifference(int16_t, int16_t);"
            "u16vec2 absoluteDifference(i16vec2, i16vec2);"
            "u16vec3 absoluteDifference(i16vec3, i16vec3);"
            "u16vec4 absoluteDifference(i16vec4, i16vec4);"

            "uint64_t absoluteDifference(int64_t, int64_t);"
            "u64vec2 absoluteDifference(i64vec2, i64vec2);"
            "u64vec3 absoluteDifference(i64vec3, i64vec3);"
            "u64vec4 absoluteDifference(i64vec4, i64vec4);"

            "uint absoluteDifference(uint, uint);"
            "uvec2 absoluteDifference(uvec2, uvec2);"
            "uvec3 absoluteDifference(uvec3, uvec3);"
            "uvec4 absoluteDifference(uvec4, uvec4);"

            "uint16_t absoluteDifference(uint16_t, uint16_t);"
            "u16vec2 absoluteDifference(u16vec2, u16vec2);"
            "u16vec3 absoluteDifference(u16vec3, u16vec3);"
            "u16vec4 absoluteDifference(u16vec4, u16vec4);"

            "uint64_t absoluteDifference(uint64_t, uint64_t);"
            "u64vec2 absoluteDifference(u64vec2, u64vec2);"
            "u64vec3 absoluteDifference(u64vec3, u64vec3);"
            "u64vec4 absoluteDifference(u64vec4, u64vec4);"

            "int addSaturate(int, int);"
            "ivec2 addSaturate(ivec2, ivec2);"
            "ivec3 addSaturate(ivec3, ivec3);"
            "ivec4 addSaturate(ivec4, ivec4);"

            "int16_t addSaturate(int16_t, int16_t);"
            "i16vec2 addSaturate(i16vec2, i16vec2);"
            "i16vec3 addSaturate(i16vec3, i16vec3);"
            "i16vec4 addSaturate(i16vec4, i16vec4);"

            "int64_t addSaturate(int64_t, int64_t);"
            "i64vec2 addSaturate(i64vec2, i64vec2);"
            "i64vec3 addSaturate(i64vec3, i64vec3);"
            "i64vec4 addSaturate(i64vec4, i64vec4);"

            "uint addSaturate(uint, uint);"
            "uvec2 addSaturate(uvec2, uvec2);"
            "uvec3 addSaturate(uvec3, uvec3);"
            "uvec4 addSaturate(uvec4, uvec4);"

            "uint16_t addSaturate(uint16_t, uint16_t);"
            "u16vec2 addSaturate(u16vec2, u16vec2);"
            "u16vec3 addSaturate(u16vec3, u16vec3);"
            "u16vec4 addSaturate(u16vec4, u16vec4);"

            "uint64_t addSaturate(uint64_t, uint64_t);"
            "u64vec2 addSaturate(u64vec2, u64vec2);"
            "u64vec3 addSaturate(u64vec3, u64vec3);"
            "u64vec4 addSaturate(u64vec4, u64vec4);"

            "int subtractSaturate(int, int);"
            "ivec2 subtractSaturate(ivec2, ivec2);"
            "ivec3 subtractSaturate(ivec3, ivec3);"
            "ivec4 subtractSaturate(ivec4, ivec4);"

            "int16_t subtractSaturate(int16_t, int16_t);"
            "i16vec2 subtractSaturate(i16vec2, i16vec2);"
            "i16vec3 subtractSaturate(i16vec3, i16vec3);"
            "i16vec4 subtractSaturate(i16vec4, i16vec4);"

            "int64_t subtractSaturate(int64_t, int64_t);"
            "i64vec2 subtractSaturate(i64vec2, i64vec2);"
            "i64vec3 subtractSaturate(i64vec3, i64vec3);"
            "i64vec4 subtractSaturate(i64vec4, i64vec4);"

            "uint subtractSaturate(uint, uint);"
            "uvec2 subtractSaturate(uvec2, uvec2);"
            "uvec3 subtractSaturate(uvec3, uvec3);"
            "uvec4 subtractSaturate(uvec4, uvec4);"

            "uint16_t subtractSaturate(uint16_t, uint16_t);"
            "u16vec2 subtractSaturate(u16vec2, u16vec2);"
            "u16vec3 subtractSaturate(u16vec3, u16vec3);"
            "u16vec4 subtractSaturate(u16vec4, u16vec4);"

            "uint64_t subtractSaturate(uint64_t, uint64_t);"
            "u64vec2 subtractSaturate(u64vec2, u64vec2);"
            "u64vec3 subtractSaturate(u64vec3, u64vec3);"
            "u64vec4 subtractSaturate(u64vec4, u64vec4);"

            "int average(int, int);"
            "ivec2 average(ivec2, ivec2);"
            "ivec3 average(ivec3, ivec3);"
            "ivec4 average(ivec4, ivec4);"

            "int16_t average(int16_t, int16_t);"
            "i16vec2 average(i16vec2, i16vec2);"
            "i16vec3 average(i16vec3, i16vec3);"
            "i16vec4 average(i16vec4, i16vec4);"

            "int64_t average(int64_t, int64_t);"
            "i64vec2 average(i64vec2, i64vec2);"
            "i64vec3 average(i64vec3, i64vec3);"
            "i64vec4 average(i64vec4, i64vec4);"

            "uint average(uint, uint);"
            "uvec2 average(uvec2, uvec2);"
            "uvec3 average(uvec3, uvec3);"
            "uvec4 average(uvec4, uvec4);"

            "uint16_t average(uint16_t, uint16_t);"
            "u16vec2 average(u16vec2, u16vec2);"
            "u16vec3 average(u16vec3, u16vec3);"
            "u16vec4 average(u16vec4, u16vec4);"

            "uint64_t average(uint64_t, uint64_t);"
            "u64vec2 average(u64vec2, u64vec2);"
            "u64vec3 average(u64vec3, u64vec3);"
            "u64vec4 average(u64vec4, u64vec4);"

            "int averageRounded(int, int);"
            "ivec2 averageRounded(ivec2, ivec2);"
            "ivec3 averageRounded(ivec3, ivec3);"
            "ivec4 averageRounded(ivec4, ivec4);"

            "int16_t averageRounded(int16_t, int16_t);"
            "i16vec2 averageRounded(i16vec2, i16vec2);"
            "i16vec3 averageRounded(i16vec3, i16vec3);"
            "i16vec4 averageRounded(i16vec4, i16vec4);"

            "int64_t averageRounded(int64_t, int64_t);"
            "i64vec2 averageRounded(i64vec2, i64vec2);"
            "i64vec3 averageRounded(i64vec3, i64vec3);"
            "i64vec4 averageRounded(i64vec4, i64vec4);"

            "uint averageRounded(uint, uint);"
            "uvec2 averageRounded(uvec2, uvec2);"
            "uvec3 averageRounded(uvec3, uvec3);"
            "uvec4 averageRounded(uvec4, uvec4);"

            "uint16_t averageRounded(uint16_t, uint16_t);"
            "u16vec2 averageRounded(u16vec2, u16vec2);"
            "u16vec3 averageRounded(u16vec3, u16vec3);"
            "u16vec4 averageRounded(u16vec4, u16vec4);"

            "uint64_t averageRounded(uint64_t, uint64_t);"
            "u64vec2 averageRounded(u64vec2, u64vec2);"
            "u64vec3 averageRounded(u64vec3, u64vec3);"
            "u64vec4 averageRounded(u64vec4, u64vec4);"

            "int multiply32x16(int, int);"
            "ivec2 multiply32x16(ivec2, ivec2);"
            "ivec3 multiply32x16(ivec3, ivec3);"
            "ivec4 multiply32x16(ivec4, ivec4);"

            "uint multiply32x16(uint, uint);"
            "uvec2 multiply32x16(uvec2, uvec2);"
            "uvec3 multiply32x16(uvec3, uvec3);"
            "uvec4 multiply32x16(uvec4, uvec4);"
            "\n");
    }

    if ((profile != EEsProfile && version >= 450) ||
        (profile == EEsProfile && version >= 320)) {
        commonBuiltins.append(
            "struct gl_TextureFootprint2DLW {"
                "uvec2 anchor;"
                "uvec2 offset;"
                "uvec2 mask;"
                "uint lod;"
                "uint granularity;"
            "};"

            "struct gl_TextureFootprint3DLW {"
                "uvec3 anchor;"
                "uvec3 offset;"
                "uvec2 mask;"
                "uint lod;"
                "uint granularity;"
            "};"
            "bool textureFootprintLW(sampler2D, vec2, int, bool, out gl_TextureFootprint2DLW);"
            "bool textureFootprintLW(sampler3D, vec3, int, bool, out gl_TextureFootprint3DLW);"
            "bool textureFootprintLW(sampler2D, vec2, int, bool, out gl_TextureFootprint2DLW, float);"
            "bool textureFootprintLW(sampler3D, vec3, int, bool, out gl_TextureFootprint3DLW, float);"
            "bool textureFootprintClampLW(sampler2D, vec2, float, int, bool, out gl_TextureFootprint2DLW);"
            "bool textureFootprintClampLW(sampler3D, vec3, float, int, bool, out gl_TextureFootprint3DLW);"
            "bool textureFootprintClampLW(sampler2D, vec2, float, int, bool, out gl_TextureFootprint2DLW, float);"
            "bool textureFootprintClampLW(sampler3D, vec3, float, int, bool, out gl_TextureFootprint3DLW, float);"
            "bool textureFootprintLodLW(sampler2D, vec2, float, int, bool, out gl_TextureFootprint2DLW);"
            "bool textureFootprintLodLW(sampler3D, vec3, float, int, bool, out gl_TextureFootprint3DLW);"
            "bool textureFootprintGradLW(sampler2D, vec2, vec2, vec2, int, bool, out gl_TextureFootprint2DLW);"
            "bool textureFootprintGradClampLW(sampler2D, vec2, vec2, vec2, float, int, bool, out gl_TextureFootprint2DLW);"
            "\n");
    }

    if ((profile == EEsProfile && version >= 300 && version < 310) ||
        (profile != EEsProfile && version >= 150 && version < 450)) { // GL_EXT_shader_integer_mix
        commonBuiltins.append("int mix(int, int, bool);"
                              "ivec2 mix(ivec2, ivec2, bvec2);"
                              "ivec3 mix(ivec3, ivec3, bvec3);"
                              "ivec4 mix(ivec4, ivec4, bvec4);"
                              "uint  mix(uint,  uint,  bool );"
                              "uvec2 mix(uvec2, uvec2, bvec2);"
                              "uvec3 mix(uvec3, uvec3, bvec3);"
                              "uvec4 mix(uvec4, uvec4, bvec4);"
                              "bool  mix(bool,  bool,  bool );"
                              "bvec2 mix(bvec2, bvec2, bvec2);"
                              "bvec3 mix(bvec3, bvec3, bvec3);"
                              "bvec4 mix(bvec4, bvec4, bvec4);"

                              "\n");
    }

    // GL_AMD_gpu_shader_half_float/Explicit types
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "float16_t radians(float16_t);"
            "f16vec2   radians(f16vec2);"
            "f16vec3   radians(f16vec3);"
            "f16vec4   radians(f16vec4);"

            "float16_t degrees(float16_t);"
            "f16vec2   degrees(f16vec2);"
            "f16vec3   degrees(f16vec3);"
            "f16vec4   degrees(f16vec4);"

            "float16_t sin(float16_t);"
            "f16vec2   sin(f16vec2);"
            "f16vec3   sin(f16vec3);"
            "f16vec4   sin(f16vec4);"

            "float16_t cos(float16_t);"
            "f16vec2   cos(f16vec2);"
            "f16vec3   cos(f16vec3);"
            "f16vec4   cos(f16vec4);"

            "float16_t tan(float16_t);"
            "f16vec2   tan(f16vec2);"
            "f16vec3   tan(f16vec3);"
            "f16vec4   tan(f16vec4);"

            "float16_t asin(float16_t);"
            "f16vec2   asin(f16vec2);"
            "f16vec3   asin(f16vec3);"
            "f16vec4   asin(f16vec4);"

            "float16_t acos(float16_t);"
            "f16vec2   acos(f16vec2);"
            "f16vec3   acos(f16vec3);"
            "f16vec4   acos(f16vec4);"

            "float16_t atan(float16_t, float16_t);"
            "f16vec2   atan(f16vec2,   f16vec2);"
            "f16vec3   atan(f16vec3,   f16vec3);"
            "f16vec4   atan(f16vec4,   f16vec4);"

            "float16_t atan(float16_t);"
            "f16vec2   atan(f16vec2);"
            "f16vec3   atan(f16vec3);"
            "f16vec4   atan(f16vec4);"

            "float16_t sinh(float16_t);"
            "f16vec2   sinh(f16vec2);"
            "f16vec3   sinh(f16vec3);"
            "f16vec4   sinh(f16vec4);"

            "float16_t cosh(float16_t);"
            "f16vec2   cosh(f16vec2);"
            "f16vec3   cosh(f16vec3);"
            "f16vec4   cosh(f16vec4);"

            "float16_t tanh(float16_t);"
            "f16vec2   tanh(f16vec2);"
            "f16vec3   tanh(f16vec3);"
            "f16vec4   tanh(f16vec4);"

            "float16_t asinh(float16_t);"
            "f16vec2   asinh(f16vec2);"
            "f16vec3   asinh(f16vec3);"
            "f16vec4   asinh(f16vec4);"

            "float16_t acosh(float16_t);"
            "f16vec2   acosh(f16vec2);"
            "f16vec3   acosh(f16vec3);"
            "f16vec4   acosh(f16vec4);"

            "float16_t atanh(float16_t);"
            "f16vec2   atanh(f16vec2);"
            "f16vec3   atanh(f16vec3);"
            "f16vec4   atanh(f16vec4);"

            "float16_t pow(float16_t, float16_t);"
            "f16vec2   pow(f16vec2,   f16vec2);"
            "f16vec3   pow(f16vec3,   f16vec3);"
            "f16vec4   pow(f16vec4,   f16vec4);"

            "float16_t exp(float16_t);"
            "f16vec2   exp(f16vec2);"
            "f16vec3   exp(f16vec3);"
            "f16vec4   exp(f16vec4);"

            "float16_t log(float16_t);"
            "f16vec2   log(f16vec2);"
            "f16vec3   log(f16vec3);"
            "f16vec4   log(f16vec4);"

            "float16_t exp2(float16_t);"
            "f16vec2   exp2(f16vec2);"
            "f16vec3   exp2(f16vec3);"
            "f16vec4   exp2(f16vec4);"

            "float16_t log2(float16_t);"
            "f16vec2   log2(f16vec2);"
            "f16vec3   log2(f16vec3);"
            "f16vec4   log2(f16vec4);"

            "float16_t sqrt(float16_t);"
            "f16vec2   sqrt(f16vec2);"
            "f16vec3   sqrt(f16vec3);"
            "f16vec4   sqrt(f16vec4);"

            "float16_t ilwersesqrt(float16_t);"
            "f16vec2   ilwersesqrt(f16vec2);"
            "f16vec3   ilwersesqrt(f16vec3);"
            "f16vec4   ilwersesqrt(f16vec4);"

            "float16_t abs(float16_t);"
            "f16vec2   abs(f16vec2);"
            "f16vec3   abs(f16vec3);"
            "f16vec4   abs(f16vec4);"

            "float16_t sign(float16_t);"
            "f16vec2   sign(f16vec2);"
            "f16vec3   sign(f16vec3);"
            "f16vec4   sign(f16vec4);"

            "float16_t floor(float16_t);"
            "f16vec2   floor(f16vec2);"
            "f16vec3   floor(f16vec3);"
            "f16vec4   floor(f16vec4);"

            "float16_t trunc(float16_t);"
            "f16vec2   trunc(f16vec2);"
            "f16vec3   trunc(f16vec3);"
            "f16vec4   trunc(f16vec4);"

            "float16_t round(float16_t);"
            "f16vec2   round(f16vec2);"
            "f16vec3   round(f16vec3);"
            "f16vec4   round(f16vec4);"

            "float16_t roundEven(float16_t);"
            "f16vec2   roundEven(f16vec2);"
            "f16vec3   roundEven(f16vec3);"
            "f16vec4   roundEven(f16vec4);"

            "float16_t ceil(float16_t);"
            "f16vec2   ceil(f16vec2);"
            "f16vec3   ceil(f16vec3);"
            "f16vec4   ceil(f16vec4);"

            "float16_t fract(float16_t);"
            "f16vec2   fract(f16vec2);"
            "f16vec3   fract(f16vec3);"
            "f16vec4   fract(f16vec4);"

            "float16_t mod(float16_t, float16_t);"
            "f16vec2   mod(f16vec2,   float16_t);"
            "f16vec3   mod(f16vec3,   float16_t);"
            "f16vec4   mod(f16vec4,   float16_t);"
            "f16vec2   mod(f16vec2,   f16vec2);"
            "f16vec3   mod(f16vec3,   f16vec3);"
            "f16vec4   mod(f16vec4,   f16vec4);"

            "float16_t modf(float16_t, out float16_t);"
            "f16vec2   modf(f16vec2,   out f16vec2);"
            "f16vec3   modf(f16vec3,   out f16vec3);"
            "f16vec4   modf(f16vec4,   out f16vec4);"

            "float16_t min(float16_t, float16_t);"
            "f16vec2   min(f16vec2,   float16_t);"
            "f16vec3   min(f16vec3,   float16_t);"
            "f16vec4   min(f16vec4,   float16_t);"
            "f16vec2   min(f16vec2,   f16vec2);"
            "f16vec3   min(f16vec3,   f16vec3);"
            "f16vec4   min(f16vec4,   f16vec4);"

            "float16_t max(float16_t, float16_t);"
            "f16vec2   max(f16vec2,   float16_t);"
            "f16vec3   max(f16vec3,   float16_t);"
            "f16vec4   max(f16vec4,   float16_t);"
            "f16vec2   max(f16vec2,   f16vec2);"
            "f16vec3   max(f16vec3,   f16vec3);"
            "f16vec4   max(f16vec4,   f16vec4);"

            "float16_t clamp(float16_t, float16_t, float16_t);"
            "f16vec2   clamp(f16vec2,   float16_t, float16_t);"
            "f16vec3   clamp(f16vec3,   float16_t, float16_t);"
            "f16vec4   clamp(f16vec4,   float16_t, float16_t);"
            "f16vec2   clamp(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   clamp(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   clamp(f16vec4,   f16vec4,   f16vec4);"

            "float16_t mix(float16_t, float16_t, float16_t);"
            "f16vec2   mix(f16vec2,   f16vec2,   float16_t);"
            "f16vec3   mix(f16vec3,   f16vec3,   float16_t);"
            "f16vec4   mix(f16vec4,   f16vec4,   float16_t);"
            "f16vec2   mix(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   mix(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   mix(f16vec4,   f16vec4,   f16vec4);"
            "float16_t mix(float16_t, float16_t, bool);"
            "f16vec2   mix(f16vec2,   f16vec2,   bvec2);"
            "f16vec3   mix(f16vec3,   f16vec3,   bvec3);"
            "f16vec4   mix(f16vec4,   f16vec4,   bvec4);"

            "float16_t step(float16_t, float16_t);"
            "f16vec2   step(f16vec2,   f16vec2);"
            "f16vec3   step(f16vec3,   f16vec3);"
            "f16vec4   step(f16vec4,   f16vec4);"
            "f16vec2   step(float16_t, f16vec2);"
            "f16vec3   step(float16_t, f16vec3);"
            "f16vec4   step(float16_t, f16vec4);"

            "float16_t smoothstep(float16_t, float16_t, float16_t);"
            "f16vec2   smoothstep(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   smoothstep(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   smoothstep(f16vec4,   f16vec4,   f16vec4);"
            "f16vec2   smoothstep(float16_t, float16_t, f16vec2);"
            "f16vec3   smoothstep(float16_t, float16_t, f16vec3);"
            "f16vec4   smoothstep(float16_t, float16_t, f16vec4);"

            "bool  isnan(float16_t);"
            "bvec2 isnan(f16vec2);"
            "bvec3 isnan(f16vec3);"
            "bvec4 isnan(f16vec4);"

            "bool  isinf(float16_t);"
            "bvec2 isinf(f16vec2);"
            "bvec3 isinf(f16vec3);"
            "bvec4 isinf(f16vec4);"

            "float16_t fma(float16_t, float16_t, float16_t);"
            "f16vec2   fma(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   fma(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   fma(f16vec4,   f16vec4,   f16vec4);"

            "float16_t frexp(float16_t, out int);"
            "f16vec2   frexp(f16vec2,   out ivec2);"
            "f16vec3   frexp(f16vec3,   out ivec3);"
            "f16vec4   frexp(f16vec4,   out ivec4);"

            "float16_t ldexp(float16_t, in int);"
            "f16vec2   ldexp(f16vec2,   in ivec2);"
            "f16vec3   ldexp(f16vec3,   in ivec3);"
            "f16vec4   ldexp(f16vec4,   in ivec4);"

            "uint    packFloat2x16(f16vec2);"
            "f16vec2 unpackFloat2x16(uint);"

            "float16_t length(float16_t);"
            "float16_t length(f16vec2);"
            "float16_t length(f16vec3);"
            "float16_t length(f16vec4);"

            "float16_t distance(float16_t, float16_t);"
            "float16_t distance(f16vec2,   f16vec2);"
            "float16_t distance(f16vec3,   f16vec3);"
            "float16_t distance(f16vec4,   f16vec4);"

            "float16_t dot(float16_t, float16_t);"
            "float16_t dot(f16vec2,   f16vec2);"
            "float16_t dot(f16vec3,   f16vec3);"
            "float16_t dot(f16vec4,   f16vec4);"

            "f16vec3 cross(f16vec3, f16vec3);"

            "float16_t normalize(float16_t);"
            "f16vec2   normalize(f16vec2);"
            "f16vec3   normalize(f16vec3);"
            "f16vec4   normalize(f16vec4);"

            "float16_t faceforward(float16_t, float16_t, float16_t);"
            "f16vec2   faceforward(f16vec2,   f16vec2,   f16vec2);"
            "f16vec3   faceforward(f16vec3,   f16vec3,   f16vec3);"
            "f16vec4   faceforward(f16vec4,   f16vec4,   f16vec4);"

            "float16_t reflect(float16_t, float16_t);"
            "f16vec2   reflect(f16vec2,   f16vec2);"
            "f16vec3   reflect(f16vec3,   f16vec3);"
            "f16vec4   reflect(f16vec4,   f16vec4);"

            "float16_t refract(float16_t, float16_t, float16_t);"
            "f16vec2   refract(f16vec2,   f16vec2,   float16_t);"
            "f16vec3   refract(f16vec3,   f16vec3,   float16_t);"
            "f16vec4   refract(f16vec4,   f16vec4,   float16_t);"

            "f16mat2   matrixCompMult(f16mat2,   f16mat2);"
            "f16mat3   matrixCompMult(f16mat3,   f16mat3);"
            "f16mat4   matrixCompMult(f16mat4,   f16mat4);"
            "f16mat2x3 matrixCompMult(f16mat2x3, f16mat2x3);"
            "f16mat2x4 matrixCompMult(f16mat2x4, f16mat2x4);"
            "f16mat3x2 matrixCompMult(f16mat3x2, f16mat3x2);"
            "f16mat3x4 matrixCompMult(f16mat3x4, f16mat3x4);"
            "f16mat4x2 matrixCompMult(f16mat4x2, f16mat4x2);"
            "f16mat4x3 matrixCompMult(f16mat4x3, f16mat4x3);"

            "f16mat2   outerProduct(f16vec2, f16vec2);"
            "f16mat3   outerProduct(f16vec3, f16vec3);"
            "f16mat4   outerProduct(f16vec4, f16vec4);"
            "f16mat2x3 outerProduct(f16vec3, f16vec2);"
            "f16mat3x2 outerProduct(f16vec2, f16vec3);"
            "f16mat2x4 outerProduct(f16vec4, f16vec2);"
            "f16mat4x2 outerProduct(f16vec2, f16vec4);"
            "f16mat3x4 outerProduct(f16vec4, f16vec3);"
            "f16mat4x3 outerProduct(f16vec3, f16vec4);"

            "f16mat2   transpose(f16mat2);"
            "f16mat3   transpose(f16mat3);"
            "f16mat4   transpose(f16mat4);"
            "f16mat2x3 transpose(f16mat3x2);"
            "f16mat3x2 transpose(f16mat2x3);"
            "f16mat2x4 transpose(f16mat4x2);"
            "f16mat4x2 transpose(f16mat2x4);"
            "f16mat3x4 transpose(f16mat4x3);"
            "f16mat4x3 transpose(f16mat3x4);"

            "float16_t determinant(f16mat2);"
            "float16_t determinant(f16mat3);"
            "float16_t determinant(f16mat4);"

            "f16mat2 ilwerse(f16mat2);"
            "f16mat3 ilwerse(f16mat3);"
            "f16mat4 ilwerse(f16mat4);"

            "bvec2 lessThan(f16vec2, f16vec2);"
            "bvec3 lessThan(f16vec3, f16vec3);"
            "bvec4 lessThan(f16vec4, f16vec4);"

            "bvec2 lessThanEqual(f16vec2, f16vec2);"
            "bvec3 lessThanEqual(f16vec3, f16vec3);"
            "bvec4 lessThanEqual(f16vec4, f16vec4);"

            "bvec2 greaterThan(f16vec2, f16vec2);"
            "bvec3 greaterThan(f16vec3, f16vec3);"
            "bvec4 greaterThan(f16vec4, f16vec4);"

            "bvec2 greaterThanEqual(f16vec2, f16vec2);"
            "bvec3 greaterThanEqual(f16vec3, f16vec3);"
            "bvec4 greaterThanEqual(f16vec4, f16vec4);"

            "bvec2 equal(f16vec2, f16vec2);"
            "bvec3 equal(f16vec3, f16vec3);"
            "bvec4 equal(f16vec4, f16vec4);"

            "bvec2 notEqual(f16vec2, f16vec2);"
            "bvec3 notEqual(f16vec3, f16vec3);"
            "bvec4 notEqual(f16vec4, f16vec4);"

            "\n");
    }

    // Explicit types
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "int8_t abs(int8_t);"
            "i8vec2 abs(i8vec2);"
            "i8vec3 abs(i8vec3);"
            "i8vec4 abs(i8vec4);"

            "int8_t sign(int8_t);"
            "i8vec2 sign(i8vec2);"
            "i8vec3 sign(i8vec3);"
            "i8vec4 sign(i8vec4);"

            "int8_t min(int8_t x, int8_t y);"
            "i8vec2 min(i8vec2 x, int8_t y);"
            "i8vec3 min(i8vec3 x, int8_t y);"
            "i8vec4 min(i8vec4 x, int8_t y);"
            "i8vec2 min(i8vec2 x, i8vec2 y);"
            "i8vec3 min(i8vec3 x, i8vec3 y);"
            "i8vec4 min(i8vec4 x, i8vec4 y);"

            "uint8_t min(uint8_t x, uint8_t y);"
            "u8vec2 min(u8vec2 x, uint8_t y);"
            "u8vec3 min(u8vec3 x, uint8_t y);"
            "u8vec4 min(u8vec4 x, uint8_t y);"
            "u8vec2 min(u8vec2 x, u8vec2 y);"
            "u8vec3 min(u8vec3 x, u8vec3 y);"
            "u8vec4 min(u8vec4 x, u8vec4 y);"

            "int8_t max(int8_t x, int8_t y);"
            "i8vec2 max(i8vec2 x, int8_t y);"
            "i8vec3 max(i8vec3 x, int8_t y);"
            "i8vec4 max(i8vec4 x, int8_t y);"
            "i8vec2 max(i8vec2 x, i8vec2 y);"
            "i8vec3 max(i8vec3 x, i8vec3 y);"
            "i8vec4 max(i8vec4 x, i8vec4 y);"

            "uint8_t max(uint8_t x, uint8_t y);"
            "u8vec2 max(u8vec2 x, uint8_t y);"
            "u8vec3 max(u8vec3 x, uint8_t y);"
            "u8vec4 max(u8vec4 x, uint8_t y);"
            "u8vec2 max(u8vec2 x, u8vec2 y);"
            "u8vec3 max(u8vec3 x, u8vec3 y);"
            "u8vec4 max(u8vec4 x, u8vec4 y);"

            "int8_t    clamp(int8_t x, int8_t milwal, int8_t maxVal);"
            "i8vec2  clamp(i8vec2  x, int8_t milwal, int8_t maxVal);"
            "i8vec3  clamp(i8vec3  x, int8_t milwal, int8_t maxVal);"
            "i8vec4  clamp(i8vec4  x, int8_t milwal, int8_t maxVal);"
            "i8vec2  clamp(i8vec2  x, i8vec2  milwal, i8vec2  maxVal);"
            "i8vec3  clamp(i8vec3  x, i8vec3  milwal, i8vec3  maxVal);"
            "i8vec4  clamp(i8vec4  x, i8vec4  milwal, i8vec4  maxVal);"

            "uint8_t   clamp(uint8_t x, uint8_t milwal, uint8_t maxVal);"
            "u8vec2  clamp(u8vec2  x, uint8_t milwal, uint8_t maxVal);"
            "u8vec3  clamp(u8vec3  x, uint8_t milwal, uint8_t maxVal);"
            "u8vec4  clamp(u8vec4  x, uint8_t milwal, uint8_t maxVal);"
            "u8vec2  clamp(u8vec2  x, u8vec2  milwal, u8vec2  maxVal);"
            "u8vec3  clamp(u8vec3  x, u8vec3  milwal, u8vec3  maxVal);"
            "u8vec4  clamp(u8vec4  x, u8vec4  milwal, u8vec4  maxVal);"

            "int8_t  mix(int8_t,  int8_t,  bool);"
            "i8vec2  mix(i8vec2,  i8vec2,  bvec2);"
            "i8vec3  mix(i8vec3,  i8vec3,  bvec3);"
            "i8vec4  mix(i8vec4,  i8vec4,  bvec4);"
            "uint8_t mix(uint8_t, uint8_t, bool);"
            "u8vec2  mix(u8vec2,  u8vec2,  bvec2);"
            "u8vec3  mix(u8vec3,  u8vec3,  bvec3);"
            "u8vec4  mix(u8vec4,  u8vec4,  bvec4);"

            "bvec2 lessThan(i8vec2, i8vec2);"
            "bvec3 lessThan(i8vec3, i8vec3);"
            "bvec4 lessThan(i8vec4, i8vec4);"
            "bvec2 lessThan(u8vec2, u8vec2);"
            "bvec3 lessThan(u8vec3, u8vec3);"
            "bvec4 lessThan(u8vec4, u8vec4);"

            "bvec2 lessThanEqual(i8vec2, i8vec2);"
            "bvec3 lessThanEqual(i8vec3, i8vec3);"
            "bvec4 lessThanEqual(i8vec4, i8vec4);"
            "bvec2 lessThanEqual(u8vec2, u8vec2);"
            "bvec3 lessThanEqual(u8vec3, u8vec3);"
            "bvec4 lessThanEqual(u8vec4, u8vec4);"

            "bvec2 greaterThan(i8vec2, i8vec2);"
            "bvec3 greaterThan(i8vec3, i8vec3);"
            "bvec4 greaterThan(i8vec4, i8vec4);"
            "bvec2 greaterThan(u8vec2, u8vec2);"
            "bvec3 greaterThan(u8vec3, u8vec3);"
            "bvec4 greaterThan(u8vec4, u8vec4);"

            "bvec2 greaterThanEqual(i8vec2, i8vec2);"
            "bvec3 greaterThanEqual(i8vec3, i8vec3);"
            "bvec4 greaterThanEqual(i8vec4, i8vec4);"
            "bvec2 greaterThanEqual(u8vec2, u8vec2);"
            "bvec3 greaterThanEqual(u8vec3, u8vec3);"
            "bvec4 greaterThanEqual(u8vec4, u8vec4);"

            "bvec2 equal(i8vec2, i8vec2);"
            "bvec3 equal(i8vec3, i8vec3);"
            "bvec4 equal(i8vec4, i8vec4);"
            "bvec2 equal(u8vec2, u8vec2);"
            "bvec3 equal(u8vec3, u8vec3);"
            "bvec4 equal(u8vec4, u8vec4);"

            "bvec2 notEqual(i8vec2, i8vec2);"
            "bvec3 notEqual(i8vec3, i8vec3);"
            "bvec4 notEqual(i8vec4, i8vec4);"
            "bvec2 notEqual(u8vec2, u8vec2);"
            "bvec3 notEqual(u8vec3, u8vec3);"
            "bvec4 notEqual(u8vec4, u8vec4);"

            "  int8_t bitfieldExtract(  int8_t, int8_t, int8_t);"
            "i8vec2 bitfieldExtract(i8vec2, int8_t, int8_t);"
            "i8vec3 bitfieldExtract(i8vec3, int8_t, int8_t);"
            "i8vec4 bitfieldExtract(i8vec4, int8_t, int8_t);"

            " uint8_t bitfieldExtract( uint8_t, int8_t, int8_t);"
            "u8vec2 bitfieldExtract(u8vec2, int8_t, int8_t);"
            "u8vec3 bitfieldExtract(u8vec3, int8_t, int8_t);"
            "u8vec4 bitfieldExtract(u8vec4, int8_t, int8_t);"

            "  int8_t bitfieldInsert(  int8_t base,   int8_t, int8_t, int8_t);"
            "i8vec2 bitfieldInsert(i8vec2 base, i8vec2, int8_t, int8_t);"
            "i8vec3 bitfieldInsert(i8vec3 base, i8vec3, int8_t, int8_t);"
            "i8vec4 bitfieldInsert(i8vec4 base, i8vec4, int8_t, int8_t);"

            " uint8_t bitfieldInsert( uint8_t base,  uint8_t, int8_t, int8_t);"
            "u8vec2 bitfieldInsert(u8vec2 base, u8vec2, int8_t, int8_t);"
            "u8vec3 bitfieldInsert(u8vec3 base, u8vec3, int8_t, int8_t);"
            "u8vec4 bitfieldInsert(u8vec4 base, u8vec4, int8_t, int8_t);"

            "  int8_t bitCount(  int8_t);"
            "i8vec2 bitCount(i8vec2);"
            "i8vec3 bitCount(i8vec3);"
            "i8vec4 bitCount(i8vec4);"

            "  int8_t bitCount( uint8_t);"
            "i8vec2 bitCount(u8vec2);"
            "i8vec3 bitCount(u8vec3);"
            "i8vec4 bitCount(u8vec4);"

            "  int8_t findLSB(  int8_t);"
            "i8vec2 findLSB(i8vec2);"
            "i8vec3 findLSB(i8vec3);"
            "i8vec4 findLSB(i8vec4);"

            "  int8_t findLSB( uint8_t);"
            "i8vec2 findLSB(u8vec2);"
            "i8vec3 findLSB(u8vec3);"
            "i8vec4 findLSB(u8vec4);"

            "  int8_t findMSB(  int8_t);"
            "i8vec2 findMSB(i8vec2);"
            "i8vec3 findMSB(i8vec3);"
            "i8vec4 findMSB(i8vec4);"

            "  int8_t findMSB( uint8_t);"
            "i8vec2 findMSB(u8vec2);"
            "i8vec3 findMSB(u8vec3);"
            "i8vec4 findMSB(u8vec4);"

            "int16_t abs(int16_t);"
            "i16vec2 abs(i16vec2);"
            "i16vec3 abs(i16vec3);"
            "i16vec4 abs(i16vec4);"

            "int16_t sign(int16_t);"
            "i16vec2 sign(i16vec2);"
            "i16vec3 sign(i16vec3);"
            "i16vec4 sign(i16vec4);"

            "int16_t min(int16_t x, int16_t y);"
            "i16vec2 min(i16vec2 x, int16_t y);"
            "i16vec3 min(i16vec3 x, int16_t y);"
            "i16vec4 min(i16vec4 x, int16_t y);"
            "i16vec2 min(i16vec2 x, i16vec2 y);"
            "i16vec3 min(i16vec3 x, i16vec3 y);"
            "i16vec4 min(i16vec4 x, i16vec4 y);"

            "uint16_t min(uint16_t x, uint16_t y);"
            "u16vec2 min(u16vec2 x, uint16_t y);"
            "u16vec3 min(u16vec3 x, uint16_t y);"
            "u16vec4 min(u16vec4 x, uint16_t y);"
            "u16vec2 min(u16vec2 x, u16vec2 y);"
            "u16vec3 min(u16vec3 x, u16vec3 y);"
            "u16vec4 min(u16vec4 x, u16vec4 y);"

            "int16_t max(int16_t x, int16_t y);"
            "i16vec2 max(i16vec2 x, int16_t y);"
            "i16vec3 max(i16vec3 x, int16_t y);"
            "i16vec4 max(i16vec4 x, int16_t y);"
            "i16vec2 max(i16vec2 x, i16vec2 y);"
            "i16vec3 max(i16vec3 x, i16vec3 y);"
            "i16vec4 max(i16vec4 x, i16vec4 y);"

            "uint16_t max(uint16_t x, uint16_t y);"
            "u16vec2 max(u16vec2 x, uint16_t y);"
            "u16vec3 max(u16vec3 x, uint16_t y);"
            "u16vec4 max(u16vec4 x, uint16_t y);"
            "u16vec2 max(u16vec2 x, u16vec2 y);"
            "u16vec3 max(u16vec3 x, u16vec3 y);"
            "u16vec4 max(u16vec4 x, u16vec4 y);"

            "int16_t    clamp(int16_t x, int16_t milwal, int16_t maxVal);"
            "i16vec2  clamp(i16vec2  x, int16_t milwal, int16_t maxVal);"
            "i16vec3  clamp(i16vec3  x, int16_t milwal, int16_t maxVal);"
            "i16vec4  clamp(i16vec4  x, int16_t milwal, int16_t maxVal);"
            "i16vec2  clamp(i16vec2  x, i16vec2  milwal, i16vec2  maxVal);"
            "i16vec3  clamp(i16vec3  x, i16vec3  milwal, i16vec3  maxVal);"
            "i16vec4  clamp(i16vec4  x, i16vec4  milwal, i16vec4  maxVal);"

            "uint16_t   clamp(uint16_t x, uint16_t milwal, uint16_t maxVal);"
            "u16vec2  clamp(u16vec2  x, uint16_t milwal, uint16_t maxVal);"
            "u16vec3  clamp(u16vec3  x, uint16_t milwal, uint16_t maxVal);"
            "u16vec4  clamp(u16vec4  x, uint16_t milwal, uint16_t maxVal);"
            "u16vec2  clamp(u16vec2  x, u16vec2  milwal, u16vec2  maxVal);"
            "u16vec3  clamp(u16vec3  x, u16vec3  milwal, u16vec3  maxVal);"
            "u16vec4  clamp(u16vec4  x, u16vec4  milwal, u16vec4  maxVal);"

            "int16_t  mix(int16_t,  int16_t,  bool);"
            "i16vec2  mix(i16vec2,  i16vec2,  bvec2);"
            "i16vec3  mix(i16vec3,  i16vec3,  bvec3);"
            "i16vec4  mix(i16vec4,  i16vec4,  bvec4);"
            "uint16_t mix(uint16_t, uint16_t, bool);"
            "u16vec2  mix(u16vec2,  u16vec2,  bvec2);"
            "u16vec3  mix(u16vec3,  u16vec3,  bvec3);"
            "u16vec4  mix(u16vec4,  u16vec4,  bvec4);"

            "float16_t frexp(float16_t, out int16_t);"
            "f16vec2   frexp(f16vec2,   out i16vec2);"
            "f16vec3   frexp(f16vec3,   out i16vec3);"
            "f16vec4   frexp(f16vec4,   out i16vec4);"

            "float16_t ldexp(float16_t, int16_t);"
            "f16vec2   ldexp(f16vec2,   i16vec2);"
            "f16vec3   ldexp(f16vec3,   i16vec3);"
            "f16vec4   ldexp(f16vec4,   i16vec4);"

            "int16_t halfBitsToInt16(float16_t);"
            "i16vec2 halfBitsToInt16(f16vec2);"
            "i16vec3 halhBitsToInt16(f16vec3);"
            "i16vec4 halfBitsToInt16(f16vec4);"

            "uint16_t halfBitsToUint16(float16_t);"
            "u16vec2  halfBitsToUint16(f16vec2);"
            "u16vec3  halfBitsToUint16(f16vec3);"
            "u16vec4  halfBitsToUint16(f16vec4);"

            "int16_t float16BitsToInt16(float16_t);"
            "i16vec2 float16BitsToInt16(f16vec2);"
            "i16vec3 float16BitsToInt16(f16vec3);"
            "i16vec4 float16BitsToInt16(f16vec4);"

            "uint16_t float16BitsToUint16(float16_t);"
            "u16vec2  float16BitsToUint16(f16vec2);"
            "u16vec3  float16BitsToUint16(f16vec3);"
            "u16vec4  float16BitsToUint16(f16vec4);"

            "float16_t int16BitsToFloat16(int16_t);"
            "f16vec2   int16BitsToFloat16(i16vec2);"
            "f16vec3   int16BitsToFloat16(i16vec3);"
            "f16vec4   int16BitsToFloat16(i16vec4);"

            "float16_t uint16BitsToFloat16(uint16_t);"
            "f16vec2   uint16BitsToFloat16(u16vec2);"
            "f16vec3   uint16BitsToFloat16(u16vec3);"
            "f16vec4   uint16BitsToFloat16(u16vec4);"

            "float16_t int16BitsToHalf(int16_t);"
            "f16vec2   int16BitsToHalf(i16vec2);"
            "f16vec3   int16BitsToHalf(i16vec3);"
            "f16vec4   int16BitsToHalf(i16vec4);"

            "float16_t uint16BitsToHalf(uint16_t);"
            "f16vec2   uint16BitsToHalf(u16vec2);"
            "f16vec3   uint16BitsToHalf(u16vec3);"
            "f16vec4   uint16BitsToHalf(u16vec4);"

            "int      packInt2x16(i16vec2);"
            "uint     packUint2x16(u16vec2);"
            "int64_t  packInt4x16(i16vec4);"
            "uint64_t packUint4x16(u16vec4);"
            "i16vec2  unpackInt2x16(int);"
            "u16vec2  unpackUint2x16(uint);"
            "i16vec4  unpackInt4x16(int64_t);"
            "u16vec4  unpackUint4x16(uint64_t);"

            "bvec2 lessThan(i16vec2, i16vec2);"
            "bvec3 lessThan(i16vec3, i16vec3);"
            "bvec4 lessThan(i16vec4, i16vec4);"
            "bvec2 lessThan(u16vec2, u16vec2);"
            "bvec3 lessThan(u16vec3, u16vec3);"
            "bvec4 lessThan(u16vec4, u16vec4);"

            "bvec2 lessThanEqual(i16vec2, i16vec2);"
            "bvec3 lessThanEqual(i16vec3, i16vec3);"
            "bvec4 lessThanEqual(i16vec4, i16vec4);"
            "bvec2 lessThanEqual(u16vec2, u16vec2);"
            "bvec3 lessThanEqual(u16vec3, u16vec3);"
            "bvec4 lessThanEqual(u16vec4, u16vec4);"

            "bvec2 greaterThan(i16vec2, i16vec2);"
            "bvec3 greaterThan(i16vec3, i16vec3);"
            "bvec4 greaterThan(i16vec4, i16vec4);"
            "bvec2 greaterThan(u16vec2, u16vec2);"
            "bvec3 greaterThan(u16vec3, u16vec3);"
            "bvec4 greaterThan(u16vec4, u16vec4);"

            "bvec2 greaterThanEqual(i16vec2, i16vec2);"
            "bvec3 greaterThanEqual(i16vec3, i16vec3);"
            "bvec4 greaterThanEqual(i16vec4, i16vec4);"
            "bvec2 greaterThanEqual(u16vec2, u16vec2);"
            "bvec3 greaterThanEqual(u16vec3, u16vec3);"
            "bvec4 greaterThanEqual(u16vec4, u16vec4);"

            "bvec2 equal(i16vec2, i16vec2);"
            "bvec3 equal(i16vec3, i16vec3);"
            "bvec4 equal(i16vec4, i16vec4);"
            "bvec2 equal(u16vec2, u16vec2);"
            "bvec3 equal(u16vec3, u16vec3);"
            "bvec4 equal(u16vec4, u16vec4);"

            "bvec2 notEqual(i16vec2, i16vec2);"
            "bvec3 notEqual(i16vec3, i16vec3);"
            "bvec4 notEqual(i16vec4, i16vec4);"
            "bvec2 notEqual(u16vec2, u16vec2);"
            "bvec3 notEqual(u16vec3, u16vec3);"
            "bvec4 notEqual(u16vec4, u16vec4);"

            "  int16_t bitfieldExtract(  int16_t, int16_t, int16_t);"
            "i16vec2 bitfieldExtract(i16vec2, int16_t, int16_t);"
            "i16vec3 bitfieldExtract(i16vec3, int16_t, int16_t);"
            "i16vec4 bitfieldExtract(i16vec4, int16_t, int16_t);"

            " uint16_t bitfieldExtract( uint16_t, int16_t, int16_t);"
            "u16vec2 bitfieldExtract(u16vec2, int16_t, int16_t);"
            "u16vec3 bitfieldExtract(u16vec3, int16_t, int16_t);"
            "u16vec4 bitfieldExtract(u16vec4, int16_t, int16_t);"

            "  int16_t bitfieldInsert(  int16_t base,   int16_t, int16_t, int16_t);"
            "i16vec2 bitfieldInsert(i16vec2 base, i16vec2, int16_t, int16_t);"
            "i16vec3 bitfieldInsert(i16vec3 base, i16vec3, int16_t, int16_t);"
            "i16vec4 bitfieldInsert(i16vec4 base, i16vec4, int16_t, int16_t);"

            " uint16_t bitfieldInsert( uint16_t base,  uint16_t, int16_t, int16_t);"
            "u16vec2 bitfieldInsert(u16vec2 base, u16vec2, int16_t, int16_t);"
            "u16vec3 bitfieldInsert(u16vec3 base, u16vec3, int16_t, int16_t);"
            "u16vec4 bitfieldInsert(u16vec4 base, u16vec4, int16_t, int16_t);"

            "  int16_t bitCount(  int16_t);"
            "i16vec2 bitCount(i16vec2);"
            "i16vec3 bitCount(i16vec3);"
            "i16vec4 bitCount(i16vec4);"

            "  int16_t bitCount( uint16_t);"
            "i16vec2 bitCount(u16vec2);"
            "i16vec3 bitCount(u16vec3);"
            "i16vec4 bitCount(u16vec4);"

            "  int16_t findLSB(  int16_t);"
            "i16vec2 findLSB(i16vec2);"
            "i16vec3 findLSB(i16vec3);"
            "i16vec4 findLSB(i16vec4);"

            "  int16_t findLSB( uint16_t);"
            "i16vec2 findLSB(u16vec2);"
            "i16vec3 findLSB(u16vec3);"
            "i16vec4 findLSB(u16vec4);"

            "  int16_t findMSB(  int16_t);"
            "i16vec2 findMSB(i16vec2);"
            "i16vec3 findMSB(i16vec3);"
            "i16vec4 findMSB(i16vec4);"

            "  int16_t findMSB( uint16_t);"
            "i16vec2 findMSB(u16vec2);"
            "i16vec3 findMSB(u16vec3);"
            "i16vec4 findMSB(u16vec4);"

            "int16_t  pack16(i8vec2);"
            "uint16_t pack16(u8vec2);"
            "int32_t  pack32(i8vec4);"
            "uint32_t pack32(u8vec4);"
            "int32_t  pack32(i16vec2);"
            "uint32_t pack32(u16vec2);"
            "int64_t  pack64(i16vec4);"
            "uint64_t pack64(u16vec4);"
            "int64_t  pack64(i32vec2);"
            "uint64_t pack64(u32vec2);"

            "i8vec2   unpack8(int16_t);"
            "u8vec2   unpack8(uint16_t);"
            "i8vec4   unpack8(int32_t);"
            "u8vec4   unpack8(uint32_t);"
            "i16vec2  unpack16(int32_t);"
            "u16vec2  unpack16(uint32_t);"
            "i16vec4  unpack16(int64_t);"
            "u16vec4  unpack16(uint64_t);"
            "i32vec2  unpack32(int64_t);"
            "u32vec2  unpack32(uint64_t);"

            "float64_t radians(float64_t);"
            "f64vec2   radians(f64vec2);"
            "f64vec3   radians(f64vec3);"
            "f64vec4   radians(f64vec4);"

            "float64_t degrees(float64_t);"
            "f64vec2   degrees(f64vec2);"
            "f64vec3   degrees(f64vec3);"
            "f64vec4   degrees(f64vec4);"

            "float64_t sin(float64_t);"
            "f64vec2   sin(f64vec2);"
            "f64vec3   sin(f64vec3);"
            "f64vec4   sin(f64vec4);"

            "float64_t cos(float64_t);"
            "f64vec2   cos(f64vec2);"
            "f64vec3   cos(f64vec3);"
            "f64vec4   cos(f64vec4);"

            "float64_t tan(float64_t);"
            "f64vec2   tan(f64vec2);"
            "f64vec3   tan(f64vec3);"
            "f64vec4   tan(f64vec4);"

            "float64_t asin(float64_t);"
            "f64vec2   asin(f64vec2);"
            "f64vec3   asin(f64vec3);"
            "f64vec4   asin(f64vec4);"

            "float64_t acos(float64_t);"
            "f64vec2   acos(f64vec2);"
            "f64vec3   acos(f64vec3);"
            "f64vec4   acos(f64vec4);"

            "float64_t atan(float64_t, float64_t);"
            "f64vec2   atan(f64vec2,   f64vec2);"
            "f64vec3   atan(f64vec3,   f64vec3);"
            "f64vec4   atan(f64vec4,   f64vec4);"

            "float64_t atan(float64_t);"
            "f64vec2   atan(f64vec2);"
            "f64vec3   atan(f64vec3);"
            "f64vec4   atan(f64vec4);"

            "float64_t sinh(float64_t);"
            "f64vec2   sinh(f64vec2);"
            "f64vec3   sinh(f64vec3);"
            "f64vec4   sinh(f64vec4);"

            "float64_t cosh(float64_t);"
            "f64vec2   cosh(f64vec2);"
            "f64vec3   cosh(f64vec3);"
            "f64vec4   cosh(f64vec4);"

            "float64_t tanh(float64_t);"
            "f64vec2   tanh(f64vec2);"
            "f64vec3   tanh(f64vec3);"
            "f64vec4   tanh(f64vec4);"

            "float64_t asinh(float64_t);"
            "f64vec2   asinh(f64vec2);"
            "f64vec3   asinh(f64vec3);"
            "f64vec4   asinh(f64vec4);"

            "float64_t acosh(float64_t);"
            "f64vec2   acosh(f64vec2);"
            "f64vec3   acosh(f64vec3);"
            "f64vec4   acosh(f64vec4);"

            "float64_t atanh(float64_t);"
            "f64vec2   atanh(f64vec2);"
            "f64vec3   atanh(f64vec3);"
            "f64vec4   atanh(f64vec4);"

            "float64_t pow(float64_t, float64_t);"
            "f64vec2   pow(f64vec2,   f64vec2);"
            "f64vec3   pow(f64vec3,   f64vec3);"
            "f64vec4   pow(f64vec4,   f64vec4);"

            "float64_t exp(float64_t);"
            "f64vec2   exp(f64vec2);"
            "f64vec3   exp(f64vec3);"
            "f64vec4   exp(f64vec4);"

            "float64_t log(float64_t);"
            "f64vec2   log(f64vec2);"
            "f64vec3   log(f64vec3);"
            "f64vec4   log(f64vec4);"

            "float64_t exp2(float64_t);"
            "f64vec2   exp2(f64vec2);"
            "f64vec3   exp2(f64vec3);"
            "f64vec4   exp2(f64vec4);"

            "float64_t log2(float64_t);"
            "f64vec2   log2(f64vec2);"
            "f64vec3   log2(f64vec3);"
            "f64vec4   log2(f64vec4);"
            "\n");
        }
        if (profile != EEsProfile && version >= 450) {
            stageBuiltins[EShLangFragment].append(derivativesAndControl64bits);
            stageBuiltins[EShLangFragment].append(
                "float64_t interpolateAtCentroid(float64_t);"
                "f64vec2   interpolateAtCentroid(f64vec2);"
                "f64vec3   interpolateAtCentroid(f64vec3);"
                "f64vec4   interpolateAtCentroid(f64vec4);"

                "float64_t interpolateAtSample(float64_t, int);"
                "f64vec2   interpolateAtSample(f64vec2,   int);"
                "f64vec3   interpolateAtSample(f64vec3,   int);"
                "f64vec4   interpolateAtSample(f64vec4,   int);"

                "float64_t interpolateAtOffset(float64_t, f64vec2);"
                "f64vec2   interpolateAtOffset(f64vec2,   f64vec2);"
                "f64vec3   interpolateAtOffset(f64vec3,   f64vec2);"
                "f64vec4   interpolateAtOffset(f64vec4,   f64vec2);"

                "\n");

    }

    //============================================================================
    //
    // Prototypes for built-in functions seen by vertex shaders only.
    // (Except legacy lod functions, where it depends which release they are
    // vertex only.)
    //
    //============================================================================

    //
    // Geometric Functions.
    //
    if (spvVersion.vulkan == 0 && IncludeLegacy(version, profile, spvVersion))
        stageBuiltins[EShLangVertex].append("vec4 ftransform();");

    //
    // Original-style texture Functions with lod.
    //
    TString* s;
    if (version == 100)
        s = &stageBuiltins[EShLangVertex];
    else
        s = &commonBuiltins;
    if ((profile == EEsProfile && version == 100) ||
         profile == ECompatibilityProfile ||
        (profile == ECoreProfile && version < 420) ||
         profile == ENoProfile) {
        if (spvVersion.spv == 0) {
            s->append(
                "vec4 texture2DLod(sampler2D, vec2, float);"         // GL_ARB_shader_texture_lod
                "vec4 texture2DProjLod(sampler2D, vec3, float);"     // GL_ARB_shader_texture_lod
                "vec4 texture2DProjLod(sampler2D, vec4, float);"     // GL_ARB_shader_texture_lod
                "vec4 texture3DLod(sampler3D, vec3, float);"         // GL_ARB_shader_texture_lod  // OES_texture_3D, but caught by keyword check
                "vec4 texture3DProjLod(sampler3D, vec4, float);"     // GL_ARB_shader_texture_lod  // OES_texture_3D, but caught by keyword check
                "vec4 textureLwbeLod(samplerLwbe, vec3, float);"     // GL_ARB_shader_texture_lod

                "\n");
        }
    }
    if ( profile == ECompatibilityProfile ||
        (profile == ECoreProfile && version < 420) ||
         profile == ENoProfile) {
        if (spvVersion.spv == 0) {
            s->append(
                "vec4 texture1DLod(sampler1D, float, float);"                          // GL_ARB_shader_texture_lod
                "vec4 texture1DProjLod(sampler1D, vec2, float);"                       // GL_ARB_shader_texture_lod
                "vec4 texture1DProjLod(sampler1D, vec4, float);"                       // GL_ARB_shader_texture_lod
                "vec4 shadow1DLod(sampler1DShadow, vec3, float);"                      // GL_ARB_shader_texture_lod
                "vec4 shadow2DLod(sampler2DShadow, vec3, float);"                      // GL_ARB_shader_texture_lod
                "vec4 shadow1DProjLod(sampler1DShadow, vec4, float);"                  // GL_ARB_shader_texture_lod
                "vec4 shadow2DProjLod(sampler2DShadow, vec4, float);"                  // GL_ARB_shader_texture_lod

                "vec4 texture1DGradARB(sampler1D, float, float, float);"               // GL_ARB_shader_texture_lod
                "vec4 texture1DProjGradARB(sampler1D, vec2, float, float);"            // GL_ARB_shader_texture_lod
                "vec4 texture1DProjGradARB(sampler1D, vec4, float, float);"            // GL_ARB_shader_texture_lod
                "vec4 texture2DGradARB(sampler2D, vec2, vec2, vec2);"                  // GL_ARB_shader_texture_lod
                "vec4 texture2DProjGradARB(sampler2D, vec3, vec2, vec2);"              // GL_ARB_shader_texture_lod
                "vec4 texture2DProjGradARB(sampler2D, vec4, vec2, vec2);"              // GL_ARB_shader_texture_lod
                "vec4 texture3DGradARB(sampler3D, vec3, vec3, vec3);"                  // GL_ARB_shader_texture_lod
                "vec4 texture3DProjGradARB(sampler3D, vec4, vec3, vec3);"              // GL_ARB_shader_texture_lod
                "vec4 textureLwbeGradARB(samplerLwbe, vec3, vec3, vec3);"              // GL_ARB_shader_texture_lod
                "vec4 shadow1DGradARB(sampler1DShadow, vec3, float, float);"           // GL_ARB_shader_texture_lod
                "vec4 shadow1DProjGradARB( sampler1DShadow, vec4, float, float);"      // GL_ARB_shader_texture_lod
                "vec4 shadow2DGradARB(sampler2DShadow, vec3, vec2, vec2);"             // GL_ARB_shader_texture_lod
                "vec4 shadow2DProjGradARB( sampler2DShadow, vec4, vec2, vec2);"        // GL_ARB_shader_texture_lod
                "vec4 texture2DRectGradARB(sampler2DRect, vec2, vec2, vec2);"          // GL_ARB_shader_texture_lod
                "vec4 texture2DRectProjGradARB( sampler2DRect, vec3, vec2, vec2);"     // GL_ARB_shader_texture_lod
                "vec4 texture2DRectProjGradARB( sampler2DRect, vec4, vec2, vec2);"     // GL_ARB_shader_texture_lod
                "vec4 shadow2DRectGradARB( sampler2DRectShadow, vec3, vec2, vec2);"    // GL_ARB_shader_texture_lod
                "vec4 shadow2DRectProjGradARB(sampler2DRectShadow, vec4, vec2, vec2);" // GL_ARB_shader_texture_lod

                "\n");
        }
    }

    if ((profile != EEsProfile && version >= 150) ||
        (profile == EEsProfile && version >= 310)) {
        //============================================================================
        //
        // Prototypes for built-in functions seen by geometry shaders only.
        //
        //============================================================================

        if (profile != EEsProfile && version >= 400) {
            stageBuiltins[EShLangGeometry].append(
                "void EmitStreamVertex(int);"
                "void EndStreamPrimitive(int);"
                );
        }
        stageBuiltins[EShLangGeometry].append(
            "void EmitVertex();"
            "void EndPrimitive();"
            "\n");
    }
#endif

    //============================================================================
    //
    // Prototypes for all control functions.
    //
    //============================================================================
    bool esBarrier = (profile == EEsProfile && version >= 310);
    if ((profile != EEsProfile && version >= 150) || esBarrier)
        stageBuiltins[EShLangTessControl].append(
            "void barrier();"
            );
    if ((profile != EEsProfile && version >= 420) || esBarrier)
        stageBuiltins[EShLangCompute].append(
            "void barrier();"
            );
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        stageBuiltins[EShLangMeshLW].append(
            "void barrier();"
            );
        stageBuiltins[EShLangTaskLW].append(
            "void barrier();"
            );
    }
    if ((profile != EEsProfile && version >= 130) || esBarrier)
        commonBuiltins.append(
            "void memoryBarrier();"
            );
    if ((profile != EEsProfile && version >= 420) || esBarrier) {
        commonBuiltins.append(
            "void memoryBarrierBuffer();"
            );
        stageBuiltins[EShLangCompute].append(
            "void memoryBarrierShared();"
            "void groupMemoryBarrier();"
            );
    }
#ifndef GLSLANG_WEB
    if ((profile != EEsProfile && version >= 420) || esBarrier) {
        if (spvVersion.vulkan == 0) {
            commonBuiltins.append("void memoryBarrierAtomicCounter();");
        }
        commonBuiltins.append("void memoryBarrierImage();");
    }
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        stageBuiltins[EShLangMeshLW].append(
            "void memoryBarrierShared();"
            "void groupMemoryBarrier();"
        );
        stageBuiltins[EShLangTaskLW].append(
            "void memoryBarrierShared();"
            "void groupMemoryBarrier();"
        );
    }

    commonBuiltins.append("void controlBarrier(int, int, int, int);\n"
                          "void memoryBarrier(int, int, int);\n");

    commonBuiltins.append("void debugPrintfEXT();\n");

    if (profile != EEsProfile && version >= 450) {
        // coopMatStoreLW perhaps ought to have "out" on the buf parameter, but
        // adding it introduces undesirable tempArgs on the stack. What we want
        // is more like "buf" thought of as a pointer value being an in parameter.
        stageBuiltins[EShLangCompute].append(
            "void coopMatLoadLW(out fcoopmatLW m, volatile coherent float16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out fcoopmatLW m, volatile coherent float[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out fcoopmatLW m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out fcoopmatLW m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out fcoopmatLW m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out fcoopmatLW m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out fcoopmatLW m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out fcoopmatLW m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "void coopMatStoreLW(fcoopmatLW m, volatile coherent float16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(fcoopmatLW m, volatile coherent float[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(fcoopmatLW m, volatile coherent float64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(fcoopmatLW m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(fcoopmatLW m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(fcoopmatLW m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(fcoopmatLW m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(fcoopmatLW m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(fcoopmatLW m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "fcoopmatLW coopMatMulAddLW(fcoopmatLW A, fcoopmatLW B, fcoopmatLW C);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent int8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent int16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent int[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent int64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent ivec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent ivec4[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out icoopmatLW m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent int8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent int16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent int[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent int64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent ivec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent ivec4[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatLoadLW(out ucoopmatLW m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "void coopMatStoreLW(icoopmatLW m, volatile coherent int8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent int16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent int[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent int64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent ivec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent ivec4[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(icoopmatLW m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "void coopMatStoreLW(ucoopmatLW m, volatile coherent int8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent int16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent int[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent int64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent ivec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent ivec4[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent uint8_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent uint16_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent uint[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent uint64_t[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent uvec2[] buf, uint element, uint stride, bool colMajor);\n"
            "void coopMatStoreLW(ucoopmatLW m, volatile coherent uvec4[] buf, uint element, uint stride, bool colMajor);\n"

            "icoopmatLW coopMatMulAddLW(icoopmatLW A, icoopmatLW B, icoopmatLW C);\n"
            "ucoopmatLW coopMatMulAddLW(ucoopmatLW A, ucoopmatLW B, ucoopmatLW C);\n"
            );
    }

    //============================================================================
    //
    // Prototypes for built-in functions seen by fragment shaders only.
    //
    //============================================================================

    //
    // Original-style texture Functions with bias.
    //
    if (spvVersion.spv == 0 && (profile != EEsProfile || version == 100)) {
        stageBuiltins[EShLangFragment].append(
            "vec4 texture2D(sampler2D, vec2, float);"
            "vec4 texture2DProj(sampler2D, vec3, float);"
            "vec4 texture2DProj(sampler2D, vec4, float);"
            "vec4 texture3D(sampler3D, vec3, float);"        // OES_texture_3D
            "vec4 texture3DProj(sampler3D, vec4, float);"    // OES_texture_3D
            "vec4 textureLwbe(samplerLwbe, vec3, float);"

            "\n");
    }
    if (spvVersion.spv == 0 && (profile != EEsProfile && version > 100)) {
        stageBuiltins[EShLangFragment].append(
            "vec4 texture1D(sampler1D, float, float);"
            "vec4 texture1DProj(sampler1D, vec2, float);"
            "vec4 texture1DProj(sampler1D, vec4, float);"
            "vec4 shadow1D(sampler1DShadow, vec3, float);"
            "vec4 shadow2D(sampler2DShadow, vec3, float);"
            "vec4 shadow1DProj(sampler1DShadow, vec4, float);"
            "vec4 shadow2DProj(sampler2DShadow, vec4, float);"

            "\n");
    }
    if (spvVersion.spv == 0 && profile == EEsProfile) {
        stageBuiltins[EShLangFragment].append(
            "vec4 texture2DLodEXT(sampler2D, vec2, float);"      // GL_EXT_shader_texture_lod
            "vec4 texture2DProjLodEXT(sampler2D, vec3, float);"  // GL_EXT_shader_texture_lod
            "vec4 texture2DProjLodEXT(sampler2D, vec4, float);"  // GL_EXT_shader_texture_lod
            "vec4 textureLwbeLodEXT(samplerLwbe, vec3, float);"  // GL_EXT_shader_texture_lod

            "\n");
    }

    // GL_ARB_derivative_control
    if (profile != EEsProfile && version >= 400) {
        stageBuiltins[EShLangFragment].append(derivativeControls);
        stageBuiltins[EShLangFragment].append("\n");
    }

    // GL_OES_shader_multisample_interpolation
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 400)) {
        stageBuiltins[EShLangFragment].append(
            "float interpolateAtCentroid(float);"
            "vec2  interpolateAtCentroid(vec2);"
            "vec3  interpolateAtCentroid(vec3);"
            "vec4  interpolateAtCentroid(vec4);"

            "float interpolateAtSample(float, int);"
            "vec2  interpolateAtSample(vec2,  int);"
            "vec3  interpolateAtSample(vec3,  int);"
            "vec4  interpolateAtSample(vec4,  int);"

            "float interpolateAtOffset(float, vec2);"
            "vec2  interpolateAtOffset(vec2,  vec2);"
            "vec3  interpolateAtOffset(vec3,  vec2);"
            "vec4  interpolateAtOffset(vec4,  vec2);"

            "\n");
    }

    stageBuiltins[EShLangFragment].append(
        "void beginIlwocationInterlockARB(void);"
        "void endIlwocationInterlockARB(void);");

    stageBuiltins[EShLangFragment].append(
        "bool helperIlwocationEXT();"
        "\n");

    // GL_AMD_shader_explicit_vertex_parameter
    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangFragment].append(
            "float interpolateAtVertexAMD(float, uint);"
            "vec2  interpolateAtVertexAMD(vec2,  uint);"
            "vec3  interpolateAtVertexAMD(vec3,  uint);"
            "vec4  interpolateAtVertexAMD(vec4,  uint);"

            "int   interpolateAtVertexAMD(int,   uint);"
            "ivec2 interpolateAtVertexAMD(ivec2, uint);"
            "ivec3 interpolateAtVertexAMD(ivec3, uint);"
            "ivec4 interpolateAtVertexAMD(ivec4, uint);"

            "uint  interpolateAtVertexAMD(uint,  uint);"
            "uvec2 interpolateAtVertexAMD(uvec2, uint);"
            "uvec3 interpolateAtVertexAMD(uvec3, uint);"
            "uvec4 interpolateAtVertexAMD(uvec4, uint);"

            "float16_t interpolateAtVertexAMD(float16_t, uint);"
            "f16vec2   interpolateAtVertexAMD(f16vec2,   uint);"
            "f16vec3   interpolateAtVertexAMD(f16vec3,   uint);"
            "f16vec4   interpolateAtVertexAMD(f16vec4,   uint);"

            "\n");
    }

    // GL_AMD_gpu_shader_half_float
    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangFragment].append(derivativesAndControl16bits);
        stageBuiltins[EShLangFragment].append("\n");

        stageBuiltins[EShLangFragment].append(
            "float16_t interpolateAtCentroid(float16_t);"
            "f16vec2   interpolateAtCentroid(f16vec2);"
            "f16vec3   interpolateAtCentroid(f16vec3);"
            "f16vec4   interpolateAtCentroid(f16vec4);"

            "float16_t interpolateAtSample(float16_t, int);"
            "f16vec2   interpolateAtSample(f16vec2,   int);"
            "f16vec3   interpolateAtSample(f16vec3,   int);"
            "f16vec4   interpolateAtSample(f16vec4,   int);"

            "float16_t interpolateAtOffset(float16_t, f16vec2);"
            "f16vec2   interpolateAtOffset(f16vec2,   f16vec2);"
            "f16vec3   interpolateAtOffset(f16vec3,   f16vec2);"
            "f16vec4   interpolateAtOffset(f16vec4,   f16vec2);"

            "\n");
    }

    // GL_ARB_shader_clock & GL_EXT_shader_realtime_clock
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append(
            "uvec2 clock2x32ARB();"
            "uint64_t clockARB();"
            "uvec2 clockRealtime2x32EXT();"
            "uint64_t clockRealtimeEXT();"
            "\n");
    }

    // GL_AMD_shader_fragment_mask
    if (profile != EEsProfile && version >= 450 && spvVersion.vulkan > 0) {
        stageBuiltins[EShLangFragment].append(
            "uint fragmentMaskFetchAMD(subpassInputMS);"
            "uint fragmentMaskFetchAMD(isubpassInputMS);"
            "uint fragmentMaskFetchAMD(usubpassInputMS);"

            "vec4  fragmentFetchAMD(subpassInputMS,  uint);"
            "ivec4 fragmentFetchAMD(isubpassInputMS, uint);"
            "uvec4 fragmentFetchAMD(usubpassInputMS, uint);"

            "\n");
        }

    // Builtins for GL_LW_ray_tracing/GL_EXT_ray_tracing/GL_EXT_ray_query
    if (profile != EEsProfile && version >= 460) {
         commonBuiltins.append("void rayQueryInitializeEXT(rayQueryEXT, accelerationStructureEXT, uint, uint, vec3, float, vec3, float);"
            "void rayQueryTerminateEXT(rayQueryEXT);"
            "void rayQueryGenerateIntersectionEXT(rayQueryEXT, float);"
            "void rayQueryConfirmIntersectionEXT(rayQueryEXT);"
            "bool rayQueryProceedEXT(rayQueryEXT);"
            "uint rayQueryGetIntersectionTypeEXT(rayQueryEXT, bool);"
            "float rayQueryGetRayTMinEXT(rayQueryEXT);"
            "uint rayQueryGetRayFlagsEXT(rayQueryEXT);"
            "vec3 rayQueryGetWorldRayOriginEXT(rayQueryEXT);"
            "vec3 rayQueryGetWorldRayDirectionEXT(rayQueryEXT);"
            "float rayQueryGetIntersectionTEXT(rayQueryEXT, bool);"
            "int rayQueryGetIntersectionInstanceLwstomIndexEXT(rayQueryEXT, bool);"
            "int rayQueryGetIntersectionInstanceIdEXT(rayQueryEXT, bool);"
            "uint rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(rayQueryEXT, bool);"
            "int rayQueryGetIntersectionGeometryIndexEXT(rayQueryEXT, bool);"
            "int rayQueryGetIntersectionPrimitiveIndexEXT(rayQueryEXT, bool);"
            "vec2 rayQueryGetIntersectionBarycentricsEXT(rayQueryEXT, bool);"
            "bool rayQueryGetIntersectionFrontFaceEXT(rayQueryEXT, bool);"
            "bool rayQueryGetIntersectionCandidateAABBOpaqueEXT(rayQueryEXT);"
            "vec3 rayQueryGetIntersectionObjectRayDirectionEXT(rayQueryEXT, bool);"
            "vec3 rayQueryGetIntersectionObjectRayOriginEXT(rayQueryEXT, bool);"
            "mat4x3 rayQueryGetIntersectionObjectToWorldEXT(rayQueryEXT, bool);"
            "mat4x3 rayQueryGetIntersectionWorldToObjectEXT(rayQueryEXT, bool);"
            "\n");

        stageBuiltins[EShLangRayGen].append(
            "void traceLW(accelerationStructureLW,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void traceRayEXT(accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void exelwteCallableLW(uint, int);"
            "void exelwteCallableEXT(uint, int);"
            "\n");
        stageBuiltins[EShLangIntersect].append(
            "bool reportIntersectionLW(float, uint);"
            "bool reportIntersectionEXT(float, uint);"
            "\n");
        stageBuiltins[EShLangAnyHit].append(
            "void ignoreIntersectionLW();"
            "void ignoreIntersectionEXT();"
            "void terminateRayLW();"
            "void terminateRayEXT();"
            "\n");
        stageBuiltins[EShLangClosestHit].append(
            "void traceLW(accelerationStructureLW,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void traceRayEXT(accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void exelwteCallableLW(uint, int);"
            "void exelwteCallableEXT(uint, int);"
            "\n");
        stageBuiltins[EShLangMiss].append(
            "void traceLW(accelerationStructureLW,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void traceRayEXT(accelerationStructureEXT,uint,uint,uint,uint,uint,vec3,float,vec3,float,int);"
            "void exelwteCallableLW(uint, int);"
            "void exelwteCallableEXT(uint, int);"
            "\n");
        stageBuiltins[EShLangCallable].append(
            "void exelwteCallableLW(uint, int);"
            "void exelwteCallableEXT(uint, int);"
            "\n");
    }

    //E_SPV_LW_compute_shader_derivatives
    if ((profile == EEsProfile && version >= 320) || (profile != EEsProfile && version >= 450)) {
        stageBuiltins[EShLangCompute].append(derivativeControls);
        stageBuiltins[EShLangCompute].append("\n");
    }
    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangCompute].append(derivativesAndControl16bits);
        stageBuiltins[EShLangCompute].append(derivativesAndControl64bits);
        stageBuiltins[EShLangCompute].append("\n");
    }

    // Builtins for GL_LW_mesh_shader
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        stageBuiltins[EShLangMeshLW].append(
            "void writePackedPrimitiveIndices4x8LW(uint, uint);"
            "\n");
    }
#endif

    //============================================================================
    //
    // Standard Uniforms
    //
    //============================================================================

    //
    // Depth range in window coordinates, p. 33
    //
    if (spvVersion.spv == 0) {
        commonBuiltins.append(
            "struct gl_DepthRangeParameters {"
            );
        if (profile == EEsProfile) {
            commonBuiltins.append(
                "highp float near;"   // n
                "highp float far;"    // f
                "highp float diff;"   // f - n
                );
        } else {
#ifndef GLSLANG_WEB
            commonBuiltins.append(
                "float near;"  // n
                "float far;"   // f
                "float diff;"  // f - n
                );
#endif
        }

        commonBuiltins.append(
            "};"
            "uniform gl_DepthRangeParameters gl_DepthRange;"
            "\n");
    }

#ifndef GLSLANG_WEB
    if (spvVersion.spv == 0 && IncludeLegacy(version, profile, spvVersion)) {
        //
        // Matrix state. p. 31, 32, 37, 39, 40.
        //
        commonBuiltins.append(
            "uniform mat4  gl_ModelViewMatrix;"
            "uniform mat4  gl_ProjectionMatrix;"
            "uniform mat4  gl_ModelViewProjectionMatrix;"

            //
            // Derived matrix state that provides ilwerse and transposed versions
            // of the matrices above.
            //
            "uniform mat3  gl_NormalMatrix;"

            "uniform mat4  gl_ModelViewMatrixIlwerse;"
            "uniform mat4  gl_ProjectionMatrixIlwerse;"
            "uniform mat4  gl_ModelViewProjectionMatrixIlwerse;"

            "uniform mat4  gl_ModelViewMatrixTranspose;"
            "uniform mat4  gl_ProjectionMatrixTranspose;"
            "uniform mat4  gl_ModelViewProjectionMatrixTranspose;"

            "uniform mat4  gl_ModelViewMatrixIlwerseTranspose;"
            "uniform mat4  gl_ProjectionMatrixIlwerseTranspose;"
            "uniform mat4  gl_ModelViewProjectionMatrixIlwerseTranspose;"

            //
            // Normal scaling p. 39.
            //
            "uniform float gl_NormalScale;"

            //
            // Point Size, p. 66, 67.
            //
            "struct gl_PointParameters {"
                "float size;"
                "float sizeMin;"
                "float sizeMax;"
                "float fadeThresholdSize;"
                "float distanceConstantAttenuation;"
                "float distanceLinearAttenuation;"
                "float distanceQuadraticAttenuation;"
            "};"

            "uniform gl_PointParameters gl_Point;"

            //
            // Material State p. 50, 55.
            //
            "struct gl_MaterialParameters {"
                "vec4  emission;"    // Ecm
                "vec4  ambient;"     // Acm
                "vec4  diffuse;"     // Dcm
                "vec4  spelwlar;"    // Scm
                "float shininess;"   // Srm
            "};"
            "uniform gl_MaterialParameters  gl_FrontMaterial;"
            "uniform gl_MaterialParameters  gl_BackMaterial;"

            //
            // Light State p 50, 53, 55.
            //
            "struct gl_LightSourceParameters {"
                "vec4  ambient;"             // Acli
                "vec4  diffuse;"             // Dcli
                "vec4  spelwlar;"            // Scli
                "vec4  position;"            // Ppli
                "vec4  halfVector;"          // Derived: Hi
                "vec3  spotDirection;"       // Sdli
                "float spotExponent;"        // Srli
                "float spotLwtoff;"          // Crli
                                                        // (range: [0.0,90.0], 180.0)
                "float spotCosLwtoff;"       // Derived: cos(Crli)
                                                        // (range: [1.0,0.0],-1.0)
                "float constantAttenuation;" // K0
                "float linearAttenuation;"   // K1
                "float quadraticAttenuation;"// K2
            "};"

            "struct gl_LightModelParameters {"
                "vec4  ambient;"       // Acs
            "};"

            "uniform gl_LightModelParameters  gl_LightModel;"

            //
            // Derived state from products of light and material.
            //
            "struct gl_LightModelProducts {"
                "vec4  sceneColor;"     // Derived. Ecm + Acm * Acs
            "};"

            "uniform gl_LightModelProducts gl_FrontLightModelProduct;"
            "uniform gl_LightModelProducts gl_BackLightModelProduct;"

            "struct gl_LightProducts {"
                "vec4  ambient;"        // Acm * Acli
                "vec4  diffuse;"        // Dcm * Dcli
                "vec4  spelwlar;"       // Scm * Scli
            "};"

            //
            // Fog p. 161
            //
            "struct gl_FogParameters {"
                "vec4  color;"
                "float density;"
                "float start;"
                "float end;"
                "float scale;"   //  1 / (gl_FogEnd - gl_FogStart)
            "};"

            "uniform gl_FogParameters gl_Fog;"

            "\n");
    }
#endif

    //============================================================================
    //
    // Define the interface to the compute shader.
    //
    //============================================================================

    if ((profile != EEsProfile && version >= 420) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangCompute].append(
            "in    highp uvec3 gl_NumWorkGroups;"
            "const highp uvec3 gl_WorkGroupSize = uvec3(1,1,1);"

            "in highp uvec3 gl_WorkGroupID;"
            "in highp uvec3 gl_LocalIlwocationID;"

            "in highp uvec3 gl_GlobalIlwocationID;"
            "in highp uint gl_LocalIlwocationIndex;"

            "\n");
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangCompute].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "\n");
    }

#ifndef GLSLANG_WEB
    //============================================================================
    //
    // Define the interface to the mesh/task shader.
    //
    //============================================================================

    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        // per-vertex attributes
        stageBuiltins[EShLangMeshLW].append(
            "out gl_MeshPerVertexLW {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                "float gl_LwllDistance[];"
                "perviewLW vec4 gl_PositionPerViewLW[];"
                "perviewLW float gl_ClipDistancePerViewLW[][];"
                "perviewLW float gl_LwllDistancePerViewLW[][];"
            "} gl_MeshVerticesLW[];"
        );

        // per-primitive attributes
        stageBuiltins[EShLangMeshLW].append(
            "perprimitiveLW out gl_MeshPerPrimitiveLW {"
                "int gl_PrimitiveID;"
                "int gl_Layer;"
                "int gl_ViewportIndex;"
                "int gl_ViewportMask[];"
                "perviewLW int gl_LayerPerViewLW[];"
                "perviewLW int gl_ViewportMaskPerViewLW[][];"
            "} gl_MeshPrimitivesLW[];"
        );

        stageBuiltins[EShLangMeshLW].append(
            "out uint gl_PrimitiveCountLW;"
            "out uint gl_PrimitiveIndicesLW[];"

            "in uint gl_MeshViewCountLW;"
            "in uint gl_MeshViewIndicesLW[4];"

            "const highp uvec3 gl_WorkGroupSize = uvec3(1,1,1);"

            "in highp uvec3 gl_WorkGroupID;"
            "in highp uvec3 gl_LocalIlwocationID;"

            "in highp uvec3 gl_GlobalIlwocationID;"
            "in highp uint gl_LocalIlwocationIndex;"

            "\n");

        stageBuiltins[EShLangTaskLW].append(
            "out uint gl_TaskCountLW;"

            "const highp uvec3 gl_WorkGroupSize = uvec3(1,1,1);"

            "in highp uvec3 gl_WorkGroupID;"
            "in highp uvec3 gl_LocalIlwocationID;"

            "in highp uvec3 gl_GlobalIlwocationID;"
            "in highp uint gl_LocalIlwocationIndex;"

            "in uint gl_MeshViewCountLW;"
            "in uint gl_MeshViewIndicesLW[4];"

            "\n");
    }

    if (profile != EEsProfile && version >= 450) {
        stageBuiltins[EShLangMeshLW].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in int gl_DrawIDARB;"             // GL_ARB_shader_draw_parameters
            "\n");

        stageBuiltins[EShLangTaskLW].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in int gl_DrawIDARB;"             // GL_ARB_shader_draw_parameters
            "\n");

        if (version >= 460) {
            stageBuiltins[EShLangMeshLW].append(
                "in int gl_DrawID;"
                "\n");

            stageBuiltins[EShLangTaskLW].append(
                "in int gl_DrawID;"
                "\n");
        }
    }

    //============================================================================
    //
    // Define the interface to the vertex shader.
    //
    //============================================================================

    if (profile != EEsProfile) {
        if (version < 130) {
            stageBuiltins[EShLangVertex].append(
                "attribute vec4  gl_Color;"
                "attribute vec4  gl_SecondaryColor;"
                "attribute vec3  gl_Normal;"
                "attribute vec4  gl_Vertex;"
                "attribute vec4  gl_MultiTexCoord0;"
                "attribute vec4  gl_MultiTexCoord1;"
                "attribute vec4  gl_MultiTexCoord2;"
                "attribute vec4  gl_MultiTexCoord3;"
                "attribute vec4  gl_MultiTexCoord4;"
                "attribute vec4  gl_MultiTexCoord5;"
                "attribute vec4  gl_MultiTexCoord6;"
                "attribute vec4  gl_MultiTexCoord7;"
                "attribute float gl_FogCoord;"
                "\n");
        } else if (IncludeLegacy(version, profile, spvVersion)) {
            stageBuiltins[EShLangVertex].append(
                "in vec4  gl_Color;"
                "in vec4  gl_SecondaryColor;"
                "in vec3  gl_Normal;"
                "in vec4  gl_Vertex;"
                "in vec4  gl_MultiTexCoord0;"
                "in vec4  gl_MultiTexCoord1;"
                "in vec4  gl_MultiTexCoord2;"
                "in vec4  gl_MultiTexCoord3;"
                "in vec4  gl_MultiTexCoord4;"
                "in vec4  gl_MultiTexCoord5;"
                "in vec4  gl_MultiTexCoord6;"
                "in vec4  gl_MultiTexCoord7;"
                "in float gl_FogCoord;"
                "\n");
        }

        if (version < 150) {
            if (version < 130) {
                stageBuiltins[EShLangVertex].append(
                    "        vec4  gl_ClipVertex;"       // needs qualifier fixed later
                    "varying vec4  gl_FrontColor;"
                    "varying vec4  gl_BackColor;"
                    "varying vec4  gl_FrontSecondaryColor;"
                    "varying vec4  gl_BackSecondaryColor;"
                    "varying vec4  gl_TexCoord[];"
                    "varying float gl_FogFragCoord;"
                    "\n");
            } else if (IncludeLegacy(version, profile, spvVersion)) {
                stageBuiltins[EShLangVertex].append(
                    "    vec4  gl_ClipVertex;"       // needs qualifier fixed later
                    "out vec4  gl_FrontColor;"
                    "out vec4  gl_BackColor;"
                    "out vec4  gl_FrontSecondaryColor;"
                    "out vec4  gl_BackSecondaryColor;"
                    "out vec4  gl_TexCoord[];"
                    "out float gl_FogFragCoord;"
                    "\n");
            }
            stageBuiltins[EShLangVertex].append(
                "vec4 gl_Position;"   // needs qualifier fixed later
                "float gl_PointSize;" // needs qualifier fixed later
                );

            if (version == 130 || version == 140)
                stageBuiltins[EShLangVertex].append(
                    "out float gl_ClipDistance[];"
                    );
        } else {
            // version >= 150
            stageBuiltins[EShLangVertex].append(
                "out gl_PerVertex {"
                    "vec4 gl_Position;"     // needs qualifier fixed later
                    "float gl_PointSize;"   // needs qualifier fixed later
                    "float gl_ClipDistance[];"
                    );
            if (IncludeLegacy(version, profile, spvVersion))
                stageBuiltins[EShLangVertex].append(
                    "vec4 gl_ClipVertex;"   // needs qualifier fixed later
                    "vec4 gl_FrontColor;"
                    "vec4 gl_BackColor;"
                    "vec4 gl_FrontSecondaryColor;"
                    "vec4 gl_BackSecondaryColor;"
                    "vec4 gl_TexCoord[];"
                    "float gl_FogFragCoord;"
                    );
            if (version >= 450)
                stageBuiltins[EShLangVertex].append(
                    "float gl_LwllDistance[];"
                    );
            stageBuiltins[EShLangVertex].append(
                "};"
                "\n");
        }
        if (version >= 130 && spvVersion.vulkan == 0)
            stageBuiltins[EShLangVertex].append(
                "int gl_VertexID;"            // needs qualifier fixed later
                );
        if (version >= 140 && spvVersion.vulkan == 0)
            stageBuiltins[EShLangVertex].append(
                "int gl_InstanceID;"          // needs qualifier fixed later
                );
        if (spvVersion.vulkan > 0 && version >= 140)
            stageBuiltins[EShLangVertex].append(
                "in int gl_VertexIndex;"
                "in int gl_InstanceIndex;"
                );
        if (version >= 440) {
            stageBuiltins[EShLangVertex].append(
                "in int gl_BaseVertexARB;"
                "in int gl_BaseInstanceARB;"
                "in int gl_DrawIDARB;"
                );
        }
        if (version >= 410) {
            stageBuiltins[EShLangVertex].append(
                "out int gl_ViewportIndex;"
                "out int gl_Layer;"
                );
        }
        if (version >= 460) {
            stageBuiltins[EShLangVertex].append(
                "in int gl_BaseVertex;"
                "in int gl_BaseInstance;"
                "in int gl_DrawID;"
                );
        }

        if (version >= 450)
            stageBuiltins[EShLangVertex].append(
                "out int gl_ViewportMask[];"             // GL_LW_viewport_array2
                "out int gl_SecondaryViewportMaskLW[];"  // GL_LW_stereo_view_rendering
                "out vec4 gl_SecondaryPositionLW;"       // GL_LW_stereo_view_rendering
                "out vec4 gl_PositionPerViewLW[];"       // GL_LWX_multiview_per_view_attributes
                "out int  gl_ViewportMaskPerViewLW[];"   // GL_LWX_multiview_per_view_attributes
                );
    } else {
        // ES profile
        if (version == 100) {
            stageBuiltins[EShLangVertex].append(
                "highp   vec4  gl_Position;"  // needs qualifier fixed later
                "mediump float gl_PointSize;" // needs qualifier fixed later
                );
        } else {
            if (spvVersion.vulkan == 0)
                stageBuiltins[EShLangVertex].append(
                    "in highp int gl_VertexID;"      // needs qualifier fixed later
                    "in highp int gl_InstanceID;"    // needs qualifier fixed later
                    );
            if (spvVersion.vulkan > 0)
#endif
                stageBuiltins[EShLangVertex].append(
                    "in highp int gl_VertexIndex;"
                    "in highp int gl_InstanceIndex;"
                    );
#ifndef GLSLANG_WEB
            if (version < 310)
#endif
                stageBuiltins[EShLangVertex].append(
                    "highp vec4  gl_Position;"    // needs qualifier fixed later
                    "highp float gl_PointSize;"   // needs qualifier fixed later
                    );
#ifndef GLSLANG_WEB
            else
                stageBuiltins[EShLangVertex].append(
                    "out gl_PerVertex {"
                        "highp vec4  gl_Position;"    // needs qualifier fixed later
                        "highp float gl_PointSize;"   // needs qualifier fixed later
                    "};"
                    );
        }
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangVertex].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    if (version >= 300 /* both ES and non-ES */) {
        stageBuiltins[EShLangVertex].append(
            "in highp uint gl_ViewID_OVR;"     // GL_OVR_multiview, GL_OVR_multiview2
            "\n");
    }


    //============================================================================
    //
    // Define the interface to the geometry shader.
    //
    //============================================================================

    if (profile == ECoreProfile || profile == ECompatibilityProfile) {
        stageBuiltins[EShLangGeometry].append(
            "in gl_PerVertex {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                );
        if (profile == ECompatibilityProfile)
            stageBuiltins[EShLangGeometry].append(
                "vec4 gl_ClipVertex;"
                "vec4 gl_FrontColor;"
                "vec4 gl_BackColor;"
                "vec4 gl_FrontSecondaryColor;"
                "vec4 gl_BackSecondaryColor;"
                "vec4 gl_TexCoord[];"
                "float gl_FogFragCoord;"
                );
        if (version >= 450)
            stageBuiltins[EShLangGeometry].append(
                "float gl_LwllDistance[];"
                "vec4 gl_SecondaryPositionLW;"   // GL_LW_stereo_view_rendering
                "vec4 gl_PositionPerViewLW[];"   // GL_LWX_multiview_per_view_attributes
                );
        stageBuiltins[EShLangGeometry].append(
            "} gl_in[];"

            "in int gl_PrimitiveIDIn;"
            "out gl_PerVertex {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                "\n");
        if (profile == ECompatibilityProfile && version >= 400)
            stageBuiltins[EShLangGeometry].append(
                "vec4 gl_ClipVertex;"
                "vec4 gl_FrontColor;"
                "vec4 gl_BackColor;"
                "vec4 gl_FrontSecondaryColor;"
                "vec4 gl_BackSecondaryColor;"
                "vec4 gl_TexCoord[];"
                "float gl_FogFragCoord;"
                );
        if (version >= 450)
            stageBuiltins[EShLangGeometry].append(
                "float gl_LwllDistance[];"
                );
        stageBuiltins[EShLangGeometry].append(
            "};"

            "out int gl_PrimitiveID;"
            "out int gl_Layer;");

        if (version >= 150)
            stageBuiltins[EShLangGeometry].append(
            "out int gl_ViewportIndex;"
            );

        if (profile == ECompatibilityProfile && version < 400)
            stageBuiltins[EShLangGeometry].append(
            "out vec4 gl_ClipVertex;"
            );

        if (version >= 400)
            stageBuiltins[EShLangGeometry].append(
            "in int gl_IlwocationID;"
            );

        if (version >= 450)
            stageBuiltins[EShLangGeometry].append(
                "out int gl_ViewportMask[];"               // GL_LW_viewport_array2
                "out int gl_SecondaryViewportMaskLW[];"    // GL_LW_stereo_view_rendering
                "out vec4 gl_SecondaryPositionLW;"         // GL_LW_stereo_view_rendering
                "out vec4 gl_PositionPerViewLW[];"         // GL_LWX_multiview_per_view_attributes
                "out int  gl_ViewportMaskPerViewLW[];"     // GL_LWX_multiview_per_view_attributes
            );

        stageBuiltins[EShLangGeometry].append("\n");
    } else if (profile == EEsProfile && version >= 310) {
        stageBuiltins[EShLangGeometry].append(
            "in gl_PerVertex {"
                "highp vec4 gl_Position;"
                "highp float gl_PointSize;"
            "} gl_in[];"
            "\n"
            "in highp int gl_PrimitiveIDIn;"
            "in highp int gl_IlwocationID;"
            "\n"
            "out gl_PerVertex {"
                "highp vec4 gl_Position;"
                "highp float gl_PointSize;"
            "};"
            "\n"
            "out highp int gl_PrimitiveID;"
            "out highp int gl_Layer;"
            "\n"
            );
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangGeometry].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    //============================================================================
    //
    // Define the interface to the tessellation control shader.
    //
    //============================================================================

    if (profile != EEsProfile && version >= 150) {
        // Note:  "in gl_PerVertex {...} gl_in[gl_MaxPatchVertices];" is declared in initialize() below,
        // as it depends on the resource sizing of gl_MaxPatchVertices.

        stageBuiltins[EShLangTessControl].append(
            "in int gl_PatchVerticesIn;"
            "in int gl_PrimitiveID;"
            "in int gl_IlwocationID;"

            "out gl_PerVertex {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
                );
        if (profile == ECompatibilityProfile)
            stageBuiltins[EShLangTessControl].append(
                "vec4 gl_ClipVertex;"
                "vec4 gl_FrontColor;"
                "vec4 gl_BackColor;"
                "vec4 gl_FrontSecondaryColor;"
                "vec4 gl_BackSecondaryColor;"
                "vec4 gl_TexCoord[];"
                "float gl_FogFragCoord;"
                );
        if (version >= 450)
            stageBuiltins[EShLangTessControl].append(
                "float gl_LwllDistance[];"
                "int  gl_ViewportMask[];"             // GL_LW_viewport_array2
                "vec4 gl_SecondaryPositionLW;"        // GL_LW_stereo_view_rendering
                "int  gl_SecondaryViewportMaskLW[];"  // GL_LW_stereo_view_rendering
                "vec4 gl_PositionPerViewLW[];"        // GL_LWX_multiview_per_view_attributes
                "int  gl_ViewportMaskPerViewLW[];"    // GL_LWX_multiview_per_view_attributes
                );
        stageBuiltins[EShLangTessControl].append(
            "} gl_out[];"

            "patch out float gl_TessLevelOuter[4];"
            "patch out float gl_TessLevelInner[2];"
            "\n");

        if (version >= 410)
            stageBuiltins[EShLangTessControl].append(
                "out int gl_ViewportIndex;"
                "out int gl_Layer;"
                "\n");

    } else {
        // Note:  "in gl_PerVertex {...} gl_in[gl_MaxPatchVertices];" is declared in initialize() below,
        // as it depends on the resource sizing of gl_MaxPatchVertices.

        stageBuiltins[EShLangTessControl].append(
            "in highp int gl_PatchVerticesIn;"
            "in highp int gl_PrimitiveID;"
            "in highp int gl_IlwocationID;"

            "out gl_PerVertex {"
                "highp vec4 gl_Position;"
                "highp float gl_PointSize;"
                );
        stageBuiltins[EShLangTessControl].append(
            "} gl_out[];"

            "patch out highp float gl_TessLevelOuter[4];"
            "patch out highp float gl_TessLevelInner[2];"
            "patch out highp vec4 gl_BoundingBoxOES[2];"
            "patch out highp vec4 gl_BoundingBoxEXT[2];"
            "\n");
        if (profile == EEsProfile && version >= 320) {
            stageBuiltins[EShLangTessControl].append(
                "patch out highp vec4 gl_BoundingBox[2];"
                "\n"
            );
        }
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangTessControl].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    //============================================================================
    //
    // Define the interface to the tessellation evaluation shader.
    //
    //============================================================================

    if (profile != EEsProfile && version >= 150) {
        // Note:  "in gl_PerVertex {...} gl_in[gl_MaxPatchVertices];" is declared in initialize() below,
        // as it depends on the resource sizing of gl_MaxPatchVertices.

        stageBuiltins[EShLangTessEvaluation].append(
            "in int gl_PatchVerticesIn;"
            "in int gl_PrimitiveID;"
            "in vec3 gl_TessCoord;"

            "patch in float gl_TessLevelOuter[4];"
            "patch in float gl_TessLevelInner[2];"

            "out gl_PerVertex {"
                "vec4 gl_Position;"
                "float gl_PointSize;"
                "float gl_ClipDistance[];"
            );
        if (version >= 400 && profile == ECompatibilityProfile)
            stageBuiltins[EShLangTessEvaluation].append(
                "vec4 gl_ClipVertex;"
                "vec4 gl_FrontColor;"
                "vec4 gl_BackColor;"
                "vec4 gl_FrontSecondaryColor;"
                "vec4 gl_BackSecondaryColor;"
                "vec4 gl_TexCoord[];"
                "float gl_FogFragCoord;"
                );
        if (version >= 450)
            stageBuiltins[EShLangTessEvaluation].append(
                "float gl_LwllDistance[];"
                );
        stageBuiltins[EShLangTessEvaluation].append(
            "};"
            "\n");

        if (version >= 410)
            stageBuiltins[EShLangTessEvaluation].append(
                "out int gl_ViewportIndex;"
                "out int gl_Layer;"
                "\n");

        if (version >= 450)
            stageBuiltins[EShLangTessEvaluation].append(
                "out int  gl_ViewportMask[];"             // GL_LW_viewport_array2
                "out vec4 gl_SecondaryPositionLW;"        // GL_LW_stereo_view_rendering
                "out int  gl_SecondaryViewportMaskLW[];"  // GL_LW_stereo_view_rendering
                "out vec4 gl_PositionPerViewLW[];"        // GL_LWX_multiview_per_view_attributes
                "out int  gl_ViewportMaskPerViewLW[];"    // GL_LWX_multiview_per_view_attributes
                );

    } else if (profile == EEsProfile && version >= 310) {
        // Note:  "in gl_PerVertex {...} gl_in[gl_MaxPatchVertices];" is declared in initialize() below,
        // as it depends on the resource sizing of gl_MaxPatchVertices.

        stageBuiltins[EShLangTessEvaluation].append(
            "in highp int gl_PatchVerticesIn;"
            "in highp int gl_PrimitiveID;"
            "in highp vec3 gl_TessCoord;"

            "patch in highp float gl_TessLevelOuter[4];"
            "patch in highp float gl_TessLevelInner[2];"

            "out gl_PerVertex {"
                "highp vec4 gl_Position;"
                "highp float gl_PointSize;"
            );
        stageBuiltins[EShLangTessEvaluation].append(
            "};"
            "\n");
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangTessEvaluation].append(
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    //============================================================================
    //
    // Define the interface to the fragment shader.
    //
    //============================================================================

    if (profile != EEsProfile) {

        stageBuiltins[EShLangFragment].append(
            "vec4  gl_FragCoord;"   // needs qualifier fixed later
            "bool  gl_FrontFacing;" // needs qualifier fixed later
            "float gl_FragDepth;"   // needs qualifier fixed later
            );
        if (version >= 120)
            stageBuiltins[EShLangFragment].append(
                "vec2 gl_PointCoord;"  // needs qualifier fixed later
                );
        if (version >= 140)
            stageBuiltins[EShLangFragment].append(
                "out int gl_FragStencilRefARB;"
                );
        if (IncludeLegacy(version, profile, spvVersion) || (! ForwardCompatibility && version < 420))
            stageBuiltins[EShLangFragment].append(
                "vec4 gl_FragColor;"   // needs qualifier fixed later
                );

        if (version < 130) {
            stageBuiltins[EShLangFragment].append(
                "varying vec4  gl_Color;"
                "varying vec4  gl_SecondaryColor;"
                "varying vec4  gl_TexCoord[];"
                "varying float gl_FogFragCoord;"
                );
        } else {
            stageBuiltins[EShLangFragment].append(
                "in float gl_ClipDistance[];"
                );

            if (IncludeLegacy(version, profile, spvVersion)) {
                if (version < 150)
                    stageBuiltins[EShLangFragment].append(
                        "in float gl_FogFragCoord;"
                        "in vec4  gl_TexCoord[];"
                        "in vec4  gl_Color;"
                        "in vec4  gl_SecondaryColor;"
                        );
                else
                    stageBuiltins[EShLangFragment].append(
                        "in gl_PerFragment {"
                            "in float gl_FogFragCoord;"
                            "in vec4  gl_TexCoord[];"
                            "in vec4  gl_Color;"
                            "in vec4  gl_SecondaryColor;"
                        "};"
                        );
            }
        }

        if (version >= 150)
            stageBuiltins[EShLangFragment].append(
                "flat in int gl_PrimitiveID;"
                );

        if (version >= 130) { // ARB_sample_shading
            stageBuiltins[EShLangFragment].append(
                "flat in  int  gl_SampleID;"
                "     in  vec2 gl_SamplePosition;"
                "     out int  gl_SampleMask[];"
                );

            if (spvVersion.spv == 0) {
                stageBuiltins[EShLangFragment].append(
                    "uniform int gl_NumSamples;"
                );
            }
        }

        if (version >= 400)
            stageBuiltins[EShLangFragment].append(
                "flat in  int  gl_SampleMaskIn[];"
            );

        if (version >= 430)
            stageBuiltins[EShLangFragment].append(
                "flat in int gl_Layer;"
                "flat in int gl_ViewportIndex;"
                );

        if (version >= 450)
            stageBuiltins[EShLangFragment].append(
                "in float gl_LwllDistance[];"
                "bool gl_HelperIlwocation;"     // needs qualifier fixed later
                );

        if (version >= 450)
            stageBuiltins[EShLangFragment].append( // GL_EXT_fragment_ilwocation_density
                "flat in ivec2 gl_FragSizeEXT;"
                "flat in int   gl_FragIlwocationCountEXT;"
                );

        if (version >= 450)
            stageBuiltins[EShLangFragment].append(
                "in vec2 gl_BaryCoordNoPerspAMD;"
                "in vec2 gl_BaryCoordNoPerspCentroidAMD;"
                "in vec2 gl_BaryCoordNoPerspSampleAMD;"
                "in vec2 gl_BaryCoordSmoothAMD;"
                "in vec2 gl_BaryCoordSmoothCentroidAMD;"
                "in vec2 gl_BaryCoordSmoothSampleAMD;"
                "in vec3 gl_BaryCoordPullModelAMD;"
                );

        if (version >= 430)
            stageBuiltins[EShLangFragment].append(
                "in bool gl_FragFullyCoveredLW;"
                );
        if (version >= 450)
            stageBuiltins[EShLangFragment].append(
                "flat in ivec2 gl_FragmentSizeLW;"          // GL_LW_shading_rate_image
                "flat in int   gl_IlwocationsPerPixelLW;"
                "in vec3 gl_BaryCoordLW;"                   // GL_LW_fragment_shader_barycentric
                "in vec3 gl_BaryCoordNoPerspLW;"
                );

    } else {
        // ES profile

        if (version == 100) {
            stageBuiltins[EShLangFragment].append(
                "mediump vec4 gl_FragCoord;"    // needs qualifier fixed later
                "        bool gl_FrontFacing;"  // needs qualifier fixed later
                "mediump vec4 gl_FragColor;"    // needs qualifier fixed later
                "mediump vec2 gl_PointCoord;"   // needs qualifier fixed later
                );
        }
#endif
        if (version >= 300) {
            stageBuiltins[EShLangFragment].append(
                "highp   vec4  gl_FragCoord;"    // needs qualifier fixed later
                "        bool  gl_FrontFacing;"  // needs qualifier fixed later
                "mediump vec2  gl_PointCoord;"   // needs qualifier fixed later
                "highp   float gl_FragDepth;"    // needs qualifier fixed later
                );
        }
#ifndef GLSLANG_WEB
        if (version >= 310) {
            stageBuiltins[EShLangFragment].append(
                "bool gl_HelperIlwocation;"          // needs qualifier fixed later
                "flat in highp int gl_PrimitiveID;"  // needs qualifier fixed later
                "flat in highp int gl_Layer;"        // needs qualifier fixed later
                );

            stageBuiltins[EShLangFragment].append(  // GL_OES_sample_variables
                "flat  in lowp     int gl_SampleID;"
                "      in mediump vec2 gl_SamplePosition;"
                "flat  in highp    int gl_SampleMaskIn[];"
                "     out highp    int gl_SampleMask[];"
                );
            if (spvVersion.spv == 0)
                stageBuiltins[EShLangFragment].append(  // GL_OES_sample_variables
                    "uniform lowp int gl_NumSamples;"
                    );
        }
        stageBuiltins[EShLangFragment].append(
            "highp float gl_FragDepthEXT;"       // GL_EXT_frag_depth
            );

        if (version >= 310)
            stageBuiltins[EShLangFragment].append( // GL_EXT_fragment_ilwocation_density
                "flat in ivec2 gl_FragSizeEXT;"
                "flat in int   gl_FragIlwocationCountEXT;"
            );
        if (version >= 320)
            stageBuiltins[EShLangFragment].append( // GL_LW_shading_rate_image
                "flat in ivec2 gl_FragmentSizeLW;"
                "flat in int   gl_IlwocationsPerPixelLW;"
            );
        if (version >= 320)
            stageBuiltins[EShLangFragment].append(
                "in vec3 gl_BaryCoordLW;"
                "in vec3 gl_BaryCoordNoPerspLW;"
                );
    }
#endif

    stageBuiltins[EShLangFragment].append("\n");

    if (version >= 130)
        add2ndGenerationSamplingImaging(version, profile, spvVersion);

#ifndef GLSLANG_WEB

    // GL_ARB_shader_ballot
    if (profile != EEsProfile && version >= 450) {
        const char* ballotDecls =
            "uniform uint gl_SubGroupSizeARB;"
            "in uint     gl_SubGroupIlwocationARB;"
            "in uint64_t gl_SubGroupEqMaskARB;"
            "in uint64_t gl_SubGroupGeMaskARB;"
            "in uint64_t gl_SubGroupGtMaskARB;"
            "in uint64_t gl_SubGroupLeMaskARB;"
            "in uint64_t gl_SubGroupLtMaskARB;"
            "\n";
        const char* fragmentBallotDecls =
            "uniform uint gl_SubGroupSizeARB;"
            "flat in uint     gl_SubGroupIlwocationARB;"
            "flat in uint64_t gl_SubGroupEqMaskARB;"
            "flat in uint64_t gl_SubGroupGeMaskARB;"
            "flat in uint64_t gl_SubGroupGtMaskARB;"
            "flat in uint64_t gl_SubGroupLeMaskARB;"
            "flat in uint64_t gl_SubGroupLtMaskARB;"
            "\n";
        stageBuiltins[EShLangVertex]        .append(ballotDecls);
        stageBuiltins[EShLangTessControl]   .append(ballotDecls);
        stageBuiltins[EShLangTessEvaluation].append(ballotDecls);
        stageBuiltins[EShLangGeometry]      .append(ballotDecls);
        stageBuiltins[EShLangCompute]       .append(ballotDecls);
        stageBuiltins[EShLangFragment]      .append(fragmentBallotDecls);
        stageBuiltins[EShLangMeshLW]        .append(ballotDecls);
        stageBuiltins[EShLangTaskLW]        .append(ballotDecls);
    }

    if ((profile != EEsProfile && version >= 140) ||
        (profile == EEsProfile && version >= 310)) {
        stageBuiltins[EShLangFragment].append(
            "flat in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "flat in highp int gl_ViewIndex;"       // GL_EXT_multiview
            "\n");
    }

    // GL_KHR_shader_subgroup
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 140)) {
        const char* subgroupDecls =
            "in mediump uint  gl_SubgroupSize;"
            "in mediump uint  gl_SubgroupIlwocationID;"
            "in highp   uvec4 gl_SubgroupEqMask;"
            "in highp   uvec4 gl_SubgroupGeMask;"
            "in highp   uvec4 gl_SubgroupGtMask;"
            "in highp   uvec4 gl_SubgroupLeMask;"
            "in highp   uvec4 gl_SubgroupLtMask;"
            // GL_LW_shader_sm_builtins
            "in highp   uint  gl_WarpsPerSMLW;"
            "in highp   uint  gl_SMCountLW;"
            "in highp   uint  gl_WarpIDLW;"
            "in highp   uint  gl_SMIDLW;"
            "\n";
        const char* fragmentSubgroupDecls =
            "flat in mediump uint  gl_SubgroupSize;"
            "flat in mediump uint  gl_SubgroupIlwocationID;"
            "flat in highp   uvec4 gl_SubgroupEqMask;"
            "flat in highp   uvec4 gl_SubgroupGeMask;"
            "flat in highp   uvec4 gl_SubgroupGtMask;"
            "flat in highp   uvec4 gl_SubgroupLeMask;"
            "flat in highp   uvec4 gl_SubgroupLtMask;"
            // GL_LW_shader_sm_builtins
            "flat in highp   uint  gl_WarpsPerSMLW;"
            "flat in highp   uint  gl_SMCountLW;"
            "flat in highp   uint  gl_WarpIDLW;"
            "flat in highp   uint  gl_SMIDLW;"
            "\n";
        const char* computeSubgroupDecls =
            "in highp   uint  gl_NumSubgroups;"
            "in highp   uint  gl_SubgroupID;"
            "\n";

        stageBuiltins[EShLangVertex]        .append(subgroupDecls);
        stageBuiltins[EShLangTessControl]   .append(subgroupDecls);
        stageBuiltins[EShLangTessEvaluation].append(subgroupDecls);
        stageBuiltins[EShLangGeometry]      .append(subgroupDecls);
        stageBuiltins[EShLangCompute]       .append(subgroupDecls);
        stageBuiltins[EShLangCompute]       .append(computeSubgroupDecls);
        stageBuiltins[EShLangFragment]      .append(fragmentSubgroupDecls);
        stageBuiltins[EShLangMeshLW]        .append(subgroupDecls);
        stageBuiltins[EShLangMeshLW]        .append(computeSubgroupDecls);
        stageBuiltins[EShLangTaskLW]        .append(subgroupDecls);
        stageBuiltins[EShLangTaskLW]        .append(computeSubgroupDecls);
        stageBuiltins[EShLangRayGen]        .append(subgroupDecls);
        stageBuiltins[EShLangIntersect]     .append(subgroupDecls);
        stageBuiltins[EShLangAnyHit]        .append(subgroupDecls);
        stageBuiltins[EShLangClosestHit]    .append(subgroupDecls);
        stageBuiltins[EShLangMiss]          .append(subgroupDecls);
        stageBuiltins[EShLangCallable]      .append(subgroupDecls);
    }

    // GL_LW_ray_tracing/GL_EXT_ray_tracing
    if (profile != EEsProfile && version >= 460) {

        const char *constRayFlags =
            "const uint gl_RayFlagsNoneLW = 0U;"
            "const uint gl_RayFlagsNoneEXT = 0U;"
            "const uint gl_RayFlagsOpaqueLW = 1U;"
            "const uint gl_RayFlagsOpaqueEXT = 1U;"
            "const uint gl_RayFlagsNoOpaqueLW = 2U;"
            "const uint gl_RayFlagsNoOpaqueEXT = 2U;"
            "const uint gl_RayFlagsTerminateOnFirstHitLW = 4U;"
            "const uint gl_RayFlagsTerminateOnFirstHitEXT = 4U;"
            "const uint gl_RayFlagsSkipClosestHitShaderLW = 8U;"
            "const uint gl_RayFlagsSkipClosestHitShaderEXT = 8U;"
            "const uint gl_RayFlagsLwllBackFacingTrianglesLW = 16U;"
            "const uint gl_RayFlagsLwllBackFacingTrianglesEXT = 16U;"
            "const uint gl_RayFlagsLwllFrontFacingTrianglesLW = 32U;"
            "const uint gl_RayFlagsLwllFrontFacingTrianglesEXT = 32U;"
            "const uint gl_RayFlagsLwllOpaqueLW = 64U;"
            "const uint gl_RayFlagsLwllOpaqueEXT = 64U;"
            "const uint gl_RayFlagsLwllNoOpaqueLW = 128U;"
            "const uint gl_RayFlagsLwllNoOpaqueEXT = 128U;"
            "const uint gl_RayFlagsSkipTrianglesEXT = 256U;"
            "const uint gl_RayFlagsSkipAABBEXT = 512U;"
            "const uint gl_HitKindFrontFacingTriangleEXT = 254U;"
            "const uint gl_HitKindBackFacingTriangleEXT = 255U;"
            "\n";

        const char *constRayQueryIntersection =
            "const uint gl_RayQueryCandidateIntersectionEXT = 0U;"
            "const uint gl_RayQueryCommittedIntersectionEXT = 1U;"
            "const uint gl_RayQueryCommittedIntersectionNoneEXT = 0U;"
            "const uint gl_RayQueryCommittedIntersectionTriangleEXT = 1U;"
            "const uint gl_RayQueryCommittedIntersectionGeneratedEXT = 2U;"
            "const uint gl_RayQueryCandidateIntersectionTriangleEXT = 0U;"
            "const uint gl_RayQueryCandidateIntersectionAABBEXT = 1U;"
            "\n";

        const char *rayGenDecls =
            "in    uvec3  gl_LaunchIDLW;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeLW;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "\n";
        const char *intersectDecls =
            "in    uvec3  gl_LaunchIDLW;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeLW;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "in     int   gl_PrimitiveID;"
            "in     int   gl_InstanceID;"
            "in     int   gl_InstanceLwstomIndexLW;"
            "in     int   gl_InstanceLwstomIndexEXT;"
            "in     int   gl_GeometryIndexEXT;"
            "in    vec3   gl_WorldRayOriginLW;"
            "in    vec3   gl_WorldRayOriginEXT;"
            "in    vec3   gl_WorldRayDirectionLW;"
            "in    vec3   gl_WorldRayDirectionEXT;"
            "in    vec3   gl_ObjectRayOriginLW;"
            "in    vec3   gl_ObjectRayOriginEXT;"
            "in    vec3   gl_ObjectRayDirectionLW;"
            "in    vec3   gl_ObjectRayDirectionEXT;"
            "in    float  gl_RayTminLW;"
            "in    float  gl_RayTminEXT;"
            "in    float  gl_RayTmaxLW;"
            "in    float  gl_RayTmaxEXT;"
            "in    mat4x3 gl_ObjectToWorldLW;"
            "in    mat4x3 gl_ObjectToWorldEXT;"
            "in    mat3x4 gl_ObjectToWorld3x4EXT;"
            "in    mat4x3 gl_WorldToObjectLW;"
            "in    mat4x3 gl_WorldToObjectEXT;"
            "in    mat3x4 gl_WorldToObject3x4EXT;"
            "in    uint   gl_IncomingRayFlagsLW;"
            "in    uint   gl_IncomingRayFlagsEXT;"
            "\n";
        const char *hitDecls =
            "in    uvec3  gl_LaunchIDLW;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeLW;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "in     int   gl_PrimitiveID;"
            "in     int   gl_InstanceID;"
            "in     int   gl_InstanceLwstomIndexLW;"
            "in     int   gl_InstanceLwstomIndexEXT;"
            "in     int   gl_GeometryIndexEXT;"
            "in    vec3   gl_WorldRayOriginLW;"
            "in    vec3   gl_WorldRayOriginEXT;"
            "in    vec3   gl_WorldRayDirectionLW;"
            "in    vec3   gl_WorldRayDirectionEXT;"
            "in    vec3   gl_ObjectRayOriginLW;"
            "in    vec3   gl_ObjectRayOriginEXT;"
            "in    vec3   gl_ObjectRayDirectionLW;"
            "in    vec3   gl_ObjectRayDirectionEXT;"
            "in    float  gl_RayTminLW;"
            "in    float  gl_RayTminEXT;"
            "in    float  gl_RayTmaxLW;"
            "in    float  gl_RayTmaxEXT;"
            "in    float  gl_HitTLW;"
            "in    float  gl_HitTEXT;"
            "in    uint   gl_HitKindLW;"
            "in    uint   gl_HitKindEXT;"
            "in    mat4x3 gl_ObjectToWorldLW;"
            "in    mat4x3 gl_ObjectToWorldEXT;"
            "in    mat3x4 gl_ObjectToWorld3x4EXT;"
            "in    mat4x3 gl_WorldToObjectLW;"
            "in    mat4x3 gl_WorldToObjectEXT;"
            "in    mat3x4 gl_WorldToObject3x4EXT;"
            "in    uint   gl_IncomingRayFlagsLW;"
            "in    uint   gl_IncomingRayFlagsEXT;"
            "\n";
        const char *missDecls =
            "in    uvec3  gl_LaunchIDLW;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeLW;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "in    vec3   gl_WorldRayOriginLW;"
            "in    vec3   gl_WorldRayOriginEXT;"
            "in    vec3   gl_WorldRayDirectionLW;"
            "in    vec3   gl_WorldRayDirectionEXT;"
            "in    vec3   gl_ObjectRayOriginLW;"
            "in    vec3   gl_ObjectRayDirectionLW;"
            "in    float  gl_RayTminLW;"
            "in    float  gl_RayTminEXT;"
            "in    float  gl_RayTmaxLW;"
            "in    float  gl_RayTmaxEXT;"
            "in    uint   gl_IncomingRayFlagsLW;"
            "in    uint   gl_IncomingRayFlagsEXT;"
            "\n";

        const char *callableDecls =
            "in    uvec3  gl_LaunchIDLW;"
            "in    uvec3  gl_LaunchIDEXT;"
            "in    uvec3  gl_LaunchSizeLW;"
            "in    uvec3  gl_LaunchSizeEXT;"
            "\n";


        commonBuiltins.append(constRayQueryIntersection);
        commonBuiltins.append(constRayFlags);

        stageBuiltins[EShLangRayGen].append(rayGenDecls);
        stageBuiltins[EShLangIntersect].append(intersectDecls);
        stageBuiltins[EShLangAnyHit].append(hitDecls);
        stageBuiltins[EShLangClosestHit].append(hitDecls);
        stageBuiltins[EShLangMiss].append(missDecls);
        stageBuiltins[EShLangCallable].append(callableDecls);

    }
    if ((profile != EEsProfile && version >= 140)) {
        const char *deviceIndex =
            "in highp int gl_DeviceIndex;"     // GL_EXT_device_group
            "\n";

        stageBuiltins[EShLangRayGen].append(deviceIndex);
        stageBuiltins[EShLangIntersect].append(deviceIndex);
        stageBuiltins[EShLangAnyHit].append(deviceIndex);
        stageBuiltins[EShLangClosestHit].append(deviceIndex);
        stageBuiltins[EShLangMiss].append(deviceIndex);
    }

    if (version >= 300 /* both ES and non-ES */) {
        stageBuiltins[EShLangFragment].append(
            "flat in highp uint gl_ViewID_OVR;"     // GL_OVR_multiview, GL_OVR_multiview2
            "\n");
    }

    if ((profile != EEsProfile && version >= 420) ||
        (profile == EEsProfile && version >= 310)) {
        commonBuiltins.append("const int gl_ScopeDevice      = 1;\n");
        commonBuiltins.append("const int gl_ScopeWorkgroup   = 2;\n");
        commonBuiltins.append("const int gl_ScopeSubgroup    = 3;\n");
        commonBuiltins.append("const int gl_ScopeIlwocation  = 4;\n");
        commonBuiltins.append("const int gl_ScopeQueueFamily = 5;\n");
        commonBuiltins.append("const int gl_ScopeShaderCallEXT = 6;\n");

        commonBuiltins.append("const int gl_SemanticsRelaxed         = 0x0;\n");
        commonBuiltins.append("const int gl_SemanticsAcquire         = 0x2;\n");
        commonBuiltins.append("const int gl_SemanticsRelease         = 0x4;\n");
        commonBuiltins.append("const int gl_SemanticsAcquireRelease  = 0x8;\n");
        commonBuiltins.append("const int gl_SemanticsMakeAvailable   = 0x2000;\n");
        commonBuiltins.append("const int gl_SemanticsMakeVisible     = 0x4000;\n");
        commonBuiltins.append("const int gl_SemanticsVolatile        = 0x8000;\n");

        commonBuiltins.append("const int gl_StorageSemanticsNone     = 0x0;\n");
        commonBuiltins.append("const int gl_StorageSemanticsBuffer   = 0x40;\n");
        commonBuiltins.append("const int gl_StorageSemanticsShared   = 0x100;\n");
        commonBuiltins.append("const int gl_StorageSemanticsImage    = 0x800;\n");
        commonBuiltins.append("const int gl_StorageSemanticsOutput   = 0x1000;\n");
    }
#endif

    // printf("%s\n", commonBuiltins.c_str());
    // printf("%s\n", stageBuiltins[EShLangFragment].c_str());
}

//
// Helper function for initialize(), to add the second set of names for texturing,
// when adding context-independent built-in functions.
//
void TBuiltIns::add2ndGenerationSamplingImaging(int version, EProfile profile, const SpvVersion& spvVersion)
{
    //
    // In this function proper, enumerate the types, then calls the next set of functions
    // to enumerate all the uses for that type.
    //

    // enumerate all the types
#ifdef GLSLANG_WEB
    const TBasicType bTypes[] = { EbtFloat, EbtInt, EbtUint };
    bool skipBuffer = true;
    bool skipLwbeArrayed = true;
    const int image = 0;
#else
    const TBasicType bTypes[] = { EbtFloat, EbtInt, EbtUint, EbtFloat16 };
    bool skipBuffer = (profile == EEsProfile && version < 310) || (profile != EEsProfile && version < 140);
    bool skipLwbeArrayed = (profile == EEsProfile && version < 310) || (profile != EEsProfile && version < 130);
    for (int image = 0; image <= 1; ++image) // loop over "bool" image vs sampler
#endif
    {
        for (int shadow = 0; shadow <= 1; ++shadow) { // loop over "bool" shadow or not
#ifdef GLSLANG_WEB
            const int ms = 0;
#else
            for (int ms = 0; ms <= 1; ++ms) // loop over "bool" multisample or not
#endif
            {
                if ((ms || image) && shadow)
                    continue;
                if (ms && profile != EEsProfile && version < 150)
                    continue;
                if (ms && image && profile == EEsProfile)
                    continue;
                if (ms && profile == EEsProfile && version < 310)
                    continue;

                for (int arrayed = 0; arrayed <= 1; ++arrayed) { // loop over "bool" arrayed or not
#ifdef GLSLANG_WEB
                    for (int dim = Esd2D; dim <= EsdLwbe; ++dim) { // 2D, 3D, and Lwbe
#else
                    for (int dim = Esd1D; dim < EsdNumDims; ++dim) { // 1D, ..., buffer, subpass
                        if (dim == EsdSubpass && spvVersion.vulkan == 0)
                            continue;
                        if (dim == EsdSubpass && (image || shadow || arrayed))
                            continue;
                        if ((dim == Esd1D || dim == EsdRect) && profile == EEsProfile)
                            continue;
                        if (dim == EsdSubpass && spvVersion.vulkan == 0)
                            continue;
                        if (dim == EsdSubpass && (image || shadow || arrayed))
                            continue;
                        if ((dim == Esd1D || dim == EsdRect) && profile == EEsProfile)
                            continue;
                        if (dim != Esd2D && dim != EsdSubpass && ms)
                            continue;
                        if (dim == EsdBuffer && skipBuffer)
                            continue;
                        if (dim == EsdBuffer && (shadow || arrayed || ms))
                            continue;
                        if (ms && arrayed && profile == EEsProfile && version < 310)
                            continue;
#endif
                        if (dim == Esd3D && shadow)
                            continue;
                        if (dim == EsdLwbe && arrayed && skipLwbeArrayed)
                            continue;
                        if ((dim == Esd3D || dim == EsdRect) && arrayed)
                            continue;

                        // Loop over the bTypes
                        for (size_t bType = 0; bType < sizeof(bTypes)/sizeof(TBasicType); ++bType) {
#ifndef GLSLANG_WEB
                            if (bTypes[bType] == EbtFloat16 && (profile == EEsProfile || version < 450))
                                continue;
                            if (dim == EsdRect && version < 140 && bType > 0)
                                continue;
#endif
                            if (shadow && (bTypes[bType] == EbtInt || bTypes[bType] == EbtUint))
                                continue;

                            //
                            // Now, make all the function prototypes for the type we just built...
                            //
                            TSampler sampler;
#ifndef GLSLANG_WEB
                            if (dim == EsdSubpass) {
                                sampler.setSubpass(bTypes[bType], ms ? true : false);
                            } else
#endif
                            if (image) {
                                sampler.setImage(bTypes[bType], (TSamplerDim)dim, arrayed ? true : false,
                                                                                  shadow  ? true : false,
                                                                                  ms      ? true : false);
                            } else {
                                sampler.set(bTypes[bType], (TSamplerDim)dim, arrayed ? true : false,
                                                                             shadow  ? true : false,
                                                                             ms      ? true : false);
                            }

                            TString typeName = sampler.getString();

#ifndef GLSLANG_WEB
                            if (dim == EsdSubpass) {
                                addSubpassSampling(sampler, typeName, version, profile);
                                continue;
                            }
#endif

                            addQueryFunctions(sampler, typeName, version, profile);

                            if (image)
                                addImageFunctions(sampler, typeName, version, profile);
                            else {
                                addSamplingFunctions(sampler, typeName, version, profile);
#ifndef GLSLANG_WEB
                                addGatherFunctions(sampler, typeName, version, profile);
                                if (spvVersion.vulkan > 0 && sampler.isCombined() && !sampler.shadow) {
                                    // Base Vulkan allows texelFetch() for
                                    // textureBuffer (i.e. without sampler).
                                    //
                                    // GL_EXT_samplerless_texture_functions
                                    // allows texelFetch() and query functions
                                    // (other than textureQueryLod()) for all
                                    // texture types.
                                    sampler.setTexture(sampler.type, sampler.dim, sampler.arrayed, sampler.shadow,
                                                       sampler.ms);
                                    TString textureTypeName = sampler.getString();
                                    addSamplingFunctions(sampler, textureTypeName, version, profile);
                                    addQueryFunctions(sampler, textureTypeName, version, profile);
                                }
#endif
                            }
                        }
                    }
                }
            }
        }
    }

    //
    // sparseTexelsResidentARB()
    //
    if (profile != EEsProfile && version >= 450) {
        commonBuiltins.append("bool sparseTexelsResidentARB(int code);\n");
    }
}

//
// Helper function for add2ndGenerationSamplingImaging(),
// when adding context-independent built-in functions.
//
// Add all the query functions for the given type.
//
void TBuiltIns::addQueryFunctions(TSampler sampler, const TString& typeName, int version, EProfile profile)
{
    //
    // textureSize() and imageSize()
    //

    int sizeDims = dimMap[sampler.dim] + (sampler.arrayed ? 1 : 0) - (sampler.dim == EsdLwbe ? 1 : 0);

#ifdef GLSLANG_WEB
    commonBuiltins.append("highp ");
    commonBuiltins.append("ivec");
    commonBuiltins.append(postfixes[sizeDims]);
    commonBuiltins.append(" textureSize(");
    commonBuiltins.append(typeName);
    commonBuiltins.append(",int);\n");
    return;
#endif

    if (sampler.isImage() && ((profile == EEsProfile && version < 310) || (profile != EEsProfile && version < 420)))
        return;

    if (profile == EEsProfile)
        commonBuiltins.append("highp ");
    if (sizeDims == 1)
        commonBuiltins.append("int");
    else {
        commonBuiltins.append("ivec");
        commonBuiltins.append(postfixes[sizeDims]);
    }
    if (sampler.isImage())
        commonBuiltins.append(" imageSize(readonly writeonly volatile coherent ");
    else
        commonBuiltins.append(" textureSize(");
    commonBuiltins.append(typeName);
    if (! sampler.isImage() && ! sampler.isRect() && ! sampler.isBuffer() && ! sampler.isMultiSample())
        commonBuiltins.append(",int);\n");
    else
        commonBuiltins.append(");\n");

    //
    // textureSamples() and imageSamples()
    //

    // GL_ARB_shader_texture_image_samples
    // TODO: spec issue? there are no memory qualifiers; how to query a writeonly/readonly image, etc?
    if (profile != EEsProfile && version >= 430 && sampler.isMultiSample()) {
        commonBuiltins.append("int ");
        if (sampler.isImage())
            commonBuiltins.append("imageSamples(readonly writeonly volatile coherent ");
        else
            commonBuiltins.append("textureSamples(");
        commonBuiltins.append(typeName);
        commonBuiltins.append(");\n");
    }

    //
    // textureQueryLod(), fragment stage only
    // Also enabled with extension GL_ARB_texture_query_lod

    if (profile != EEsProfile && version >= 150 && sampler.isCombined() && sampler.dim != EsdRect &&
        ! sampler.isMultiSample() && ! sampler.isBuffer()) {
        for (int f16TexAddr = 0; f16TexAddr < 2; ++f16TexAddr) {
            if (f16TexAddr && sampler.type != EbtFloat16)
                continue;
            stageBuiltins[EShLangFragment].append("vec2 textureQueryLod(");
            stageBuiltins[EShLangFragment].append(typeName);
            if (dimMap[sampler.dim] == 1)
                if (f16TexAddr)
                    stageBuiltins[EShLangFragment].append(", float16_t");
                else
                    stageBuiltins[EShLangFragment].append(", float");
            else {
                if (f16TexAddr)
                    stageBuiltins[EShLangFragment].append(", f16vec");
                else
                    stageBuiltins[EShLangFragment].append(", vec");
                stageBuiltins[EShLangFragment].append(postfixes[dimMap[sampler.dim]]);
            }
            stageBuiltins[EShLangFragment].append(");\n");
        }

        stageBuiltins[EShLangCompute].append("vec2 textureQueryLod(");
        stageBuiltins[EShLangCompute].append(typeName);
        if (dimMap[sampler.dim] == 1)
            stageBuiltins[EShLangCompute].append(", float");
        else {
            stageBuiltins[EShLangCompute].append(", vec");
            stageBuiltins[EShLangCompute].append(postfixes[dimMap[sampler.dim]]);
        }
        stageBuiltins[EShLangCompute].append(");\n");
    }

    //
    // textureQueryLevels()
    //

    if (profile != EEsProfile && version >= 430 && ! sampler.isImage() && sampler.dim != EsdRect &&
        ! sampler.isMultiSample() && ! sampler.isBuffer()) {
        commonBuiltins.append("int textureQueryLevels(");
        commonBuiltins.append(typeName);
        commonBuiltins.append(");\n");
    }
}

//
// Helper function for add2ndGenerationSamplingImaging(),
// when adding context-independent built-in functions.
//
// Add all the image access functions for the given type.
//
void TBuiltIns::addImageFunctions(TSampler sampler, const TString& typeName, int version, EProfile profile)
{
    int dims = dimMap[sampler.dim];
    // most things with an array add a dimension, except for lwbemaps
    if (sampler.arrayed && sampler.dim != EsdLwbe)
        ++dims;

    TString imageParams = typeName;
    if (dims == 1)
        imageParams.append(", int");
    else {
        imageParams.append(", ivec");
        imageParams.append(postfixes[dims]);
    }
    if (sampler.isMultiSample())
        imageParams.append(", int");

    if (profile == EEsProfile)
        commonBuiltins.append("highp ");
    commonBuiltins.append(prefixes[sampler.type]);
    commonBuiltins.append("vec4 imageLoad(readonly volatile coherent ");
    commonBuiltins.append(imageParams);
    commonBuiltins.append(");\n");

    commonBuiltins.append("void imageStore(writeonly volatile coherent ");
    commonBuiltins.append(imageParams);
    commonBuiltins.append(", ");
    commonBuiltins.append(prefixes[sampler.type]);
    commonBuiltins.append("vec4);\n");

    if (! sampler.is1D() && ! sampler.isBuffer() && profile != EEsProfile && version >= 450) {
        commonBuiltins.append("int sparseImageLoadARB(readonly volatile coherent ");
        commonBuiltins.append(imageParams);
        commonBuiltins.append(", out ");
        commonBuiltins.append(prefixes[sampler.type]);
        commonBuiltins.append("vec4");
        commonBuiltins.append(");\n");
    }

    if ( profile != EEsProfile ||
        (profile == EEsProfile && version >= 310)) {
        if (sampler.type == EbtInt || sampler.type == EbtUint) {
            const char* dataType = sampler.type == EbtInt ? "highp int" : "highp uint";

            const int numBuiltins = 7;

            static const char* atomicFunc[numBuiltins] = {
                " imageAtomicAdd(volatile coherent ",
                " imageAtomicMin(volatile coherent ",
                " imageAtomicMax(volatile coherent ",
                " imageAtomicAnd(volatile coherent ",
                " imageAtomicOr(volatile coherent ",
                " imageAtomicXor(volatile coherent ",
                " imageAtomicExchange(volatile coherent "
            };

            // Loop twice to add prototypes with/without scope/semantics
            for (int j = 0; j < 2; ++j) {
                for (size_t i = 0; i < numBuiltins; ++i) {
                    commonBuiltins.append(dataType);
                    commonBuiltins.append(atomicFunc[i]);
                    commonBuiltins.append(imageParams);
                    commonBuiltins.append(", ");
                    commonBuiltins.append(dataType);
                    if (j == 1) {
                        commonBuiltins.append(", int, int, int");
                    }
                    commonBuiltins.append(");\n");
                }

                commonBuiltins.append(dataType);
                commonBuiltins.append(" imageAtomicCompSwap(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", ");
                commonBuiltins.append(dataType);
                commonBuiltins.append(", ");
                commonBuiltins.append(dataType);
                if (j == 1) {
                    commonBuiltins.append(", int, int, int, int, int");
                }
                commonBuiltins.append(");\n");
            }

            commonBuiltins.append(dataType);
            commonBuiltins.append(" imageAtomicLoad(volatile coherent ");
            commonBuiltins.append(imageParams);
            commonBuiltins.append(", int, int, int);\n");

            commonBuiltins.append("void imageAtomicStore(volatile coherent ");
            commonBuiltins.append(imageParams);
            commonBuiltins.append(", ");
            commonBuiltins.append(dataType);
            commonBuiltins.append(", int, int, int);\n");

        } else {
            // not int or uint
            // GL_ARB_ES3_1_compatibility
            // TODO: spec issue: are there restrictions on the kind of layout() that can be used?  what about dropping memory qualifiers?
            if ((profile != EEsProfile && version >= 450) ||
                (profile == EEsProfile && version >= 310)) {
                commonBuiltins.append("float imageAtomicExchange(volatile coherent ");
                commonBuiltins.append(imageParams);
                commonBuiltins.append(", float);\n");
            }
        }
    }

    if (sampler.dim == EsdRect || sampler.dim == EsdBuffer || sampler.shadow || sampler.isMultiSample())
        return;

    if (profile == EEsProfile || version < 450)
        return;

    TString imageLodParams = typeName;
    if (dims == 1)
        imageLodParams.append(", int");
    else {
        imageLodParams.append(", ivec");
        imageLodParams.append(postfixes[dims]);
    }
    imageLodParams.append(", int");

    commonBuiltins.append(prefixes[sampler.type]);
    commonBuiltins.append("vec4 imageLoadLodAMD(readonly volatile coherent ");
    commonBuiltins.append(imageLodParams);
    commonBuiltins.append(");\n");

    commonBuiltins.append("void imageStoreLodAMD(writeonly volatile coherent ");
    commonBuiltins.append(imageLodParams);
    commonBuiltins.append(", ");
    commonBuiltins.append(prefixes[sampler.type]);
    commonBuiltins.append("vec4);\n");

    if (! sampler.is1D()) {
        commonBuiltins.append("int sparseImageLoadLodAMD(readonly volatile coherent ");
        commonBuiltins.append(imageLodParams);
        commonBuiltins.append(", out ");
        commonBuiltins.append(prefixes[sampler.type]);
        commonBuiltins.append("vec4");
        commonBuiltins.append(");\n");
    }
}

//
// Helper function for initialize(),
// when adding context-independent built-in functions.
//
// Add all the subpass access functions for the given type.
//
void TBuiltIns::addSubpassSampling(TSampler sampler, const TString& typeName, int /*version*/, EProfile /*profile*/)
{
    stageBuiltins[EShLangFragment].append(prefixes[sampler.type]);
    stageBuiltins[EShLangFragment].append("vec4 subpassLoad");
    stageBuiltins[EShLangFragment].append("(");
    stageBuiltins[EShLangFragment].append(typeName.c_str());
    if (sampler.isMultiSample())
        stageBuiltins[EShLangFragment].append(", int");
    stageBuiltins[EShLangFragment].append(");\n");
}

//
// Helper function for add2ndGenerationSamplingImaging(),
// when adding context-independent built-in functions.
//
// Add all the texture lookup functions for the given type.
//
void TBuiltIns::addSamplingFunctions(TSampler sampler, const TString& typeName, int version, EProfile profile)
{
#ifdef GLSLANG_WEB
    profile = EEsProfile;
    version = 310;
#endif

    //
    // texturing
    //
    for (int proj = 0; proj <= 1; ++proj) { // loop over "bool" projective or not

        if (proj && (sampler.dim == EsdLwbe || sampler.isBuffer() || sampler.arrayed || sampler.isMultiSample()
            || !sampler.isCombined()))
            continue;

        for (int lod = 0; lod <= 1; ++lod) {

            if (lod && (sampler.isBuffer() || sampler.isRect() || sampler.isMultiSample() || !sampler.isCombined()))
                continue;
            if (lod && sampler.dim == Esd2D && sampler.arrayed && sampler.shadow)
                continue;
            if (lod && sampler.dim == EsdLwbe && sampler.shadow)
                continue;

            for (int bias = 0; bias <= 1; ++bias) {

                if (bias && (lod || sampler.isMultiSample() || !sampler.isCombined()))
                    continue;
                if (bias && (sampler.dim == Esd2D || sampler.dim == EsdLwbe) && sampler.shadow && sampler.arrayed)
                    continue;
                if (bias && (sampler.isRect() || sampler.isBuffer()))
                    continue;

                for (int offset = 0; offset <= 1; ++offset) { // loop over "bool" offset or not

                    if (proj + offset + bias + lod > 3)
                        continue;
                    if (offset && (sampler.dim == EsdLwbe || sampler.isBuffer() || sampler.isMultiSample()))
                        continue;

                    for (int fetch = 0; fetch <= 1; ++fetch) { // loop over "bool" fetch or not

                        if (proj + offset + fetch + bias + lod > 3)
                            continue;
                        if (fetch && (lod || bias))
                            continue;
                        if (fetch && (sampler.shadow || sampler.dim == EsdLwbe))
                            continue;
                        if (fetch == 0 && (sampler.isMultiSample() || sampler.isBuffer()
                            || !sampler.isCombined()))
                            continue;

                        for (int grad = 0; grad <= 1; ++grad) { // loop over "bool" grad or not

                            if (grad && (lod || bias || sampler.isMultiSample() || !sampler.isCombined()))
                                continue;
                            if (grad && sampler.isBuffer())
                                continue;
                            if (proj + offset + fetch + grad + bias + lod > 3)
                                continue;

                            for (int extraProj = 0; extraProj <= 1; ++extraProj) {
                                bool compare = false;
                                int totalDims = dimMap[sampler.dim] + (sampler.arrayed ? 1 : 0);
                                // skip dummy unused second component for 1D non-array shadows
                                if (sampler.shadow && totalDims < 2)
                                    totalDims = 2;
                                totalDims += (sampler.shadow ? 1 : 0) + proj;
                                if (totalDims > 4 && sampler.shadow) {
                                    compare = true;
                                    totalDims = 4;
                                }
                                assert(totalDims <= 4);

                                if (extraProj && ! proj)
                                    continue;
                                if (extraProj && (sampler.dim == Esd3D || sampler.shadow || !sampler.isCombined()))
                                    continue;

                                // loop over 16-bit floating-point texel addressing
#ifdef GLSLANG_WEB
                                const int f16TexAddr = 0;
#else
                                for (int f16TexAddr = 0; f16TexAddr <= 1; ++f16TexAddr)
#endif
                                {
                                    if (f16TexAddr && sampler.type != EbtFloat16)
                                        continue;
                                    if (f16TexAddr && sampler.shadow && ! compare) {
                                        compare = true; // compare argument is always present
                                        totalDims--;
                                    }
                                    // loop over "bool" lod clamp
#ifdef GLSLANG_WEB
                                    const int lodClamp = 0;
#else
                                    for (int lodClamp = 0; lodClamp <= 1 ;++lodClamp)
#endif
                                    {
                                        if (lodClamp && (profile == EEsProfile || version < 450))
                                            continue;
                                        if (lodClamp && (proj || lod || fetch))
                                            continue;

                                        // loop over "bool" sparse or not
#ifdef GLSLANG_WEB
                                        const int sparse = 0;
#else
                                        for (int sparse = 0; sparse <= 1; ++sparse)
#endif
                                        {
                                            if (sparse && (profile == EEsProfile || version < 450))
                                                continue;
                                            // Sparse sampling is not for 1D/1D array texture, buffer texture, and
                                            // projective texture
                                            if (sparse && (sampler.is1D() || sampler.isBuffer() || proj))
                                                continue;

                                            TString s;

                                            // return type
                                            if (sparse)
                                                s.append("int ");
                                            else {
                                                if (sampler.shadow)
                                                    if (sampler.type == EbtFloat16)
                                                        s.append("float16_t ");
                                                    else
                                                        s.append("float ");
                                                else {
                                                    s.append(prefixes[sampler.type]);
                                                    s.append("vec4 ");
                                                }
                                            }

                                            // name
                                            if (sparse) {
                                                if (fetch)
                                                    s.append("sparseTexel");
                                                else
                                                    s.append("sparseTexture");
                                            }
                                            else {
                                                if (fetch)
                                                    s.append("texel");
                                                else
                                                    s.append("texture");
                                            }
                                            if (proj)
                                                s.append("Proj");
                                            if (lod)
                                                s.append("Lod");
                                            if (grad)
                                                s.append("Grad");
                                            if (fetch)
                                                s.append("Fetch");
                                            if (offset)
                                                s.append("Offset");
                                            if (lodClamp)
                                                s.append("Clamp");
                                            if (lodClamp || sparse)
                                                s.append("ARB");
                                            s.append("(");

                                            // sampler type
                                            s.append(typeName);
                                            // P coordinate
                                            if (extraProj) {
                                                if (f16TexAddr)
                                                    s.append(",f16vec4");
                                                else
                                                    s.append(",vec4");
                                            } else {
                                                s.append(",");
                                                TBasicType t = fetch ? EbtInt : (f16TexAddr ? EbtFloat16 : EbtFloat);
                                                if (totalDims == 1)
                                                    s.append(TType::getBasicString(t));
                                                else {
                                                    s.append(prefixes[t]);
                                                    s.append("vec");
                                                    s.append(postfixes[totalDims]);
                                                }
                                            }
                                            // non-optional compare
                                            if (compare)
                                                s.append(",float");

                                            // non-optional lod argument (lod that's not driven by lod loop) or sample
                                            if ((fetch && !sampler.isBuffer() &&
                                                 !sampler.isRect() && !sampler.isMultiSample())
                                                 || (sampler.isMultiSample() && fetch))
                                                s.append(",int");
                                            // non-optional lod
                                            if (lod) {
                                                if (f16TexAddr)
                                                    s.append(",float16_t");
                                                else
                                                    s.append(",float");
                                            }

                                            // gradient arguments
                                            if (grad) {
                                                if (dimMap[sampler.dim] == 1) {
                                                    if (f16TexAddr)
                                                        s.append(",float16_t,float16_t");
                                                    else
                                                        s.append(",float,float");
                                                } else {
                                                    if (f16TexAddr)
                                                        s.append(",f16vec");
                                                    else
                                                        s.append(",vec");
                                                    s.append(postfixes[dimMap[sampler.dim]]);
                                                    if (f16TexAddr)
                                                        s.append(",f16vec");
                                                    else
                                                        s.append(",vec");
                                                    s.append(postfixes[dimMap[sampler.dim]]);
                                                }
                                            }
                                            // offset
                                            if (offset) {
                                                if (dimMap[sampler.dim] == 1)
                                                    s.append(",int");
                                                else {
                                                    s.append(",ivec");
                                                    s.append(postfixes[dimMap[sampler.dim]]);
                                                }
                                            }

                                            // lod clamp
                                            if (lodClamp) {
                                                if (f16TexAddr)
                                                    s.append(",float16_t");
                                                else
                                                    s.append(",float");
                                            }
                                            // texel out (for sparse texture)
                                            if (sparse) {
                                                s.append(",out ");
                                                if (sampler.shadow)
                                                    if (sampler.type == EbtFloat16)
                                                        s.append("float16_t");
                                                    else
                                                        s.append("float");
                                                else {
                                                    s.append(prefixes[sampler.type]);
                                                    s.append("vec4");
                                                }
                                            }
                                            // optional bias
                                            if (bias) {
                                                if (f16TexAddr)
                                                    s.append(",float16_t");
                                                else
                                                    s.append(",float");
                                            }
                                            s.append(");\n");

                                            // Add to the per-language set of built-ins
                                            if (bias || lodClamp) {
                                                stageBuiltins[EShLangFragment].append(s);
                                                stageBuiltins[EShLangCompute].append(s);
                                            } else
                                                commonBuiltins.append(s);

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//
// Helper function for add2ndGenerationSamplingImaging(),
// when adding context-independent built-in functions.
//
// Add all the texture gather functions for the given type.
//
void TBuiltIns::addGatherFunctions(TSampler sampler, const TString& typeName, int version, EProfile profile)
{
#ifdef GLSLANG_WEB
    profile = EEsProfile;
    version = 310;
#endif

    switch (sampler.dim) {
    case Esd2D:
    case EsdRect:
    case EsdLwbe:
        break;
    default:
        return;
    }

    if (sampler.isMultiSample())
        return;

    if (version < 140 && sampler.dim == EsdRect && sampler.type != EbtFloat)
        return;

    for (int f16TexAddr = 0; f16TexAddr <= 1; ++f16TexAddr) { // loop over 16-bit floating-point texel addressing

        if (f16TexAddr && sampler.type != EbtFloat16)
            continue;
        for (int offset = 0; offset < 3; ++offset) { // loop over three forms of offset in the call name:  none, Offset, and Offsets

            for (int comp = 0; comp < 2; ++comp) { // loop over presence of comp argument

                if (comp > 0 && sampler.shadow)
                    continue;

                if (offset > 0 && sampler.dim == EsdLwbe)
                    continue;

                for (int sparse = 0; sparse <= 1; ++sparse) { // loop over "bool" sparse or not
                    if (sparse && (profile == EEsProfile || version < 450))
                        continue;

                    TString s;

                    // return type
                    if (sparse)
                        s.append("int ");
                    else {
                        s.append(prefixes[sampler.type]);
                        s.append("vec4 ");
                    }

                    // name
                    if (sparse)
                        s.append("sparseTextureGather");
                    else
                        s.append("textureGather");
                    switch (offset) {
                    case 1:
                        s.append("Offset");
                        break;
                    case 2:
                        s.append("Offsets");
                        break;
                    default:
                        break;
                    }
                    if (sparse)
                        s.append("ARB");
                    s.append("(");

                    // sampler type argument
                    s.append(typeName);

                    // P coordinate argument
                    if (f16TexAddr)
                        s.append(",f16vec");
                    else
                        s.append(",vec");
                    int totalDims = dimMap[sampler.dim] + (sampler.arrayed ? 1 : 0);
                    s.append(postfixes[totalDims]);

                    // refZ argument
                    if (sampler.shadow)
                        s.append(",float");

                    // offset argument
                    if (offset > 0) {
                        s.append(",ivec2");
                        if (offset == 2)
                            s.append("[4]");
                    }

                    // texel out (for sparse texture)
                    if (sparse) {
                        s.append(",out ");
                        s.append(prefixes[sampler.type]);
                        s.append("vec4 ");
                    }

                    // comp argument
                    if (comp)
                        s.append(",int");

                    s.append(");\n");
                    commonBuiltins.append(s);
                }
            }
        }
    }

    if (sampler.dim == EsdRect || sampler.shadow)
        return;

    if (profile == EEsProfile || version < 450)
        return;

    for (int bias = 0; bias < 2; ++bias) { // loop over presence of bias argument

        for (int lod = 0; lod < 2; ++lod) { // loop over presence of lod argument

            if ((lod && bias) || (lod == 0 && bias == 0))
                continue;

            for (int f16TexAddr = 0; f16TexAddr <= 1; ++f16TexAddr) { // loop over 16-bit floating-point texel addressing

                if (f16TexAddr && sampler.type != EbtFloat16)
                    continue;

                for (int offset = 0; offset < 3; ++offset) { // loop over three forms of offset in the call name:  none, Offset, and Offsets

                    for (int comp = 0; comp < 2; ++comp) { // loop over presence of comp argument

                        if (comp == 0 && bias)
                            continue;

                        if (offset > 0 && sampler.dim == EsdLwbe)
                            continue;

                        for (int sparse = 0; sparse <= 1; ++sparse) { // loop over "bool" sparse or not
                            if (sparse && (profile == EEsProfile || version < 450))
                                continue;

                            TString s;

                            // return type
                            if (sparse)
                                s.append("int ");
                            else {
                                s.append(prefixes[sampler.type]);
                                s.append("vec4 ");
                            }

                            // name
                            if (sparse)
                                s.append("sparseTextureGather");
                            else
                                s.append("textureGather");

                            if (lod)
                                s.append("Lod");

                            switch (offset) {
                            case 1:
                                s.append("Offset");
                                break;
                            case 2:
                                s.append("Offsets");
                                break;
                            default:
                                break;
                            }

                            if (lod)
                                s.append("AMD");
                            else if (sparse)
                                s.append("ARB");

                            s.append("(");

                            // sampler type argument
                            s.append(typeName);

                            // P coordinate argument
                            if (f16TexAddr)
                                s.append(",f16vec");
                            else
                                s.append(",vec");
                            int totalDims = dimMap[sampler.dim] + (sampler.arrayed ? 1 : 0);
                            s.append(postfixes[totalDims]);

                            // lod argument
                            if (lod) {
                                if (f16TexAddr)
                                    s.append(",float16_t");
                                else
                                    s.append(",float");
                            }

                            // offset argument
                            if (offset > 0) {
                                s.append(",ivec2");
                                if (offset == 2)
                                    s.append("[4]");
                            }

                            // texel out (for sparse texture)
                            if (sparse) {
                                s.append(",out ");
                                s.append(prefixes[sampler.type]);
                                s.append("vec4 ");
                            }

                            // comp argument
                            if (comp)
                                s.append(",int");

                            // bias argument
                            if (bias) {
                                if (f16TexAddr)
                                    s.append(",float16_t");
                                else
                                    s.append(",float");
                            }

                            s.append(");\n");
                            if (bias)
                                stageBuiltins[EShLangFragment].append(s);
                            else
                                commonBuiltins.append(s);
                        }
                    }
                }
            }
        }
    }
}

//
// Add context-dependent built-in functions and variables that are present
// for the given version and profile.  All the results are put into just the
// commonBuiltins, because it is called for just a specific stage.  So,
// add stage-specific entries to the commonBuiltins, and only if that stage
// was requested.
//
void TBuiltIns::initialize(const TBuiltInResource &resources, int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language)
{
#ifdef GLSLANG_WEB
    version = 310;
    profile = EEsProfile;
#endif

    //
    // Initialize the context-dependent (resource-dependent) built-in strings for parsing.
    //

    //============================================================================
    //
    // Standard Uniforms
    //
    //============================================================================

    TString& s = commonBuiltins;
    const int maxSize = 200;
    char builtInConstant[maxSize];

    //
    // Build string of implementation dependent constants.
    //

    if (profile == EEsProfile) {
        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVertexAttribs = %d;", resources.maxVertexAttribs);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVertexUniformVectors = %d;", resources.maxVertexUniformVectors);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVertexTextureImageUnits = %d;", resources.maxVertexTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxCombinedTextureImageUnits = %d;", resources.maxCombinedTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxTextureImageUnits = %d;", resources.maxTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxFragmentUniformVectors = %d;", resources.maxFragmentUniformVectors);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxDrawBuffers = %d;", resources.maxDrawBuffers);
        s.append(builtInConstant);

        if (version == 100) {
            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVaryingVectors = %d;", resources.maxVaryingVectors);
            s.append(builtInConstant);
        } else {
            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxVertexOutputVectors = %d;", resources.maxVertexOutputVectors);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxFragmentInputVectors = %d;", resources.maxFragmentInputVectors);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const mediump int  gl_MinProgramTexelOffset = %d;", resources.minProgramTexelOffset);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxProgramTexelOffset = %d;", resources.maxProgramTexelOffset);
            s.append(builtInConstant);
        }

#ifndef GLSLANG_WEB
        if (version >= 310) {
            // geometry

            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryInputComponents = %d;", resources.maxGeometryInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryOutputComponents = %d;", resources.maxGeometryOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryImageUniforms = %d;", resources.maxGeometryImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryTextureImageUnits = %d;", resources.maxGeometryTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryOutputVertices = %d;", resources.maxGeometryOutputVertices);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryTotalOutputComponents = %d;", resources.maxGeometryTotalOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryUniformComponents = %d;", resources.maxGeometryUniformComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryAtomicCounters = %d;", resources.maxGeometryAtomicCounters);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryAtomicCounterBuffers = %d;", resources.maxGeometryAtomicCounterBuffers);
            s.append(builtInConstant);

            // tessellation

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlInputComponents = %d;", resources.maxTessControlInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlOutputComponents = %d;", resources.maxTessControlOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlTextureImageUnits = %d;", resources.maxTessControlTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlUniformComponents = %d;", resources.maxTessControlUniformComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlTotalOutputComponents = %d;", resources.maxTessControlTotalOutputComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationInputComponents = %d;", resources.maxTessEvaluationInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationOutputComponents = %d;", resources.maxTessEvaluationOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationTextureImageUnits = %d;", resources.maxTessEvaluationTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationUniformComponents = %d;", resources.maxTessEvaluationUniformComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessPatchComponents = %d;", resources.maxTessPatchComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxPatchVertices = %d;", resources.maxPatchVertices);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessGenLevel = %d;", resources.maxTessGenLevel);
            s.append(builtInConstant);

            // this is here instead of with the others in initialize(version, profile) due to the dependence on gl_MaxPatchVertices
            if (language == EShLangTessControl || language == EShLangTessEvaluation) {
                s.append(
                    "in gl_PerVertex {"
                        "highp vec4 gl_Position;"
                        "highp float gl_PointSize;"
                        "highp vec4 gl_SecondaryPositionLW;"  // GL_LW_stereo_view_rendering
                        "highp vec4 gl_PositionPerViewLW[];"  // GL_LWX_multiview_per_view_attributes
                    "} gl_in[gl_MaxPatchVertices];"
                    "\n");
            }
        }

        if (version >= 320) {
            // tessellation

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlImageUniforms = %d;", resources.maxTessControlImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationImageUniforms = %d;", resources.maxTessEvaluationImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlAtomicCounters = %d;", resources.maxTessControlAtomicCounters);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationAtomicCounters = %d;", resources.maxTessEvaluationAtomicCounters);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlAtomicCounterBuffers = %d;", resources.maxTessControlAtomicCounterBuffers);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationAtomicCounterBuffers = %d;", resources.maxTessEvaluationAtomicCounterBuffers);
            s.append(builtInConstant);
        }

        if (version >= 100) {
            // GL_EXT_blend_func_extended
            snprintf(builtInConstant, maxSize, "const mediump int gl_MaxDualSourceDrawBuffersEXT = %d;", resources.maxDualSourceDrawBuffersEXT);
            s.append(builtInConstant);
            // this is here instead of with the others in initialize(version, profile) due to the dependence on gl_MaxDualSourceDrawBuffersEXT
            if (language == EShLangFragment) {
                s.append(
                    "mediump vec4 gl_SecondaryFragColorEXT;"
                    "mediump vec4 gl_SecondaryFragDataEXT[gl_MaxDualSourceDrawBuffersEXT];"
                    "\n");
            }
        }
    } else {
        // non-ES profile

        if (version > 400) {
            snprintf(builtInConstant, maxSize, "const int  gl_MaxVertexUniformVectors = %d;", resources.maxVertexUniformVectors);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int  gl_MaxFragmentUniformVectors = %d;", resources.maxFragmentUniformVectors);
            s.append(builtInConstant);
        }

        snprintf(builtInConstant, maxSize, "const int  gl_MaxVertexAttribs = %d;", resources.maxVertexAttribs);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxVertexTextureImageUnits = %d;", resources.maxVertexTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxCombinedTextureImageUnits = %d;", resources.maxCombinedTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxTextureImageUnits = %d;", resources.maxTextureImageUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxDrawBuffers = %d;", resources.maxDrawBuffers);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxLights = %d;", resources.maxLights);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxClipPlanes = %d;", resources.maxClipPlanes);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxTextureUnits = %d;", resources.maxTextureUnits);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxTextureCoords = %d;", resources.maxTextureCoords);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int  gl_MaxVertexUniformComponents = %d;", resources.maxVertexUniformComponents);
        s.append(builtInConstant);

        if (version < 150 || ARBCompatibility) {
            snprintf(builtInConstant, maxSize, "const int  gl_MaxVaryingFloats = %d;", resources.maxVaryingFloats);
            s.append(builtInConstant);
        }

        snprintf(builtInConstant, maxSize, "const int  gl_MaxFragmentUniformComponents = %d;", resources.maxFragmentUniformComponents);
        s.append(builtInConstant);

        if (spvVersion.spv == 0 && IncludeLegacy(version, profile, spvVersion)) {
            //
            // OpenGL'uniform' state.  Page numbers are in reference to version
            // 1.4 of the OpenGL specification.
            //

            //
            // Matrix state. p. 31, 32, 37, 39, 40.
            //
            s.append("uniform mat4  gl_TextureMatrix[gl_MaxTextureCoords];"

            //
            // Derived matrix state that provides ilwerse and transposed versions
            // of the matrices above.
            //
                        "uniform mat4  gl_TextureMatrixIlwerse[gl_MaxTextureCoords];"

                        "uniform mat4  gl_TextureMatrixTranspose[gl_MaxTextureCoords];"

                        "uniform mat4  gl_TextureMatrixIlwerseTranspose[gl_MaxTextureCoords];"

            //
            // Clip planes p. 42.
            //
                        "uniform vec4  gl_ClipPlane[gl_MaxClipPlanes];"

            //
            // Light State p 50, 53, 55.
            //
                        "uniform gl_LightSourceParameters  gl_LightSource[gl_MaxLights];"

            //
            // Derived state from products of light.
            //
                        "uniform gl_LightProducts gl_FrontLightProduct[gl_MaxLights];"
                        "uniform gl_LightProducts gl_BackLightProduct[gl_MaxLights];"

            //
            // Texture Environment and Generation, p. 152, p. 40-42.
            //
                        "uniform vec4  gl_TextureElwColor[gl_MaxTextureImageUnits];"
                        "uniform vec4  gl_EyePlaneS[gl_MaxTextureCoords];"
                        "uniform vec4  gl_EyePlaneT[gl_MaxTextureCoords];"
                        "uniform vec4  gl_EyePlaneR[gl_MaxTextureCoords];"
                        "uniform vec4  gl_EyePlaneQ[gl_MaxTextureCoords];"
                        "uniform vec4  gl_ObjectPlaneS[gl_MaxTextureCoords];"
                        "uniform vec4  gl_ObjectPlaneT[gl_MaxTextureCoords];"
                        "uniform vec4  gl_ObjectPlaneR[gl_MaxTextureCoords];"
                        "uniform vec4  gl_ObjectPlaneQ[gl_MaxTextureCoords];");
        }

        if (version >= 130) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxClipDistances = %d;", resources.maxClipDistances);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxVaryingComponents = %d;", resources.maxVaryingComponents);
            s.append(builtInConstant);

            // GL_ARB_shading_language_420pack
            snprintf(builtInConstant, maxSize, "const mediump int  gl_MinProgramTexelOffset = %d;", resources.minProgramTexelOffset);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const mediump int  gl_MaxProgramTexelOffset = %d;", resources.maxProgramTexelOffset);
            s.append(builtInConstant);
        }

        // geometry
        if (version >= 150) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryInputComponents = %d;", resources.maxGeometryInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryOutputComponents = %d;", resources.maxGeometryOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryTextureImageUnits = %d;", resources.maxGeometryTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryOutputVertices = %d;", resources.maxGeometryOutputVertices);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryTotalOutputComponents = %d;", resources.maxGeometryTotalOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryUniformComponents = %d;", resources.maxGeometryUniformComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryVaryingComponents = %d;", resources.maxGeometryVaryingComponents);
            s.append(builtInConstant);

        }

        if (version >= 150) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxVertexOutputComponents = %d;", resources.maxVertexOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxFragmentInputComponents = %d;", resources.maxFragmentInputComponents);
            s.append(builtInConstant);
        }

        // tessellation
        if (version >= 150) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlInputComponents = %d;", resources.maxTessControlInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlOutputComponents = %d;", resources.maxTessControlOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlTextureImageUnits = %d;", resources.maxTessControlTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlUniformComponents = %d;", resources.maxTessControlUniformComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlTotalOutputComponents = %d;", resources.maxTessControlTotalOutputComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationInputComponents = %d;", resources.maxTessEvaluationInputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationOutputComponents = %d;", resources.maxTessEvaluationOutputComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationTextureImageUnits = %d;", resources.maxTessEvaluationTextureImageUnits);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationUniformComponents = %d;", resources.maxTessEvaluationUniformComponents);
            s.append(builtInConstant);

            snprintf(builtInConstant, maxSize, "const int gl_MaxTessPatchComponents = %d;", resources.maxTessPatchComponents);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessGenLevel = %d;", resources.maxTessGenLevel);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxPatchVertices = %d;", resources.maxPatchVertices);
            s.append(builtInConstant);

            // this is here instead of with the others in initialize(version, profile) due to the dependence on gl_MaxPatchVertices
            if (language == EShLangTessControl || language == EShLangTessEvaluation) {
                s.append(
                    "in gl_PerVertex {"
                        "vec4 gl_Position;"
                        "float gl_PointSize;"
                        "float gl_ClipDistance[];"
                    );
                if (profile == ECompatibilityProfile)
                    s.append(
                        "vec4 gl_ClipVertex;"
                        "vec4 gl_FrontColor;"
                        "vec4 gl_BackColor;"
                        "vec4 gl_FrontSecondaryColor;"
                        "vec4 gl_BackSecondaryColor;"
                        "vec4 gl_TexCoord[];"
                        "float gl_FogFragCoord;"
                        );
                if (profile != EEsProfile && version >= 450)
                    s.append(
                        "float gl_LwllDistance[];"
                        "vec4 gl_SecondaryPositionLW;"  // GL_LW_stereo_view_rendering
                        "vec4 gl_PositionPerViewLW[];"  // GL_LWX_multiview_per_view_attributes
                       );
                s.append(
                    "} gl_in[gl_MaxPatchVertices];"
                    "\n");
            }
        }

        if (version >= 150) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxViewports = %d;", resources.maxViewports);
            s.append(builtInConstant);
        }

        // images
        if (version >= 130) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedImageUnitsAndFragmentOutputs = %d;", resources.maxCombinedImageUnitsAndFragmentOutputs);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxImageSamples = %d;", resources.maxImageSamples);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlImageUniforms = %d;", resources.maxTessControlImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationImageUniforms = %d;", resources.maxTessEvaluationImageUniforms);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryImageUniforms = %d;", resources.maxGeometryImageUniforms);
            s.append(builtInConstant);
        }

        // enhanced layouts
        if (version >= 430) {
            snprintf(builtInConstant, maxSize, "const int gl_MaxTransformFeedbackBuffers = %d;", resources.maxTransformFeedbackBuffers);
            s.append(builtInConstant);
            snprintf(builtInConstant, maxSize, "const int gl_MaxTransformFeedbackInterleavedComponents = %d;", resources.maxTransformFeedbackInterleavedComponents);
            s.append(builtInConstant);
        }
#endif
    }

    // compute
    if ((profile == EEsProfile && version >= 310) || (profile != EEsProfile && version >= 420)) {
        snprintf(builtInConstant, maxSize, "const ivec3 gl_MaxComputeWorkGroupCount = ivec3(%d,%d,%d);", resources.maxComputeWorkGroupCountX,
                                                                                                         resources.maxComputeWorkGroupCountY,
                                                                                                         resources.maxComputeWorkGroupCountZ);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const ivec3 gl_MaxComputeWorkGroupSize = ivec3(%d,%d,%d);", resources.maxComputeWorkGroupSizeX,
                                                                                                        resources.maxComputeWorkGroupSizeY,
                                                                                                        resources.maxComputeWorkGroupSizeZ);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeUniformComponents = %d;", resources.maxComputeUniformComponents);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeTextureImageUnits = %d;", resources.maxComputeTextureImageUnits);
        s.append(builtInConstant);

        s.append("\n");
    }

#ifndef GLSLANG_WEB
    // images (some in compute below)
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 130)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxImageUnits = %d;", resources.maxImageUnits);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedShaderOutputResources = %d;", resources.maxCombinedShaderOutputResources);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxVertexImageUniforms = %d;", resources.maxVertexImageUniforms);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxFragmentImageUniforms = %d;", resources.maxFragmentImageUniforms);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedImageUniforms = %d;", resources.maxCombinedImageUniforms);
        s.append(builtInConstant);
    }

    // compute
    if ((profile == EEsProfile && version >= 310) || (profile != EEsProfile && version >= 420)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeImageUniforms = %d;", resources.maxComputeImageUniforms);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeAtomicCounters = %d;", resources.maxComputeAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxComputeAtomicCounterBuffers = %d;", resources.maxComputeAtomicCounterBuffers);
        s.append(builtInConstant);

        s.append("\n");
    }

    // atomic counters (some in compute below)
    if ((profile == EEsProfile && version >= 310) ||
        (profile != EEsProfile && version >= 420)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxVertexAtomicCounters = %d;", resources.               maxVertexAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxFragmentAtomicCounters = %d;", resources.             maxFragmentAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedAtomicCounters = %d;", resources.             maxCombinedAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxAtomicCounterBindings = %d;", resources.              maxAtomicCounterBindings);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxVertexAtomicCounterBuffers = %d;", resources.         maxVertexAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxFragmentAtomicCounterBuffers = %d;", resources.       maxFragmentAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedAtomicCounterBuffers = %d;", resources.       maxCombinedAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxAtomicCounterBufferSize = %d;", resources.            maxAtomicCounterBufferSize);
        s.append(builtInConstant);
    }
    if (profile != EEsProfile && version >= 420) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlAtomicCounters = %d;", resources.          maxTessControlAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationAtomicCounters = %d;", resources.       maxTessEvaluationAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryAtomicCounters = %d;", resources.             maxGeometryAtomicCounters);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxTessControlAtomicCounterBuffers = %d;", resources.    maxTessControlAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxTessEvaluationAtomicCounterBuffers = %d;", resources. maxTessEvaluationAtomicCounterBuffers);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxGeometryAtomicCounterBuffers = %d;", resources.       maxGeometryAtomicCounterBuffers);
        s.append(builtInConstant);

        s.append("\n");
    }

    // GL_ARB_lwll_distance
    if (profile != EEsProfile && version >= 450) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxLwllDistances = %d;",                resources.maxLwllDistances);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const int gl_MaxCombinedClipAndLwllDistances = %d;", resources.maxCombinedClipAndLwllDistances);
        s.append(builtInConstant);
    }

    // GL_ARB_ES3_1_compatibility
    if ((profile != EEsProfile && version >= 450) ||
        (profile == EEsProfile && version >= 310)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxSamples = %d;", resources.maxSamples);
        s.append(builtInConstant);
    }

    // SPV_LW_mesh_shader
    if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
        snprintf(builtInConstant, maxSize, "const int gl_MaxMeshOutputVerticesLW = %d;", resources.maxMeshOutputVerticesLW);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int gl_MaxMeshOutputPrimitivesLW = %d;", resources.maxMeshOutputPrimitivesLW);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const ivec3 gl_MaxMeshWorkGroupSizeLW = ivec3(%d,%d,%d);", resources.maxMeshWorkGroupSizeX_LW,
                                                                                                       resources.maxMeshWorkGroupSizeY_LW,
                                                                                                       resources.maxMeshWorkGroupSizeZ_LW);
        s.append(builtInConstant);
        snprintf(builtInConstant, maxSize, "const ivec3 gl_MaxTaskWorkGroupSizeLW = ivec3(%d,%d,%d);", resources.maxTaskWorkGroupSizeX_LW,
                                                                                                       resources.maxTaskWorkGroupSizeY_LW,
                                                                                                       resources.maxTaskWorkGroupSizeZ_LW);
        s.append(builtInConstant);

        snprintf(builtInConstant, maxSize, "const int gl_MaxMeshViewCountLW = %d;", resources.maxMeshViewCountLW);
        s.append(builtInConstant);

        s.append("\n");
    }
#endif

    s.append("\n");
}

//
// To support special built-ins that have a special qualifier that cannot be declared textually
// in a shader, like gl_Position.
//
// This lets the type of the built-in be declared textually, and then have just its qualifier be
// updated afterward.
//
// Safe to call even if name is not present.
//
// Only use this for built-in variables that have a special qualifier in TStorageQualifier.
// New built-in variables should use a generic (textually declarable) qualifier in
// TStoraregQualifier and only call BuiltIlwariable().
//
static void SpecialQualifier(const char* name, TStorageQualifier qualifier, TBuiltIlwariable builtIn, TSymbolTable& symbolTable)
{
    TSymbol* symbol = symbolTable.find(name);
    if (symbol == nullptr)
        return;

    TQualifier& symQualifier = symbol->getWritableType().getQualifier();
    symQualifier.storage = qualifier;
    symQualifier.builtIn = builtIn;
}

//
// To tag built-in variables with their TBuiltIlwariable enum.  Use this when the
// normal declaration text already gets the qualifier right, and all that's needed
// is setting the builtIn field.  This should be the normal way for all new
// built-in variables.
//
// If SpecialQualifier() was called, this does not need to be called.
//
// Safe to call even if name is not present.
//
static void BuiltIlwariable(const char* name, TBuiltIlwariable builtIn, TSymbolTable& symbolTable)
{
    TSymbol* symbol = symbolTable.find(name);
    if (symbol == nullptr)
        return;

    TQualifier& symQualifier = symbol->getWritableType().getQualifier();
    symQualifier.builtIn = builtIn;
}

//
// For built-in variables inside a named block.
// SpecialQualifier() won't ever go inside a block; their member's qualifier come
// from the qualification of the block.
//
// See comments above for other detail.
//
static void BuiltIlwariable(const char* blockName, const char* name, TBuiltIlwariable builtIn, TSymbolTable& symbolTable)
{
    TSymbol* symbol = symbolTable.find(blockName);
    if (symbol == nullptr)
        return;

    TTypeList& structure = *symbol->getWritableType().getWritableStruct();
    for (int i = 0; i < (int)structure.size(); ++i) {
        if (structure[i].type->getFieldName().compare(name) == 0) {
            structure[i].type->getQualifier().builtIn = builtIn;
            return;
        }
    }
}

//
// Finish adding/processing context-independent built-in symbols.
// 1) Programmatically add symbols that could not be added by simple text strings above.
// 2) Map built-in functions to operators, for those that will turn into an operation node
//    instead of remaining a function call.
// 3) Tag extension-related symbols added to their base version with their extensions, so
//    that if an early version has the extension turned off, there is an error reported on use.
//
void TBuiltIns::identifyBuiltIns(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language, TSymbolTable& symbolTable)
{
#ifdef GLSLANG_WEB
    version = 310;
    profile = EEsProfile;
#endif

    //
    // Tag built-in variables and functions with additional qualifier and extension information
    // that cannot be declared with the text strings.
    //

    // N.B.: a symbol should only be tagged once, and this function is called multiple times, once
    // per stage that's used for this profile.  So
    //  - generally, stick common ones in the fragment stage to ensure they are tagged exactly once
    //  - for ES, which has different precisions for different stages, the coarsest-grained tagging
    //    for a built-in used in many stages needs to be once for the fragment stage and once for
    //    the vertex stage

    switch(language) {
    case EShLangVertex:
        if (spvVersion.vulkan > 0) {
            BuiltIlwariable("gl_VertexIndex",   EbvVertexIndex,   symbolTable);
            BuiltIlwariable("gl_InstanceIndex", EbvInstanceIndex, symbolTable);
        }

#ifndef GLSLANG_WEB
        if (spvVersion.vulkan == 0) {
            SpecialQualifier("gl_VertexID",   EvqVertexId,   EbvVertexId,   symbolTable);
            SpecialQualifier("gl_InstanceID", EvqInstanceId, EbvInstanceId, symbolTable);
        }

        if (profile != EEsProfile) {
            if (version >= 440) {
                symbolTable.setVariableExtensions("gl_BaseVertexARB",   1, &E_GL_ARB_shader_draw_parameters);
                symbolTable.setVariableExtensions("gl_BaseInstanceARB", 1, &E_GL_ARB_shader_draw_parameters);
                symbolTable.setVariableExtensions("gl_DrawIDARB",       1, &E_GL_ARB_shader_draw_parameters);
                BuiltIlwariable("gl_BaseVertexARB",   EbvBaseVertex,   symbolTable);
                BuiltIlwariable("gl_BaseInstanceARB", EbvBaseInstance, symbolTable);
                BuiltIlwariable("gl_DrawIDARB",       EbvDrawId,       symbolTable);
            }
            if (version >= 460) {
                BuiltIlwariable("gl_BaseVertex",   EbvBaseVertex,   symbolTable);
                BuiltIlwariable("gl_BaseInstance", EbvBaseInstance, symbolTable);
                BuiltIlwariable("gl_DrawID",       EbvDrawId,       symbolTable);
            }
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupIlwocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            symbolTable.setFunctionExtensions("ballotARB",              1, &E_GL_ARB_shader_ballot);
            symbolTable.setFunctionExtensions("readIlwocationARB",      1, &E_GL_ARB_shader_ballot);
            symbolTable.setFunctionExtensions("readFirstIlwocationARB", 1, &E_GL_ARB_shader_ballot);

            if (version >= 430) {
                symbolTable.setFunctionExtensions("anyIlwocationARB",       1, &E_GL_ARB_shader_group_vote);
                symbolTable.setFunctionExtensions("allIlwocationsARB",      1, &E_GL_ARB_shader_group_vote);
                symbolTable.setFunctionExtensions("allIlwocationsEqualARB", 1, &E_GL_ARB_shader_group_vote);
            }
        }


        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("minIlwocationsAMD",                1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxIlwocationsAMD",                1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addIlwocationsAMD",                1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("minIlwocationsNonUniformAMD",      1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxIlwocationsNonUniformAMD",      1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addIlwocationsNonUniformAMD",      1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("swizzleIlwocationsAMD",            1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("swizzleIlwocationsWithPatternAMD", 1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("writeIlwocationAMD",               1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("mbcntAMD",                         1, &E_GL_AMD_shader_ballot);

            symbolTable.setFunctionExtensions("minIlwocationsInclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxIlwocationsInclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addIlwocationsInclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("minIlwocationsInclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxIlwocationsInclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addIlwocationsInclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("minIlwocationsExclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxIlwocationsExclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addIlwocationsExclusiveScanAMD",             1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("minIlwocationsExclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("maxIlwocationsExclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
            symbolTable.setFunctionExtensions("addIlwocationsExclusiveScanNonUniformAMD",   1, &E_GL_AMD_shader_ballot);
        }

        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("min3", 1, &E_GL_AMD_shader_trinary_minmax);
            symbolTable.setFunctionExtensions("max3", 1, &E_GL_AMD_shader_trinary_minmax);
            symbolTable.setFunctionExtensions("mid3", 1, &E_GL_AMD_shader_trinary_minmax);
        }

        if (profile != EEsProfile) {
            symbolTable.setVariableExtensions("gl_SIMDGroupSizeAMD", 1, &E_GL_AMD_gcn_shader);
            SpecialQualifier("gl_SIMDGroupSizeAMD", EvqVaryingIn, EbvSubGroupSize, symbolTable);

            symbolTable.setFunctionExtensions("lwbeFaceIndexAMD", 1, &E_GL_AMD_gcn_shader);
            symbolTable.setFunctionExtensions("lwbeFaceCoordAMD", 1, &E_GL_AMD_gcn_shader);
            symbolTable.setFunctionExtensions("timeAMD",          1, &E_GL_AMD_gcn_shader);
        }

        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("fragmentMaskFetchAMD", 1, &E_GL_AMD_shader_fragment_mask);
            symbolTable.setFunctionExtensions("fragmentFetchAMD",     1, &E_GL_AMD_shader_fragment_mask);
        }

        symbolTable.setFunctionExtensions("countLeadingZeros",  1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("countTrailingZeros", 1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("absoluteDifference", 1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("addSaturate",        1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("subtractSaturate",   1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("average",            1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("averageRounded",     1, &E_GL_INTEL_shader_integer_functions2);
        symbolTable.setFunctionExtensions("multiply32x16",      1, &E_GL_INTEL_shader_integer_functions2);

        symbolTable.setFunctionExtensions("textureFootprintLW",          1, &E_GL_LW_shader_texture_footprint);
        symbolTable.setFunctionExtensions("textureFootprintClampLW",     1, &E_GL_LW_shader_texture_footprint);
        symbolTable.setFunctionExtensions("textureFootprintLodLW",       1, &E_GL_LW_shader_texture_footprint);
        symbolTable.setFunctionExtensions("textureFootprintGradLW",      1, &E_GL_LW_shader_texture_footprint);
        symbolTable.setFunctionExtensions("textureFootprintGradClampLW", 1, &E_GL_LW_shader_texture_footprint);
        // Compatibility variables, vertex only
        if (spvVersion.spv == 0) {
            BuiltIlwariable("gl_Color",          EbvColor,          symbolTable);
            BuiltIlwariable("gl_SecondaryColor", EbvSecondaryColor, symbolTable);
            BuiltIlwariable("gl_Normal",         EbvNormal,         symbolTable);
            BuiltIlwariable("gl_Vertex",         EbvVertex,         symbolTable);
            BuiltIlwariable("gl_MultiTexCoord0", EbvMultiTexCoord0, symbolTable);
            BuiltIlwariable("gl_MultiTexCoord1", EbvMultiTexCoord1, symbolTable);
            BuiltIlwariable("gl_MultiTexCoord2", EbvMultiTexCoord2, symbolTable);
            BuiltIlwariable("gl_MultiTexCoord3", EbvMultiTexCoord3, symbolTable);
            BuiltIlwariable("gl_MultiTexCoord4", EbvMultiTexCoord4, symbolTable);
            BuiltIlwariable("gl_MultiTexCoord5", EbvMultiTexCoord5, symbolTable);
            BuiltIlwariable("gl_MultiTexCoord6", EbvMultiTexCoord6, symbolTable);
            BuiltIlwariable("gl_MultiTexCoord7", EbvMultiTexCoord7, symbolTable);
            BuiltIlwariable("gl_FogCoord",       EbvFogFragCoord,   symbolTable);
        }

        if (profile == EEsProfile) {
            if (spvVersion.spv == 0) {
                symbolTable.setFunctionExtensions("texture2DGradEXT",     1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DProjGradEXT", 1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("textureLwbeGradEXT",   1, &E_GL_EXT_shader_texture_lod);
                if (version == 310)
                    symbolTable.setFunctionExtensions("textureGatherOffsets", Num_AEP_gpu_shader5, AEP_gpu_shader5);
            }
            if (version == 310)
                symbolTable.setFunctionExtensions("fma", Num_AEP_gpu_shader5, AEP_gpu_shader5);
        }

        if (profile == EEsProfile && version < 320) {
            symbolTable.setFunctionExtensions("imageAtomicAdd",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicMin",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicMax",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicAnd",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicOr",       1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicXor",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicExchange", 1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicCompSwap", 1, &E_GL_OES_shader_image_atomic);
        }

        if (version >= 300 /* both ES and non-ES */) {
            symbolTable.setVariableExtensions("gl_ViewID_OVR", Num_OVR_multiview_EXTs, OVR_multiview_EXTs);
            BuiltIlwariable("gl_ViewID_OVR", EbvViewIndex, symbolTable);
        }

        if (profile == EEsProfile) {
            symbolTable.setFunctionExtensions("shadow2DEXT",        1, &E_GL_EXT_shadow_samplers);
            symbolTable.setFunctionExtensions("shadow2DProjEXT",    1, &E_GL_EXT_shadow_samplers);
        }
        // Fall through

    case EShLangTessControl:
        if (profile == EEsProfile && version >= 310) {
            BuiltIlwariable("gl_BoundingBoxEXT", EbvBoundingBox, symbolTable);
            symbolTable.setVariableExtensions("gl_BoundingBoxEXT", 1,
                                              &E_GL_EXT_primitive_bounding_box);
            BuiltIlwariable("gl_BoundingBoxOES", EbvBoundingBox, symbolTable);
            symbolTable.setVariableExtensions("gl_BoundingBoxOES", 1,
                                              &E_GL_OES_primitive_bounding_box);

            if (version >= 320) {
                BuiltIlwariable("gl_BoundingBox", EbvBoundingBox, symbolTable);
            }
        }
        // Fall through

    case EShLangTessEvaluation:
    case EShLangGeometry:
#endif
        SpecialQualifier("gl_Position",   EvqPosition,   EbvPosition,   symbolTable);
        SpecialQualifier("gl_PointSize",  EvqPointSize,  EbvPointSize,  symbolTable);

        BuiltIlwariable("gl_in",  "gl_Position",     EbvPosition,     symbolTable);
        BuiltIlwariable("gl_in",  "gl_PointSize",    EbvPointSize,    symbolTable);

        BuiltIlwariable("gl_out", "gl_Position",     EbvPosition,     symbolTable);
        BuiltIlwariable("gl_out", "gl_PointSize",    EbvPointSize,    symbolTable);

#ifndef GLSLANG_WEB
        SpecialQualifier("gl_ClipVertex", EvqClipVertex, EbvClipVertex, symbolTable);

        BuiltIlwariable("gl_in",  "gl_ClipDistance", EbvClipDistance, symbolTable);
        BuiltIlwariable("gl_in",  "gl_LwllDistance", EbvLwllDistance, symbolTable);

        BuiltIlwariable("gl_out", "gl_ClipDistance", EbvClipDistance, symbolTable);
        BuiltIlwariable("gl_out", "gl_LwllDistance", EbvLwllDistance, symbolTable);

        BuiltIlwariable("gl_ClipDistance",    EbvClipDistance,   symbolTable);
        BuiltIlwariable("gl_LwllDistance",    EbvLwllDistance,   symbolTable);
        BuiltIlwariable("gl_PrimitiveIDIn",   EbvPrimitiveId,    symbolTable);
        BuiltIlwariable("gl_PrimitiveID",     EbvPrimitiveId,    symbolTable);
        BuiltIlwariable("gl_IlwocationID",    EbvIlwocationId,   symbolTable);
        BuiltIlwariable("gl_Layer",           EbvLayer,          symbolTable);
        BuiltIlwariable("gl_ViewportIndex",   EbvViewportIndex,  symbolTable);

        if (language != EShLangGeometry) {
            symbolTable.setVariableExtensions("gl_Layer",         Num_viewportEXTs, viewportEXTs);
            symbolTable.setVariableExtensions("gl_ViewportIndex", Num_viewportEXTs, viewportEXTs);
        }
        symbolTable.setVariableExtensions("gl_ViewportMask",            1, &E_GL_LW_viewport_array2);
        symbolTable.setVariableExtensions("gl_SecondaryPositionLW",     1, &E_GL_LW_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_SecondaryViewportMaskLW", 1, &E_GL_LW_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_PositionPerViewLW",       1, &E_GL_LWX_multiview_per_view_attributes);
        symbolTable.setVariableExtensions("gl_ViewportMaskPerViewLW",   1, &E_GL_LWX_multiview_per_view_attributes);

        BuiltIlwariable("gl_ViewportMask",              EbvViewportMaskLW,          symbolTable);
        BuiltIlwariable("gl_SecondaryPositionLW",       EbvSecondaryPositionLW,     symbolTable);
        BuiltIlwariable("gl_SecondaryViewportMaskLW",   EbvSecondaryViewportMaskLW, symbolTable);
        BuiltIlwariable("gl_PositionPerViewLW",         EbvPositionPerViewLW,       symbolTable);
        BuiltIlwariable("gl_ViewportMaskPerViewLW",     EbvViewportMaskPerViewLW,   symbolTable);

        if (language == EShLangVertex || language == EShLangGeometry) {
            symbolTable.setVariableExtensions("gl_in", "gl_SecondaryPositionLW", 1, &E_GL_LW_stereo_view_rendering);
            symbolTable.setVariableExtensions("gl_in", "gl_PositionPerViewLW",   1, &E_GL_LWX_multiview_per_view_attributes);

            BuiltIlwariable("gl_in", "gl_SecondaryPositionLW", EbvSecondaryPositionLW, symbolTable);
            BuiltIlwariable("gl_in", "gl_PositionPerViewLW",   EbvPositionPerViewLW,   symbolTable);
        }
        symbolTable.setVariableExtensions("gl_out", "gl_ViewportMask",            1, &E_GL_LW_viewport_array2);
        symbolTable.setVariableExtensions("gl_out", "gl_SecondaryPositionLW",     1, &E_GL_LW_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_out", "gl_SecondaryViewportMaskLW", 1, &E_GL_LW_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_out", "gl_PositionPerViewLW",       1, &E_GL_LWX_multiview_per_view_attributes);
        symbolTable.setVariableExtensions("gl_out", "gl_ViewportMaskPerViewLW",   1, &E_GL_LWX_multiview_per_view_attributes);

        BuiltIlwariable("gl_out", "gl_ViewportMask",            EbvViewportMaskLW,          symbolTable);
        BuiltIlwariable("gl_out", "gl_SecondaryPositionLW",     EbvSecondaryPositionLW,     symbolTable);
        BuiltIlwariable("gl_out", "gl_SecondaryViewportMaskLW", EbvSecondaryViewportMaskLW, symbolTable);
        BuiltIlwariable("gl_out", "gl_PositionPerViewLW",       EbvPositionPerViewLW,       symbolTable);
        BuiltIlwariable("gl_out", "gl_ViewportMaskPerViewLW",   EbvViewportMaskPerViewLW,   symbolTable);

        BuiltIlwariable("gl_PatchVerticesIn", EbvPatchVertices,  symbolTable);
        BuiltIlwariable("gl_TessLevelOuter",  EbvTessLevelOuter, symbolTable);
        BuiltIlwariable("gl_TessLevelInner",  EbvTessLevelInner, symbolTable);
        BuiltIlwariable("gl_TessCoord",       EbvTessCoord,      symbolTable);

        if (version < 410)
            symbolTable.setVariableExtensions("gl_ViewportIndex", 1, &E_GL_ARB_viewport_array);

        // Compatibility variables

        BuiltIlwariable("gl_in", "gl_ClipVertex",          EbvClipVertex,          symbolTable);
        BuiltIlwariable("gl_in", "gl_FrontColor",          EbvFrontColor,          symbolTable);
        BuiltIlwariable("gl_in", "gl_BackColor",           EbvBackColor,           symbolTable);
        BuiltIlwariable("gl_in", "gl_FrontSecondaryColor", EbvFrontSecondaryColor, symbolTable);
        BuiltIlwariable("gl_in", "gl_BackSecondaryColor",  EbvBackSecondaryColor,  symbolTable);
        BuiltIlwariable("gl_in", "gl_TexCoord",            EbvTexCoord,            symbolTable);
        BuiltIlwariable("gl_in", "gl_FogFragCoord",        EbvFogFragCoord,        symbolTable);

        BuiltIlwariable("gl_out", "gl_ClipVertex",          EbvClipVertex,          symbolTable);
        BuiltIlwariable("gl_out", "gl_FrontColor",          EbvFrontColor,          symbolTable);
        BuiltIlwariable("gl_out", "gl_BackColor",           EbvBackColor,           symbolTable);
        BuiltIlwariable("gl_out", "gl_FrontSecondaryColor", EbvFrontSecondaryColor, symbolTable);
        BuiltIlwariable("gl_out", "gl_BackSecondaryColor",  EbvBackSecondaryColor,  symbolTable);
        BuiltIlwariable("gl_out", "gl_TexCoord",            EbvTexCoord,            symbolTable);
        BuiltIlwariable("gl_out", "gl_FogFragCoord",        EbvFogFragCoord,        symbolTable);

        BuiltIlwariable("gl_ClipVertex",          EbvClipVertex,          symbolTable);
        BuiltIlwariable("gl_FrontColor",          EbvFrontColor,          symbolTable);
        BuiltIlwariable("gl_BackColor",           EbvBackColor,           symbolTable);
        BuiltIlwariable("gl_FrontSecondaryColor", EbvFrontSecondaryColor, symbolTable);
        BuiltIlwariable("gl_BackSecondaryColor",  EbvBackSecondaryColor,  symbolTable);
        BuiltIlwariable("gl_TexCoord",            EbvTexCoord,            symbolTable);
        BuiltIlwariable("gl_FogFragCoord",        EbvFogFragCoord,        symbolTable);

        // gl_PointSize, when it needs to be tied to an extension, is always a member of a block.
        // (Sometimes with an instance name, sometimes anonymous).
        if (profile == EEsProfile) {
            if (language == EShLangGeometry) {
                symbolTable.setVariableExtensions("gl_PointSize", Num_AEP_geometry_point_size, AEP_geometry_point_size);
                symbolTable.setVariableExtensions("gl_in", "gl_PointSize", Num_AEP_geometry_point_size, AEP_geometry_point_size);
            } else if (language == EShLangTessEvaluation || language == EShLangTessControl) {
                // gl_in tessellation settings of gl_PointSize are in the context-dependent paths
                symbolTable.setVariableExtensions("gl_PointSize", Num_AEP_tessellation_point_size, AEP_tessellation_point_size);
                symbolTable.setVariableExtensions("gl_out", "gl_PointSize", Num_AEP_tessellation_point_size, AEP_tessellation_point_size);
            }
        }

        if ((profile != EEsProfile && version >= 140) ||
            (profile == EEsProfile && version >= 310)) {
            symbolTable.setVariableExtensions("gl_DeviceIndex",  1, &E_GL_EXT_device_group);
            BuiltIlwariable("gl_DeviceIndex", EbvDeviceIndex, symbolTable);
            symbolTable.setVariableExtensions("gl_ViewIndex", 1, &E_GL_EXT_multiview);
            BuiltIlwariable("gl_ViewIndex", EbvViewIndex, symbolTable);
        }

	if (profile != EEsProfile) {
            BuiltIlwariable("gl_SubGroupIlwocationARB", EbvSubGroupIlwocation, symbolTable);
            BuiltIlwariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0)
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
            else
                BuiltIlwariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupIlwocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltIlwariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltIlwariable("gl_SubgroupIlwocationID", EbvSubgroupIlwocation2, symbolTable);
            BuiltIlwariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            // GL_LW_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMLW",         1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountLW",            1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDLW",             1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDLW",               1, &E_GL_LW_shader_sm_builtins);
            BuiltIlwariable("gl_WarpsPerSMLW",          EbvWarpsPerSM,      symbolTable);
            BuiltIlwariable("gl_SMCountLW",             EbvSMCount,         symbolTable);
            BuiltIlwariable("gl_WarpIDLW",              EbvWarpID,          symbolTable);
            BuiltIlwariable("gl_SMIDLW",                EbvSMID,            symbolTable);
        }
#endif
        break;

    case EShLangFragment:
        SpecialQualifier("gl_FrontFacing",      EvqFace,       EbvFace,             symbolTable);
        SpecialQualifier("gl_FragCoord",        EvqFragCoord,  EbvFragCoord,        symbolTable);
        SpecialQualifier("gl_PointCoord",       EvqPointCoord, EbvPointCoord,       symbolTable);
        if (spvVersion.spv == 0)
            SpecialQualifier("gl_FragColor",    EvqFragColor,  EbvFragColor,        symbolTable);
        else {
            TSymbol* symbol = symbolTable.find("gl_FragColor");
            if (symbol) {
                symbol->getWritableType().getQualifier().storage = EvqVaryingOut;
                symbol->getWritableType().getQualifier().layoutLocation = 0;
            }
        }
        SpecialQualifier("gl_FragDepth",        EvqFragDepth,  EbvFragDepth,        symbolTable);
#ifndef GLSLANG_WEB
        SpecialQualifier("gl_FragDepthEXT",     EvqFragDepth,  EbvFragDepth,        symbolTable);
        SpecialQualifier("gl_HelperIlwocation", EvqVaryingIn,  EbvHelperIlwocation, symbolTable);

        BuiltIlwariable("gl_ClipDistance",    EbvClipDistance,   symbolTable);
        BuiltIlwariable("gl_LwllDistance",    EbvLwllDistance,   symbolTable);
        BuiltIlwariable("gl_PrimitiveID",     EbvPrimitiveId,    symbolTable);

        if (profile != EEsProfile && version >= 140) {
            symbolTable.setVariableExtensions("gl_FragStencilRefARB", 1, &E_GL_ARB_shader_stencil_export);
            BuiltIlwariable("gl_FragStencilRefARB", EbvFragStencilRef, symbolTable);
        }

        if (profile != EEsProfile && version < 400) {
            symbolTable.setFunctionExtensions("textureQueryLod", 1, &E_GL_ARB_texture_query_lod);
        }

        if (profile != EEsProfile && version >= 460) {
            symbolTable.setFunctionExtensions("rayQueryInitializeEXT",                                            1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryTerminateEXT",                                             1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGenerateIntersectionEXT",                                  1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryConfirmIntersectionEXT",                                   1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryProceedEXT",                                               1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionTypeEXT",                                   1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionTEXT",                                      1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetRayFlagsEXT",                                           1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetRayTMinEXT",                                            1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionInstanceLwstomIndexEXT",                    1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionInstanceIdEXT",                             1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT", 1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionGeometryIndexEXT",                          1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionPrimitiveIndexEXT",                         1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionBarycentricsEXT",                           1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionFrontFaceEXT",                              1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionCandidateAABBOpaqueEXT",                    1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionObjectRayDirectionEXT",                     1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionObjectRayOriginEXT",                        1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionObjectToWorldEXT",                          1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetIntersectionWorldToObjectEXT",                          1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetWorldRayOriginEXT",                                     1, &E_GL_EXT_ray_query);
            symbolTable.setFunctionExtensions("rayQueryGetWorldRayDirectionEXT",                                  1, &E_GL_EXT_ray_query);
            symbolTable.setVariableExtensions("gl_RayFlagsSkipAABBEXT",                         1, &E_GL_EXT_ray_flags_primitive_lwlling);
            symbolTable.setVariableExtensions("gl_RayFlagsSkipTrianglesEXT",                    1, &E_GL_EXT_ray_flags_primitive_lwlling);
        }

        if ((profile != EEsProfile && version >= 130) ||
            (profile == EEsProfile && version >= 310)) {
            BuiltIlwariable("gl_SampleID",           EbvSampleId,       symbolTable);
            BuiltIlwariable("gl_SamplePosition",     EbvSamplePosition, symbolTable);
            BuiltIlwariable("gl_SampleMask",         EbvSampleMask,     symbolTable);

            if (profile != EEsProfile && version < 400) {
                BuiltIlwariable("gl_NumSamples",     EbvSampleMask,     symbolTable);

                symbolTable.setVariableExtensions("gl_SampleMask",     1, &E_GL_ARB_sample_shading);
                symbolTable.setVariableExtensions("gl_SampleID",       1, &E_GL_ARB_sample_shading);
                symbolTable.setVariableExtensions("gl_SamplePosition", 1, &E_GL_ARB_sample_shading);
                symbolTable.setVariableExtensions("gl_NumSamples",     1, &E_GL_ARB_sample_shading);
            } else {
                BuiltIlwariable("gl_SampleMaskIn",    EbvSampleMask,     symbolTable);

                if (profile == EEsProfile && version < 320) {
                    symbolTable.setVariableExtensions("gl_SampleID", 1, &E_GL_OES_sample_variables);
                    symbolTable.setVariableExtensions("gl_SamplePosition", 1, &E_GL_OES_sample_variables);
                    symbolTable.setVariableExtensions("gl_SampleMaskIn", 1, &E_GL_OES_sample_variables);
                    symbolTable.setVariableExtensions("gl_SampleMask", 1, &E_GL_OES_sample_variables);
                    symbolTable.setVariableExtensions("gl_NumSamples", 1, &E_GL_OES_sample_variables);
                }
            }
        }

        BuiltIlwariable("gl_Layer",           EbvLayer,          symbolTable);
        BuiltIlwariable("gl_ViewportIndex",   EbvViewportIndex,  symbolTable);

        // Compatibility variables

        BuiltIlwariable("gl_in", "gl_FogFragCoord",   EbvFogFragCoord,   symbolTable);
        BuiltIlwariable("gl_in", "gl_TexCoord",       EbvTexCoord,       symbolTable);
        BuiltIlwariable("gl_in", "gl_Color",          EbvColor,          symbolTable);
        BuiltIlwariable("gl_in", "gl_SecondaryColor", EbvSecondaryColor, symbolTable);

        BuiltIlwariable("gl_FogFragCoord",   EbvFogFragCoord,   symbolTable);
        BuiltIlwariable("gl_TexCoord",       EbvTexCoord,       symbolTable);
        BuiltIlwariable("gl_Color",          EbvColor,          symbolTable);
        BuiltIlwariable("gl_SecondaryColor", EbvSecondaryColor, symbolTable);

        // built-in functions

        if (profile == EEsProfile) {
            if (spvVersion.spv == 0) {
                symbolTable.setFunctionExtensions("texture2DLodEXT",      1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DProjLodEXT",  1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("textureLwbeLodEXT",    1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DGradEXT",     1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DProjGradEXT", 1, &E_GL_EXT_shader_texture_lod);
                symbolTable.setFunctionExtensions("textureLwbeGradEXT",   1, &E_GL_EXT_shader_texture_lod);
                if (version < 320)
                    symbolTable.setFunctionExtensions("textureGatherOffsets", Num_AEP_gpu_shader5, AEP_gpu_shader5);
            }
            if (version == 100) {
                symbolTable.setFunctionExtensions("dFdx",   1, &E_GL_OES_standard_derivatives);
                symbolTable.setFunctionExtensions("dFdy",   1, &E_GL_OES_standard_derivatives);
                symbolTable.setFunctionExtensions("fwidth", 1, &E_GL_OES_standard_derivatives);
            }
            if (version == 310) {
                symbolTable.setFunctionExtensions("fma", Num_AEP_gpu_shader5, AEP_gpu_shader5);
                symbolTable.setFunctionExtensions("interpolateAtCentroid", 1, &E_GL_OES_shader_multisample_interpolation);
                symbolTable.setFunctionExtensions("interpolateAtSample",   1, &E_GL_OES_shader_multisample_interpolation);
                symbolTable.setFunctionExtensions("interpolateAtOffset",   1, &E_GL_OES_shader_multisample_interpolation);
            }
        } else if (version < 130) {
            if (spvVersion.spv == 0) {
                symbolTable.setFunctionExtensions("texture1DLod",        1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DLod",        1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture3DLod",        1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("textureLwbeLod",      1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture1DProjLod",    1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture2DProjLod",    1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("texture3DProjLod",    1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("shadow1DLod",         1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("shadow2DLod",         1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("shadow1DProjLod",     1, &E_GL_ARB_shader_texture_lod);
                symbolTable.setFunctionExtensions("shadow2DProjLod",     1, &E_GL_ARB_shader_texture_lod);
            }
        }

        // E_GL_ARB_shader_texture_lod functions usable only with the extension enabled
        if (profile != EEsProfile && spvVersion.spv == 0) {
            symbolTable.setFunctionExtensions("texture1DGradARB",         1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture1DProjGradARB",     1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture2DGradARB",         1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture2DProjGradARB",     1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture3DGradARB",         1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture3DProjGradARB",     1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("textureLwbeGradARB",       1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow1DGradARB",          1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow1DProjGradARB",      1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow2DGradARB",          1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow2DProjGradARB",      1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture2DRectGradARB",     1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("texture2DRectProjGradARB", 1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow2DRectGradARB",      1, &E_GL_ARB_shader_texture_lod);
            symbolTable.setFunctionExtensions("shadow2DRectProjGradARB",  1, &E_GL_ARB_shader_texture_lod);
        }

        // E_GL_ARB_shader_image_load_store
        if (profile != EEsProfile && version < 420)
            symbolTable.setFunctionExtensions("memoryBarrier", 1, &E_GL_ARB_shader_image_load_store);
        // All the image access functions are protected by checks on the type of the first argument.

        // E_GL_ARB_shader_atomic_counters
        if (profile != EEsProfile && version < 420) {
            symbolTable.setFunctionExtensions("atomicCounterIncrement", 1, &E_GL_ARB_shader_atomic_counters);
            symbolTable.setFunctionExtensions("atomicCounterDecrement", 1, &E_GL_ARB_shader_atomic_counters);
            symbolTable.setFunctionExtensions("atomicCounter"         , 1, &E_GL_ARB_shader_atomic_counters);
        }

        // E_GL_ARB_derivative_control
        if (profile != EEsProfile && version < 450) {
            symbolTable.setFunctionExtensions("dFdxFine",     1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("dFdyFine",     1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("fwidthFine",   1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("dFdxCoarse",   1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("dFdyCoarse",   1, &E_GL_ARB_derivative_control);
            symbolTable.setFunctionExtensions("fwidthCoarse", 1, &E_GL_ARB_derivative_control);
        }

        // E_GL_ARB_sparse_texture2
        if (profile != EEsProfile)
        {
            symbolTable.setFunctionExtensions("sparseTextureARB",              1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureLodARB",           1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureOffsetARB",        1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTexelFetchARB",           1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTexelFetchOffsetARB",     1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureLodOffsetARB",     1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGradARB",          1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGradOffsetARB",    1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGatherARB",        1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGatherOffsetARB",  1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTextureGatherOffsetsARB", 1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseImageLoadARB",            1, &E_GL_ARB_sparse_texture2);
            symbolTable.setFunctionExtensions("sparseTexelsResident",          1, &E_GL_ARB_sparse_texture2);
        }

        // E_GL_ARB_sparse_texture_clamp
        if (profile != EEsProfile)
        {
            symbolTable.setFunctionExtensions("sparseTextureClampARB",              1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("sparseTextureOffsetClampARB",        1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("sparseTextureGradClampARB",          1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("sparseTextureGradOffsetClampARB",    1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("textureClampARB",                    1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("textureOffsetClampARB",              1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("textureGradClampARB",                1, &E_GL_ARB_sparse_texture_clamp);
            symbolTable.setFunctionExtensions("textureGradOffsetClampARB",          1, &E_GL_ARB_sparse_texture_clamp);
        }

        // E_GL_AMD_shader_explicit_vertex_parameter
        if (profile != EEsProfile) {
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspAMD",         1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspCentroidAMD", 1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspSampleAMD",   1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordSmoothAMD",          1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordSmoothCentroidAMD",  1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordSmoothSampleAMD",    1, &E_GL_AMD_shader_explicit_vertex_parameter);
            symbolTable.setVariableExtensions("gl_BaryCoordPullModelAMD",       1, &E_GL_AMD_shader_explicit_vertex_parameter);

            symbolTable.setFunctionExtensions("interpolateAtVertexAMD",         1, &E_GL_AMD_shader_explicit_vertex_parameter);

            BuiltIlwariable("gl_BaryCoordNoPerspAMD",           EbvBaryCoordNoPersp,         symbolTable);
            BuiltIlwariable("gl_BaryCoordNoPerspCentroidAMD",   EbvBaryCoordNoPerspCentroid, symbolTable);
            BuiltIlwariable("gl_BaryCoordNoPerspSampleAMD",     EbvBaryCoordNoPerspSample,   symbolTable);
            BuiltIlwariable("gl_BaryCoordSmoothAMD",            EbvBaryCoordSmooth,          symbolTable);
            BuiltIlwariable("gl_BaryCoordSmoothCentroidAMD",    EbvBaryCoordSmoothCentroid,  symbolTable);
            BuiltIlwariable("gl_BaryCoordSmoothSampleAMD",      EbvBaryCoordSmoothSample,    symbolTable);
            BuiltIlwariable("gl_BaryCoordPullModelAMD",         EbvBaryCoordPullModel,       symbolTable);
        }

        // E_GL_AMD_texture_gather_bias_lod
        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("textureGatherLodAMD",                1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("textureGatherLodOffsetAMD",          1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("textureGatherLodOffsetsAMD",         1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("sparseTextureGatherLodAMD",          1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("sparseTextureGatherLodOffsetAMD",    1, &E_GL_AMD_texture_gather_bias_lod);
            symbolTable.setFunctionExtensions("sparseTextureGatherLodOffsetsAMD",   1, &E_GL_AMD_texture_gather_bias_lod);
        }

        // E_GL_AMD_shader_image_load_store_lod
        if (profile != EEsProfile) {
            symbolTable.setFunctionExtensions("imageLoadLodAMD",        1, &E_GL_AMD_shader_image_load_store_lod);
            symbolTable.setFunctionExtensions("imageStoreLodAMD",       1, &E_GL_AMD_shader_image_load_store_lod);
            symbolTable.setFunctionExtensions("sparseImageLoadLodAMD",  1, &E_GL_AMD_shader_image_load_store_lod);
        }
        if (profile != EEsProfile && version >= 430) {
            symbolTable.setVariableExtensions("gl_FragFullyCoveredLW", 1, &E_GL_LW_conservative_raster_underestimation);
            BuiltIlwariable("gl_FragFullyCoveredLW", EbvFragFullyCoveredLW, symbolTable);
        }
        if ((profile != EEsProfile && version >= 450) ||
            (profile == EEsProfile && version >= 320)) {
            symbolTable.setVariableExtensions("gl_FragmentSizeLW",        1, &E_GL_LW_shading_rate_image);
            symbolTable.setVariableExtensions("gl_IlwocationsPerPixelLW", 1, &E_GL_LW_shading_rate_image);
            BuiltIlwariable("gl_FragmentSizeLW",        EbvFragmentSizeLW, symbolTable);
            BuiltIlwariable("gl_IlwocationsPerPixelLW", EbvIlwocationsPerPixelLW, symbolTable);
            symbolTable.setVariableExtensions("gl_BaryCoordLW",        1, &E_GL_LW_fragment_shader_barycentric);
            symbolTable.setVariableExtensions("gl_BaryCoordNoPerspLW", 1, &E_GL_LW_fragment_shader_barycentric);
            BuiltIlwariable("gl_BaryCoordLW",        EbvBaryCoordLW,        symbolTable);
            BuiltIlwariable("gl_BaryCoordNoPerspLW", EbvBaryCoordNoPerspLW, symbolTable);
        }

        if ((profile != EEsProfile && version >= 450) ||
            (profile == EEsProfile && version >= 310)) {
            symbolTable.setVariableExtensions("gl_FragSizeEXT",            1, &E_GL_EXT_fragment_ilwocation_density);
            symbolTable.setVariableExtensions("gl_FragIlwocationCountEXT", 1, &E_GL_EXT_fragment_ilwocation_density);
            BuiltIlwariable("gl_FragSizeEXT",            EbvFragSizeEXT, symbolTable);
            BuiltIlwariable("gl_FragIlwocationCountEXT", EbvFragIlwocationCountEXT, symbolTable);
        }

        symbolTable.setVariableExtensions("gl_FragDepthEXT", 1, &E_GL_EXT_frag_depth);

        symbolTable.setFunctionExtensions("clockARB",     1, &E_GL_ARB_shader_clock);
        symbolTable.setFunctionExtensions("clock2x32ARB", 1, &E_GL_ARB_shader_clock);

        symbolTable.setFunctionExtensions("clockRealtimeEXT",     1, &E_GL_EXT_shader_realtime_clock);
        symbolTable.setFunctionExtensions("clockRealtime2x32EXT", 1, &E_GL_EXT_shader_realtime_clock);

        if (profile == EEsProfile && version < 320) {
            symbolTable.setVariableExtensions("gl_PrimitiveID",  Num_AEP_geometry_shader, AEP_geometry_shader);
            symbolTable.setVariableExtensions("gl_Layer",        Num_AEP_geometry_shader, AEP_geometry_shader);
        }

        if (profile == EEsProfile && version < 320) {
            symbolTable.setFunctionExtensions("imageAtomicAdd",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicMin",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicMax",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicAnd",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicOr",       1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicXor",      1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicExchange", 1, &E_GL_OES_shader_image_atomic);
            symbolTable.setFunctionExtensions("imageAtomicCompSwap", 1, &E_GL_OES_shader_image_atomic);
        }

        if (profile != EEsProfile && version < 330 ) {
            symbolTable.setFunctionExtensions("floatBitsToInt", 1, &E_GL_ARB_shader_bit_encoding);
            symbolTable.setFunctionExtensions("floatBitsToUint", 1, &E_GL_ARB_shader_bit_encoding);
            symbolTable.setFunctionExtensions("intBitsToFloat", 1, &E_GL_ARB_shader_bit_encoding);
            symbolTable.setFunctionExtensions("uintBitsToFloat", 1, &E_GL_ARB_shader_bit_encoding);
        }

        if (profile != EEsProfile && version < 430 ) {
            symbolTable.setFunctionExtensions("imageSize", 1, &E_GL_ARB_shader_image_size);
        }

        // GL_ARB_shader_storage_buffer_object
        if (profile != EEsProfile && version < 430 ) {
            symbolTable.setFunctionExtensions("atomicAdd", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicMin", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicMax", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicAnd", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicOr", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicXor", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicExchange", 1, &E_GL_ARB_shader_storage_buffer_object);
            symbolTable.setFunctionExtensions("atomicCompSwap", 1, &E_GL_ARB_shader_storage_buffer_object);
        }

        // GL_ARB_shading_language_packing
        if (profile != EEsProfile && version < 400 ) {
            symbolTable.setFunctionExtensions("packUnorm2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackUnorm2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("packSnorm4x8", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("packUnorm4x8", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackSnorm4x8", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackUnorm4x8", 1, &E_GL_ARB_shading_language_packing);
        }
        if (profile != EEsProfile && version < 420 ) {
            symbolTable.setFunctionExtensions("packSnorm2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackSnorm2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("unpackHalf2x16", 1, &E_GL_ARB_shading_language_packing);
            symbolTable.setFunctionExtensions("packHalf2x16", 1, &E_GL_ARB_shading_language_packing);
        }

        symbolTable.setVariableExtensions("gl_DeviceIndex",  1, &E_GL_EXT_device_group);
        BuiltIlwariable("gl_DeviceIndex", EbvDeviceIndex, symbolTable);
        symbolTable.setVariableExtensions("gl_ViewIndex", 1, &E_GL_EXT_multiview);
        BuiltIlwariable("gl_ViewIndex", EbvViewIndex, symbolTable);
        if (version >= 300 /* both ES and non-ES */) {
            symbolTable.setVariableExtensions("gl_ViewID_OVR", Num_OVR_multiview_EXTs, OVR_multiview_EXTs);
            BuiltIlwariable("gl_ViewID_OVR", EbvViewIndex, symbolTable);
        }

        // GL_ARB_shader_ballot
        if (profile != EEsProfile) {
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupIlwocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltIlwariable("gl_SubGroupIlwocationARB", EbvSubGroupIlwocation, symbolTable);
            BuiltIlwariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0)
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
            else
                BuiltIlwariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupIlwocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltIlwariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltIlwariable("gl_SubgroupIlwocationID", EbvSubgroupIlwocation2, symbolTable);
            BuiltIlwariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            symbolTable.setFunctionExtensions("subgroupBarrier",                 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupMemoryBarrier",           1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupMemoryBarrierBuffer",     1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupMemoryBarrierImage",      1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupElect",                   1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setFunctionExtensions("subgroupAll",                     1, &E_GL_KHR_shader_subgroup_vote);
            symbolTable.setFunctionExtensions("subgroupAny",                     1, &E_GL_KHR_shader_subgroup_vote);
            symbolTable.setFunctionExtensions("subgroupAllEqual",                1, &E_GL_KHR_shader_subgroup_vote);
            symbolTable.setFunctionExtensions("subgroupBroadcast",               1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBroadcastFirst",          1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallot",                  1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupIlwerseBallot",           1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotBitExtract",        1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotBitCount",          1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotInclusiveBitCount", 1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotExclusiveBitCount", 1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotFindLSB",           1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupBallotFindMSB",           1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setFunctionExtensions("subgroupShuffle",                 1, &E_GL_KHR_shader_subgroup_shuffle);
            symbolTable.setFunctionExtensions("subgroupShuffleXor",              1, &E_GL_KHR_shader_subgroup_shuffle);
            symbolTable.setFunctionExtensions("subgroupShuffleUp",               1, &E_GL_KHR_shader_subgroup_shuffle_relative);
            symbolTable.setFunctionExtensions("subgroupShuffleDown",             1, &E_GL_KHR_shader_subgroup_shuffle_relative);
            symbolTable.setFunctionExtensions("subgroupAdd",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupMul",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupMin",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupMax",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupAnd",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupOr",                      1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupXor",                     1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveAdd",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveMul",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveMin",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveMax",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveAnd",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveOr",             1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupInclusiveXor",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveAdd",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveMul",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveMin",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveMax",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveAnd",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveOr",             1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupExclusiveXor",            1, &E_GL_KHR_shader_subgroup_arithmetic);
            symbolTable.setFunctionExtensions("subgroupClusteredAdd",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredMul",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredMin",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredMax",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredAnd",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredOr",             1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupClusteredXor",            1, &E_GL_KHR_shader_subgroup_clustered);
            symbolTable.setFunctionExtensions("subgroupQuadBroadcast",           1, &E_GL_KHR_shader_subgroup_quad);
            symbolTable.setFunctionExtensions("subgroupQuadSwapHorizontal",      1, &E_GL_KHR_shader_subgroup_quad);
            symbolTable.setFunctionExtensions("subgroupQuadSwapVertical",        1, &E_GL_KHR_shader_subgroup_quad);
            symbolTable.setFunctionExtensions("subgroupQuadSwapDiagonal",        1, &E_GL_KHR_shader_subgroup_quad);
            symbolTable.setFunctionExtensions("subgroupPartitionLW",                          1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedAddLW",                     1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedMulLW",                     1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedMinLW",                     1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedMaxLW",                     1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedAndLW",                     1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedOrLW",                      1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedXorLW",                     1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveAddLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveMulLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveMinLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveMaxLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveAndLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveOrLW",             1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedInclusiveXorLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveAddLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveMulLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveMinLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveMaxLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveAndLW",            1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveOrLW",             1, &E_GL_LW_shader_subgroup_partitioned);
            symbolTable.setFunctionExtensions("subgroupPartitionedExclusiveXorLW",            1, &E_GL_LW_shader_subgroup_partitioned);

            // GL_LW_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMLW",         1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountLW",            1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDLW",             1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDLW",               1, &E_GL_LW_shader_sm_builtins);
            BuiltIlwariable("gl_WarpsPerSMLW",          EbvWarpsPerSM,      symbolTable);
            BuiltIlwariable("gl_SMCountLW",             EbvSMCount,         symbolTable);
            BuiltIlwariable("gl_WarpIDLW",              EbvWarpID,          symbolTable);
            BuiltIlwariable("gl_SMIDLW",                EbvSMID,            symbolTable);
        }

        if (profile == EEsProfile) {
            symbolTable.setFunctionExtensions("shadow2DEXT",        1, &E_GL_EXT_shadow_samplers);
            symbolTable.setFunctionExtensions("shadow2DProjEXT",    1, &E_GL_EXT_shadow_samplers);
        }

        if (spvVersion.vulkan > 0) {
            symbolTable.setVariableExtensions("gl_ScopeDevice",             1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_ScopeWorkgroup",          1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_ScopeSubgroup",           1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_ScopeIlwocation",         1, &E_GL_KHR_memory_scope_semantics);

            symbolTable.setVariableExtensions("gl_SemanticsRelaxed",        1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsAcquire",        1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsRelease",        1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsAcquireRelease", 1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsMakeAvailable",  1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsMakeVisible",    1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_SemanticsVolatile",       1, &E_GL_KHR_memory_scope_semantics);

            symbolTable.setVariableExtensions("gl_StorageSemanticsNone",    1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_StorageSemanticsBuffer",  1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_StorageSemanticsShared",  1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_StorageSemanticsImage",   1, &E_GL_KHR_memory_scope_semantics);
            symbolTable.setVariableExtensions("gl_StorageSemanticsOutput",  1, &E_GL_KHR_memory_scope_semantics);
        }

        symbolTable.setFunctionExtensions("helperIlwocationEXT",            1, &E_GL_EXT_demote_to_helper_ilwocation);
#endif
        break;

    case EShLangCompute:
        BuiltIlwariable("gl_NumWorkGroups",         EbvNumWorkGroups,        symbolTable);
        BuiltIlwariable("gl_WorkGroupSize",         EbvWorkGroupSize,        symbolTable);
        BuiltIlwariable("gl_WorkGroupID",           EbvWorkGroupId,          symbolTable);
        BuiltIlwariable("gl_LocalIlwocationID",     EbvLocalIlwocationId,    symbolTable);
        BuiltIlwariable("gl_GlobalIlwocationID",    EbvGlobalIlwocationId,   symbolTable);
        BuiltIlwariable("gl_LocalIlwocationIndex",  EbvLocalIlwocationIndex, symbolTable);
        BuiltIlwariable("gl_DeviceIndex",           EbvDeviceIndex,          symbolTable);
        BuiltIlwariable("gl_ViewIndex",             EbvViewIndex,            symbolTable);

#ifndef GLSLANG_WEB
        if ((profile != EEsProfile && version >= 140) ||
            (profile == EEsProfile && version >= 310)) {
            symbolTable.setVariableExtensions("gl_DeviceIndex",  1, &E_GL_EXT_device_group);
            symbolTable.setVariableExtensions("gl_ViewIndex",    1, &E_GL_EXT_multiview);
        }

        if (profile != EEsProfile && version < 430) {
            symbolTable.setVariableExtensions("gl_NumWorkGroups",        1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_WorkGroupSize",        1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_WorkGroupID",          1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_LocalIlwocationID",    1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_GlobalIlwocationID",   1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_LocalIlwocationIndex", 1, &E_GL_ARB_compute_shader);

            symbolTable.setVariableExtensions("gl_MaxComputeWorkGroupCount",       1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeWorkGroupSize",        1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeUniformComponents",    1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeTextureImageUnits",    1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeImageUniforms",        1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeAtomicCounters",       1, &E_GL_ARB_compute_shader);
            symbolTable.setVariableExtensions("gl_MaxComputeAtomicCounterBuffers", 1, &E_GL_ARB_compute_shader);

            symbolTable.setFunctionExtensions("barrier",                    1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("memoryBarrierAtomicCounter", 1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("memoryBarrierBuffer",        1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("memoryBarrierImage",         1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("memoryBarrierShared",        1, &E_GL_ARB_compute_shader);
            symbolTable.setFunctionExtensions("groupMemoryBarrier",         1, &E_GL_ARB_compute_shader);
        }


        symbolTable.setFunctionExtensions("controlBarrier",                 1, &E_GL_KHR_memory_scope_semantics);
        symbolTable.setFunctionExtensions("debugPrintfEXT",                 1, &E_GL_EXT_debug_printf);

        // GL_ARB_shader_ballot
        if (profile != EEsProfile) {
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupIlwocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltIlwariable("gl_SubGroupIlwocationARB", EbvSubGroupIlwocation, symbolTable);
            BuiltIlwariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0)
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
            else
                BuiltIlwariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupIlwocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltIlwariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltIlwariable("gl_SubgroupIlwocationID", EbvSubgroupIlwocation2, symbolTable);
            BuiltIlwariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            // GL_LW_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMLW",         1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountLW",            1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDLW",             1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDLW",               1, &E_GL_LW_shader_sm_builtins);
            BuiltIlwariable("gl_WarpsPerSMLW",          EbvWarpsPerSM,      symbolTable);
            BuiltIlwariable("gl_SMCountLW",             EbvSMCount,         symbolTable);
            BuiltIlwariable("gl_WarpIDLW",              EbvWarpID,          symbolTable);
            BuiltIlwariable("gl_SMIDLW",                EbvSMID,            symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_NumSubgroups", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupID",   1, &E_GL_KHR_shader_subgroup_basic);

            BuiltIlwariable("gl_NumSubgroups", EbvNumSubgroups, symbolTable);
            BuiltIlwariable("gl_SubgroupID",   EbvSubgroupID,   symbolTable);

            symbolTable.setFunctionExtensions("subgroupMemoryBarrierShared", 1, &E_GL_KHR_shader_subgroup_basic);
        }

        {
            const char *coopExt[2] = { E_GL_LW_cooperative_matrix, E_GL_LW_integer_cooperative_matrix };
            symbolTable.setFunctionExtensions("coopMatLoadLW",   2, coopExt);
            symbolTable.setFunctionExtensions("coopMatStoreLW",  2, coopExt);
            symbolTable.setFunctionExtensions("coopMatMulAddLW", 2, coopExt);
        }

        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            symbolTable.setFunctionExtensions("dFdx",                   1, &E_GL_LW_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdy",                   1, &E_GL_LW_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("fwidth",                 1, &E_GL_LW_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdxFine",               1, &E_GL_LW_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdyFine",               1, &E_GL_LW_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("fwidthFine",             1, &E_GL_LW_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdxCoarse",             1, &E_GL_LW_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("dFdyCoarse",             1, &E_GL_LW_compute_shader_derivatives);
            symbolTable.setFunctionExtensions("fwidthCoarse",           1, &E_GL_LW_compute_shader_derivatives);
        }
#endif
        break;

#ifndef GLSLANG_WEB
    case EShLangRayGen:
    case EShLangIntersect:
    case EShLangAnyHit:
    case EShLangClosestHit:
    case EShLangMiss:
    case EShLangCallable:
        if (profile != EEsProfile && version >= 460) {
            const char *rtexts[] = { E_GL_LW_ray_tracing, E_GL_EXT_ray_tracing };
            symbolTable.setVariableExtensions("gl_LaunchIDLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_LaunchIDEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_LaunchSizeLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_LaunchSizeEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_PrimitiveID", 2, rtexts);
            symbolTable.setVariableExtensions("gl_InstanceID", 2, rtexts);
            symbolTable.setVariableExtensions("gl_InstanceLwstomIndexLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_InstanceLwstomIndexEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_GeometryIndexEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldRayOriginLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldRayOriginEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldRayDirectionLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldRayDirectionEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectRayOriginLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectRayOriginEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectRayDirectionLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectRayDirectionEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_RayTminLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_RayTminEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_RayTmaxLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_RayTmaxEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_HitTLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_HitTEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_HitKindLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_HitKindEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectToWorldLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectToWorldEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_ObjectToWorld3x4EXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldToObjectLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldToObjectEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_WorldToObject3x4EXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setVariableExtensions("gl_IncomingRayFlagsLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setVariableExtensions("gl_IncomingRayFlagsEXT", 1, &E_GL_EXT_ray_tracing);

            symbolTable.setVariableExtensions("gl_DeviceIndex", 1, &E_GL_EXT_device_group);


            symbolTable.setFunctionExtensions("traceLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setFunctionExtensions("traceRayEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setFunctionExtensions("reportIntersectionLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setFunctionExtensions("reportIntersectionEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setFunctionExtensions("ignoreIntersectionLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setFunctionExtensions("ignoreIntersectionEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setFunctionExtensions("terminateRayLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setFunctionExtensions("terminateRayEXT", 1, &E_GL_EXT_ray_tracing);
            symbolTable.setFunctionExtensions("exelwteCallableLW", 1, &E_GL_LW_ray_tracing);
            symbolTable.setFunctionExtensions("exelwteCallableEXT", 1, &E_GL_EXT_ray_tracing);


            BuiltIlwariable("gl_LaunchIDLW",             EbvLaunchId,           symbolTable);
            BuiltIlwariable("gl_LaunchIDEXT",            EbvLaunchId,           symbolTable);
            BuiltIlwariable("gl_LaunchSizeLW",           EbvLaunchSize,         symbolTable);
            BuiltIlwariable("gl_LaunchSizeEXT",          EbvLaunchSize,         symbolTable);
            BuiltIlwariable("gl_PrimitiveID",            EbvPrimitiveId,        symbolTable);
            BuiltIlwariable("gl_InstanceID",             EbvInstanceId,         symbolTable);
            BuiltIlwariable("gl_InstanceLwstomIndexLW",  EbvInstanceLwstomIndex,symbolTable);
            BuiltIlwariable("gl_InstanceLwstomIndexEXT", EbvInstanceLwstomIndex,symbolTable);
            BuiltIlwariable("gl_GeometryIndexEXT",       EbvGeometryIndex,      symbolTable);
            BuiltIlwariable("gl_WorldRayOriginLW",       EbvWorldRayOrigin,     symbolTable);
            BuiltIlwariable("gl_WorldRayOriginEXT",      EbvWorldRayOrigin,     symbolTable);
            BuiltIlwariable("gl_WorldRayDirectionLW",    EbvWorldRayDirection,  symbolTable);
            BuiltIlwariable("gl_WorldRayDirectionEXT",   EbvWorldRayDirection,  symbolTable);
            BuiltIlwariable("gl_ObjectRayOriginLW",      EbvObjectRayOrigin,    symbolTable);
            BuiltIlwariable("gl_ObjectRayOriginEXT",     EbvObjectRayOrigin,    symbolTable);
            BuiltIlwariable("gl_ObjectRayDirectionLW",   EbvObjectRayDirection, symbolTable);
            BuiltIlwariable("gl_ObjectRayDirectionEXT",  EbvObjectRayDirection, symbolTable);
            BuiltIlwariable("gl_RayTminLW",              EbvRayTmin,            symbolTable);
            BuiltIlwariable("gl_RayTminEXT",             EbvRayTmin,            symbolTable);
            BuiltIlwariable("gl_RayTmaxLW",              EbvRayTmax,            symbolTable);
            BuiltIlwariable("gl_RayTmaxEXT",             EbvRayTmax,            symbolTable);
            BuiltIlwariable("gl_HitTLW",                 EbvHitT,               symbolTable);
            BuiltIlwariable("gl_HitTEXT",                EbvHitT,               symbolTable);
            BuiltIlwariable("gl_HitKindLW",              EbvHitKind,            symbolTable);
            BuiltIlwariable("gl_HitKindEXT",             EbvHitKind,            symbolTable);
            BuiltIlwariable("gl_ObjectToWorldLW",        EbvObjectToWorld,      symbolTable);
            BuiltIlwariable("gl_ObjectToWorldEXT",       EbvObjectToWorld,      symbolTable);
            BuiltIlwariable("gl_ObjectToWorld3x4EXT",    EbvObjectToWorld3x4,   symbolTable);
            BuiltIlwariable("gl_WorldToObjectLW",        EbvWorldToObject,      symbolTable);
            BuiltIlwariable("gl_WorldToObjectEXT",       EbvWorldToObject,      symbolTable);
            BuiltIlwariable("gl_WorldToObject3x4EXT",    EbvWorldToObject3x4,   symbolTable);
            BuiltIlwariable("gl_IncomingRayFlagsLW",     EbvIncomingRayFlags,   symbolTable);
            BuiltIlwariable("gl_IncomingRayFlagsEXT",    EbvIncomingRayFlags,   symbolTable);
            BuiltIlwariable("gl_DeviceIndex",            EbvDeviceIndex,        symbolTable);

            // GL_ARB_shader_ballot
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupIlwocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltIlwariable("gl_SubGroupIlwocationARB", EbvSubGroupIlwocation, symbolTable);
            BuiltIlwariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0)
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
            else
                BuiltIlwariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);

            // GL_KHR_shader_subgroup
            symbolTable.setVariableExtensions("gl_NumSubgroups",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupID",           1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupIlwocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltIlwariable("gl_NumSubgroups",         EbvNumSubgroups,        symbolTable);
            BuiltIlwariable("gl_SubgroupID",           EbvSubgroupID,          symbolTable);
            BuiltIlwariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltIlwariable("gl_SubgroupIlwocationID", EbvSubgroupIlwocation2, symbolTable);
            BuiltIlwariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            // GL_LW_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMLW",         1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountLW",            1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDLW",             1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDLW",               1, &E_GL_LW_shader_sm_builtins);
            BuiltIlwariable("gl_WarpsPerSMLW",          EbvWarpsPerSM,      symbolTable);
            BuiltIlwariable("gl_SMCountLW",             EbvSMCount,         symbolTable);
            BuiltIlwariable("gl_WarpIDLW",              EbvWarpID,          symbolTable);
            BuiltIlwariable("gl_SMIDLW",                EbvSMID,            symbolTable);
        }
        break;

    case EShLangMeshLW:
        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            // per-vertex builtins
            symbolTable.setVariableExtensions("gl_MeshVerticesLW", "gl_Position",     1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesLW", "gl_PointSize",    1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesLW", "gl_ClipDistance", 1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesLW", "gl_LwllDistance", 1, &E_GL_LW_mesh_shader);

            BuiltIlwariable("gl_MeshVerticesLW", "gl_Position",     EbvPosition,     symbolTable);
            BuiltIlwariable("gl_MeshVerticesLW", "gl_PointSize",    EbvPointSize,    symbolTable);
            BuiltIlwariable("gl_MeshVerticesLW", "gl_ClipDistance", EbvClipDistance, symbolTable);
            BuiltIlwariable("gl_MeshVerticesLW", "gl_LwllDistance", EbvLwllDistance, symbolTable);

            symbolTable.setVariableExtensions("gl_MeshVerticesLW", "gl_PositionPerViewLW",     1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesLW", "gl_ClipDistancePerViewLW", 1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshVerticesLW", "gl_LwllDistancePerViewLW", 1, &E_GL_LW_mesh_shader);

            BuiltIlwariable("gl_MeshVerticesLW", "gl_PositionPerViewLW",     EbvPositionPerViewLW,     symbolTable);
            BuiltIlwariable("gl_MeshVerticesLW", "gl_ClipDistancePerViewLW", EbvClipDistancePerViewLW, symbolTable);
            BuiltIlwariable("gl_MeshVerticesLW", "gl_LwllDistancePerViewLW", EbvLwllDistancePerViewLW, symbolTable);

            // per-primitive builtins
            symbolTable.setVariableExtensions("gl_MeshPrimitivesLW", "gl_PrimitiveID",   1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesLW", "gl_Layer",         1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesLW", "gl_ViewportIndex", 1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesLW", "gl_ViewportMask",  1, &E_GL_LW_mesh_shader);

            BuiltIlwariable("gl_MeshPrimitivesLW", "gl_PrimitiveID",   EbvPrimitiveId,    symbolTable);
            BuiltIlwariable("gl_MeshPrimitivesLW", "gl_Layer",         EbvLayer,          symbolTable);
            BuiltIlwariable("gl_MeshPrimitivesLW", "gl_ViewportIndex", EbvViewportIndex,  symbolTable);
            BuiltIlwariable("gl_MeshPrimitivesLW", "gl_ViewportMask",  EbvViewportMaskLW, symbolTable);

            // per-view per-primitive builtins
            symbolTable.setVariableExtensions("gl_MeshPrimitivesLW", "gl_LayerPerViewLW",        1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshPrimitivesLW", "gl_ViewportMaskPerViewLW", 1, &E_GL_LW_mesh_shader);

            BuiltIlwariable("gl_MeshPrimitivesLW", "gl_LayerPerViewLW",        EbvLayerPerViewLW,        symbolTable);
            BuiltIlwariable("gl_MeshPrimitivesLW", "gl_ViewportMaskPerViewLW", EbvViewportMaskPerViewLW, symbolTable);

            // other builtins
            symbolTable.setVariableExtensions("gl_PrimitiveCountLW",     1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_PrimitiveIndicesLW",   1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshViewCountLW",      1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshViewIndicesLW",    1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_WorkGroupSize",        1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_WorkGroupID",          1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_LocalIlwocationID",    1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_GlobalIlwocationID",   1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_LocalIlwocationIndex", 1, &E_GL_LW_mesh_shader);

            BuiltIlwariable("gl_PrimitiveCountLW",     EbvPrimitiveCountLW,     symbolTable);
            BuiltIlwariable("gl_PrimitiveIndicesLW",   EbvPrimitiveIndicesLW,   symbolTable);
            BuiltIlwariable("gl_MeshViewCountLW",      EbvMeshViewCountLW,      symbolTable);
            BuiltIlwariable("gl_MeshViewIndicesLW",    EbvMeshViewIndicesLW,    symbolTable);
            BuiltIlwariable("gl_WorkGroupSize",        EbvWorkGroupSize,        symbolTable);
            BuiltIlwariable("gl_WorkGroupID",          EbvWorkGroupId,          symbolTable);
            BuiltIlwariable("gl_LocalIlwocationID",    EbvLocalIlwocationId,    symbolTable);
            BuiltIlwariable("gl_GlobalIlwocationID",   EbvGlobalIlwocationId,   symbolTable);
            BuiltIlwariable("gl_LocalIlwocationIndex", EbvLocalIlwocationIndex, symbolTable);

            // builtin constants
            symbolTable.setVariableExtensions("gl_MaxMeshOutputVerticesLW",   1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MaxMeshOutputPrimitivesLW", 1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MaxMeshWorkGroupSizeLW",    1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MaxMeshViewCountLW",        1, &E_GL_LW_mesh_shader);

            // builtin functions
            symbolTable.setFunctionExtensions("barrier",                      1, &E_GL_LW_mesh_shader);
            symbolTable.setFunctionExtensions("memoryBarrierShared",          1, &E_GL_LW_mesh_shader);
            symbolTable.setFunctionExtensions("groupMemoryBarrier",           1, &E_GL_LW_mesh_shader);
        }

        if (profile != EEsProfile && version >= 450) {
            // GL_EXT_device_group
            symbolTable.setVariableExtensions("gl_DeviceIndex", 1, &E_GL_EXT_device_group);
            BuiltIlwariable("gl_DeviceIndex", EbvDeviceIndex, symbolTable);

            // GL_ARB_shader_draw_parameters
            symbolTable.setVariableExtensions("gl_DrawIDARB", 1, &E_GL_ARB_shader_draw_parameters);
            BuiltIlwariable("gl_DrawIDARB", EbvDrawId, symbolTable);
            if (version >= 460) {
                BuiltIlwariable("gl_DrawID", EbvDrawId, symbolTable);
            }

            // GL_ARB_shader_ballot
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupIlwocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltIlwariable("gl_SubGroupIlwocationARB", EbvSubGroupIlwocation, symbolTable);
            BuiltIlwariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0)
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
            else
                BuiltIlwariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_NumSubgroups",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupID",           1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupIlwocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltIlwariable("gl_NumSubgroups",         EbvNumSubgroups,        symbolTable);
            BuiltIlwariable("gl_SubgroupID",           EbvSubgroupID,          symbolTable);
            BuiltIlwariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltIlwariable("gl_SubgroupIlwocationID", EbvSubgroupIlwocation2, symbolTable);
            BuiltIlwariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            symbolTable.setFunctionExtensions("subgroupMemoryBarrierShared", 1, &E_GL_KHR_shader_subgroup_basic);

            // GL_LW_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMLW",         1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountLW",            1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDLW",             1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDLW",               1, &E_GL_LW_shader_sm_builtins);
            BuiltIlwariable("gl_WarpsPerSMLW",          EbvWarpsPerSM,      symbolTable);
            BuiltIlwariable("gl_SMCountLW",             EbvSMCount,         symbolTable);
            BuiltIlwariable("gl_WarpIDLW",              EbvWarpID,          symbolTable);
            BuiltIlwariable("gl_SMIDLW",                EbvSMID,            symbolTable);
        }
        break;

    case EShLangTaskLW:
        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            symbolTable.setVariableExtensions("gl_TaskCountLW",          1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_WorkGroupSize",        1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_WorkGroupID",          1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_LocalIlwocationID",    1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_GlobalIlwocationID",   1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_LocalIlwocationIndex", 1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshViewCountLW",      1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MeshViewIndicesLW",    1, &E_GL_LW_mesh_shader);

            BuiltIlwariable("gl_TaskCountLW",          EbvTaskCountLW,          symbolTable);
            BuiltIlwariable("gl_WorkGroupSize",        EbvWorkGroupSize,        symbolTable);
            BuiltIlwariable("gl_WorkGroupID",          EbvWorkGroupId,          symbolTable);
            BuiltIlwariable("gl_LocalIlwocationID",    EbvLocalIlwocationId,    symbolTable);
            BuiltIlwariable("gl_GlobalIlwocationID",   EbvGlobalIlwocationId,   symbolTable);
            BuiltIlwariable("gl_LocalIlwocationIndex", EbvLocalIlwocationIndex, symbolTable);
            BuiltIlwariable("gl_MeshViewCountLW",      EbvMeshViewCountLW,      symbolTable);
            BuiltIlwariable("gl_MeshViewIndicesLW",    EbvMeshViewIndicesLW,    symbolTable);

            symbolTable.setVariableExtensions("gl_MaxTaskWorkGroupSizeLW", 1, &E_GL_LW_mesh_shader);
            symbolTable.setVariableExtensions("gl_MaxMeshViewCountLW",     1, &E_GL_LW_mesh_shader);

            symbolTable.setFunctionExtensions("barrier",                   1, &E_GL_LW_mesh_shader);
            symbolTable.setFunctionExtensions("memoryBarrierShared",       1, &E_GL_LW_mesh_shader);
            symbolTable.setFunctionExtensions("groupMemoryBarrier",        1, &E_GL_LW_mesh_shader);
        }

        if (profile != EEsProfile && version >= 450) {
            // GL_EXT_device_group
            symbolTable.setVariableExtensions("gl_DeviceIndex", 1, &E_GL_EXT_device_group);
            BuiltIlwariable("gl_DeviceIndex", EbvDeviceIndex, symbolTable);

            // GL_ARB_shader_draw_parameters
            symbolTable.setVariableExtensions("gl_DrawIDARB", 1, &E_GL_ARB_shader_draw_parameters);
            BuiltIlwariable("gl_DrawIDARB", EbvDrawId, symbolTable);
            if (version >= 460) {
                BuiltIlwariable("gl_DrawID", EbvDrawId, symbolTable);
            }

            // GL_ARB_shader_ballot
            symbolTable.setVariableExtensions("gl_SubGroupSizeARB",       1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupIlwocationARB", 1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupEqMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupGtMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLeMaskARB",     1, &E_GL_ARB_shader_ballot);
            symbolTable.setVariableExtensions("gl_SubGroupLtMaskARB",     1, &E_GL_ARB_shader_ballot);

            BuiltIlwariable("gl_SubGroupIlwocationARB", EbvSubGroupIlwocation, symbolTable);
            BuiltIlwariable("gl_SubGroupEqMaskARB",     EbvSubGroupEqMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGeMaskARB",     EbvSubGroupGeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupGtMaskARB",     EbvSubGroupGtMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLeMaskARB",     EbvSubGroupLeMask,     symbolTable);
            BuiltIlwariable("gl_SubGroupLtMaskARB",     EbvSubGroupLtMask,     symbolTable);

            if (spvVersion.vulkan > 0)
                // Treat "gl_SubGroupSizeARB" as shader input instead of uniform for Vulkan
                SpecialQualifier("gl_SubGroupSizeARB", EvqVaryingIn, EbvSubGroupSize, symbolTable);
            else
                BuiltIlwariable("gl_SubGroupSizeARB", EbvSubGroupSize, symbolTable);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.setVariableExtensions("gl_NumSubgroups",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupID",           1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupSize",         1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupIlwocationID", 1, &E_GL_KHR_shader_subgroup_basic);
            symbolTable.setVariableExtensions("gl_SubgroupEqMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupGtMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLeMask",       1, &E_GL_KHR_shader_subgroup_ballot);
            symbolTable.setVariableExtensions("gl_SubgroupLtMask",       1, &E_GL_KHR_shader_subgroup_ballot);

            BuiltIlwariable("gl_NumSubgroups",         EbvNumSubgroups,        symbolTable);
            BuiltIlwariable("gl_SubgroupID",           EbvSubgroupID,          symbolTable);
            BuiltIlwariable("gl_SubgroupSize",         EbvSubgroupSize2,       symbolTable);
            BuiltIlwariable("gl_SubgroupIlwocationID", EbvSubgroupIlwocation2, symbolTable);
            BuiltIlwariable("gl_SubgroupEqMask",       EbvSubgroupEqMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGeMask",       EbvSubgroupGeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupGtMask",       EbvSubgroupGtMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLeMask",       EbvSubgroupLeMask2,     symbolTable);
            BuiltIlwariable("gl_SubgroupLtMask",       EbvSubgroupLtMask2,     symbolTable);

            symbolTable.setFunctionExtensions("subgroupMemoryBarrierShared", 1, &E_GL_KHR_shader_subgroup_basic);

            // GL_LW_shader_sm_builtins
            symbolTable.setVariableExtensions("gl_WarpsPerSMLW",         1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMCountLW",            1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_WarpIDLW",             1, &E_GL_LW_shader_sm_builtins);
            symbolTable.setVariableExtensions("gl_SMIDLW",               1, &E_GL_LW_shader_sm_builtins);
            BuiltIlwariable("gl_WarpsPerSMLW",          EbvWarpsPerSM,      symbolTable);
            BuiltIlwariable("gl_SMCountLW",             EbvSMCount,         symbolTable);
            BuiltIlwariable("gl_WarpIDLW",              EbvWarpID,          symbolTable);
            BuiltIlwariable("gl_SMIDLW",                EbvSMID,            symbolTable);
        }
        break;
#endif

    default:
        assert(false && "Language not supported");
        break;
    }

    //
    // Next, identify which built-ins have a mapping to an operator.
    // If PureOperatorBuiltins is false, those that are not identified as such are
    // expected to be resolved through a library of functions, versus as
    // operations.
    //

    relateTabledBuiltins(version, profile, spvVersion, language, symbolTable);

#ifndef GLSLANG_WEB
    symbolTable.relateToOperator("doubleBitsToInt64",  EOpDoubleBitsToInt64);
    symbolTable.relateToOperator("doubleBitsToUint64", EOpDoubleBitsToUint64);
    symbolTable.relateToOperator("int64BitsToDouble",  EOpInt64BitsToDouble);
    symbolTable.relateToOperator("uint64BitsToDouble", EOpUint64BitsToDouble);
    symbolTable.relateToOperator("halfBitsToInt16",  EOpFloat16BitsToInt16);
    symbolTable.relateToOperator("halfBitsToUint16", EOpFloat16BitsToUint16);
    symbolTable.relateToOperator("float16BitsToInt16",  EOpFloat16BitsToInt16);
    symbolTable.relateToOperator("float16BitsToUint16", EOpFloat16BitsToUint16);
    symbolTable.relateToOperator("int16BitsToFloat16",  EOpInt16BitsToFloat16);
    symbolTable.relateToOperator("uint16BitsToFloat16", EOpUint16BitsToFloat16);

    symbolTable.relateToOperator("int16BitsToHalf",  EOpInt16BitsToFloat16);
    symbolTable.relateToOperator("uint16BitsToHalf", EOpUint16BitsToFloat16);

    symbolTable.relateToOperator("packSnorm4x8",    EOpPackSnorm4x8);
    symbolTable.relateToOperator("unpackSnorm4x8",  EOpUnpackSnorm4x8);
    symbolTable.relateToOperator("packUnorm4x8",    EOpPackUnorm4x8);
    symbolTable.relateToOperator("unpackUnorm4x8",  EOpUnpackUnorm4x8);

    symbolTable.relateToOperator("packDouble2x32",    EOpPackDouble2x32);
    symbolTable.relateToOperator("unpackDouble2x32",  EOpUnpackDouble2x32);

    symbolTable.relateToOperator("packInt2x32",     EOpPackInt2x32);
    symbolTable.relateToOperator("unpackInt2x32",   EOpUnpackInt2x32);
    symbolTable.relateToOperator("packUint2x32",    EOpPackUint2x32);
    symbolTable.relateToOperator("unpackUint2x32",  EOpUnpackUint2x32);

    symbolTable.relateToOperator("packInt2x16",     EOpPackInt2x16);
    symbolTable.relateToOperator("unpackInt2x16",   EOpUnpackInt2x16);
    symbolTable.relateToOperator("packUint2x16",    EOpPackUint2x16);
    symbolTable.relateToOperator("unpackUint2x16",  EOpUnpackUint2x16);

    symbolTable.relateToOperator("packInt4x16",     EOpPackInt4x16);
    symbolTable.relateToOperator("unpackInt4x16",   EOpUnpackInt4x16);
    symbolTable.relateToOperator("packUint4x16",    EOpPackUint4x16);
    symbolTable.relateToOperator("unpackUint4x16",  EOpUnpackUint4x16);
    symbolTable.relateToOperator("packFloat2x16",   EOpPackFloat2x16);
    symbolTable.relateToOperator("unpackFloat2x16", EOpUnpackFloat2x16);

    symbolTable.relateToOperator("pack16",          EOpPack16);
    symbolTable.relateToOperator("pack32",          EOpPack32);
    symbolTable.relateToOperator("pack64",          EOpPack64);

    symbolTable.relateToOperator("unpack32",        EOpUnpack32);
    symbolTable.relateToOperator("unpack16",        EOpUnpack16);
    symbolTable.relateToOperator("unpack8",         EOpUnpack8);

    symbolTable.relateToOperator("controlBarrier",             EOpBarrier);
    symbolTable.relateToOperator("memoryBarrierAtomicCounter", EOpMemoryBarrierAtomicCounter);
    symbolTable.relateToOperator("memoryBarrierImage",         EOpMemoryBarrierImage);

    symbolTable.relateToOperator("atomicLoad",     EOpAtomicLoad);
    symbolTable.relateToOperator("atomicStore",    EOpAtomicStore);

    symbolTable.relateToOperator("atomicCounterIncrement", EOpAtomicCounterIncrement);
    symbolTable.relateToOperator("atomicCounterDecrement", EOpAtomicCounterDecrement);
    symbolTable.relateToOperator("atomicCounter",          EOpAtomicCounter);

    symbolTable.relateToOperator("clockARB",     EOpReadClockSubgroupKHR);
    symbolTable.relateToOperator("clock2x32ARB", EOpReadClockSubgroupKHR);

    symbolTable.relateToOperator("clockRealtimeEXT",     EOpReadClockDeviceKHR);
    symbolTable.relateToOperator("clockRealtime2x32EXT", EOpReadClockDeviceKHR);

    if (profile != EEsProfile && version >= 460) {
        symbolTable.relateToOperator("atomicCounterAdd",      EOpAtomicCounterAdd);
        symbolTable.relateToOperator("atomicCounterSubtract", EOpAtomicCounterSubtract);
        symbolTable.relateToOperator("atomicCounterMin",      EOpAtomicCounterMin);
        symbolTable.relateToOperator("atomicCounterMax",      EOpAtomicCounterMax);
        symbolTable.relateToOperator("atomicCounterAnd",      EOpAtomicCounterAnd);
        symbolTable.relateToOperator("atomicCounterOr",       EOpAtomicCounterOr);
        symbolTable.relateToOperator("atomicCounterXor",      EOpAtomicCounterXor);
        symbolTable.relateToOperator("atomicCounterExchange", EOpAtomicCounterExchange);
        symbolTable.relateToOperator("atomicCounterCompSwap", EOpAtomicCounterCompSwap);
    }

    symbolTable.relateToOperator("fma",               EOpFma);
    symbolTable.relateToOperator("frexp",             EOpFrexp);
    symbolTable.relateToOperator("ldexp",             EOpLdexp);
    symbolTable.relateToOperator("uaddCarry",         EOpAddCarry);
    symbolTable.relateToOperator("usubBorrow",        EOpSubBorrow);
    symbolTable.relateToOperator("umulExtended",      EOpUMulExtended);
    symbolTable.relateToOperator("imulExtended",      EOpIMulExtended);
    symbolTable.relateToOperator("bitfieldExtract",   EOpBitfieldExtract);
    symbolTable.relateToOperator("bitfieldInsert",    EOpBitfieldInsert);
    symbolTable.relateToOperator("bitfieldReverse",   EOpBitFieldReverse);
    symbolTable.relateToOperator("bitCount",          EOpBitCount);
    symbolTable.relateToOperator("findLSB",           EOpFindLSB);
    symbolTable.relateToOperator("findMSB",           EOpFindMSB);

    symbolTable.relateToOperator("helperIlwocationEXT",  EOpIsHelperIlwocation);

    symbolTable.relateToOperator("countLeadingZeros",  EOpCountLeadingZeros);
    symbolTable.relateToOperator("countTrailingZeros", EOpCountTrailingZeros);
    symbolTable.relateToOperator("absoluteDifference", EOpAbsDifference);
    symbolTable.relateToOperator("addSaturate",        EOpAddSaturate);
    symbolTable.relateToOperator("subtractSaturate",   EOpSubSaturate);
    symbolTable.relateToOperator("average",            EOpAverage);
    symbolTable.relateToOperator("averageRounded",     EOpAverageRounded);
    symbolTable.relateToOperator("multiply32x16",      EOpMul32x16);
    symbolTable.relateToOperator("debugPrintfEXT",     EOpDebugPrintf);


    if (PureOperatorBuiltins) {
        symbolTable.relateToOperator("imageSize",               EOpImageQuerySize);
        symbolTable.relateToOperator("imageSamples",            EOpImageQuerySamples);
        symbolTable.relateToOperator("imageLoad",               EOpImageLoad);
        symbolTable.relateToOperator("imageStore",              EOpImageStore);
        symbolTable.relateToOperator("imageAtomicAdd",          EOpImageAtomicAdd);
        symbolTable.relateToOperator("imageAtomicMin",          EOpImageAtomicMin);
        symbolTable.relateToOperator("imageAtomicMax",          EOpImageAtomicMax);
        symbolTable.relateToOperator("imageAtomicAnd",          EOpImageAtomicAnd);
        symbolTable.relateToOperator("imageAtomicOr",           EOpImageAtomicOr);
        symbolTable.relateToOperator("imageAtomicXor",          EOpImageAtomicXor);
        symbolTable.relateToOperator("imageAtomicExchange",     EOpImageAtomicExchange);
        symbolTable.relateToOperator("imageAtomicCompSwap",     EOpImageAtomicCompSwap);
        symbolTable.relateToOperator("imageAtomicLoad",         EOpImageAtomicLoad);
        symbolTable.relateToOperator("imageAtomicStore",        EOpImageAtomicStore);

        symbolTable.relateToOperator("subpassLoad",             EOpSubpassLoad);
        symbolTable.relateToOperator("subpassLoadMS",           EOpSubpassLoadMS);

        symbolTable.relateToOperator("textureGather",           EOpTextureGather);
        symbolTable.relateToOperator("textureGatherOffset",     EOpTextureGatherOffset);
        symbolTable.relateToOperator("textureGatherOffsets",    EOpTextureGatherOffsets);

        symbolTable.relateToOperator("noise1", EOpNoise);
        symbolTable.relateToOperator("noise2", EOpNoise);
        symbolTable.relateToOperator("noise3", EOpNoise);
        symbolTable.relateToOperator("noise4", EOpNoise);

        symbolTable.relateToOperator("textureFootprintLW",          EOpImageSampleFootprintLW);
        symbolTable.relateToOperator("textureFootprintClampLW",     EOpImageSampleFootprintClampLW);
        symbolTable.relateToOperator("textureFootprintLodLW",       EOpImageSampleFootprintLodLW);
        symbolTable.relateToOperator("textureFootprintGradLW",      EOpImageSampleFootprintGradLW);
        symbolTable.relateToOperator("textureFootprintGradClampLW", EOpImageSampleFootprintGradClampLW);

        if (spvVersion.spv == 0 && IncludeLegacy(version, profile, spvVersion))
            symbolTable.relateToOperator("ftransform", EOpFtransform);

        if (spvVersion.spv == 0 && (IncludeLegacy(version, profile, spvVersion) ||
            (profile == EEsProfile && version == 100))) {

            symbolTable.relateToOperator("texture1D",                EOpTexture);
            symbolTable.relateToOperator("texture1DGradARB",         EOpTextureGrad);
            symbolTable.relateToOperator("texture1DProj",            EOpTextureProj);
            symbolTable.relateToOperator("texture1DProjGradARB",     EOpTextureProjGrad);
            symbolTable.relateToOperator("texture1DLod",             EOpTextureLod);
            symbolTable.relateToOperator("texture1DProjLod",         EOpTextureProjLod);

            symbolTable.relateToOperator("texture2DRect",            EOpTexture);
            symbolTable.relateToOperator("texture2DRectProj",        EOpTextureProj);
            symbolTable.relateToOperator("texture2DRectGradARB",     EOpTextureGrad);
            symbolTable.relateToOperator("texture2DRectProjGradARB", EOpTextureProjGrad);
            symbolTable.relateToOperator("shadow2DRect",             EOpTexture);
            symbolTable.relateToOperator("shadow2DRectProj",         EOpTextureProj);
            symbolTable.relateToOperator("shadow2DRectGradARB",      EOpTextureGrad);
            symbolTable.relateToOperator("shadow2DRectProjGradARB",  EOpTextureProjGrad);

            symbolTable.relateToOperator("texture2D",                EOpTexture);
            symbolTable.relateToOperator("texture2DProj",            EOpTextureProj);
            symbolTable.relateToOperator("texture2DGradEXT",         EOpTextureGrad);
            symbolTable.relateToOperator("texture2DGradARB",         EOpTextureGrad);
            symbolTable.relateToOperator("texture2DProjGradEXT",     EOpTextureProjGrad);
            symbolTable.relateToOperator("texture2DProjGradARB",     EOpTextureProjGrad);
            symbolTable.relateToOperator("texture2DLod",             EOpTextureLod);
            symbolTable.relateToOperator("texture2DLodEXT",          EOpTextureLod);
            symbolTable.relateToOperator("texture2DProjLod",         EOpTextureProjLod);
            symbolTable.relateToOperator("texture2DProjLodEXT",      EOpTextureProjLod);

            symbolTable.relateToOperator("texture3D",                EOpTexture);
            symbolTable.relateToOperator("texture3DGradARB",         EOpTextureGrad);
            symbolTable.relateToOperator("texture3DProj",            EOpTextureProj);
            symbolTable.relateToOperator("texture3DProjGradARB",     EOpTextureProjGrad);
            symbolTable.relateToOperator("texture3DLod",             EOpTextureLod);
            symbolTable.relateToOperator("texture3DProjLod",         EOpTextureProjLod);
            symbolTable.relateToOperator("textureLwbe",              EOpTexture);
            symbolTable.relateToOperator("textureLwbeGradEXT",       EOpTextureGrad);
            symbolTable.relateToOperator("textureLwbeGradARB",       EOpTextureGrad);
            symbolTable.relateToOperator("textureLwbeLod",           EOpTextureLod);
            symbolTable.relateToOperator("textureLwbeLodEXT",        EOpTextureLod);
            symbolTable.relateToOperator("shadow1D",                 EOpTexture);
            symbolTable.relateToOperator("shadow1DGradARB",          EOpTextureGrad);
            symbolTable.relateToOperator("shadow2D",                 EOpTexture);
            symbolTable.relateToOperator("shadow2DGradARB",          EOpTextureGrad);
            symbolTable.relateToOperator("shadow1DProj",             EOpTextureProj);
            symbolTable.relateToOperator("shadow2DProj",             EOpTextureProj);
            symbolTable.relateToOperator("shadow1DProjGradARB",      EOpTextureProjGrad);
            symbolTable.relateToOperator("shadow2DProjGradARB",      EOpTextureProjGrad);
            symbolTable.relateToOperator("shadow1DLod",              EOpTextureLod);
            symbolTable.relateToOperator("shadow2DLod",              EOpTextureLod);
            symbolTable.relateToOperator("shadow1DProjLod",          EOpTextureProjLod);
            symbolTable.relateToOperator("shadow2DProjLod",          EOpTextureProjLod);
        }

        if (profile != EEsProfile) {
            symbolTable.relateToOperator("sparseTextureARB",                EOpSparseTexture);
            symbolTable.relateToOperator("sparseTextureLodARB",             EOpSparseTextureLod);
            symbolTable.relateToOperator("sparseTextureOffsetARB",          EOpSparseTextureOffset);
            symbolTable.relateToOperator("sparseTexelFetchARB",             EOpSparseTextureFetch);
            symbolTable.relateToOperator("sparseTexelFetchOffsetARB",       EOpSparseTextureFetchOffset);
            symbolTable.relateToOperator("sparseTextureLodOffsetARB",       EOpSparseTextureLodOffset);
            symbolTable.relateToOperator("sparseTextureGradARB",            EOpSparseTextureGrad);
            symbolTable.relateToOperator("sparseTextureGradOffsetARB",      EOpSparseTextureGradOffset);
            symbolTable.relateToOperator("sparseTextureGatherARB",          EOpSparseTextureGather);
            symbolTable.relateToOperator("sparseTextureGatherOffsetARB",    EOpSparseTextureGatherOffset);
            symbolTable.relateToOperator("sparseTextureGatherOffsetsARB",   EOpSparseTextureGatherOffsets);
            symbolTable.relateToOperator("sparseImageLoadARB",              EOpSparseImageLoad);
            symbolTable.relateToOperator("sparseTexelsResidentARB",         EOpSparseTexelsResident);

            symbolTable.relateToOperator("sparseTextureClampARB",           EOpSparseTextureClamp);
            symbolTable.relateToOperator("sparseTextureOffsetClampARB",     EOpSparseTextureOffsetClamp);
            symbolTable.relateToOperator("sparseTextureGradClampARB",       EOpSparseTextureGradClamp);
            symbolTable.relateToOperator("sparseTextureGradOffsetClampARB", EOpSparseTextureGradOffsetClamp);
            symbolTable.relateToOperator("textureClampARB",                 EOpTextureClamp);
            symbolTable.relateToOperator("textureOffsetClampARB",           EOpTextureOffsetClamp);
            symbolTable.relateToOperator("textureGradClampARB",             EOpTextureGradClamp);
            symbolTable.relateToOperator("textureGradOffsetClampARB",       EOpTextureGradOffsetClamp);

            symbolTable.relateToOperator("ballotARB",                       EOpBallot);
            symbolTable.relateToOperator("readIlwocationARB",               EOpReadIlwocation);
            symbolTable.relateToOperator("readFirstIlwocationARB",          EOpReadFirstIlwocation);

            if (version >= 430) {
                symbolTable.relateToOperator("anyIlwocationARB",            EOpAnyIlwocation);
                symbolTable.relateToOperator("allIlwocationsARB",           EOpAllIlwocations);
                symbolTable.relateToOperator("allIlwocationsEqualARB",      EOpAllIlwocationsEqual);
            }
            if (version >= 460) {
                symbolTable.relateToOperator("anyIlwocation",               EOpAnyIlwocation);
                symbolTable.relateToOperator("allIlwocations",              EOpAllIlwocations);
                symbolTable.relateToOperator("allIlwocationsEqual",         EOpAllIlwocationsEqual);
            }
            symbolTable.relateToOperator("minIlwocationsAMD",                           EOpMinIlwocations);
            symbolTable.relateToOperator("maxIlwocationsAMD",                           EOpMaxIlwocations);
            symbolTable.relateToOperator("addIlwocationsAMD",                           EOpAddIlwocations);
            symbolTable.relateToOperator("minIlwocationsNonUniformAMD",                 EOpMinIlwocationsNonUniform);
            symbolTable.relateToOperator("maxIlwocationsNonUniformAMD",                 EOpMaxIlwocationsNonUniform);
            symbolTable.relateToOperator("addIlwocationsNonUniformAMD",                 EOpAddIlwocationsNonUniform);
            symbolTable.relateToOperator("minIlwocationsInclusiveScanAMD",              EOpMinIlwocationsInclusiveScan);
            symbolTable.relateToOperator("maxIlwocationsInclusiveScanAMD",              EOpMaxIlwocationsInclusiveScan);
            symbolTable.relateToOperator("addIlwocationsInclusiveScanAMD",              EOpAddIlwocationsInclusiveScan);
            symbolTable.relateToOperator("minIlwocationsInclusiveScanNonUniformAMD",    EOpMinIlwocationsInclusiveScanNonUniform);
            symbolTable.relateToOperator("maxIlwocationsInclusiveScanNonUniformAMD",    EOpMaxIlwocationsInclusiveScanNonUniform);
            symbolTable.relateToOperator("addIlwocationsInclusiveScanNonUniformAMD",    EOpAddIlwocationsInclusiveScanNonUniform);
            symbolTable.relateToOperator("minIlwocationsExclusiveScanAMD",              EOpMinIlwocationsExclusiveScan);
            symbolTable.relateToOperator("maxIlwocationsExclusiveScanAMD",              EOpMaxIlwocationsExclusiveScan);
            symbolTable.relateToOperator("addIlwocationsExclusiveScanAMD",              EOpAddIlwocationsExclusiveScan);
            symbolTable.relateToOperator("minIlwocationsExclusiveScanNonUniformAMD",    EOpMinIlwocationsExclusiveScanNonUniform);
            symbolTable.relateToOperator("maxIlwocationsExclusiveScanNonUniformAMD",    EOpMaxIlwocationsExclusiveScanNonUniform);
            symbolTable.relateToOperator("addIlwocationsExclusiveScanNonUniformAMD",    EOpAddIlwocationsExclusiveScanNonUniform);
            symbolTable.relateToOperator("swizzleIlwocationsAMD",                       EOpSwizzleIlwocations);
            symbolTable.relateToOperator("swizzleIlwocationsMaskedAMD",                 EOpSwizzleIlwocationsMasked);
            symbolTable.relateToOperator("writeIlwocationAMD",                          EOpWriteIlwocation);
            symbolTable.relateToOperator("mbcntAMD",                                    EOpMbcnt);

            symbolTable.relateToOperator("min3",    EOpMin3);
            symbolTable.relateToOperator("max3",    EOpMax3);
            symbolTable.relateToOperator("mid3",    EOpMid3);

            symbolTable.relateToOperator("lwbeFaceIndexAMD",    EOpLwbeFaceIndex);
            symbolTable.relateToOperator("lwbeFaceCoordAMD",    EOpLwbeFaceCoord);
            symbolTable.relateToOperator("timeAMD",             EOpTime);

            symbolTable.relateToOperator("textureGatherLodAMD",                 EOpTextureGatherLod);
            symbolTable.relateToOperator("textureGatherLodOffsetAMD",           EOpTextureGatherLodOffset);
            symbolTable.relateToOperator("textureGatherLodOffsetsAMD",          EOpTextureGatherLodOffsets);
            symbolTable.relateToOperator("sparseTextureGatherLodAMD",           EOpSparseTextureGatherLod);
            symbolTable.relateToOperator("sparseTextureGatherLodOffsetAMD",     EOpSparseTextureGatherLodOffset);
            symbolTable.relateToOperator("sparseTextureGatherLodOffsetsAMD",    EOpSparseTextureGatherLodOffsets);

            symbolTable.relateToOperator("imageLoadLodAMD",                     EOpImageLoadLod);
            symbolTable.relateToOperator("imageStoreLodAMD",                    EOpImageStoreLod);
            symbolTable.relateToOperator("sparseImageLoadLodAMD",               EOpSparseImageLoadLod);

            symbolTable.relateToOperator("fragmentMaskFetchAMD",                EOpFragmentMaskFetch);
            symbolTable.relateToOperator("fragmentFetchAMD",                    EOpFragmentFetch);
        }

        // GL_KHR_shader_subgroup
        if ((profile == EEsProfile && version >= 310) ||
            (profile != EEsProfile && version >= 140)) {
            symbolTable.relateToOperator("subgroupBarrier",                 EOpSubgroupBarrier);
            symbolTable.relateToOperator("subgroupMemoryBarrier",           EOpSubgroupMemoryBarrier);
            symbolTable.relateToOperator("subgroupMemoryBarrierBuffer",     EOpSubgroupMemoryBarrierBuffer);
            symbolTable.relateToOperator("subgroupMemoryBarrierImage",      EOpSubgroupMemoryBarrierImage);
            symbolTable.relateToOperator("subgroupElect",                   EOpSubgroupElect);
            symbolTable.relateToOperator("subgroupAll",                     EOpSubgroupAll);
            symbolTable.relateToOperator("subgroupAny",                     EOpSubgroupAny);
            symbolTable.relateToOperator("subgroupAllEqual",                EOpSubgroupAllEqual);
            symbolTable.relateToOperator("subgroupBroadcast",               EOpSubgroupBroadcast);
            symbolTable.relateToOperator("subgroupBroadcastFirst",          EOpSubgroupBroadcastFirst);
            symbolTable.relateToOperator("subgroupBallot",                  EOpSubgroupBallot);
            symbolTable.relateToOperator("subgroupIlwerseBallot",           EOpSubgroupIlwerseBallot);
            symbolTable.relateToOperator("subgroupBallotBitExtract",        EOpSubgroupBallotBitExtract);
            symbolTable.relateToOperator("subgroupBallotBitCount",          EOpSubgroupBallotBitCount);
            symbolTable.relateToOperator("subgroupBallotInclusiveBitCount", EOpSubgroupBallotInclusiveBitCount);
            symbolTable.relateToOperator("subgroupBallotExclusiveBitCount", EOpSubgroupBallotExclusiveBitCount);
            symbolTable.relateToOperator("subgroupBallotFindLSB",           EOpSubgroupBallotFindLSB);
            symbolTable.relateToOperator("subgroupBallotFindMSB",           EOpSubgroupBallotFindMSB);
            symbolTable.relateToOperator("subgroupShuffle",                 EOpSubgroupShuffle);
            symbolTable.relateToOperator("subgroupShuffleXor",              EOpSubgroupShuffleXor);
            symbolTable.relateToOperator("subgroupShuffleUp",               EOpSubgroupShuffleUp);
            symbolTable.relateToOperator("subgroupShuffleDown",             EOpSubgroupShuffleDown);
            symbolTable.relateToOperator("subgroupAdd",                     EOpSubgroupAdd);
            symbolTable.relateToOperator("subgroupMul",                     EOpSubgroupMul);
            symbolTable.relateToOperator("subgroupMin",                     EOpSubgroupMin);
            symbolTable.relateToOperator("subgroupMax",                     EOpSubgroupMax);
            symbolTable.relateToOperator("subgroupAnd",                     EOpSubgroupAnd);
            symbolTable.relateToOperator("subgroupOr",                      EOpSubgroupOr);
            symbolTable.relateToOperator("subgroupXor",                     EOpSubgroupXor);
            symbolTable.relateToOperator("subgroupInclusiveAdd",            EOpSubgroupInclusiveAdd);
            symbolTable.relateToOperator("subgroupInclusiveMul",            EOpSubgroupInclusiveMul);
            symbolTable.relateToOperator("subgroupInclusiveMin",            EOpSubgroupInclusiveMin);
            symbolTable.relateToOperator("subgroupInclusiveMax",            EOpSubgroupInclusiveMax);
            symbolTable.relateToOperator("subgroupInclusiveAnd",            EOpSubgroupInclusiveAnd);
            symbolTable.relateToOperator("subgroupInclusiveOr",             EOpSubgroupInclusiveOr);
            symbolTable.relateToOperator("subgroupInclusiveXor",            EOpSubgroupInclusiveXor);
            symbolTable.relateToOperator("subgroupExclusiveAdd",            EOpSubgroupExclusiveAdd);
            symbolTable.relateToOperator("subgroupExclusiveMul",            EOpSubgroupExclusiveMul);
            symbolTable.relateToOperator("subgroupExclusiveMin",            EOpSubgroupExclusiveMin);
            symbolTable.relateToOperator("subgroupExclusiveMax",            EOpSubgroupExclusiveMax);
            symbolTable.relateToOperator("subgroupExclusiveAnd",            EOpSubgroupExclusiveAnd);
            symbolTable.relateToOperator("subgroupExclusiveOr",             EOpSubgroupExclusiveOr);
            symbolTable.relateToOperator("subgroupExclusiveXor",            EOpSubgroupExclusiveXor);
            symbolTable.relateToOperator("subgroupClusteredAdd",            EOpSubgroupClusteredAdd);
            symbolTable.relateToOperator("subgroupClusteredMul",            EOpSubgroupClusteredMul);
            symbolTable.relateToOperator("subgroupClusteredMin",            EOpSubgroupClusteredMin);
            symbolTable.relateToOperator("subgroupClusteredMax",            EOpSubgroupClusteredMax);
            symbolTable.relateToOperator("subgroupClusteredAnd",            EOpSubgroupClusteredAnd);
            symbolTable.relateToOperator("subgroupClusteredOr",             EOpSubgroupClusteredOr);
            symbolTable.relateToOperator("subgroupClusteredXor",            EOpSubgroupClusteredXor);
            symbolTable.relateToOperator("subgroupQuadBroadcast",           EOpSubgroupQuadBroadcast);
            symbolTable.relateToOperator("subgroupQuadSwapHorizontal",      EOpSubgroupQuadSwapHorizontal);
            symbolTable.relateToOperator("subgroupQuadSwapVertical",        EOpSubgroupQuadSwapVertical);
            symbolTable.relateToOperator("subgroupQuadSwapDiagonal",        EOpSubgroupQuadSwapDiagonal);

            symbolTable.relateToOperator("subgroupPartitionLW",                          EOpSubgroupPartition);
            symbolTable.relateToOperator("subgroupPartitionedAddLW",                     EOpSubgroupPartitionedAdd);
            symbolTable.relateToOperator("subgroupPartitionedMulLW",                     EOpSubgroupPartitionedMul);
            symbolTable.relateToOperator("subgroupPartitionedMinLW",                     EOpSubgroupPartitionedMin);
            symbolTable.relateToOperator("subgroupPartitionedMaxLW",                     EOpSubgroupPartitionedMax);
            symbolTable.relateToOperator("subgroupPartitionedAndLW",                     EOpSubgroupPartitionedAnd);
            symbolTable.relateToOperator("subgroupPartitionedOrLW",                      EOpSubgroupPartitionedOr);
            symbolTable.relateToOperator("subgroupPartitionedXorLW",                     EOpSubgroupPartitionedXor);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveAddLW",            EOpSubgroupPartitionedInclusiveAdd);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveMulLW",            EOpSubgroupPartitionedInclusiveMul);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveMinLW",            EOpSubgroupPartitionedInclusiveMin);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveMaxLW",            EOpSubgroupPartitionedInclusiveMax);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveAndLW",            EOpSubgroupPartitionedInclusiveAnd);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveOrLW",             EOpSubgroupPartitionedInclusiveOr);
            symbolTable.relateToOperator("subgroupPartitionedInclusiveXorLW",            EOpSubgroupPartitionedInclusiveXor);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveAddLW",            EOpSubgroupPartitionedExclusiveAdd);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveMulLW",            EOpSubgroupPartitionedExclusiveMul);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveMinLW",            EOpSubgroupPartitionedExclusiveMin);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveMaxLW",            EOpSubgroupPartitionedExclusiveMax);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveAndLW",            EOpSubgroupPartitionedExclusiveAnd);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveOrLW",             EOpSubgroupPartitionedExclusiveOr);
            symbolTable.relateToOperator("subgroupPartitionedExclusiveXorLW",            EOpSubgroupPartitionedExclusiveXor);
        }

        if (profile == EEsProfile) {
            symbolTable.relateToOperator("shadow2DEXT",              EOpTexture);
            symbolTable.relateToOperator("shadow2DProjEXT",          EOpTextureProj);
        }
    }

    switch(language) {
    case EShLangVertex:
        break;

    case EShLangTessControl:
    case EShLangTessEvaluation:
        break;

    case EShLangGeometry:
        symbolTable.relateToOperator("EmitStreamVertex",   EOpEmitStreamVertex);
        symbolTable.relateToOperator("EndStreamPrimitive", EOpEndStreamPrimitive);
        symbolTable.relateToOperator("EmitVertex",         EOpEmitVertex);
        symbolTable.relateToOperator("EndPrimitive",       EOpEndPrimitive);
        break;

    case EShLangFragment:
        if (profile != EEsProfile && version >= 400) {
            symbolTable.relateToOperator("dFdxFine",     EOpDPdxFine);
            symbolTable.relateToOperator("dFdyFine",     EOpDPdyFine);
            symbolTable.relateToOperator("fwidthFine",   EOpFwidthFine);
            symbolTable.relateToOperator("dFdxCoarse",   EOpDPdxCoarse);
            symbolTable.relateToOperator("dFdyCoarse",   EOpDPdyCoarse);
            symbolTable.relateToOperator("fwidthCoarse", EOpFwidthCoarse);
        }

        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("rayQueryInitializeEXT",                                             EOpRayQueryInitialize);
            symbolTable.relateToOperator("rayQueryTerminateEXT",                                              EOpRayQueryTerminate);
            symbolTable.relateToOperator("rayQueryGenerateIntersectionEXT",                                   EOpRayQueryGenerateIntersection);
            symbolTable.relateToOperator("rayQueryConfirmIntersectionEXT",                                    EOpRayQueryConfirmIntersection);
            symbolTable.relateToOperator("rayQueryProceedEXT",                                                EOpRayQueryProceed);
            symbolTable.relateToOperator("rayQueryGetIntersectionTypeEXT",                                    EOpRayQueryGetIntersectionType);
            symbolTable.relateToOperator("rayQueryGetRayTMinEXT",                                             EOpRayQueryGetRayTMin);
            symbolTable.relateToOperator("rayQueryGetRayFlagsEXT",                                            EOpRayQueryGetRayFlags);
            symbolTable.relateToOperator("rayQueryGetIntersectionTEXT",                                       EOpRayQueryGetIntersectionT);
            symbolTable.relateToOperator("rayQueryGetIntersectionInstanceLwstomIndexEXT",                     EOpRayQueryGetIntersectionInstanceLwstomIndex);
            symbolTable.relateToOperator("rayQueryGetIntersectionInstanceIdEXT",                              EOpRayQueryGetIntersectionInstanceId);
            symbolTable.relateToOperator("rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT",  EOpRayQueryGetIntersectionInstanceShaderBindingTableRecordOffset);
            symbolTable.relateToOperator("rayQueryGetIntersectionGeometryIndexEXT",                           EOpRayQueryGetIntersectionGeometryIndex);
            symbolTable.relateToOperator("rayQueryGetIntersectionPrimitiveIndexEXT",                          EOpRayQueryGetIntersectionPrimitiveIndex);
            symbolTable.relateToOperator("rayQueryGetIntersectionBarycentricsEXT",                            EOpRayQueryGetIntersectionBarycentrics);
            symbolTable.relateToOperator("rayQueryGetIntersectionFrontFaceEXT",                               EOpRayQueryGetIntersectionFrontFace);
            symbolTable.relateToOperator("rayQueryGetIntersectionCandidateAABBOpaqueEXT",                     EOpRayQueryGetIntersectionCandidateAABBOpaque);
            symbolTable.relateToOperator("rayQueryGetIntersectionObjectRayDirectionEXT",                      EOpRayQueryGetIntersectionObjectRayDirection);
            symbolTable.relateToOperator("rayQueryGetIntersectionObjectRayOriginEXT",                         EOpRayQueryGetIntersectionObjectRayOrigin);
            symbolTable.relateToOperator("rayQueryGetWorldRayDirectionEXT",                                   EOpRayQueryGetWorldRayDirection);
            symbolTable.relateToOperator("rayQueryGetWorldRayOriginEXT",                                      EOpRayQueryGetWorldRayOrigin);
            symbolTable.relateToOperator("rayQueryGetIntersectionObjectToWorldEXT",                           EOpRayQueryGetIntersectionObjectToWorld);
            symbolTable.relateToOperator("rayQueryGetIntersectionWorldToObjectEXT",                           EOpRayQueryGetIntersectionWorldToObject);
        }

        symbolTable.relateToOperator("interpolateAtCentroid", EOpInterpolateAtCentroid);
        symbolTable.relateToOperator("interpolateAtSample",   EOpInterpolateAtSample);
        symbolTable.relateToOperator("interpolateAtOffset",   EOpInterpolateAtOffset);

        if (profile != EEsProfile)
            symbolTable.relateToOperator("interpolateAtVertexAMD", EOpInterpolateAtVertex);

        symbolTable.relateToOperator("beginIlwocationInterlockARB", EOpBeginIlwocationInterlock);
        symbolTable.relateToOperator("endIlwocationInterlockARB",   EOpEndIlwocationInterlock);

        break;

    case EShLangCompute:
        symbolTable.relateToOperator("subgroupMemoryBarrierShared", EOpSubgroupMemoryBarrierShared);
        if ((profile != EEsProfile && version >= 450) ||
            (profile == EEsProfile && version >= 320)) {
            symbolTable.relateToOperator("dFdx",        EOpDPdx);
            symbolTable.relateToOperator("dFdy",        EOpDPdy);
            symbolTable.relateToOperator("fwidth",      EOpFwidth);
            symbolTable.relateToOperator("dFdxFine",    EOpDPdxFine);
            symbolTable.relateToOperator("dFdyFine",    EOpDPdyFine);
            symbolTable.relateToOperator("fwidthFine",  EOpFwidthFine);
            symbolTable.relateToOperator("dFdxCoarse",  EOpDPdxCoarse);
            symbolTable.relateToOperator("dFdyCoarse",  EOpDPdyCoarse);
            symbolTable.relateToOperator("fwidthCoarse",EOpFwidthCoarse);
        }
        symbolTable.relateToOperator("coopMatLoadLW",              EOpCooperativeMatrixLoad);
        symbolTable.relateToOperator("coopMatStoreLW",             EOpCooperativeMatrixStore);
        symbolTable.relateToOperator("coopMatMulAddLW",            EOpCooperativeMatrixMulAdd);
        break;

    case EShLangRayGen:
    case EShLangClosestHit:
    case EShLangMiss:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("traceLW", EOpTrace);
            symbolTable.relateToOperator("traceRayEXT", EOpTrace);
            symbolTable.relateToOperator("exelwteCallableLW", EOpExelwteCallable);
            symbolTable.relateToOperator("exelwteCallableEXT", EOpExelwteCallable);
        }
        break;
    case EShLangIntersect:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("reportIntersectionLW", EOpReportIntersection);
            symbolTable.relateToOperator("reportIntersectionEXT", EOpReportIntersection);
	}
        break;
    case EShLangAnyHit:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("ignoreIntersectionLW", EOpIgnoreIntersection);
            symbolTable.relateToOperator("ignoreIntersectionEXT", EOpIgnoreIntersection);
            symbolTable.relateToOperator("terminateRayLW", EOpTerminateRay);
            symbolTable.relateToOperator("terminateRayEXT", EOpTerminateRay);
        }
        break;
    case EShLangCallable:
        if (profile != EEsProfile && version >= 460) {
            symbolTable.relateToOperator("exelwteCallableLW", EOpExelwteCallable);
            symbolTable.relateToOperator("exelwteCallableEXT", EOpExelwteCallable);
        }
        break;
    case EShLangMeshLW:
        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            symbolTable.relateToOperator("writePackedPrimitiveIndices4x8LW", EOpWritePackedPrimitiveIndices4x8LW);
        }
        // fall through
    case EShLangTaskLW:
        if ((profile != EEsProfile && version >= 450) || (profile == EEsProfile && version >= 320)) {
            symbolTable.relateToOperator("memoryBarrierShared", EOpMemoryBarrierShared);
            symbolTable.relateToOperator("groupMemoryBarrier", EOpGroupMemoryBarrier);
            symbolTable.relateToOperator("subgroupMemoryBarrierShared", EOpSubgroupMemoryBarrierShared);
        }
        break;

    default:
        assert(false && "Language not supported");
    }
#endif
}

//
// Add context-dependent (resource-specific) built-ins not handled by the above.  These
// would be ones that need to be programmatically added because they cannot
// be added by simple text strings.  For these, also
// 1) Map built-in functions to operators, for those that will turn into an operation node
//    instead of remaining a function call.
// 2) Tag extension-related symbols added to their base version with their extensions, so
//    that if an early version has the extension turned off, there is an error reported on use.
//
void TBuiltIns::identifyBuiltIns(int version, EProfile profile, const SpvVersion& spvVersion, EShLanguage language, TSymbolTable& symbolTable, const TBuiltInResource &resources)
{
#ifndef GLSLANG_WEB
    if (profile != EEsProfile && version >= 430 && version < 440) {
        symbolTable.setVariableExtensions("gl_MaxTransformFeedbackBuffers", 1, &E_GL_ARB_enhanced_layouts);
        symbolTable.setVariableExtensions("gl_MaxTransformFeedbackInterleavedComponents", 1, &E_GL_ARB_enhanced_layouts);
    }
    if (profile != EEsProfile && version >= 130 && version < 420) {
        symbolTable.setVariableExtensions("gl_MinProgramTexelOffset", 1, &E_GL_ARB_shading_language_420pack);
        symbolTable.setVariableExtensions("gl_MaxProgramTexelOffset", 1, &E_GL_ARB_shading_language_420pack);
    }
    if (profile != EEsProfile && version >= 150 && version < 410)
        symbolTable.setVariableExtensions("gl_MaxViewports", 1, &E_GL_ARB_viewport_array);

    switch(language) {
    case EShLangFragment:
        // Set up gl_FragData based on current array size.
        if (version == 100 || IncludeLegacy(version, profile, spvVersion) || (! ForwardCompatibility && profile != EEsProfile && version < 420)) {
            TPrecisionQualifier pq = profile == EEsProfile ? EpqMedium : EpqNone;
            TType fragData(EbtFloat, EvqFragColor, pq, 4);
            TArraySizes* arraySizes = new TArraySizes;
            arraySizes->addInnerSize(resources.maxDrawBuffers);
            fragData.transferArraySizes(arraySizes);
            symbolTable.insert(*new TVariable(NewPoolTString("gl_FragData"), fragData));
            SpecialQualifier("gl_FragData", EvqFragColor, EbvFragData, symbolTable);
        }

        // GL_EXT_blend_func_extended
        if (profile == EEsProfile && version >= 100) {
           symbolTable.setVariableExtensions("gl_MaxDualSourceDrawBuffersEXT",    1, &E_GL_EXT_blend_func_extended);
           symbolTable.setVariableExtensions("gl_SecondaryFragColorEXT",    1, &E_GL_EXT_blend_func_extended);
           symbolTable.setVariableExtensions("gl_SecondaryFragDataEXT",    1, &E_GL_EXT_blend_func_extended);
           SpecialQualifier("gl_SecondaryFragColorEXT", EvqVaryingOut, EbvSecondaryFragColorEXT, symbolTable);
           SpecialQualifier("gl_SecondaryFragDataEXT", EvqVaryingOut, EbvSecondaryFragDataEXT, symbolTable);
        }

        break;

    case EShLangTessControl:
    case EShLangTessEvaluation:
        // Because of the context-dependent array size (gl_MaxPatchVertices),
        // these variables were added later than the others and need to be mapped now.

        // standard members
        BuiltIlwariable("gl_in", "gl_Position",     EbvPosition,     symbolTable);
        BuiltIlwariable("gl_in", "gl_PointSize",    EbvPointSize,    symbolTable);
        BuiltIlwariable("gl_in", "gl_ClipDistance", EbvClipDistance, symbolTable);
        BuiltIlwariable("gl_in", "gl_LwllDistance", EbvLwllDistance, symbolTable);

        // compatibility members
        BuiltIlwariable("gl_in", "gl_ClipVertex",          EbvClipVertex,          symbolTable);
        BuiltIlwariable("gl_in", "gl_FrontColor",          EbvFrontColor,          symbolTable);
        BuiltIlwariable("gl_in", "gl_BackColor",           EbvBackColor,           symbolTable);
        BuiltIlwariable("gl_in", "gl_FrontSecondaryColor", EbvFrontSecondaryColor, symbolTable);
        BuiltIlwariable("gl_in", "gl_BackSecondaryColor",  EbvBackSecondaryColor,  symbolTable);
        BuiltIlwariable("gl_in", "gl_TexCoord",            EbvTexCoord,            symbolTable);
        BuiltIlwariable("gl_in", "gl_FogFragCoord",        EbvFogFragCoord,        symbolTable);

        symbolTable.setVariableExtensions("gl_in", "gl_SecondaryPositionLW", 1, &E_GL_LW_stereo_view_rendering);
        symbolTable.setVariableExtensions("gl_in", "gl_PositionPerViewLW",   1, &E_GL_LWX_multiview_per_view_attributes);

        BuiltIlwariable("gl_in", "gl_SecondaryPositionLW", EbvSecondaryPositionLW, symbolTable);
        BuiltIlwariable("gl_in", "gl_PositionPerViewLW",   EbvPositionPerViewLW,   symbolTable);

        // extension requirements
        if (profile == EEsProfile) {
            symbolTable.setVariableExtensions("gl_in", "gl_PointSize", Num_AEP_tessellation_point_size, AEP_tessellation_point_size);
        }

        break;

    default:
        break;
    }
#endif
}

} // end namespace glslang
