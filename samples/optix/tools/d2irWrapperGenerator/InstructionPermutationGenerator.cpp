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

#include <InstructionPermutationGenerator.h>

#include <FrontEnd/PTX/Intrinsics/IntrinsicHelpers.h>
#include <FrontEnd/PTX/Intrinsics/PTXToLLVMHelpers.h>

#include <stdexcept>
#include <unordered_map>

namespace {
bool startsWith( const std::string& someString, const std::string& prefix )
{
    if( prefix.size() > someString.size() )
        return false;
    auto res = std::mismatch( prefix.begin(), prefix.end(), someString.begin() );
    return res.first == prefix.end();
}

std::string removePrefix( const std::string& someString, const std::string& prefix )
{
    return someString.substr( prefix.size(), someString.size() - prefix.size() );
}

std::string getNoftzString( optix::PTXIntrinsics::Noftz noftz )
{
    switch( noftz )
    {
    case optix::PTXIntrinsics::Noftz::noftz:
            return "noftz";
        default:
            throw std::runtime_error( "Unhandled noftz" );
    }
}

}

using namespace optix::PTXIntrinsics;

struct PossibleArgumentTypes
{
    // List of types that can be used for each argument
    std::vector<OperandType> possibleTypes;
    // If this is from ptxInstrTypes, this contains the index it originated from
    int typeIndex;
};

llvm::Type* getLlvmTypeForPtxType( llvm::LLVMContext& context, ptxInstructionType ptxType, unsigned int typeSize )
{
    if( ptxType == ptxFloatIType )
    {
        switch( typeSize )
        {
            case 2:
                return llvm::Type::getHalfTy( context );
            case 4:
                return llvm::Type::getFloatTy( context );
            case 8:
                return llvm::Type::getDoubleTy( context );
        }
    }
    else if( ptxType == ptxPackedHalfFloatIType )
        return llvm::Type::getFloatTy( context );

    // Special case: predicate types are represented with i1, even though their typeSize is 4
    if( ptxType == ptxPredicateIType )
        return llvm::IntegerType::get( context, 1 );

    // Bit, unsigned and signed integer
    return llvm::IntegerType::get( context, typeSize * 8 );
}

std::string getTypePrefixForPtxType( ptxInstructionType ptxType, bool isUnsigned, const std::string& unsupportedTypeStr )
{
    switch( ptxType )
    {
        case ptxFloatIType:
            return "f";
        case ptxIntIType:
            return isUnsigned ? "u" : "s";
        case ptxBitIType:
        {
            std::string retValue = "b";
            if( unsupportedTypeStr == "bf16" )
                retValue = "bf";
            else if( unsupportedTypeStr == "bf16x2" )
                retValue = "bf";
            else if( unsupportedTypeStr == "tf32" )
                retValue = "tf";
            return retValue;
        }
        case ptxPredicateIType:
            return "pred";
        default:
            throw std::runtime_error( "Encountered unknown PTX instruction type: " + std::to_string( ptxType ) );
    }
}

void ptxTypeToOperandTys( llvm::LLVMContext&        context,
                          bool                      includeUnsigned,
                          ptxInstructionType        ptxType,
                          stdBitSet_t               ptxTypeSizes,
                          std::vector<OperandType>& possibleTypes,
                          const std::string&        unsupportedTypeStr )
{
    // f16x2 is a special case; it represents two halfs packed into a float.
    if( ptxType == ptxPackedHalfFloatIType )
    {
        OperandType newType{};
        newType.llvmType = llvm::IntegerType::get( context, 32 );
        newType.ptxType  = "f16x2";
        possibleTypes.push_back( newType );
        return;
    }

    std::string signedTypePrefix        = getTypePrefixForPtxType( ptxType, /*isUnsigned=*/false, unsupportedTypeStr );
    std::string unsignedTypePrefix      = getTypePrefixForPtxType( ptxType, /*isUnsigned=*/true, unsupportedTypeStr );
    bool        hasSeparateUnsignedType = signedTypePrefix.compare( unsignedTypePrefix ) != 0;

    // PTX doesn't support 8-bit float types.
    int minTypeSize = ptxType == ptxFloatIType ? 2 : 1;
    for( int lwrrTypeSize = minTypeSize; lwrrTypeSize <= 8; lwrrTypeSize *= 2 )
    {
        if( !bitSetElement( ptxTypeSizes, lwrrTypeSize * 8 ) )
            continue;

        llvm::Type* llvmType = getLlvmTypeForPtxType( context, ptxType, lwrrTypeSize );
        OperandType newType{};
        newType.llvmType = llvmType;
        newType.ptxType  = signedTypePrefix;

        // Special case: pred type doesn't have bit width; only add it for other types
        if( ptxType != ptxPredicateIType )
        {
            // another special case is bf16x2
            if( newType.ptxType == "bf" && lwrrTypeSize == 4 )  // ie, (lwrrTypeSize * 8) == 32
                newType.ptxType = "bf16x2";
            else
                newType.ptxType += std::to_string( lwrrTypeSize * 8 );
        }
        possibleTypes.push_back( newType );

        if( includeUnsigned && hasSeparateUnsignedType )
        {
            newType.ptxType = unsignedTypePrefix + std::to_string( lwrrTypeSize * 8 );
            possibleTypes.push_back( newType );
        }
    }
}

/**
 * Expand the argument data in the given instruction template into a single list of possible arguments.
 */
void expandInstructionArguments( llvm::LLVMContext& context,
                                 const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData,
                                 std::vector<PossibleArgumentTypes>& args )
{
    const ptxInstructionTemplate&       instructionTemplate = instTemplateData.first;
    const std::vector<UnsupportedType>& unsupportedArgTypes = instTemplateData.second;
    for( unsigned int argIndex = 0; argIndex < instructionTemplate->nrofArguments; ++argIndex )
    {
        args.emplace_back();
        PossibleArgumentTypes& lwrrArg = args.back();

        if( instructionTemplate->argType[argIndex] != ptxFollowAType )
        {
            lwrrArg.possibleTypes.push_back( ptxArgToOperandTy( context, instructionTemplate->argType[argIndex] ) );
            lwrrArg.typeIndex = -1;
        }
        else
        {
            unsigned int followIndex = instructionTemplate->followMap[argIndex];

            stdBitSet_t possibleSizes = instructionTemplate->instrTypeSizes[followIndex];

            std::string unsupportedTypeStr;
            if( !unsupportedArgTypes.empty() && followIndex < unsupportedArgTypes.size() )
            {
                UnsupportedType t = unsupportedArgTypes[followIndex];
                if( t == UNSUPPORTED_TYPE_A16 )
                    unsupportedTypeStr = "bf16";
                else if( t == UNSUPPORTED_TYPE_A32 )
                    unsupportedTypeStr = "bf16x2";
                else if( t == UNSUPPORTED_TYPE_T32 )
                    unsupportedTypeStr = "tf32";
            }

            ptxTypeToOperandTys( context, !instructionTemplate->features.SIGNED, instructionTemplate->instrType[followIndex],
                                 possibleSizes, lwrrArg.possibleTypes, unsupportedTypeStr );
            lwrrArg.typeIndex = followIndex;
        }
    }
}

void getPossibleSignatures( const std::vector<PossibleArgumentTypes>& instructionArguments,
                            std::unordered_map<int, OperandType> existingTypes,
                            InstructionSignature               lwrrSignature,
                            unsigned int                       lwrrArg,
                            OperandType                        lwrrOperand,
                            int                                lwrrOperandTypeIndex,
                            std::vector<InstructionSignature>& possibleSignatures )
{
    lwrrSignature.push_back( lwrrOperand );

    if( lwrrArg >= instructionArguments.size() )
    {
        if( !lwrrSignature.empty() )
            possibleSignatures.push_back( lwrrSignature );
        return;
    }

    if( lwrrOperandTypeIndex != -1 )
        existingTypes[lwrrOperandTypeIndex] = lwrrOperand;

    for( OperandType nextOperand : instructionArguments[lwrrArg].possibleTypes )
    {
        if( existingTypes.find( instructionArguments[lwrrArg].typeIndex ) != existingTypes.end()
            && existingTypes[instructionArguments[lwrrArg].typeIndex] != nextOperand )
            continue;

        getPossibleSignatures( instructionArguments, existingTypes, lwrrSignature, lwrrArg + 1, nextOperand,
                               instructionArguments[lwrrArg].typeIndex, possibleSignatures );
    }
}

void getPossibleSignatures( llvm::LLVMContext&                        context,
                            const std::vector<PossibleArgumentTypes>& instructionArguments,
                            std::vector<InstructionSignature>&        possibleSignatures,
                            const bool                                hasResult )
{
    InstructionSignature lwrrSignature;
    if( !hasResult )
        lwrrSignature.push_back( {"void", llvm::Type::getVoidTy( context ), false} );
    std::unordered_map<int, OperandType> existingTypes;

    for( OperandType lwrrType : instructionArguments[0].possibleTypes )
        getPossibleSignatures( instructionArguments, existingTypes, lwrrSignature, 1, lwrrType,
                               instructionArguments[0].typeIndex, possibleSignatures );
}

std::vector<InstructionSignature> generatePossibleSignatures( llvm::LLVMContext& context,
                                                              const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate&      instTemplate = instTemplateData.first;
    std::vector<PossibleArgumentTypes> instructionArguments;
    expandInstructionArguments( context, instTemplateData, instructionArguments );

    std::vector<InstructionSignature> possibleSignatures;
    getPossibleSignatures( context, instructionArguments, possibleSignatures, instTemplate->features.RESULT );

    return possibleSignatures;
}

lwvm::RoundingMode getRoundModeFromString( const std::string& roundMode )
{
    if( startsWith( roundMode, "rn" ) )
        return lwvm::ROUND_RN;
    if( startsWith( roundMode, "rm" ) )
        return lwvm::ROUND_RM;
    if( startsWith( roundMode, "rp" ) )
        return lwvm::ROUND_RP;
    if( startsWith( roundMode, "rz" ) )
        return lwvm::ROUND_RZ;

    return lwvm::ROUND_RN;
}

// TODO: Switch this to take an enum for roundMode (and probably separate float and int round mode enums in header)
void addMathModifiers( const RoundMode                     roundMode,
                       const VectorSize                    vectorSize,
                       const Approx                        approx,
                       const Ftz                           ftz,
                       const Sat                           sat,
                       std::vector<PTXIntrinsicModifiers>& outModifiers )
{
    // approx instructions don't support explicit round modes; skip them
    if( approx == Approx::approx && roundMode != RoundMode::unspecified )
        return;

    PTXIntrinsicModifiers modifiers{};
    modifiers.vectorSize = vectorSize;
    modifiers.approx     = approx;
    modifiers.roundMode  = roundMode;
    modifiers.ftz        = ftz;
    modifiers.sat        = sat;

    outModifiers.push_back( modifiers );
}

std::vector<PTXIntrinsicModifiers> generatePossibleMathModifiers( const ptxInstructionTemplate& instTemplate )
{
    std::vector<RoundMode> roundModes = {RoundMode::unspecified};
    if( instTemplate->features.ROUNDI )
    {
        roundModes.push_back( RoundMode::rni );
        roundModes.push_back( RoundMode::rmi );
        roundModes.push_back( RoundMode::rzi );
        roundModes.push_back( RoundMode::rpi );
    }
    if( instTemplate->features.ROUNDF )
    {
        roundModes.push_back( RoundMode::rn );
        roundModes.push_back( RoundMode::rm );
        roundModes.push_back( RoundMode::rz );
        roundModes.push_back( RoundMode::rp );
    }

    std::vector<VectorSize> vectorSizes = {VectorSize::unspecified};
    if( instTemplate->features.VECTORIZABLE )
    {
        vectorSizes.push_back( VectorSize::v2 );
        vectorSizes.push_back( VectorSize::v4 );
    }

    std::vector<Approx> approxValues = {Approx::unspecified};
    if( instTemplate->features.APRX )
        approxValues.push_back( Approx::approx );

    std::vector<Ftz> ftzValues = {Ftz::unspecified};
    if( instTemplate->features.FTZ )
        ftzValues.push_back( Ftz::ftz );

    std::vector<Sat> satValues = {Sat::unspecified};
    if( instTemplate->features.SAT || instTemplate->features.SATF )
        satValues.push_back( Sat::sat );

    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    for( const RoundMode roundMode : roundModes )
        for( const VectorSize vectorSize : vectorSizes )
            for( const Approx approx : approxValues )
                for( const Ftz ftz : ftzValues )
                    for( const Sat sat : satValues )
                        addMathModifiers( roundMode, vectorSize, approx, ftz, sat, possibleModifiers );

    return possibleModifiers;
}

std::string getRoundModeString( RoundMode roundMode )
{
    switch( roundMode )
    {
        case RoundMode::unspecified:
            return "";
        case RoundMode::rn:
            return "rn";
        case RoundMode::rm:
            return "rm";
        case RoundMode::rp:
            return "rp";
        case RoundMode::rz:
            return "rz";
        case RoundMode::rni:
            return "rni";
        case RoundMode::rmi:
            return "rmi";
        case RoundMode::rpi:
            return "rpi";
        case RoundMode::rzi:
            return "rzi";
    }
    throw std::runtime_error( "Unrecognized round mode" );
}

std::string getMathInstructionName( const std::string& baseName, const PTXIntrinsicModifiers& modifiers, const InstructionSignature& signature )
{
    std::string instructionName( baseName );

    if( modifiers.approx == Approx::approx )
        instructionName += ".approx";

    if( modifiers.roundMode != RoundMode::unspecified )
        instructionName += "." + getRoundModeString( modifiers.roundMode );

    if( modifiers.ftz == Ftz::ftz )
        instructionName += ".ftz";

    if( modifiers.sat == Sat::sat )
        instructionName += ".sat";

    instructionName += "." + signature[0].ptxType;

    return instructionName;
}

std::vector<PTXIntrinsicInfo> getMathInstructionsFromTemplate( llvm::LLVMContext& context,
                                                               const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> mathInstructions;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleMathModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            InstructionSignature expandedSignature =
                expandMathInstructionSignature( context, signature, instTemplate->features.DOUBLERES );

            mathInstructions.push_back( {getMathInstructionName( instTemplate->name, modifiers, signature ),
                                         (ptxInstructionCode)instTemplate->code, modifiers, expandedSignature} );
        }
    }

    return mathInstructions;
}

void addCvtModifiers( const RoundMode roundMode, Ftz ftz, Sat sat, std::vector<PTXIntrinsicModifiers>& outModifiers )
{
    PTXIntrinsicModifiers modifiers{};
    modifiers.roundMode = roundMode;
    modifiers.ftz       = ftz;
    modifiers.sat       = sat;

    outModifiers.push_back( modifiers );
}

std::vector<PTXIntrinsicModifiers> generatePossibleCvtModifiers( const ptxInstructionTemplate& instTemplate )
{
    std::vector<RoundMode> roundModes = {RoundMode::unspecified};
    if( instTemplate->features.ROUNDI )
    {
        roundModes.push_back( RoundMode::rni );
        roundModes.push_back( RoundMode::rmi );
        roundModes.push_back( RoundMode::rzi );
        roundModes.push_back( RoundMode::rpi );
    }
    if( instTemplate->features.ROUNDF )
    {
        roundModes.push_back( RoundMode::rn );
        roundModes.push_back( RoundMode::rm );
        roundModes.push_back( RoundMode::rz );
        roundModes.push_back( RoundMode::rp );
    }

    std::vector<Ftz> ftzValues = {Ftz::unspecified};
    if( instTemplate->features.FTZ )
        ftzValues.push_back( Ftz::ftz );

    std::vector<Sat> satValues = {Sat::unspecified};
    if( instTemplate->features.SAT || instTemplate->features.SATF )
        satValues.push_back( Sat::sat );

    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    for( const RoundMode roundMode : roundModes )
        for( const Ftz ftz : ftzValues )
            for( const Sat sat : satValues )
                addCvtModifiers( roundMode, ftz, sat, possibleModifiers );

    return possibleModifiers;
}

std::string getCvtInstructionName( const std::string& baseName, const PTXIntrinsicModifiers& modifiers, const InstructionSignature& signature )
{
    std::string instructionName( baseName );

    if( modifiers.roundMode != RoundMode::unspecified )
        instructionName += "." + getRoundModeString( modifiers.roundMode );

    if( modifiers.ftz == Ftz::ftz )
        instructionName += ".ftz";

    if( modifiers.sat == Sat::sat )
        instructionName += ".sat";

    instructionName += "." + signature[0].ptxType + "." + signature[1].ptxType;

    return instructionName;
}

std::vector<PTXIntrinsicInfo> getCvtInstructionsFromTemplate( llvm::LLVMContext& context,
                                                              const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleCvtModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            // CVT only supports colwerting f16x2 types to or from f32.
            if( signature[0].ptxType.compare("f16x2") == 0 && !signature[1].llvmType->isFloatTy() )
                continue;
            else if( signature[1].ptxType.compare("f16x2") == 0 && !signature[0].llvmType->isFloatTy() )
                continue;

            std::string instructionName = getCvtInstructionName( instTemplate->name, modifiers, signature );
            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, signature} );
        }
    }

    return instructions;
}

std::vector<PTXIntrinsicModifiers> generatePossibleTexModifiers( const ptxInstructionTemplate& instTemplate )
{
    std::vector<TextureDimensionality> dimensionalities = {
        TextureDimensionality::dim1D,       TextureDimensionality::dim1DArray, TextureDimensionality::dim2D,
        TextureDimensionality::dim2DArray,  TextureDimensionality::dim3D,      TextureDimensionality::dimLwbe,
        TextureDimensionality::dimLwbeArray};

    std::vector<VectorSize> vectorSizes = {VectorSize::unspecified, VectorSize::v2, VectorSize::v4};

    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    for( const TextureDimensionality dim : dimensionalities )
    {
        for( const VectorSize vectorSize : vectorSizes )
        {
            PTXIntrinsicModifiers lwrrModifiers{};
            lwrrModifiers.texDim     = dim;
            lwrrModifiers.vectorSize = vectorSize;

            possibleModifiers.push_back( lwrrModifiers );
        }
    }

    return possibleModifiers;
}

std::string getTexDimensionalityString( TextureDimensionality dim )
{
    switch( dim )
    {
        case TextureDimensionality::dim1D:
            return "1d";
        case TextureDimensionality::dim1DArray:
            return "a1d";
        case TextureDimensionality::dim2D:
            return "2d";
        case TextureDimensionality::dim2DArray:
            return "a2d";
        case TextureDimensionality::dim3D:
            return "3d";
        case TextureDimensionality::dimLwbe:
            return "lwbe";
        case TextureDimensionality::dimLwbeArray:
            return "alwbe";
        default:
            throw std::runtime_error( "Unrecognized texture dimensionality!" );
    }
    throw std::runtime_error( "Unrecognized texture dimensionality!" );
}

std::string getTexInstructionName( const std::string&           baseName,
                                   const PTXIntrinsicModifiers& modifiers,
                                   const InstructionSignature&  signature,
                                   bool                         isSparse )
{
    std::string suffix = getTexDimensionalityString( modifiers.texDim );
    if( modifiers.vectorSize != VectorSize::unspecified )
        // TODO(Kincaid): Rename vectorSizeToInt to ptxVectorSizeToInt
        suffix += ".v" + std::to_string( vectorSizeToInt( modifiers.vectorSize ) );
    suffix += "." + signature[0].ptxType;
    suffix += "." + signature[2].ptxType;

    if( isSparse )
        return std::string( baseName ) + "." + suffix;
    else
        return "nonsparse." + std::string( baseName ) + "." + suffix;
}

std::vector<PTXIntrinsicInfo> getTexInstructionsFromTemplate( llvm::LLVMContext& context,
                                                              const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleTexModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            std::string instructionName = getTexInstructionName( instTemplate->name, modifiers, signature, /*isSparse*/ true );
            InstructionSignature lwrrSignature =
                expandTexInstructionSignature( context, signature, modifiers, (ptxInstructionCode)instTemplate->code,
                                               /*isSparse*/ true );
            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, lwrrSignature, true} );

            // For every tex instruction, we need to add a version that does
            // not include the optional result predicate. This is done to
            // support older architectures, as well as to allow for both types
            // of instructions to be called by a program. The term "nonsparse"
            // refers to the term used by the lwvm intrinsics to differentiate
            // these variants.
            std::string instructionNameNonsparse =
                getTexInstructionName( instTemplate->name, modifiers, signature, /*isSparse*/ false );
            InstructionSignature lwrrSignatureNonsparse =
                expandTexInstructionSignature( context, signature, modifiers, (ptxInstructionCode)instTemplate->code,
                                               /*isSparse*/ false );
            instructions.push_back( {instructionNameNonsparse, (ptxInstructionCode)instTemplate->code, modifiers,
                                     lwrrSignatureNonsparse, false} );
        }
    }

    return instructions;
}

std::vector<PTXIntrinsicModifiers> generatePossibleTld4Modifiers( const ptxInstructionTemplate& instTemplate )
{
    std::vector<TextureDimensionality> dimensionalities = {TextureDimensionality::dim2D, TextureDimensionality::dim2DArray,
                                                           TextureDimensionality::dimLwbe, TextureDimensionality::dimLwbeArray};
    std::vector<RgbaComponent> components = {RgbaComponent::r, RgbaComponent::g, RgbaComponent::b, RgbaComponent::a};

    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    for( const TextureDimensionality dim : dimensionalities )
    {
        for( const RgbaComponent comp : components )
        {
            PTXIntrinsicModifiers lwrrModifiers{};
            lwrrModifiers.texDim        = dim;
            lwrrModifiers.rgbaComponent = comp;

            possibleModifiers.push_back( lwrrModifiers );
        }
    }

    return possibleModifiers;
}

std::string getRgbaComponentString( RgbaComponent comp )
{
    switch( comp )
    {
        case RgbaComponent::r:
            return "r";
        case RgbaComponent::g:
            return "g";
        case RgbaComponent::b:
            return "b";
        case RgbaComponent::a:
            return "a";
        default:
            throw std::runtime_error( "Unrecognized rgba component" );
    }
    throw std::runtime_error( "Unrecognized rgba component" );
}

std::string getTld4InstructionName( const std::string&           baseName,
                                    const PTXIntrinsicModifiers& modifiers,
                                    const InstructionSignature&  signature,
                                    bool                         isSparse )
{
    // TODO(Kincaid): Handle non-sparse names.
    if( !isSparse )
        throw std::runtime_error( "non-sparse TLD4 not handled" );

    // This doesn't match the PTX spec, but our LWPTX generation script switches the component and
    // dimensionality
    std::string name = std::string( baseName ) + "." + getTexDimensionalityString( modifiers.texDim ) + "."
                       + getRgbaComponentString( modifiers.rgbaComponent ) + ".v4." + signature[0].ptxType + ".f32";
    return name;
}

std::vector<PTXIntrinsicInfo> getTld4InstructionsFromTemplate( llvm::LLVMContext& context,
                                                               const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleTld4Modifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            // Skip instructions with explicit sampler argument.
            if( signature.size() > 3 )
                continue;

            std::string instructionName = getTld4InstructionName( instTemplate->name, modifiers, signature, /*isSparse*/ true );
            InstructionSignature lwrrSignature =
                expandTld4InstructionSignature( context, signature, modifiers, (ptxInstructionCode)instTemplate->code,
                                                /*isSparse*/ true );
            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, lwrrSignature,
                                     /*hasPredicateOutput*/ true} );

            // TODO(Kincaid): We should support the non-sparse version of TLD4.  However, we don't
            // lwrrently generate those on the LWPTX path, so I'll punt the issue for now.
        }
    }

    return instructions;
}

std::vector<PTXIntrinsicModifiers> generatePossibleTxqModifiers( const ptxInstructionTemplate& instTemplate )
{
    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    if( (ptxInstructionCode)instTemplate->code == ptx_txq_Instr )
    {
        std::vector<TextureQuery> textureQueries = {TextureQuery::width,
                                                    TextureQuery::height,
                                                    TextureQuery::depth,
                                                    TextureQuery::numMipmapLevels,
                                                    TextureQuery::numSamples,
                                                    TextureQuery::arraySize,
                                                    TextureQuery::normalizedCoords,
                                                    TextureQuery::channelOrder,
                                                    TextureQuery::channelDataType};

        for( const TextureQuery query : textureQueries )
        {
            PTXIntrinsicModifiers lwrrModifiers{};
            lwrrModifiers.texQuery = query;
            possibleModifiers.push_back( lwrrModifiers );
        }
    }

    if( (ptxInstructionCode)instTemplate->code == ptx_txq_level_Instr )
    {
        std::vector<TextureQuery> textureQueries = {TextureQuery::width, TextureQuery::height, TextureQuery::depth};

        for( const TextureQuery query : textureQueries )
        {
            PTXIntrinsicModifiers lwrrModifiers{};
            lwrrModifiers.texQuery = query;
            possibleModifiers.push_back( lwrrModifiers );
        }
    }

    return possibleModifiers;
}

std::string getTexSurfQueryString( TextureQuery query )
{
    switch( query )
    {
        case TextureQuery::width:
            return "width";
        case TextureQuery::height:
            return "height";
        case TextureQuery::depth:
            return "depth";
        case TextureQuery::numMipmapLevels:
            return "num_mipmap_levels";
        case TextureQuery::numSamples:
            return "num_samples";
        case TextureQuery::arraySize:
            return "array_size";
        case TextureQuery::normalizedCoords:
            return "normalized_coords";
        case TextureQuery::channelOrder:
            return "channel_order";
        case TextureQuery::channelDataType:
            return "channel_data_type";
        default:
            throw std::runtime_error( "Unrecognized texture query" );
    }
    throw std::runtime_error( "Unrecognized texture query" );
}

std::string getTxqInstructionName( const ptxInstructionTemplate& instTemplate, const PTXIntrinsicModifiers& modifiers )
{
    std::string name = instTemplate->name;

    if( (ptxInstructionCode)instTemplate->code == ptx_txq_level_Instr )
        name += ".level";

    name += "." + getTexSurfQueryString( modifiers.texQuery );

    return name + ".b32";
}

std::vector<PTXIntrinsicInfo> getTxqInstructionsFromTemplate( llvm::LLVMContext& context,
                                                              const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleTxqModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            std::string instructionName = getTxqInstructionName( instTemplate, modifiers );
            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, signature} );
        }
    }

    return instructions;
}

std::string getClampModeString( ClampMode mode )
{
    switch( mode )
    {
        case ClampMode::zero:
            return "zero";
        case ClampMode::clamp:
            return "clamp";
        case ClampMode::trap:
            return "trap";
        default:
            throw std::runtime_error( "Unhandled clamp mode" );
    }
    throw std::runtime_error( "Unhandled clamp mode" );
}

std::string getCacheOperatorString( CacheOp op )
{
    switch( op )
    {
        case CacheOp::unspecified:
            throw std::runtime_error( "unspecified cache operator" );
        case CacheOp::cg:
            return "cg";
        case CacheOp::cs:
            return "cs";
        case CacheOp::ca:
            return "ca";
        case CacheOp::lu:
            return "lu";
        case CacheOp::cv:
            return "cv";
        case CacheOp::ci:
            return "ci";
        case CacheOp::wb:
            return "wb";
        case CacheOp::wt:
            return "wt";
    }
    throw std::runtime_error( "Unhandled cache operator" );
}

std::string getSurfInstructionName( const std::string&           baseName,
                                    ptxInstructionCode           opCode,
                                    const PTXIntrinsicModifiers& modifiers,
                                    const InstructionSignature&  signature )
{
    // The suffix we apply to our wrappers in the ptx_instr Python
    // scripts doesn't match the format described in the PTX ISA. For
    // now, we match our scripts, for consistency
    std::string suffix = getTexDimensionalityString( modifiers.texDim );

    suffix += "." + getClampModeString( modifiers.clampMode );

    if( modifiers.cacheOp != CacheOp::unspecified )
        suffix += "." + getCacheOperatorString( modifiers.cacheOp );

    if( modifiers.vectorSize != VectorSize::unspecified )
        suffix += ".v" + std::to_string( vectorSizeToInt( modifiers.vectorSize ) );

    if( opCode == ptx_suld_b_Instr )
        suffix += "." + signature[0].ptxType;
    else
        suffix += "." + signature[signature.size() - 1].ptxType;

    std::string instructionName = std::string( baseName ) + "." + suffix;

    return instructionName;
}

std::vector<PTXIntrinsicModifiers> generatePossibleSurfModifiers( const ptxInstructionTemplate& instTemplate )
{
    // See https://docs.lwpu.com/lwca/parallel-thread-exelwtion/index.html#surface-instructions-suld
    // for valid modifier combinations

    std::vector<TextureDimensionality> dimensionalities = {
        TextureDimensionality::dim1D,       TextureDimensionality::dim1DArray, TextureDimensionality::dim2D,
        TextureDimensionality::dim2DArray,  TextureDimensionality::dim3D,      TextureDimensionality::dimLwbe,
        TextureDimensionality::dimLwbeArray};

    std::vector<ClampMode>  clampModes  = {ClampMode::clamp, ClampMode::trap, ClampMode::zero};
    std::vector<VectorSize> vectorSizes = {VectorSize::unspecified, VectorSize::v2, VectorSize::v4};

    // Load and store instructions support different cache modifiers.
    std::vector<CacheOp> cacheOps;
    if( instTemplate->code == ptx_suld_b_Instr )
        cacheOps = {CacheOp::unspecified, CacheOp::ca, CacheOp::cg, CacheOp::cs, CacheOp::cv};
    else if( instTemplate->code == ptx_sust_b_Instr || instTemplate->code == ptx_sust_p_Instr )
        cacheOps = {CacheOp::unspecified, CacheOp::wb, CacheOp::cg, CacheOp::cs, CacheOp::wt};
    else
        throw std::runtime_error( "unrecognized PTX op code" );

    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    for( const TextureDimensionality dim : dimensionalities )
    {
        for( const ClampMode clampMode : clampModes )
        {
            for( const VectorSize vectorSize : vectorSizes )
            {
                for( const CacheOp cacheOp : cacheOps )
                {
                    PTXIntrinsicModifiers lwrrModifiers{};
                    lwrrModifiers.texDim     = dim;
                    lwrrModifiers.cacheOp    = cacheOp;
                    lwrrModifiers.vectorSize = vectorSize;
                    lwrrModifiers.clampMode  = clampMode;

                    possibleModifiers.push_back( lwrrModifiers );
                }
            }
        }
    }

    return possibleModifiers;
}

std::vector<PTXIntrinsicInfo> getSurfaceInstructionsFromTemplate( llvm::LLVMContext& context,
                                                                  const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleSurfModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    std::vector<PTXIntrinsicInfo> instructions;

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            std::string instructionName =
                getSurfInstructionName( instTemplate->name, (ptxInstructionCode)instTemplate->code, modifiers, signature );
            InstructionSignature expandedSignature = expandSurfaceInstructionSignature( context, signature, modifiers );

            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, expandedSignature} );
        }
    }

    return instructions;
}

using PTXAddressSpace = optix::PTXIntrinsics::AddressSpace;

void addAtomOrRedModifiers( AtomicOperation op, MemOrdering ordering, MemScope scope, PTXAddressSpace addressSpace, std::vector<PTXIntrinsicModifiers>& outModifiers )

{
    PTXIntrinsicModifiers modifiers{};
    modifiers.memOrdering  = ordering;
    modifiers.memScope     = scope;
    modifiers.addressSpace = addressSpace;
    modifiers.atomicOp     = op;

    outModifiers.push_back( modifiers );
}

void addAtomOrRedModifiers( AtomicOperation op, MemOrdering ordering, MemScope scope, std::vector<PTXIntrinsicModifiers>& outModifiers )
{
    std::vector<PTXAddressSpace> possibleAddressSpaces = {PTXAddressSpace::unspecified, PTXAddressSpace::global, PTXAddressSpace::shared};

    for( const PTXAddressSpace addressSpace : possibleAddressSpaces )
    {
        // The generic address space can't be specified explicitly, and is used when an address space isn't specified
        addAtomOrRedModifiers( op, ordering, scope, addressSpace, outModifiers );
    }
}

void addAtomOrRedModifiers( AtomicOperation op, MemOrdering ordering, std::vector<PTXIntrinsicModifiers>& outModifiers )
{
    std::vector<MemScope> possibleScopes = {MemScope::unspecified, MemScope::gpu, MemScope::cta, MemScope::system};

    for( const MemScope scope : possibleScopes )
        addAtomOrRedModifiers( op, ordering, scope, outModifiers );
}

std::vector<PTXIntrinsicModifiers> generatePossibleAtomOrRedModifiers( const ptxInstructionTemplate& instTemplate )
{

    std::vector<MemOrdering> possibleOrderings = {MemOrdering::unspecified, MemOrdering::relaxed, MemOrdering::rel};
    // Atom instructions support more memory orderings
    if( instTemplate->code == ptx_atom_Instr )
    {
        possibleOrderings.push_back( MemOrdering::acq );
        possibleOrderings.push_back( MemOrdering::acq_rel );
    }

    std::vector<AtomicOperation> operations;
    if( instTemplate->features.ATOMOPF || instTemplate->features.ATOMOPI )
        operations.push_back( AtomicOperation::add );
    if( instTemplate->features.ATOMOPI )
    {
        operations.push_back( AtomicOperation::inc );
        operations.push_back( AtomicOperation::dec );
        operations.push_back( AtomicOperation::max );
        operations.push_back( AtomicOperation::min );
    }
    if( instTemplate->features.ATOMOPB )
    {
        operations.push_back( AtomicOperation::exch );
        operations.push_back( AtomicOperation::andOp );
        operations.push_back( AtomicOperation::orOp );
        operations.push_back( AtomicOperation::xorOp );
    }
    if( instTemplate->features.CAS )
        operations.push_back( AtomicOperation::cas );

    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    for( const AtomicOperation op : operations )
    {
        for( const MemOrdering ordering : possibleOrderings )
            addAtomOrRedModifiers( op, ordering, possibleModifiers );
    }

    if ( instTemplate->features.NOFTZ )
        std::for_each( possibleModifiers.begin(), possibleModifiers.end(), []( PTXIntrinsicModifiers& modifiers ) { modifiers.noftz = Noftz::noftz; } );

    return possibleModifiers;
}

std::string getOperationString( AtomicOperation op )
{
    switch( op )
    {
        case AtomicOperation::exch:
            return "exch";
        case AtomicOperation::add:
            return "add";
        case AtomicOperation::andOp:
            return "and";
        case AtomicOperation::orOp:
            return "or";
        case AtomicOperation::xorOp:
            return "xor";
        case AtomicOperation::max:
            return "max";
        case AtomicOperation::min:
            return "min";
        case AtomicOperation::inc:
            return "inc";
        case AtomicOperation::dec:
            return "dec";
        case AtomicOperation::cas:
            return "cas";
        default:
            throw std::runtime_error( "Unrecognized atomic operation!" );
    }
    throw std::runtime_error( "Unrecognized atomic operation!" );
}

std::string getAddressSpaceString( PTXAddressSpace addressSpace )
{
    switch( addressSpace )
    {
        case PTXAddressSpace::local:
            return "local";
        case PTXAddressSpace::global:
            return "global";
        case PTXAddressSpace::shared:
            return "shared";
        case PTXAddressSpace::constant:
            return "const";
        case PTXAddressSpace::param:
            return "param";
        default:
            throw std::runtime_error( "Unrecognized address space" );
    }
    throw std::runtime_error( "Unrecognized address space" );
}

std::string getMemOrderingString( MemOrdering ordering )
{
    switch( ordering )
    {
        case MemOrdering::weak:
            return "weak";
        case MemOrdering::relaxed:
            return "relaxed";
        case MemOrdering::acq:
            return "acquire";
        case MemOrdering::rel:
            return "release";
        case MemOrdering::acq_rel:
            return "acq_rel";
        default:
            throw std::runtime_error( "Unhandled memory ordering" );
    }
    throw std::runtime_error( "Unhandled memory ordering" );
}

std::string getMemScopeString( MemScope scope )
{
    switch( scope )
    {
        case MemScope::gpu:
            return "gpu";
        case MemScope::cta:
            return "cta";
        case MemScope::system:
            return "sys";
        default:
            throw std::runtime_error( "Unhandled memory scope" );
    }
    throw std::runtime_error( "Unhandled memory scope" );
}

std::string getAtomOrRedInstructionName( const std::string& baseName, const PTXIntrinsicModifiers& modifiers, const InstructionSignature& signature )
{
    std::string instructionName( baseName );
    if( modifiers.memOrdering != MemOrdering::unspecified )
        instructionName += "." + getMemOrderingString( modifiers.memOrdering );
    if( modifiers.memScope != MemScope::unspecified )
        instructionName += "." + getMemScopeString( modifiers.memScope );
    if( modifiers.addressSpace != PTXAddressSpace::unspecified )
        instructionName += "." + getAddressSpaceString( modifiers.addressSpace );
    instructionName += "." + getOperationString( modifiers.atomicOp );
    if( modifiers.noftz != Noftz::unspecified )
        instructionName += "." + getNoftzString( modifiers.noftz );
    instructionName += "." + signature[2].ptxType;

    return instructionName;
}

std::vector<PTXIntrinsicInfo> getAtomOrRedInstructionsFromTemplate( llvm::LLVMContext& context,
                                                                    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleAtomOrRedModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            // Adjust the signature so the memory type points to the proper address space.
            llvm::Type* memoryLlvmType =
                llvm::PointerType::get( signature[2].llvmType, ptxToLwvmAddressSpace( modifiers.addressSpace ) );
            InstructionSignature lwrrSignature;
            for( OperandType lwrrType : signature )
            {
                if( lwrrType.ptxType.compare( "memory" ) == 0 )
                    lwrrType.llvmType = memoryLlvmType;
                lwrrSignature.push_back( lwrrType );
            }

            const std::string instructionName = getAtomOrRedInstructionName( instTemplate->name, modifiers, signature );

            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, lwrrSignature} );
        }
    }

    return instructions;
}

std::string getStringForCmpOp( CompareOperator cmpOp )
{
    switch( cmpOp )
    {
        case CompareOperator::unspecified:
            throw std::runtime_error( "unspecified comparison predicate" );
        case CompareOperator::eq:
            return "eq";
        case CompareOperator::ne:
            return "ne";
        case CompareOperator::lt:
            return "lt";
        case CompareOperator::le:
            return "le";
        case CompareOperator::gt:
            return "gt";
        case CompareOperator::ge:
            return "ge";
        case CompareOperator::lo:
            return "lo";
        case CompareOperator::ls:
            return "ls";
        case CompareOperator::hi:
            return "li";
        case CompareOperator::hs:
            return "hs";
        case CompareOperator::equ:
            return "equ";
        case CompareOperator::neu:
            return "neu";
        case CompareOperator::ltu:
            return "ltu";
        case CompareOperator::leu:
            return "leu";
        case CompareOperator::gtu:
            return "gtu";
        case CompareOperator::geu:
            return "geu";
        case CompareOperator::num:
            return "num";
        case CompareOperator::nan:
            return "nan";
    }
    throw std::runtime_error( "unrecognized comparison predicate" );
}

std::string getStringForBoolOp( BooleanOperator boolOp )
{
    switch( boolOp )
    {
        case BooleanOperator::unspecified:
            throw std::runtime_error( "unspecified boolean operator" );
        case BooleanOperator::andOp:
            return "and";
        case BooleanOperator::orOp:
            return "or";
        case BooleanOperator::xorOp:
            return "xor";
    }
    throw std::runtime_error( "unrecognized boolean operator" );
}

std::string getSetpInstructionName( const std::string& baseName, const PTXIntrinsicModifiers& modifiers, const InstructionSignature& signature )

{
    std::string instructionName = std::string( baseName );

    instructionName += "." + getStringForCmpOp( modifiers.cmpOp );

    if( modifiers.boolOp != BooleanOperator::unspecified )
        instructionName += "." + getStringForBoolOp( modifiers.boolOp );

    if( modifiers.ftz != Ftz::unspecified )
        instructionName += ".ftz";

    instructionName += "." + signature[1].ptxType;

    return instructionName;
}

std::string getSetInstructionName( const std::string& baseName, const PTXIntrinsicModifiers& modifiers, const InstructionSignature& signature )
{
    std::string instructionName = std::string( baseName );

    instructionName += "." + getStringForCmpOp( modifiers.cmpOp );

    if( modifiers.boolOp != BooleanOperator::unspecified )
        instructionName += "." + getStringForBoolOp( modifiers.boolOp );

    if( modifiers.ftz != Ftz::unspecified )
        instructionName += ".ftz";

    instructionName += "." + signature[0].ptxType + "." + signature[1].ptxType;

    return instructionName;
}

void addSetModifiers( BooleanOperator boolOp, CompareOperator cmpOp, Ftz ftz, std::vector<PTXIntrinsicModifiers>& outModifiers )
{
    PTXIntrinsicModifiers modifiers{};

    modifiers.boolOp = boolOp;
    modifiers.cmpOp  = cmpOp;
    modifiers.ftz    = ftz;

    outModifiers.push_back( modifiers );
}

std::vector<PTXIntrinsicModifiers> generatePossibleSetModifiers( const ptxInstructionTemplate& instTemplate )
{
    std::vector<CompareOperator> possibleCmpOps = {CompareOperator::eq,  CompareOperator::ne,  CompareOperator::lt,
                                                   CompareOperator::le,  CompareOperator::gt,  CompareOperator::ge,
                                                   CompareOperator::lo,  CompareOperator::ls,  CompareOperator::hi,
                                                   CompareOperator::hs,  CompareOperator::equ, CompareOperator::neu,
                                                   CompareOperator::ltu, CompareOperator::leu, CompareOperator::gtu,
                                                   CompareOperator::geu, CompareOperator::num, CompareOperator::nan};

    std::vector<BooleanOperator> possibleBoolOps = {BooleanOperator::unspecified, BooleanOperator::andOp,
                                                    BooleanOperator::orOp, BooleanOperator::xorOp };

    std::vector<Ftz> ftzValues = {Ftz::unspecified};
    if( instTemplate->features.FTZ )
        ftzValues.push_back( Ftz::ftz );

    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    for( const BooleanOperator boolOp : possibleBoolOps )
        for( const CompareOperator cmpOp : possibleCmpOps )
            for( const Ftz ftz : ftzValues )
                addSetModifiers( boolOp, cmpOp, ftz, possibleModifiers );

    return possibleModifiers;
}

/**
 * Return true of the given comparison operator is valid for floating point comparisons.
 */
bool isValidFloatComparison( CompareOperator op )
{
    switch( op )
    {
        case CompareOperator::unspecified:
            throw std::runtime_error( "unspecified comparison operator" );
        case CompareOperator::eq:
        case CompareOperator::ne:
        case CompareOperator::lt:
        case CompareOperator::le:
        case CompareOperator::gt:
        case CompareOperator::ge:
        case CompareOperator::equ:
        case CompareOperator::neu:
        case CompareOperator::ltu:
        case CompareOperator::leu:
        case CompareOperator::gtu:
        case CompareOperator::geu:
        case CompareOperator::num:
        case CompareOperator::nan:
            return true;
        case CompareOperator::lo:
        case CompareOperator::ls:
        case CompareOperator::hi:
        case CompareOperator::hs:
            return false;
    }
    throw std::runtime_error( "unrecognized comparison operator" );
}

/**
 * Return true if the given comparison operator is valid for integer comparisons.
 */
bool isValidIntComparison( CompareOperator op )
{
    switch( op )
    {
        case CompareOperator::unspecified:
            throw std::runtime_error( "unspecified comparison operator" );
        case CompareOperator::eq:
        case CompareOperator::ne:
        case CompareOperator::lt:
        case CompareOperator::le:
        case CompareOperator::gt:
        case CompareOperator::ge:
        case CompareOperator::lo:
        case CompareOperator::ls:
        case CompareOperator::hi:
        case CompareOperator::hs:
            return true;
        case CompareOperator::equ:
        case CompareOperator::neu:
        case CompareOperator::ltu:
        case CompareOperator::leu:
        case CompareOperator::gtu:
        case CompareOperator::geu:
        case CompareOperator::num:
        case CompareOperator::nan:
            return false;
    }
    throw std::runtime_error( "unrecognized comparison operator" );
}

bool isValidSignatureModifierCombination( llvm::LLVMContext& context, const PTXIntrinsicModifiers& modifiers, const InstructionSignature& signature )
{
    // Combination isn't valid if the comparison predicate doesn't match the
    // types (e.g. float comparisons with int types)
    const bool isFloat = signature[1].ptxType[0] == 'f';
    if( isFloat && !isValidFloatComparison( modifiers.cmpOp ) )
        return false;
    const bool isInt = signature[1].ptxType[0] == 'u' || signature[1].ptxType[0] == 's';
    if( isInt && !isValidIntComparison( modifiers.cmpOp ) )
        return false;
    // Byte types only support eq and neq
    const bool isByte = signature[1].ptxType[0] == 'b';
    if( isByte && !( modifiers.cmpOp == CompareOperator::eq || modifiers.cmpOp == CompareOperator::ne ) )
        return false;

    // Invalid if modifiers have a boolean operator but signature doesn't have an input predicate that can we can apply the operator to.
    llvm::Type* predType = llvm::IntegerType::get( context, 1 );
    if( modifiers.boolOp != BooleanOperator::unspecified && signature[signature.size() - 1].llvmType != predType )
        return false;
    // Invalid if optional predicate is present but boolean operator isn't
    if( modifiers.boolOp == BooleanOperator::unspecified && signature[signature.size() - 1].llvmType == predType )
        return false;

    return true;
}

std::vector<PTXIntrinsicInfo> getSetOrSetpInstructionsFromTemplate( llvm::LLVMContext& context,
                                                                    const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    const bool isSetp = instTemplate->code == ptx_setp_Instr;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleSetModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    std::vector<PTXIntrinsicInfo> instructions;

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            if( !isValidSignatureModifierCombination( context, modifiers, signature ) )
                continue;

            std::string instructionName = isSetp ? getSetpInstructionName( instTemplate->name, modifiers, signature ) :
                                                   getSetInstructionName( instTemplate->name, modifiers, signature );

            InstructionSignature expandedSignature =
                expandSetOrSetpInstructionSignature( context, signature, instTemplate->features.RESULTP );

            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, expandedSignature} );
        }
    }

    return instructions;
}

void addLdStModifiers( PTXIntrinsicModifiers               modifiers,
                       const std::vector<PTXAddressSpace>&    addressSpaces,
                       std::vector<PTXIntrinsicModifiers>& outModifiers )
{
    std::vector<VectorSize> vectorSizes = {VectorSize::unspecified, VectorSize::v2, VectorSize::v4};
    for( const PTXAddressSpace addressSpace : addressSpaces )
    {
        modifiers.addressSpace = addressSpace;

        for( const VectorSize vectorSize : vectorSizes )
        {
            modifiers.vectorSize = vectorSize;
            outModifiers.push_back( modifiers );
        }
    }
}

void addLdStModifiers( PTXIntrinsicModifiers               modifiers,
                       const std::vector<MemScope>&        scopes,
                       const std::vector<PTXAddressSpace>&    addressSpaces,
                       std::vector<PTXIntrinsicModifiers>& outModifiers )
{
    for( const MemScope scope : scopes )
    {
        modifiers.memScope = scope;
        addLdStModifiers( modifiers, addressSpaces, outModifiers );
    }
}

void addLdStModifiers( PTXIntrinsicModifiers               modifiers,
                       const std::vector<CacheOp>&         cacheOps,
                       const std::vector<PTXAddressSpace>&    addressSpaces,
                       std::vector<PTXIntrinsicModifiers>& outModifiers )
{
    for( const CacheOp cacheOp : cacheOps )
    {
        modifiers.cacheOp = cacheOp;
        addLdStModifiers( modifiers, addressSpaces, outModifiers );
    }
}

std::vector<PTXIntrinsicModifiers> generatePossibleLdStModifiers( const ptxInstructionTemplate& instTemplate )
{
    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    std::vector<MemScope> allScopes = {MemScope::gpu, MemScope::cta, MemScope::system};

    std::vector<PTXAddressSpace> allAddressSpaces = {PTXAddressSpace::unspecified, PTXAddressSpace::local,
                                               PTXAddressSpace::global,      PTXAddressSpace::shared,
                                               PTXAddressSpace::constant,    PTXAddressSpace::param};

    std::vector<PTXAddressSpace> orderedAddressSpaces = {PTXAddressSpace::unspecified, PTXAddressSpace::global, PTXAddressSpace::shared};

    std::vector<CacheOp> cacheOps = {CacheOp::unspecified, CacheOp::ca, CacheOp::cg,
                                     CacheOp::cs,          CacheOp::lu, CacheOp::cv};

    // ST instructions have a few additional types of modifiers
    if( (ptxInstructionCode)instTemplate->code == ptx_st_Instr )
    {
        cacheOps.push_back( CacheOp::wt );
        cacheOps.push_back( CacheOp::wb );
    }

    // Generate modifiers compatible with "ld.acquire..." instructions
    if( (ptxInstructionCode)instTemplate->code == ptx_ld_Instr )
    {
        PTXIntrinsicModifiers ldAcquireModifiers{};
        ldAcquireModifiers.memOrdering = MemOrdering::acq;
        addLdStModifiers( ldAcquireModifiers, allScopes, orderedAddressSpaces, possibleModifiers );
    }

    // Generate modifiers compatible with "ld.relaxed..." instructions
    PTXIntrinsicModifiers ldRelaxedModifiers{};
    ldRelaxedModifiers.memOrdering = MemOrdering::relaxed;
    addLdStModifiers( ldRelaxedModifiers, allScopes, orderedAddressSpaces, possibleModifiers );

    // Generate modifiers compatible with "ld.volatile..." instructions
    PTXIntrinsicModifiers ldVolatileModifiers{};
    ldVolatileModifiers.vol         = Volatile::vol;
    ldVolatileModifiers.memOrdering = MemOrdering::relaxed;
    ldVolatileModifiers.memScope    = MemScope::system;
    addLdStModifiers( ldVolatileModifiers, orderedAddressSpaces, possibleModifiers );

    // Generate modifiers compatible with "ld.{weak}..." instructions
    PTXIntrinsicModifiers ldWeakModifiers{};
    ldWeakModifiers.memOrdering = MemOrdering::weak;
    ldWeakModifiers.memScope    = MemScope::unspecified;

    addLdStModifiers( ldWeakModifiers, cacheOps, allAddressSpaces, possibleModifiers );

    ldWeakModifiers.memOrdering = MemOrdering::unspecified;
    addLdStModifiers( ldWeakModifiers, cacheOps, allAddressSpaces, possibleModifiers );

    // Generate modifiers compatible with ld.global.{cop}.nc...
    if( (ptxInstructionCode)instTemplate->code == ptx_ld_Instr )
    {
        PTXIntrinsicModifiers ldNcModifiers{};
        ldNcModifiers.texDomain = TexDomain::nc;

        std::vector<PTXAddressSpace> ldNcAddressSpaces = {PTXAddressSpace::global};

        addLdStModifiers( ldNcModifiers, cacheOps, ldNcAddressSpaces, possibleModifiers );
    }

    // TODO: Add path for "st.release" (it's not strictly necessary, since our PTX scripts don't lwrrently generate it)

    return possibleModifiers;
}

std::string getLdStInstructionName( const std::string& baseName, const PTXIntrinsicModifiers& modifiers, const OperandType& suffixType )
{
    std::string instructionName = std::string( baseName );

    // Volatile always implies ".relaxed.sys"
    if( modifiers.vol != Volatile::unspecified )
        instructionName += ".volatile";
    else
    {
        if( modifiers.memOrdering != MemOrdering::unspecified )
            instructionName += "." + getMemOrderingString( modifiers.memOrdering );
        if( modifiers.memScope != MemScope::unspecified )
            instructionName += "." + getMemScopeString( modifiers.memScope );
    }

    if( modifiers.addressSpace != PTXAddressSpace::unspecified )
        instructionName += "." + getAddressSpaceString( modifiers.addressSpace );

    // According to the PTX ISA, this should come after the cache
    // operator, but our scripts emit it before for some reason.
    if( modifiers.texDomain != TexDomain::unspecified )
        instructionName += ".nc";

    if( modifiers.cacheOp != CacheOp::unspecified )
        instructionName += "." + getCacheOperatorString( modifiers.cacheOp );

    if( modifiers.vectorSize != VectorSize::unspecified )
        instructionName += ".v" + std::to_string( vectorSizeToInt( modifiers.vectorSize ) );

    instructionName += "." + suffixType.ptxType;

    return instructionName;
}

std::vector<PTXIntrinsicInfo> getLdStInstructionsFromTemplate( llvm::LLVMContext& context,
                                                               const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleLdStModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            // TODO: Templates with more than two arguments have the "DESC"
            // flag set in their features. What does that mean? What is the
            // second argument?

            // Don't add templates with more than two arguments.
            if( signature.size() > 3 )
                continue;
            // Don't add load instructions with optional cache-policy argument
            if( instTemplate->code == ptx_ld_Instr && signature.size() > 2 )
                continue;

            // The value type is returned by ld instructions, second param for st
            const int valTypeIdx = instTemplate->code == ptx_ld_Instr ? 0 : 2;

            std::string instructionName = getLdStInstructionName( instTemplate->name, modifiers, signature[valTypeIdx] );

            InstructionSignature expandedSignature = expandLdStInstructionSignature( context, instTemplate, signature, modifiers );

            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, expandedSignature, false} );
        }
    }

    return instructions;
}

std::string getShfModeString( FunnelShiftWrapMode mode )
{
    switch( mode )
    {
        case FunnelShiftWrapMode::clamp:
            return "clamp";
        case FunnelShiftWrapMode::wrap:
            return "wrap";
        default:
            throw std::runtime_error( "Unknown funnel shift mode" );
    }
}

std::string getShfInstructionName( const std::string& baseName, const PTXIntrinsicModifiers& modifiers, const InstructionSignature& signature )
{
    std::string instructionName =
        std::string( baseName ) + "." + getShfModeString( modifiers.funnelShiftWrapMode ) + "." + signature[1].ptxType;

    return instructionName;
}

std::vector<PTXIntrinsicInfo> getShfInstructionsFromTemplate( llvm::LLVMContext& context,
                                                              const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo>     instructions;
    std::vector<InstructionSignature> possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    // shf instructions have two sets of possible modifiers: "clamp" and "wrap".
    // Shift direction is determined by the op code.
    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    PTXIntrinsicModifiers modifiers{};

    modifiers.funnelShiftWrapMode = FunnelShiftWrapMode::wrap;
    possibleModifiers.push_back( modifiers );

    modifiers.funnelShiftWrapMode = FunnelShiftWrapMode::clamp;
    possibleModifiers.push_back( modifiers );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            std::string instructionName = getShfInstructionName( instTemplate->name, modifiers, signature );
            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, signature} );
        }
    }

    return instructions;
}

std::vector<PTXIntrinsicModifiers> generatePossibleMovModifiers( const ptxInstructionTemplate& instTemplate )
{
    std::vector<VectorSize> possibleVectorSizes = {VectorSize::unspecified};
    if( instTemplate->features.VECTORIZABLE )
    {
        possibleVectorSizes.push_back( VectorSize::v2 );
        possibleVectorSizes.push_back( VectorSize::v4 );
    }

    std::vector<PTXIntrinsicModifiers> possibleModifiers;

    PTXIntrinsicModifiers m{};
    for( const VectorSize vectorSize : possibleVectorSizes )
    {
        m.vectorSize = vectorSize;
        possibleModifiers.push_back( m );
    }

    return possibleModifiers;
}

std::vector<PTXIntrinsicInfo> getMovInstructionsFromTemplate( llvm::LLVMContext& context,
                                                              const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<PTXIntrinsicModifiers> possibleModifiers  = generatePossibleMovModifiers( instTemplate );
    std::vector<InstructionSignature>  possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const PTXIntrinsicModifiers& modifiers : possibleModifiers )
    {
        for( const InstructionSignature& signature : possibleSignatures )
        {
            std::string instructionName = std::string( instTemplate->name );
            if( modifiers.vectorSize != VectorSize::unspecified )
                instructionName += ".v" + std::to_string( vectorSizeToInt( modifiers.vectorSize ) );
            instructionName += "." + signature[1].ptxType;

            // Adjust the function's signature so it matches the vector size specified by the modifiers.
            InstructionSignature adjustedSignature = signature;
            if( modifiers.vectorSize != VectorSize::unspecified )
            {
                for( size_t i = 0; i < signature.size(); ++i )
                    adjustedSignature[i].llvmType =
                        llvm::VectorType::get( signature[i].llvmType, vectorSizeToInt( modifiers.vectorSize ) );
            }

            instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, modifiers, adjustedSignature} );
        }
    }

    return instructions;
}

std::vector<PTXIntrinsicInfo> getDp2aInstructionsFromTemplate( llvm::LLVMContext& context,
                                                               const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<InstructionSignature> possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const InstructionSignature& signature : possibleSignatures )
    {
        std::string instructionName =
            std::string( instTemplate->name ) + "." + signature[1].ptxType + "." + signature[2].ptxType;

        instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, {}, signature} );
    }

    return instructions;
}

std::vector<PTXIntrinsicInfo> getDp4aInstructionsFromTemplate( llvm::LLVMContext& context,
                                                               const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo> instructions;

    std::vector<InstructionSignature> possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const InstructionSignature& signature : possibleSignatures )
    {
        std::string instructionName =
            std::string( instTemplate->name ) + "." + signature[1].ptxType + "." + signature[2].ptxType;

        instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, {}, signature, false} );
    }

    return instructions;
}

std::vector<PTXIntrinsicInfo> getInstructionsFromTemplate( llvm::LLVMContext& context,
                                                           const std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>& instTemplateData )
{
    const ptxInstructionTemplate& instTemplate = instTemplateData.first;

    std::vector<PTXIntrinsicInfo>     instructions;
    std::vector<InstructionSignature> possibleSignatures = generatePossibleSignatures( context, instTemplateData );

    for( const InstructionSignature& signature : possibleSignatures )
    {
        std::string instructionName = std::string( instTemplate->name ) + "." + signature[1].ptxType;
        instructions.push_back( {instructionName, (ptxInstructionCode)instTemplate->code, {}, signature, false} );
    }

    return instructions;
}

