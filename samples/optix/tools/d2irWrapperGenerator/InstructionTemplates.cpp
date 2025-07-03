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

/*
 * This file was forked from ptxInstructionTemplates.c. It's been adapted to
 * work with C++ programs.
 */

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <ptxIR.h>
#include <ptxInstructions.h>

#include <InstructionTemplates.h>

static std::vector<std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>> s_stdTemplates;
static std::vector<std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>> s_extTemplates;

/*----------------------------- Template Storage -----------------------------*/

static void addTemplate( ptxParseData parseData, ptxInstructionTemplate instr, Bool isExtended, const std::vector<UnsupportedType>& unsupportedTypes )
{
    if( isExtended )
        s_extTemplates.push_back( std::make_pair( instr, unsupportedTypes ) );
    else
        s_stdTemplates.push_back( std::make_pair( instr, unsupportedTypes ) );
}

/*------------------------ Parsing Template Definition -----------------------*/

static uInt countTypes( cString s )
{
    char c;
    uInt result = 0;
    while( ( c = *s++ ) )
    {
        if( isalpha( c ) )
        {
            result++;
        }
    }
    return result;
}

static void handleBitIType( ptxInstructionTemplate result, uInt i, Int t, cString type )
{
    result->instrType[t] = ptxBitIType;
    // if no size restrictions follow, default to byte sizes 2, 4, and 8
    result->instrTypeSizes[t] = bitSetCreate();
    if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
    {
        bitSetInsert( result->instrTypeSizes[t], 16 );
        bitSetInsert( result->instrTypeSizes[t], 32 );
        bitSetInsert( result->instrTypeSizes[t], 64 );
    }
    else
    {
        bitSetInsert( result->instrTypeSizes[t], 0 );
    }
}

static void addInstructionTemplate( ptxParseData          parseData,
                                    cString               type,
                                    cString               name,
                                    cString               signature,
                                    ptxInstructionFeature features,
                                    ptxInstructionCode    code,
                                    Bool                  isExtended )
{
    Int                    t;
    uInt                   i;
    uInt                   size = 0;
    ptxInstructionTemplate result;

    uInt nrofArgTypes   = ( uInt )( strlen( signature ) );
    uInt nrofInstrTypes = countTypes( type );

    if( !( nrofArgTypes <= ptxMAX_INSTR_ARGS ) )
        throw std::runtime_error( std::string("ptxMAX_INSTR_ARGS too small, ") + name );

    stdNEW( result );

    char* nonConstName = (char*)malloc( strlen( name ) + 1 );
    strncpy( nonConstName, name, strlen( name ) + 1 );

    result->name           = nonConstName;
    result->code           = code;
    result->features       = features;
    result->nrofArguments  = nrofArgTypes;
    result->nrofInstrTypes = nrofInstrTypes;

    // WAR for identifying both A16, A32 and T32
    bool                         hasUnsupportedTypes = false;
    std::vector<UnsupportedType> unsupportedInstrTypes;

    // unpack instruction types and allowed sizes

    for( i = 0, t = -1; i < strlen( type ); i++ )
    {
        switch( type[i] )
        {
            case 'F':
                result->instrType[++t] = ptxFloatIType;
                // if no size restrictions follow, default to byte sizes 4 and 8
                result->instrTypeSizes[t] = bitSetCreate();
                if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
                {
                    bitSetInsert( result->instrTypeSizes[t], 32 );
                    bitSetInsert( result->instrTypeSizes[t], 64 );
                }
                else
                {
                    bitSetInsert( result->instrTypeSizes[t], 0 );
                }
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_UNDEF );
                break;

            case 'H':
                result->instrType[++t] = ptxPackedHalfFloatIType;
                // if no size restrictions follow, default to byte sizes 4
                result->instrTypeSizes[t] = bitSetCreate();
                if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
                {
                    bitSetInsert( result->instrTypeSizes[t], 32 );
                }
                else
                {
                    bitSetInsert( result->instrTypeSizes[t], 0 );
                }
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_UNDEF );
                break;

            case 'Q':
                // 'Q' represent custom float with size 8 bit (e4m3/e5m2) types and mantissa
                result->instrType[++t] = ptxLwstomFloatE8IType;
                // if no size restrictions follow, default to byte sizes 1 and 2
                result->instrTypeSizes[t] = bitSetCreate();
                if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
                {
                    bitSetInsert( result->instrTypeSizes[t], 8 );
                    bitSetInsert( result->instrTypeSizes[t], 16 );
                }
                else
                {
                    bitSetInsert( result->instrTypeSizes[t], 0 );
                }
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_UNDEF );
                break;

            case 'I':
                result->instrType[++t] = ptxIntIType;
                // if no size restrictions follow, default to byte sizes 2, 4, and 8
                result->instrTypeSizes[t] = bitSetCreate();
                if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
                {
                    bitSetInsert( result->instrTypeSizes[t], 16 );
                    bitSetInsert( result->instrTypeSizes[t], 32 );
                    bitSetInsert( result->instrTypeSizes[t], 64 );
                }
                else
                {
                    bitSetInsert( result->instrTypeSizes[t], 0 );
                }
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_UNDEF );
                break;
            // Manually added types for BF16/BF16x2/TF32 (A16/A32/T32)
            case 'A':
                result->instrType[++t] = ptxBitIType;
                // if no size restrictions follow, default to byte sizes 2, 4
                result->instrTypeSizes[t] = bitSetCreate();
                if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
                {
                    bitSetInsert( result->instrTypeSizes[t], 16 );
                    bitSetInsert( result->instrTypeSizes[t], 32 );
                }
                else
                {
                    bitSetInsert( result->instrTypeSizes[t], 0 );
                }
                hasUnsupportedTypes          = true;
                // this will be adjusted further down when reading the bit size
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_A16 );
                break;
            case 'T':
                result->instrType[++t] = ptxBitIType;
                // if no size restrictions follow, default to byte size 4
                result->instrTypeSizes[t] = bitSetCreate();
                if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
                {
                    bitSetInsert( result->instrTypeSizes[t], 32 );
                }
                else
                {
                    bitSetInsert( result->instrTypeSizes[t], 0 );
                }
                hasUnsupportedTypes          = true;
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_T32 );
                break;
            case 'B':
                result->instrType[++t] = ptxBitIType;
                // if no size restrictions follow, default to byte sizes 2, 4, and 8
                result->instrTypeSizes[t] = bitSetCreate();
                if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
                {
                    bitSetInsert( result->instrTypeSizes[t], 16 );
                    bitSetInsert( result->instrTypeSizes[t], 32 );
                    bitSetInsert( result->instrTypeSizes[t], 64 );
                }
                else
                {
                    bitSetInsert( result->instrTypeSizes[t], 0 );
                }
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_UNDEF );
                break;
            case 'P':
                if( !( i + 1 == strlen( type ) || isalpha( type[i + 1] ) ) )
                    throw std::runtime_error( "Type size restrictions not allowed for 'P' type" );

                result->instrType[++t]    = ptxPredicateIType;
                result->instrTypeSizes[t] = bitSetCreate();
                bitSetInsert( result->instrTypeSizes[t], 32 );  // predicates have size==4
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_UNDEF );
                break;
            case 'O':
                if( !( i + 1 == strlen( type ) || isalpha( type[i + 1] ) ) )
                    throw std::runtime_error( "Type size restrictions not allowed for 'O' type" );

                result->instrType[++t]    = ptxOpaqueIType;
                result->instrTypeSizes[t] = bitSetCreate();
                bitSetInsert( result->instrTypeSizes[t], 0 );  // use size==0 for opaques and do necessary checks separately
                unsupportedInstrTypes.emplace_back( UNSUPPORTED_TYPE_UNDEF );
                break;
            case '[':
                break;
            case '|':
            case ']':
                bitSetInsert( result->instrTypeSizes[t], size );
                size = 0;
                break;
            default:
                if( '0' <= type[i] && type[i] <= '9' )
                {
                    size = size * 10 + (type[i] - '0');
                }
                else
                {
                    throw std::runtime_error( std::string( "Unknown instruction type: '") + type[i] + "'" );
                }
                if( i + 1 == strlen( type ) || isalpha( type[i + 1] ) )
                {
                    bitSetInsert( result->instrTypeSizes[t], size );
                    if( hasUnsupportedTypes && unsupportedInstrTypes.back() == UNSUPPORTED_TYPE_A16 && size == 32 )
                        unsupportedInstrTypes.back() = UNSUPPORTED_TYPE_A32;
                    size = 0;
                }
                break;
        }

        if( !( t < (Int)nrofInstrTypes ) )
            throw std::runtime_error( std::string( "Instruction type error, '" ) +  name + "'");
    }

    /*
     * A single digit indicates that the argument in the current position follows the specified
     * instruction type, where instruction types are numbered starting with zero.
     *
     * For example, the instruction SET has argument type signature of '011'.  The instruction
     * SET.eq.f32.u32 d,a,b; would therefor map dest arg 'd' to type .f32, and map both source
     * args 'a' and 'b' to type .u32.
     */
    for( i = 0; i < nrofArgTypes; i++ )
    {
        switch( signature[i] )
        {
            case 'x':
                result->argType[i] = ptxU16AType;
                break;
            case 'u':
                result->argType[i] = ptxU32AType;
                break;
            case 'U':
                result->argType[i] = ptxU64AType;
                break;
            case 'd':
                result->argType[i] = ptxB32AType;
                break;
            case 'e':
                result->argType[i] = ptxB64AType;
                break;
            case 's':
                result->argType[i] = ptxS32AType;
                break;
            case 'f':
                result->argType[i] = ptxF32AType;
                break;
            case 'l':
                result->argType[i] = ptxScalarF32AType;
                break;
            case 'i':
                result->argType[i] = ptxImageAType;
                break;
            case 'h':
                result->argType[i] = ptxF16x2AType;
                break;
            case 'C':
                result->argType[i] = ptxConstantIntAType;
                break;
            case 'D':
                result->argType[i] = ptxConstantFloatAType;
                break;
            case 'P':
                result->argType[i] = ptxPredicateAType;
                break;
            case 'Q':
                result->argType[i] = ptxPredicateVectorAType;
                break;
            case 'M':
                result->argType[i] = ptxMemoryAType;
                break;
            case 'S':
                result->argType[i] = ptxSymbolAType;
                break;
            case 'T':
                result->argType[i] = ptxTargetAType;
                break;
            case 'A':
                result->argType[i] = ptxParamListAType;
                break;
            case 'V':
                result->argType[i] = ptxVoidAType;
                break;
            case 'L':
                result->argType[i] = ptxLabelAType;
                break;
            default:
            {
                uInt tindex = signature[i] - '0';

                if( !( isdigit( signature[i] ) ) )
                    throw std::runtime_error( std::string( "Unknown argument type: '" ) +  signature[i] + "'" );
                if( !( tindex < nrofInstrTypes ) )
                    throw std::runtime_error( std::string( "Instruction type index '") + std::to_string( tindex ) + "' out of range" );

                result->argType[i]   = ptxFollowAType;
                result->followMap[i] = tindex;
            }
        }
    }
    if( !hasUnsupportedTypes )
        unsupportedInstrTypes.clear();
    addTemplate( parseData, result, isExtended, unsupportedInstrTypes );
}

void getInstructionTemplates( ptxParseData                      parseData,
                              std::vector<std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>>& stdTemplates,
                              std::vector<std::pair<ptxInstructionTemplate, std::vector<UnsupportedType>>>& extTemplates )
{
    ptxInstructionFeature features;
#include "ptxInstructionDefs.adapted.incl"

    std::copy( s_stdTemplates.begin(), s_stdTemplates.end(), std::back_inserter( stdTemplates ) );
    std::copy( s_extTemplates.begin(), s_extTemplates.end(), std::back_inserter( extTemplates ) );
}
