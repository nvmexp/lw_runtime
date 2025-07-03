// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#ifndef OPTIX_BUILD_RUNTIME
#include <string>
#include <vector>
#endif

namespace optix {

enum SemanticType
{
    // The number of SBT slots that are allocated for bound
    // callable programs depends on the order of this enum
    // since it allocates slots for all possible semantics that
    // may call bound callable programs. So it uses the value
    // of ST_BINDLESS_CALLABLE_PROGRAM to callwlate the number of slots.
    ST_RAYGEN,
    ST_EXCEPTION,
    ST_MISS,
    ST_NODE_VISIT,
    ST_INTERSECTION,
    ST_BOUNDING_BOX,
    ST_CLOSEST_HIT,
    ST_ANY_HIT,
    ST_BOUND_CALLABLE_PROGRAM,
    ST_BINDLESS_CALLABLE_PROGRAM,
    ST_ATTRIBUTE,
    // Internal program that ilwokes the bounding box programs.
    // See compute_aabb() in src/AS/ComputeAabb.lw.
    ST_INTERNAL_AABB_ITERATOR,
    // Internal exception program used together with bounding box programs.
    // See compute_aabb_exception() in src/AS/ComputeAabb.lw.
    ST_INTERNAL_AABB_EXCEPTION,
    // Semantic type that is used only as the inherited semantic
    // type to mark bindless callable programs as heavyweight
    // during compilation, for the SBTManager and in the cache.
    ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE,
    ST_ILWALID
};
static const int NUM_SEMANTIC_TYPES = ST_ILWALID + 1;

#ifndef OPTIX_BUILD_RUNTIME
bool isTransformCallLegal( SemanticType callerST );
bool isGetPrimitiveIndexLegal( SemanticType callerST );
bool isGetHitKindLegal( SemanticType callerST );
bool isGetInstanceFlagsLegal( SemanticType callerST );
bool isTraceCallLegal( SemanticType callerST, bool useRTXDataModel );
bool isThrowCallLegal( SemanticType callerST );
bool isTerminateRayCallLegal( SemanticType callerST );
bool isIgnoreIntersectionCallLegal( SemanticType callerST );
bool isIntersectChildCallLegal( SemanticType callerST );
bool isPotentialIntersectionCallLegal( SemanticType callerST );
bool isReportIntersectionCallLegal( SemanticType callerST );
bool isExceptionCodeCallLegal( SemanticType callerST );
bool isAttributeAccessLegal( SemanticType callerST );
bool isAttributeWriteLegal( SemanticType callerST );
bool isPayloadAccessLegal( SemanticType callerST );
bool isPayloadStoreLegal( SemanticType callerST );
bool isLwrrentRayAccessLegal( SemanticType callerST );
bool isGetLowestGroupChildIndexCallLegal( SemanticType callerST );
bool isIntersectionDistanceAccessLegal( SemanticType callerST );
bool isLwrrentTimeAccessLegal( SemanticType callerST );
bool isBindlessCallableCallLegal( SemanticType callerST );
bool isBoundCallableCallLegal( SemanticType callerST );
bool isBufferStoreLegal( SemanticType callerST );
bool isPointerEscapeLegal( SemanticType callerST );

std::string semanticTypeToString( SemanticType sem );
std::string semanticTypeToAbbreviationString( SemanticType sem );
SemanticType stringToSemanticType( const std::string& str );
SemanticType abbreviationStringToSemanticType( const std::string& str );
std::vector<SemanticType> semanticTypesStringToVector( const std::string& str );
#endif

}  // namespace optix
