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


#include <Objects/SemanticType.h>
#include <prodlib/exceptions/Assert.h>

#include <sstream>

using namespace optix;

bool optix::isTransformCallLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_RAYGEN:
        case ST_EXCEPTION:
        case ST_BINDLESS_CALLABLE_PROGRAM:
            return false;

        default:
            return true;
    }
}

bool optix::isGetPrimitiveIndexLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_CLOSEST_HIT:
        case ST_ANY_HIT:
        case ST_INTERSECTION:
        case ST_ATTRIBUTE:
            return true;

        default:
            return false;
    }
}

bool optix::isGetHitKindLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_CLOSEST_HIT:
        case ST_ANY_HIT:
        case ST_ATTRIBUTE:
            return true;

        default:
            return false;
    }
}

bool optix::isGetInstanceFlagsLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_INTERSECTION:
        case ST_ATTRIBUTE:
            return true;
        default:
            return false;
    }
}

bool optix::isTraceCallLegal( SemanticType callerST, bool useRTXDataModel )
{
    if( useRTXDataModel && callerST == ST_BINDLESS_CALLABLE_PROGRAM )
        return true;
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_RAYGEN:
        case ST_MISS:
        case ST_CLOSEST_HIT:
            return true;
        default:
            return false;
    }
}

bool optix::isThrowCallLegal( SemanticType callerST )
{
    return true;
}

bool optix::isTerminateRayCallLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_ANY_HIT:
            return true;
        default:
            return false;
    }
}

bool optix::isIgnoreIntersectionCallLegal( SemanticType callerST )
{
    return isTerminateRayCallLegal( callerST );
}

bool optix::isIntersectChildCallLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_NODE_VISIT:
            return true;
        default:
            return false;
    }
}

bool optix::isPotentialIntersectionCallLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_INTERSECTION:
            return true;
        default:
            return false;
    }
}

bool optix::isReportIntersectionCallLegal( SemanticType callerST )
{
    return isPotentialIntersectionCallLegal( callerST );
}

bool optix::isExceptionCodeCallLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_EXCEPTION:
        case ST_INTERNAL_AABB_EXCEPTION:
            return true;
        default:
            return false;
    }
}

bool optix::isAttributeAccessLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_CLOSEST_HIT:
        case ST_ANY_HIT:
        case ST_INTERSECTION:
        case ST_ATTRIBUTE:
            return true;
        default:
            return false;
    }
}

bool optix::isAttributeWriteLegal( SemanticType callerST )
{
    if( callerST == ST_BOUND_CALLABLE_PROGRAM )
    {
        // AM: I do not understand the meaning of this assert message.
        // I think that this means that at this point we cannot have semantic type ST_BOUND_CALLABLE_PROGRAM
        // because we always check for the semantic type of the caller program.
        RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
        return false;
    }

    return callerST == ST_INTERSECTION || callerST == ST_ATTRIBUTE;
}

bool optix::isPayloadAccessLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_CLOSEST_HIT:
        case ST_ANY_HIT:
        case ST_MISS:
        case ST_NODE_VISIT:
        case ST_INTERSECTION:
        case ST_ATTRIBUTE:
            return true;

        default:
            return false;
    }
}

bool optix::isPayloadStoreLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_CLOSEST_HIT:
        case ST_ANY_HIT:
        case ST_MISS:
        case ST_NODE_VISIT:
        case ST_INTERSECTION:
            return true;

        default:
            return false;
    }
}

bool optix::isLwrrentRayAccessLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_CLOSEST_HIT:
        case ST_ANY_HIT:
        case ST_MISS:
        case ST_NODE_VISIT:
        case ST_INTERSECTION:
        case ST_ATTRIBUTE:
            return true;

        default:
            return false;
    }
}

bool optix::isGetLowestGroupChildIndexCallLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_CLOSEST_HIT:
        case ST_ANY_HIT:
        case ST_MISS:
        case ST_NODE_VISIT:
        case ST_INTERSECTION:
        case ST_ATTRIBUTE:
            return true;

        default:
            return false;
    }
}

bool optix::isLwrrentTimeAccessLegal( SemanticType callerST )
{
    return isLwrrentRayAccessLegal( callerST );
}

bool optix::isIntersectionDistanceAccessLegal( SemanticType callerST )
{
    switch( callerST )
    {
        case ST_BOUND_CALLABLE_PROGRAM:
            RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );
            return false;

        case ST_CLOSEST_HIT:
        case ST_ANY_HIT:
        case ST_NODE_VISIT:
        case ST_INTERSECTION:
        case ST_ATTRIBUTE:
            return true;

        default:
            return false;
    }
}

bool optix::isBindlessCallableCallLegal( SemanticType callerST )
{
    if( callerST == ST_BOUND_CALLABLE_PROGRAM )
        RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );

    return callerST != ST_EXCEPTION && callerST != ST_ATTRIBUTE;
}

bool optix::isBoundCallableCallLegal( SemanticType callerST )
{
    if( callerST == ST_BOUND_CALLABLE_PROGRAM )
        RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );

    return callerST != ST_EXCEPTION && callerST != ST_ATTRIBUTE;
}

bool optix::isBufferStoreLegal( SemanticType callerST )
{
    if( callerST == ST_BOUND_CALLABLE_PROGRAM )
        RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );

    return callerST != ST_ATTRIBUTE;
}

bool optix::isPointerEscapeLegal( SemanticType callerST )
{
    if( callerST == ST_BOUND_CALLABLE_PROGRAM )
        RT_ASSERT_FAIL_MSG( "Invalid semantic type. Override ST_BOUND_CALLABLE_PROGRAM by inherited " );

    return callerST != ST_ATTRIBUTE;
}

std::string optix::semanticTypeToString( SemanticType sem )
{
    switch( sem )
    {
        case ST_RAYGEN:
            return "RAYGEN";
        case ST_EXCEPTION:
            return "EXCEPTION";
        case ST_MISS:
            return "MISS";
        case ST_NODE_VISIT:
            return "NODE_VISIT";
        case ST_INTERSECTION:
            return "INTERSECTION";
        case ST_BOUNDING_BOX:
            return "BOUNDING_BOX";
        case ST_CLOSEST_HIT:
            return "CLOSEST_HIT";
        case ST_ANY_HIT:
            return "ANY_HIT";
        case ST_ATTRIBUTE:
            return "ATTRIBUTE";
        case ST_BOUND_CALLABLE_PROGRAM:
            return "BOUND_CALLABLE_PROGRAM";
        case ST_BINDLESS_CALLABLE_PROGRAM:
        case ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE:
            return "BINDLESS_CALLABLE_PROGRAM";
        case ST_INTERNAL_AABB_ITERATOR:
            return "INTERNAL_AABB_ITERATOR";
        case ST_INTERNAL_AABB_EXCEPTION:
            return "INTERNAL_AABB_EXCEPTION";
        case ST_ILWALID:
            return "INVALID";
            // Default case intentionally omitted
    }
    RT_ASSERT_FAIL_MSG( "Invalid semantic type" );
}

std::string optix::semanticTypeToAbbreviationString( SemanticType sem )
{
    switch( sem )
    {
        case ST_RAYGEN:
            return "RG";
        case ST_EXCEPTION:
            return "EX";
        case ST_MISS:
            return "MS";
        case ST_NODE_VISIT:
            return "LW";
        case ST_INTERSECTION:
            return "IN";
        case ST_BOUNDING_BOX:
            return "BB";
        case ST_CLOSEST_HIT:
            return "CH";
        case ST_ANY_HIT:
            return "AH";
        case ST_ATTRIBUTE:
            return "AT";
        case ST_BOUND_CALLABLE_PROGRAM:
            return "CP";
        case ST_BINDLESS_CALLABLE_PROGRAM:
        case ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE:
            return "BCP";
        case ST_INTERNAL_AABB_ITERATOR:
            return "AAI";
        case ST_INTERNAL_AABB_EXCEPTION:
            return "AAE";
        case ST_ILWALID:
            return "INVALID";
            // Default case intentionally omitted
    }
    RT_ASSERT_FAIL_MSG( "Invalid semantic type" );
}

optix::SemanticType optix::stringToSemanticType( const std::string& str )
{
    if( str == "RAYGEN" )
        return ST_RAYGEN;
    if( str == "EXCEPTION" )
        return ST_EXCEPTION;
    if( str == "MISS" )
        return ST_MISS;
    if( str == "NODE_VISIT" )
        return ST_NODE_VISIT;
    if( str == "INTERSECTION" )
        return ST_INTERSECTION;
    if( str == "BOUNDING_BOX" )
        return ST_BOUNDING_BOX;
    if( str == "CLOSEST_HIT" )
        return ST_CLOSEST_HIT;
    if( str == "ANY_HIT" )
        return ST_ANY_HIT;
    if( str == "ATTRIBUTE" )
        return ST_ATTRIBUTE;
    if( str == "BOUND_CALLABLE_PROGRAM" )
        return ST_BOUND_CALLABLE_PROGRAM;
    if( str == "BINDLESS_CALLABLE_PROGRAM" )
        return ST_BINDLESS_CALLABLE_PROGRAM;
    if( str == "INTERNAL_AABB_ITERATOR" )
        return ST_INTERNAL_AABB_ITERATOR;
    if( str == "INTERNAL_AABB_EXCEPTION" )
        return ST_INTERNAL_AABB_EXCEPTION;
    return ST_ILWALID;
}

optix::SemanticType optix::abbreviationStringToSemanticType( const std::string& str )
{
    if( str == "RG" )
        return ST_RAYGEN;
    if( str == "EX" )
        return ST_EXCEPTION;
    if( str == "MS" )
        return ST_MISS;
    if( str == "LW" )
        return ST_NODE_VISIT;
    if( str == "IN" )
        return ST_INTERSECTION;
    if( str == "BB" )
        return ST_BOUNDING_BOX;
    if( str == "CH" )
        return ST_CLOSEST_HIT;
    if( str == "AH" )
        return ST_ANY_HIT;
    if( str == "AT" )
        return ST_ATTRIBUTE;
    if( str == "CP" )
        return ST_BOUND_CALLABLE_PROGRAM;
    if( str == "BCP" )
        return ST_BINDLESS_CALLABLE_PROGRAM;
    if( str == "AAI" )
        return ST_INTERNAL_AABB_ITERATOR;
    if( str == "AAE" )
        return ST_INTERNAL_AABB_EXCEPTION;
    return ST_ILWALID;
}

std::vector<SemanticType> optix::semanticTypesStringToVector( const std::string& str )
{
    std::vector<SemanticType> result;
    std::stringstream         inputStream;
    inputStream << str;
    std::string typeStr;
    while( inputStream >> typeStr )
    {
        SemanticType semanticType = abbreviationStringToSemanticType( typeStr );
        RT_ASSERT_MSG( semanticType != ST_ILWALID, "Invalid semantic type specified." );
        result.push_back( semanticType );
    }
    return result;
}
