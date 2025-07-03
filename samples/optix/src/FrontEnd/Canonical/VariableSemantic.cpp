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

#include <FrontEnd/Canonical/VariableSemantic.h>

#include <prodlib/exceptions/CompileError.h>

using namespace prodlib;
using namespace optix;

VariableSemantic optix::getVariableSemanticFromString( const std::string& semantic, const std::string& varName )
{
    if( semantic == "rtPayload" )
    {
        return VS_PAYLOAD;
    }
    else if( semantic == "rtLwrrentRay" )
    {
        return VS_LWRRENTRAY;
    }
    else if( semantic == "rtLwrrentTime" )
    {
        return VS_LWRRENTTIME;
    }
    else if( semantic == "rtIntersectionDistance" )
    {
        return VS_INTERSECTIONDISTANCE;
    }
    else if( semantic == "rtLaunchIndex" )
    {
        return VS_LAUNCHINDEX;
    }
    else if( semantic == "rtLaunchDim" )
    {
        return VS_LAUNCHDIM;
    }
    else if( semantic == "rtSubframeIndex" )
    {
        return VS_SUBFRAMEINDEX;
    }
    else if( semantic == "rt_call" )
    {
        throw CompileError( RT_EXCEPTION_INFO, "Old style of rtCallableProgram detected for function (" + varName
                                                   + ").  Please recompile with sm_20+." );
    }
    else
    {
        throw CompileError( RT_EXCEPTION_INFO, "Invalid semantic type for variable (" + varName + "): " + semantic );
    }
}
