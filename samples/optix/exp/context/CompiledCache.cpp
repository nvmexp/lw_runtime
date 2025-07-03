/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <Context/RTCore.h>
#include <exp/context/CompiledCache.h>
#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>
#include <exp/context/OptixResultOneShot.h>

namespace optix_exp {

RtcoreModule::RtcoreModule( DeviceContext* context, RtcCompiledModule rtcModule )
{
    if( context->getRtcore().compiledModuleGetHash( rtcModule, &m_rtcHash ) )
    {
        lerr << "Error getting rtcore hash\n";
        return;
    }
    m_rtcModule = context->getRtcCompiledModuleCache().emplace( m_rtcHash, rtcModule );
    if( m_rtcModule == nullptr )
    {
        // Not in the cache
        m_rtcModule = rtcModule;
        return;
    }
    // If the version was cached delete the incoming one
    if( context->getRtcore().compiledModuleDestroy( rtcModule ) )
    {
        lerr << "Error while destroying module\n";
    }
}

OptixResult RtcoreModule::destroy( DeviceContext* context, ErrorDetails& errDetails )
{
    OptixResultOneShot result;
    if( m_rtcModule == nullptr )
    {
        return result;
    }

    if( !context->getRtcCompiledModuleCache().erase( m_rtcHash ) )
        return result;

    LwdaContextPushPop lwCtx( context );
    result += lwCtx.init( errDetails );

    if( const RtcResult rtcResult = context->getRtcore().compiledModuleDestroy( m_rtcModule ) )
    {
        result += errDetails.logDetails( rtcResult, "Error while destroying module" );
    }
    result += lwCtx.destroy( errDetails );
    return result;
}

RtcCompiledModule CompiledCache::emplace( Rtlw64 hash, RtcCompiledModule rtcModule )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    auto                        pair = m_hashToRtcoreModule.emplace( hash, ModuleRefCount{rtcModule, 0} );
    ModuleRefCount&             mod  = pair.first->second;
    mod.refCount++;
    // Return nullptr if the object wasn't in the cache
    if( pair.second )
        return nullptr;
    return mod.rtcModule;
}
// Returns true when the object has been removed due to zero references.
bool CompiledCache::erase( Rtlw64 hash )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    auto                        iter = m_hashToRtcoreModule.find( hash );
    assert( iter != m_hashToRtcoreModule.end() );
    if( --( iter->second.refCount ) <= 0 )
    {
        m_hashToRtcoreModule.erase( iter );
        return true;
    }
    return false;
}
}  // end namespace optix_exp
