/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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

#include <exp/context/GpuWarmup.h>

#include <exp/context/ErrorHandling.h>
#include <exp/context/GpuWarmup_fatbin.h>

#include <src/LWCA/Context.h>
#include <src/Device/LWDADevice.h>

#include <lwda_runtime_api.h>

#if defined( DEBUG ) || defined( DEVELOP )
#define CHOOSE_STRING( debug_string, release_string ) debug_string
#else
#define CHOOSE_STRING( debug_string, release_string ) release_string
#endif

namespace optix_exp {

OptixResult GpuWarmup::init( ErrorDetails& errDetails )
{
    if( m_initialized )
        return OPTIX_SUCCESS;

    // We need to use loadDataExHidden in release builds to keep this kernel hidden from tools,
    // since 1) the kernel contains TTU instructions, and 2) the launch is a WAR for a Turing HW bug
    // that we don't want to call attention to. So it's better that the launch appears as 'LWPU
    // Internal' in Nsight, lwca-gdb, etc. The error messages are more generic in release builds for
    // the same reason.

    LWresult res;
#if defined( DEBUG ) || defined( DEVELOP )
    m_module = optix::lwca::Module::loadDataEx( optix::data::getGpuWarmupData(), 0, nullptr, nullptr, &res );
#else
    m_module = optix::lwca::Module::loadDataExHidden( optix::data::getGpuWarmupData(), 0, nullptr, nullptr, &res );
#endif
    if( res != LWDA_SUCCESS )
    {
        return errDetails.logDetails( res, CHOOSE_STRING( "Unable to load warm-up kernel module",
                                                          "Error initializing device" ) );
    }
    m_function = m_module.getFunction( std::string( "gpuWarmup" ), &res );
    if( res != LWDA_SUCCESS )
    {
        return errDetails.logDetails( res, CHOOSE_STRING( "Unable to load warm-up kernel function",
                                                          "Error initializing device" ) );
    }
    m_initialized = true;
    return OPTIX_SUCCESS;
}

OptixResult GpuWarmup::destroy( ErrorDetails& errDetails )
{
    if( m_initialized )
    {
        m_function    = optix::lwca::Function();
        m_initialized = false;

        LWresult res;
        m_module.unload( &res );
        if( res != LWDA_SUCCESS )
        {
            return errDetails.logDetails( res, CHOOSE_STRING( "Error unloading warm-up module",
                                                              "Error de-initializing device" ) );
        }
    }
    return OPTIX_SUCCESS;
}

OptixResult GpuWarmup::launch( unsigned int numBlocks, ErrorDetails& errDetails )
{
    LWresult res;
    optix::lwca::Context::synchronize( &res );
    if( res != LWDA_SUCCESS )
    {
        return errDetails.logDetails( res, CHOOSE_STRING( "Error synchronizing LWCA context before warm-up launch",
                                                          "Error preparing launch" ) );
    }

    // Launch with 48KB of shared memory to help ensure exelwtion on every SM.
    const unsigned int shmemSize = 49152;

    m_function.launchKernel( numBlocks, 1, 1, 1, 1, 1, shmemSize, optix::lwca::Stream( 0 ), nullptr, nullptr, &res );
    if( res != LWDA_SUCCESS )
    {
        return errDetails.logDetails( res, CHOOSE_STRING( "Error during warm-up launch", "Error preparing launch" ) );
    }

    optix::lwca::Context::synchronize( &res );
    if( res != LWDA_SUCCESS )
    {
        return errDetails.logDetails( res, CHOOSE_STRING( "Error synchronizing LWCA context after warm-up launch",
                                                          "Error preparing launch" ) );
    }

    return OPTIX_SUCCESS;
}

}  // namespace optix_exp
