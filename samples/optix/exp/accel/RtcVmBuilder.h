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

#pragma once

#include <optix_types.h>

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

#include <exp/context/ErrorHandling.h>
#include <exp/context/DeviceContext.h>

#include <rtcore/interface/rtcore.h>
#include <rtcore/interface/types.h>

#include <lwca.h>

#include <vector>

namespace optix_exp {

class DeviceContext;

class RtcVmBuilder
{
  public:
    RtcVmBuilder( DeviceContext* context, bool computeMemory, ErrorDetails& errDetails );
    OptixResult init( const OptixVisibilityMapArrayBuildInput* buildInput );

    OptixResult computeMemoryUsage( OptixMicromeshBufferSizes* bufferSizes );

    OptixResult build( LWstream                          stream,
                       const OptixMicromeshBuffers*      buffers,
                       const OptixMicromeshEmitDesc*     emittedProperties,
                       unsigned int                      numEmittedProperties );

  private:
    OptixResult logIlwalidValue( const std::string& description )
    {
        return m_errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, description );
    }

    DeviceContext*                           m_context;
    bool                                     m_computeMemory;
    ErrorDetails&                            m_errDetails;
    const OptixVisibilityMapArrayBuildInput* m_buildInput;
    RtcVisibilityMapArrayBuildInput          m_rtcBuildInput;
};

}  // namespace optix_exp

#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
