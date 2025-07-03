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

#pragma once

#include <optix_types.h>

#include <exp/context/OpaqueApiObject.h>

#include <atomic>

namespace optix_exp {

class DeviceContext;
class ErrorDetails;

class Task : public OpaqueApiObject
{
  public:
    Task( DeviceContext* context );

    // Pointer returned is owned by the caller
    virtual OptixResult execute( OptixTask* additionalTasks, unsigned int maxNumAdditionalTasks, unsigned int* numAdditionalTasks, ErrorDetails& errDetails ) = 0;
    virtual void logErrorDetails( OptixResult result, ErrorDetails&& errDetails );
    virtual OptixResult destroy( ErrorDetails& errDetails );

    DeviceContext* getDeviceContext() const { return m_context; }

    OptixResult markStarted( ErrorDetails& errDetails );
    OptixResult markFinished( ErrorDetails& errDetails );
  private:
    enum ExelwtionState {
        NOT_STARTED,
        STARTED,
        FINISHED,
        DESTROYED
    };

    DeviceContext*              m_context;
    std::atomic<ExelwtionState> m_exelwted;
};


inline OptixResult implCast( OptixTask taskAPI, Task*& task )
{
    task = reinterpret_cast<Task*>( taskAPI );
    // It's OK for taskAPI to be nullptr
    if( task && task->m_apiType != OpaqueApiObject::ApiType::Task )
    {
        return OPTIX_ERROR_ILWALID_VALUE;
    }
    return OPTIX_SUCCESS;
}

inline OptixTask apiCast( Task* task )
{
    return reinterpret_cast<OptixTask>( task );
}


}  // end namespace optix_exp
