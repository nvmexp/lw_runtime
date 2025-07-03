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



#include <exp/context/Task.h>
#include <exp/context/DeviceContext.h>
#include <exp/context/OpaqueApiObject.h>
#include <exp/context/ErrorHandling.h>

namespace optix_exp
{

Task::Task( DeviceContext* context )
    : OpaqueApiObject( OpaqueApiObject::ApiType::Task )
    , m_context( context )
    , m_exelwted( NOT_STARTED )
{
}

void Task::logErrorDetails( OptixResult result, ErrorDetails&& errDetails )
{
    if( !result )
        return;
    optix_exp::DeviceContextLogger& clog = getDeviceContext()->getLogger();
    clog.sendError( errDetails );
}

OptixResult Task::destroy( ErrorDetails& errDetails )
{
    ExelwtionState expected  = FINISHED;
    ExelwtionState desired   = DESTROYED;
    bool           exchanged = m_exelwted.compare_exchange_strong( expected, desired );
    if( !exchanged )
    {
        // OK to destroy unstarted tasks
        if( expected == NOT_STARTED )
            return OPTIX_SUCCESS;
        else if( expected == STARTED )
            return errDetails.logDetails( OPTIX_ERROR_ILLEGAL_DURING_TASK_EXELWTE, "OptixTask destroyed while started" );
    }
    return OPTIX_SUCCESS;
}

OptixResult Task::markStarted( ErrorDetails& errDetails )
{
    ExelwtionState expected  = NOT_STARTED;
    ExelwtionState desired   = STARTED;
    bool           exchanged = m_exelwted.compare_exchange_strong( expected, desired );
    if( !exchanged )
    {
        if( expected == STARTED )
            return errDetails.logDetails( OPTIX_ERROR_ILLEGAL_DURING_TASK_EXELWTE, "OptixTask already started" );
        else if( expected == FINISHED )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION, "OptixTask already exelwted" );
        else if( expected == DESTROYED )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION, "Cannot run destroyed OptixTask" );
    }
    return OPTIX_SUCCESS;
}

OptixResult Task::markFinished( ErrorDetails& errDetails )
{
    ExelwtionState expected  = STARTED;
    ExelwtionState desired   = FINISHED;
    bool           exchanged = m_exelwted.compare_exchange_strong( expected, desired );
    if( !exchanged )
    {
        if( expected == NOT_STARTED )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION, "OptixTask not started, but marked as finished" );
        else if( expected == FINISHED )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION, "OptixTask marked as finished more than once" );
        else if( expected == DESTROYED )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION, "Destroyed OptixTask marked as finished" );
    }
    return OPTIX_SUCCESS;
}
} // end namespace optix

extern "C" OptixResult optixTaskExelwte( OptixTask taskAPI, OptixTask* additionalTasksAPI, unsigned int maxNumAdditionalTasks, unsigned int* numAdditionalTasksCreated )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Task, task, "OptixTask" );
    SCOPED_LWTX_RANGE( task->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::TASK_EXELWTE );
    optix_exp::DeviceContextLogger& clog = task->getDeviceContext()->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( additionalTasksAPI );
    OPTIX_CHECK_ZERO_ARGUMENT( maxNumAdditionalTasks );
    OPTIX_CHECK_NULL_ARGUMENT( numAdditionalTasksCreated );

    try
    {
        optix_exp::ErrorDetails errDetails;
        for( unsigned int i = 0; i < maxNumAdditionalTasks; ++i )
            additionalTasksAPI[i] = nullptr;
        *numAdditionalTasksCreated = 0;
        if( OptixResult result = task->markStarted( errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
        const OptixResult result = task->execute( additionalTasksAPI, maxNumAdditionalTasks, numAdditionalTasksCreated, errDetails );
        task->logErrorDetails( result, std::move( errDetails ) );
        task->markFinished( errDetails );
        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}
