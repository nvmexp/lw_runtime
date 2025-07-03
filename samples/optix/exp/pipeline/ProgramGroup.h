/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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
#include <rtcore/interface/types.h>

#include <exp/context/OpaqueApiObject.h>

#include <list>
#include <string>

namespace optix_exp {

class DeviceContext;
class ErrorDetails;

class ProgramGroup : public OpaqueApiObject
{
  public:
    ProgramGroup( DeviceContext* context, const OptixProgramGroupDesc& impl, const CompilePayloadType* payloadType );

    OptixResult destroy( ErrorDetails& errDetails ) { return destroy( true, errDetails ); }

    OptixResult destroyWithoutUnregistration( ErrorDetails& errDetails ) { return destroy( false, errDetails ); }

    DeviceContext* getDeviceContext() const { return m_context; }

    const OptixProgramGroupDesc& getImpl() const { return m_impl; }

    // To be used by DeviceContext only.
    struct DeviceContextIndex_fn
    {
        int& operator()( const ProgramGroup* programGroup ) { return programGroup->m_deviceContextIndex; }
    };

  private:
    OptixResult destroy( bool doUnregisterProgramGroup, ErrorDetails& errDetails );

    // Mangles and duplicates the passed string and returns a copy that is valid for the lifetime of this instance.
    // Returns nullptr for nullptr arguments.
    const char* mangleAndDuplicate( const char* s, OptixModule module, const CompilePayloadType* payloadType );

    DeviceContext* m_context;

    // Ilwariants:
    // - For callables, either DC or CC needs to be valid.
    // - If a module is not null, then the entry function name is not null, and vice versa.
    // - All modules (if present) belong to m_context.
    // - All entry function names (if present) have the correct prefix.
    //
    // TODO This reference will probably be replaced by lwvm-rt code.
    OptixProgramGroupDesc m_impl;

    // Storage used by mangleAndDuplicate(). Make sure to use a container that does not ilwalidate pointers to its content.
    std::list<std::string> m_strings;

    mutable int m_deviceContextIndex = -1;
};

inline OptixResult implCast( OptixProgramGroup programGroupAPI, ProgramGroup*& programGroup )
{
    programGroup = reinterpret_cast<ProgramGroup*>( programGroupAPI );
    // It's OK for programGroupAPI to be nullptr
    if( programGroup && programGroup->m_apiType != OpaqueApiObject::ApiType::ProgramGroup )
    {
        return OPTIX_ERROR_ILWALID_VALUE;
    }
    return OPTIX_SUCCESS;
}

inline OptixProgramGroup apiCast( ProgramGroup* programGroup )
{
    return reinterpret_cast<OptixProgramGroup>( programGroup );
}

}  // end namespace optix_exp
