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
#include <vector>

namespace optix_exp {

class DeviceContext;
class ErrorDetails;
class Module;
class RtcoreModule;
struct CompileBoundValueEntry;

struct EntryInfo
{
    size_t offset;
    size_t size;
    std::string moduleId;
    std::string annotation;
};

struct LaunchParamSpecialization
{
    std::vector<char> data;
    std::vector<EntryInfo> specializedRanges;

    void clear();

    void addSpecialization( const CompileBoundValueEntry& entry, const std::string& moduleId );
    void addSpecialization( const CompileBoundValueEntry& entry, size_t offsetInEntryData, const std::string& moduleId );
    // <------>      mod1
    //    <--- --->  mod2
    //
    // <-----><--->
};

class Pipeline : public OpaqueApiObject
{
  public:
    struct NamedConstant;

    Pipeline( DeviceContext*                        context,
              RtcPipeline                           rtcPipeline,
              RtcCompiledModule                     launchParamModule,
              Module*                               defaultExceptionModule,
              OptixPipelineCompileOptions           pipelineCompileOptions,
              OptixPipelineLinkOptions              pipelineLinkOptions,
              size_t                                launchBufferSizeInBytes,
              size_t                                launchBufferAlignment,
              size_t                                launchParamOffset,
              size_t                                launchParamSizeInBytes,
              size_t                                toolsOutputSizeInBytes,
              const RtcPipelineInfoShaderInfo&      shaderInfo,
              const RtcPipelineInfoProfileMetadata& profileMetadata,
              bool                                  ignorePipelineLaunchParamsAtLaunch,
              LaunchParamSpecialization&&           computedSpecializations,
              LWevent                               conlwrrentLaunchEvent,
              std::vector<NamedConstant*>&&         namedConstants,
              bool                                  hasDebugInformation );

    OptixResult destroy( ErrorDetails& errDetails ) { return destroy( true, errDetails ); }

    OptixResult destroyWithoutUnregistration( ErrorDetails& errDetails ) { return destroy( false, errDetails ); }

    DeviceContext* getDeviceContext() const { return m_context; }

    RtcPipeline getRtcPipeline() const { return m_rtcPipeline; }

    const OptixPipelineCompileOptions& getPipelineCompileOptions() const { return m_pipelineCompileOptions; }

    const OptixPipelineLinkOptions& getPipelineLinkOptions() const { return m_pipelineLinkOptions; }

    size_t getLaunchBufferSizeInBytes() const { return m_launchBufferSizeInBytes; }

    size_t getLaunchBufferAlignment() const { return m_launchBufferAlignment; }

    size_t getLaunchParamOffset() const { return m_launchParamOffset; }

    size_t getLaunchParamSizeInBytes() const { return m_launchParamSizeInBytes; }

    size_t getToolsOutputSizeInBytes() const { return m_toolsOutputSizeInBytes; }

    const RtcPipelineInfoShaderInfo& getShaderInfo() const { return m_shaderInfo; }

    const RtcPipelineInfoProfileMetadata& getProfileMetadata() const { return m_profileMetadata; }

    bool getIgnorePipelineLaunchParamsAtLaunch() const { return m_ignorePipelineLaunchParamsAtLaunch; }

    LWevent getConlwrrentLaunchEvent() const { return m_conlwrrentLaunchEvent; }

    Module* getDefaultExceptionModule() const { return m_defaultExceptionModule; }

    std::mutex& getRtcoreMutex() { return m_rtcoreMutex; }

    // To be used by DeviceContext only.
    struct DeviceContextIndex_fn
    {
        int& operator()( const Pipeline* pipeline ) { return pipeline->m_deviceContextIndex; }
    };

    struct NamedConstant
    {
        NamedConstant( const std::string& name, size_t size )
            : m_name( name )
            , m_size( size )
        {
        }
        std::string       m_name;
        size_t            m_size;
        void*             m_hostPtr{};
        RtcCompiledModule m_rtcModule{};
        // storing values returned from rtcPipelineGetNamedConstantInfo
        Rtlw64 m_memoryOffset{};
        Rtlw64 m_memorySizeInBytes{};
        // for proper resource release/destroy of additional resources
        std::function<OptixResult( ErrorDetails&, NamedConstant& )> m_destroyFunctor;
    };
    const std::vector<NamedConstant*>& getAdditionalNamedConstants() const { return m_additionalNamedConstants; }

    const LaunchParamSpecialization& getComputedSpecializations() { return m_computedSpecializations; }

    // Retrieve debug information state.
    bool hasDebugInformation() const { return m_hasDebugInformation; }

  private:
    OptixResult destroy( bool doUnregisterPipeline, ErrorDetails& errDetails );

    // Duplicates the passed string and returns a copy that is valid for the lifetime of this instance.
    // Returns nullptr for nullptr arguments.
    const char* duplicate( const char* s );

    DeviceContext* m_context = nullptr;

    RtcPipeline       m_rtcPipeline            = nullptr;
    RtcCompiledModule m_launchParamModule      = nullptr;
    Module*           m_defaultExceptionModule = nullptr;

    OptixPipelineCompileOptions m_pipelineCompileOptions;
    OptixPipelineLinkOptions    m_pipelineLinkOptions;

    size_t m_launchBufferSizeInBytes = 0;
    size_t m_launchBufferAlignment   = 0;
    size_t m_launchParamOffset       = 0;
    size_t m_launchParamSizeInBytes  = 0;
    size_t m_toolsOutputSizeInBytes  = 0;

    RtcPipelineInfoShaderInfo      m_shaderInfo;
    RtcPipelineInfoProfileMetadata m_profileMetadata;

    bool m_ignorePipelineLaunchParamsAtLaunch = false;

    // Temporary work around to prevent a pipeline from being used conlwrrently.
    LWevent m_conlwrrentLaunchEvent = 0;

    std::vector<NamedConstant*> m_additionalNamedConstants;

    std::mutex m_rtcoreMutex;

    // Storage used by duplicate(). Make sure to use a container that does not ilwalidate pointers to its content.
    std::list<std::string> m_strings;

    // Used for launch parameter consistency checks
    LaunchParamSpecialization m_computedSpecializations;

    mutable int m_deviceContextIndex = -1;

    // Was any module built with debug information?
    bool m_hasDebugInformation = false;
};

inline OptixResult implCast( OptixPipeline pipelineAPI, Pipeline*& pipeline )
{
    pipeline = reinterpret_cast<Pipeline*>( pipelineAPI );
    // It's OK for pipelineAPI to be nullptr
    if( pipeline && pipeline->m_apiType != OpaqueApiObject::ApiType::Pipeline )
    {
        return OPTIX_ERROR_ILWALID_VALUE;
    }
    return OPTIX_SUCCESS;
}

inline OptixPipeline apiCast( Pipeline* pipeline )
{
    return reinterpret_cast<OptixPipeline>( pipeline );
}

}  // end namespace optix_exp
