/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
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

// lwoptix_EXPORTS overrides any setting for RTAPI and forces export of symbols in this file
#if defined( lwoptix_EXPORTS )

// Undefine any existing definition for RTAPI
#if defined( RTAPI )
#undef RTAPI
#endif

// On Windows, we're going to export functions with the generated .def file, so don't
// double export them here and define RTAPI to be empty.
#if defined( _WIN32 )
#define RTAPI

// On Linux flavors, we need to set the visibility attribute to 'default'.
#elif defined( __linux__ ) || defined( __CYGWIN__ )
#define RTAPI __attribute__( ( visibility( "default" ) ) )
#elif defined( __APPLE__ ) && defined( __MACH__ )
#define RTAPI __attribute__( ( visibility( "default" ) ) )

#else
#error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#endif

#endif

#if !defined( _WIN32 )
#include <common/inc/lwUnixVersion.h>
#endif

#include <o6/optix.h>

#if defined( _WIN32 )
#include <optix_d3d10_interop.h>
#include <optix_d3d11_interop.h>
#include <optix_d3d9_interop.h>
#endif

#include <optix_lwda_interop.h>
#include <optix_gl_interop.h>

#include <private/lwoptix.h>

#include <Context/RTCore.h>
#include <exp/context/ForceDeprecatedCompiler.h>

#include <prodlib/system/Logger.h>

#include <lwselwreloadlibrary/SelwreExelwtableModule.h>

#include <corelib/system/System.h>

#include <string>

extern "C" RTresult rtLog( int level, const char* msg )
{
    llog( level ) << msg << std::endl;
    return RT_SUCCESS;
}

static RTresult rtLog( int level, const std::string& msg )
{
    return rtLog( level, msg.c_str() );
}

static RTresult rtDeprecatedApiFunction()
{
    rtLog( prodlib::log::LEV_FATAL, "OptiX: Deprecated API function called" );
    return RT_ERROR_NOT_SUPPORTED;
}

static RTresult rtGeometryTrianglesSetIndexedTriangles_backCompatible( RTgeometrytriangles  geometrytriangles_api,
                                                                       unsigned int         num_triangles,
                                                                       RTbuffer             index_buffer_api,
                                                                       RTsize               index_buffer_byte_offset,
                                                                       RTsize               tri_indices_byte_stride,
                                                                       RTformat             tri_indices_format,
                                                                       unsigned int         num_vertices,
                                                                       RTbuffer             vertex_buffer_api,
                                                                       RTsize               vertex_buffer_byte_offset,
                                                                       RTsize               vertex_byte_stride,
                                                                       RTformat             position_format,
                                                                       RTgeometrybuildflags build_flags )
{
    RTresult result;
    result = rtGeometryTrianglesSetPrimitiveCount( geometrytriangles_api, num_triangles );
    if( result != RT_SUCCESS )
        return result;
    result = rtGeometryTrianglesSetTriangleIndices( geometrytriangles_api, index_buffer_api, index_buffer_byte_offset,
                                                    tri_indices_byte_stride, tri_indices_format );
    if( result != RT_SUCCESS )
        return result;
    result = rtGeometryTrianglesSetVertices( geometrytriangles_api, num_vertices, vertex_buffer_api,
                                             vertex_buffer_byte_offset, vertex_byte_stride, position_format );
    if( result != RT_SUCCESS )
        return result;
    result = rtGeometryTrianglesSetBuildFlags( geometrytriangles_api, build_flags );
    if( result != RT_SUCCESS )
        return result;
    return RT_SUCCESS;
}

static RTresult rtGeometryTrianglesSetTriangles_backCompatible( RTgeometrytriangles  geometrytriangles_api,
                                                                unsigned int         num_triangles,
                                                                RTbuffer             vertex_buffer_api,
                                                                RTsize               vertex_buffer_byte_offset,
                                                                RTsize               vertex_byte_stride,
                                                                RTformat             position_format,
                                                                RTgeometrybuildflags build_flags )
{
    RTresult result;
    result = rtGeometryTrianglesSetPrimitiveCount( geometrytriangles_api, num_triangles );
    if( result != RT_SUCCESS )
        return result;
    result = rtGeometryTrianglesSetVertices( geometrytriangles_api, num_triangles * 3, vertex_buffer_api,
                                             vertex_buffer_byte_offset, vertex_byte_stride, position_format );
    if( result != RT_SUCCESS )
        return result;
    result = rtGeometryTrianglesSetBuildFlags( geometrytriangles_api, build_flags );
    if( result != RT_SUCCESS )
        return result;
    return RT_SUCCESS;
}

static RTresult rtGeometryTrianglesSetMotiolwerticesMultiBuffer_backCompatible( RTgeometrytriangles geometrytriangles_api,
                                                                                unsigned int        num_vertices,
                                                                                RTbuffer*           vertex_buffers,
                                                                                RTsize   vertex_buffer_byte_offset,
                                                                                RTsize   vertex_byte_stride,
                                                                                RTformat position_format )
{
    unsigned int motionSteps = 0;
    RTresult     result      = rtGeometryTrianglesGetMotionSteps( geometrytriangles_api, &motionSteps );
    if( result != RT_SUCCESS )
        return result;
    // Warning, this is flawed as we do not know if #motionSteps == #vertexBuffers.
    // That is why this function got replaced by a version that takes the number of vertex buffers as a parameter.
    result = rtGeometryTrianglesSetMotiolwerticesMultiBuffer( geometrytriangles_api, num_vertices, vertex_buffers, motionSteps,
                                                              vertex_buffer_byte_offset, vertex_byte_stride, position_format );
    return result;
}


extern "C" RTresult RTAPI rtPostProcessingStageCreateBuiltinInternal( RTcontext              context,
                                                                      const char*            builtin_name,
                                                                      void*                  denoiser,
                                                                      void*                  ssim_predictor,
                                                                      RTpostprocessingstage* stage );

extern "C" RTresult RTAPI rtGetBuildVersion( const char** result );

extern "C" RTresult RTAPI rtOverridesOtherVersion( const char* otherVersion, int* result );

extern "C" RTresult RTAPI rtSetLibraryVariant( bool isLibraryFromSdk );

extern "C" RTresult RTAPI rtSupportsLwrrentDriver();

extern "C" RTresult RTAPI rtContextCreateABI16( RTcontext* context );
extern "C" RTresult RTAPI rtContextCreateABI17( RTcontext* context );
extern "C" RTresult RTAPI rtContextCreateABI18( RTcontext* context );

extern "C" RTresult RTAPI
rtGetSymbolTable( unsigned int abiVersion, unsigned int numOptions, RToptixoptions* options, void** optiolwalues, OptiXAPI_t* api )
{
    rtLog( prodlib::log::LEV_PRINT, "Requested ABI version " + std::to_string( abiVersion ) + '\n' );

    if( api == nullptr )
        return RT_ERROR_ILWALID_VALUE;

    if( abiVersion > LWOPTIX_ABI_VERSION )
        return RT_ERROR_NOT_SUPPORTED;

    bool skipDriverVersionCheck = false;

    for( unsigned int i = 0; i < numOptions; ++i )
    {
        switch( options[i] )
        {
            case RT_OPTIX_OPTION_FROM_SDK:
                // This option must be set before any calls to rtOverridesOtherVersion() are made.
                rtSetLibraryVariant( /*isLibraryFromSdk*/ *static_cast<bool*>( optiolwalues[i] ) );
                break;
            case RT_OPTIX_OPTION_SKIP_DRIVER_VERSION_CHECK:
                skipDriverVersionCheck = *static_cast<bool*>( optiolwalues[i] );
                break;
            default:
                return RT_ERROR_NOT_SUPPORTED;
        }
    }

    if( !skipDriverVersionCheck && ( rtSupportsLwrrentDriver() != RT_SUCCESS ) )
        return RT_ERROR_NOT_SUPPORTED;

    api->version = LWOPTIX_ABI_VERSION;

// Use these macros to make the exceptions stand out more clearly.
#define NORMAL_API_FUNCTION( name_ ) api->name_     = name_
#define DEPRECATED_API_FUNCTION( name_ ) api->name_ = rtDeprecatedApiFunction

    NORMAL_API_FUNCTION( rtAccelerationCreate );
    NORMAL_API_FUNCTION( rtAccelerationDestroy );
    NORMAL_API_FUNCTION( rtAccelerationGetBuilder );
    NORMAL_API_FUNCTION( rtAccelerationGetContext );
    NORMAL_API_FUNCTION( rtAccelerationGetData );
    NORMAL_API_FUNCTION( rtAccelerationGetDataSize );
    NORMAL_API_FUNCTION( rtAccelerationGetProperty );
    NORMAL_API_FUNCTION( rtAccelerationGetTraverser );
    NORMAL_API_FUNCTION( rtAccelerationIsDirty );
    NORMAL_API_FUNCTION( rtAccelerationMarkDirty );
    NORMAL_API_FUNCTION( rtAccelerationSetBuilder );
    NORMAL_API_FUNCTION( rtAccelerationSetData );
    NORMAL_API_FUNCTION( rtAccelerationSetProperty );
    NORMAL_API_FUNCTION( rtAccelerationSetTraverser );
    NORMAL_API_FUNCTION( rtAcceleratiolwalidate );
    NORMAL_API_FUNCTION( rtBufferCreate );
    NORMAL_API_FUNCTION( rtBufferCreateForLWDA );
    NORMAL_API_FUNCTION( rtBufferCreateFromGLBO );
    NORMAL_API_FUNCTION( rtBufferDestroy );
    NORMAL_API_FUNCTION( rtBufferGLRegister );
    NORMAL_API_FUNCTION( rtBufferGLUnregister );
    NORMAL_API_FUNCTION( rtBufferGetContext );
    NORMAL_API_FUNCTION( rtBufferGetDevicePointer );
    NORMAL_API_FUNCTION( rtBufferGetDimensionality );
    NORMAL_API_FUNCTION( rtBufferGetElementSize );
    NORMAL_API_FUNCTION( rtBufferGetFormat );
    NORMAL_API_FUNCTION( rtBufferGetGLBOId );
    NORMAL_API_FUNCTION( rtBufferGetId );
    NORMAL_API_FUNCTION( rtBufferGetMipLevelCount );
    NORMAL_API_FUNCTION( rtBufferGetMipLevelSize1D );
    NORMAL_API_FUNCTION( rtBufferGetMipLevelSize2D );
    NORMAL_API_FUNCTION( rtBufferGetMipLevelSize3D );
    NORMAL_API_FUNCTION( rtBufferGetSize1D );
    NORMAL_API_FUNCTION( rtBufferGetSize2D );
    NORMAL_API_FUNCTION( rtBufferGetSize3D );
    NORMAL_API_FUNCTION( rtBufferGetSizev );
    NORMAL_API_FUNCTION( rtBufferMap );
    NORMAL_API_FUNCTION( rtBufferMapEx );
    NORMAL_API_FUNCTION( rtBufferMarkDirty );
    NORMAL_API_FUNCTION( rtBufferSetDevicePointer );
    NORMAL_API_FUNCTION( rtBufferSetElementSize );
    NORMAL_API_FUNCTION( rtBufferSetFormat );
    NORMAL_API_FUNCTION( rtBufferSetMipLevelCount );
    NORMAL_API_FUNCTION( rtBufferSetSize1D );
    NORMAL_API_FUNCTION( rtBufferSetSize2D );
    NORMAL_API_FUNCTION( rtBufferSetSize3D );
    NORMAL_API_FUNCTION( rtBufferSetSizev );
    NORMAL_API_FUNCTION( rtBufferUnmap );
    NORMAL_API_FUNCTION( rtBufferUnmapEx );
    NORMAL_API_FUNCTION( rtBufferValidate );
    NORMAL_API_FUNCTION( rtBufferGetProgressiveUpdateReady );
    NORMAL_API_FUNCTION( rtBufferBindProgressiveStream );
    NORMAL_API_FUNCTION( rtBufferSetAttribute );
    NORMAL_API_FUNCTION( rtBufferGetAttribute );
    NORMAL_API_FUNCTION( rtCommandListCreate );
    NORMAL_API_FUNCTION( rtCommandListDestroy );
    NORMAL_API_FUNCTION( rtCommandListAppendPostprocessingStage );
    NORMAL_API_FUNCTION( rtCommandListAppendLaunch2D );
    NORMAL_API_FUNCTION( rtCommandListFinalize );
    NORMAL_API_FUNCTION( rtCommandListExelwte );
    NORMAL_API_FUNCTION( rtCommandListGetContext );
    NORMAL_API_FUNCTION( rtContextCompile );
    NORMAL_API_FUNCTION( rtContextDeclareVariable );
    NORMAL_API_FUNCTION( rtContextDestroy );
    NORMAL_API_FUNCTION( rtContextSetAttribute );
    NORMAL_API_FUNCTION( rtContextGetAttribute );
    NORMAL_API_FUNCTION( rtContextGetBufferFromId );
    NORMAL_API_FUNCTION( rtContextGetDeviceCount );
    NORMAL_API_FUNCTION( rtContextGetDevices );
    NORMAL_API_FUNCTION( rtContextGetEntryPointCount );
    NORMAL_API_FUNCTION( rtContextGetErrorString );
    NORMAL_API_FUNCTION( rtContextGetExceptionEnabled );
    NORMAL_API_FUNCTION( rtContextGetExceptionProgram );
    NORMAL_API_FUNCTION( rtContextGetMissProgram );
    NORMAL_API_FUNCTION( rtContextGetPrintBufferSize );
    NORMAL_API_FUNCTION( rtContextGetPrintEnabled );
    NORMAL_API_FUNCTION( rtContextGetPrintLaunchIndex );
    NORMAL_API_FUNCTION( rtContextGetProgramFromId );
    NORMAL_API_FUNCTION( rtContextGetRayGenerationProgram );
    NORMAL_API_FUNCTION( rtContextGetRayTypeCount );
    NORMAL_API_FUNCTION( rtContextGetRunningState );
    NORMAL_API_FUNCTION( rtContextGetStackSize );
    NORMAL_API_FUNCTION( rtContextGetTextureSamplerFromId );
    NORMAL_API_FUNCTION( rtContextGetVariable );
    NORMAL_API_FUNCTION( rtContextGetVariableCount );
    NORMAL_API_FUNCTION( rtContextLaunch1D );
    NORMAL_API_FUNCTION( rtContextLaunch2D );
    NORMAL_API_FUNCTION( rtContextLaunch3D );
    NORMAL_API_FUNCTION( rtContextLaunchProgressive2D );
    NORMAL_API_FUNCTION( rtContextStopProgressive );
    NORMAL_API_FUNCTION( rtContextQueryVariable );
    NORMAL_API_FUNCTION( rtContextRemoveVariable );
    NORMAL_API_FUNCTION( rtContextSetDevices );
    NORMAL_API_FUNCTION( rtContextSetEntryPointCount );
    NORMAL_API_FUNCTION( rtContextSetExceptionEnabled );
    NORMAL_API_FUNCTION( rtContextSetExceptionProgram );
    NORMAL_API_FUNCTION( rtContextSetMissProgram );
    NORMAL_API_FUNCTION( rtContextSetPrintBufferSize );
    NORMAL_API_FUNCTION( rtContextSetPrintEnabled );
    NORMAL_API_FUNCTION( rtContextSetPrintLaunchIndex );
    NORMAL_API_FUNCTION( rtContextSetRayGenerationProgram );
    NORMAL_API_FUNCTION( rtContextSetRayTypeCount );
    DEPRECATED_API_FUNCTION( rtContextSetRemoteDevice );
    NORMAL_API_FUNCTION( rtContextSetStackSize );
    NORMAL_API_FUNCTION( rtContextSetTimeoutCallback );
    NORMAL_API_FUNCTION( rtContextSetUsageReportCallback );
    NORMAL_API_FUNCTION( rtContextValidate );
    NORMAL_API_FUNCTION( rtDeviceGetAttribute );
    NORMAL_API_FUNCTION( rtDeviceGetDeviceCount );
    NORMAL_API_FUNCTION( rtGeometryCreate );
    NORMAL_API_FUNCTION( rtGeometryDeclareVariable );
    NORMAL_API_FUNCTION( rtGeometryDestroy );
    NORMAL_API_FUNCTION( rtGeometryGetBoundingBoxProgram );
    NORMAL_API_FUNCTION( rtGeometryGetContext );
    NORMAL_API_FUNCTION( rtGeometryGetIntersectionProgram );
    NORMAL_API_FUNCTION( rtGeometryGetPrimitiveCount );
    NORMAL_API_FUNCTION( rtGeometryGetPrimitiveIndexOffset );
    NORMAL_API_FUNCTION( rtGeometryGetVariable );
    NORMAL_API_FUNCTION( rtGeometryGetVariableCount );
    NORMAL_API_FUNCTION( rtGeometryGroupCreate );
    NORMAL_API_FUNCTION( rtGeometryGroupDestroy );
    NORMAL_API_FUNCTION( rtGeometryGroupGetAcceleration );
    NORMAL_API_FUNCTION( rtGeometryGroupGetChild );
    NORMAL_API_FUNCTION( rtGeometryGroupGetChildCount );
    NORMAL_API_FUNCTION( rtGeometryGroupGetContext );
    NORMAL_API_FUNCTION( rtGeometryGroupSetAcceleration );
    NORMAL_API_FUNCTION( rtGeometryGroupSetChild );
    NORMAL_API_FUNCTION( rtGeometryGroupSetChildCount );
    NORMAL_API_FUNCTION( rtGeometryGroupValidate );
    NORMAL_API_FUNCTION( rtGeometryInstanceCreate );
    NORMAL_API_FUNCTION( rtGeometryInstanceDeclareVariable );
    NORMAL_API_FUNCTION( rtGeometryInstanceDestroy );
    NORMAL_API_FUNCTION( rtGeometryInstanceGetContext );
    NORMAL_API_FUNCTION( rtGeometryInstanceGetGeometry );
    NORMAL_API_FUNCTION( rtGeometryInstanceGetMaterial );
    NORMAL_API_FUNCTION( rtGeometryInstanceGetMaterialCount );
    NORMAL_API_FUNCTION( rtGeometryInstanceGetVariable );
    NORMAL_API_FUNCTION( rtGeometryInstanceGetVariableCount );
    NORMAL_API_FUNCTION( rtGeometryInstanceQueryVariable );
    NORMAL_API_FUNCTION( rtGeometryInstanceRemoveVariable );
    NORMAL_API_FUNCTION( rtGeometryInstanceSetGeometry );
    NORMAL_API_FUNCTION( rtGeometryInstanceSetMaterial );
    NORMAL_API_FUNCTION( rtGeometryInstanceSetMaterialCount );
    NORMAL_API_FUNCTION( rtGeometryInstanceValidate );
    NORMAL_API_FUNCTION( rtGeometryIsDirty );
    NORMAL_API_FUNCTION( rtGeometryMarkDirty );
    NORMAL_API_FUNCTION( rtGeometryQueryVariable );
    NORMAL_API_FUNCTION( rtGeometryRemoveVariable );
    NORMAL_API_FUNCTION( rtGeometrySetBoundingBoxProgram );
    NORMAL_API_FUNCTION( rtGeometrySetIntersectionProgram );
    NORMAL_API_FUNCTION( rtGeometrySetPrimitiveCount );
    NORMAL_API_FUNCTION( rtGeometrySetPrimitiveIndexOffset );
    NORMAL_API_FUNCTION( rtGeometrySetMotionRange );
    NORMAL_API_FUNCTION( rtGeometryGetMotionRange );
    NORMAL_API_FUNCTION( rtGeometrySetMotionBorderMode );
    NORMAL_API_FUNCTION( rtGeometryGetMotionBorderMode );
    NORMAL_API_FUNCTION( rtGeometrySetMotionSteps );
    NORMAL_API_FUNCTION( rtGeometryGetMotionSteps );
    NORMAL_API_FUNCTION( rtGeometryValidate );
    NORMAL_API_FUNCTION( rtGetVersion );
    NORMAL_API_FUNCTION( rtGlobalSetAttribute );
    NORMAL_API_FUNCTION( rtGlobalGetAttribute );
    NORMAL_API_FUNCTION( rtGroupCreate );
    NORMAL_API_FUNCTION( rtGroupDestroy );
    NORMAL_API_FUNCTION( rtGroupGetAcceleration );
    NORMAL_API_FUNCTION( rtGroupGetChild );
    NORMAL_API_FUNCTION( rtGroupGetChildCount );
    NORMAL_API_FUNCTION( rtGroupGetChildType );
    NORMAL_API_FUNCTION( rtGroupGetContext );
    NORMAL_API_FUNCTION( rtGroupSetAcceleration );
    NORMAL_API_FUNCTION( rtGroupSetChild );
    NORMAL_API_FUNCTION( rtGroupSetChildCount );
    NORMAL_API_FUNCTION( rtGroupValidate );
    NORMAL_API_FUNCTION( rtMaterialCreate );
    NORMAL_API_FUNCTION( rtMaterialDeclareVariable );
    NORMAL_API_FUNCTION( rtMaterialDestroy );
    NORMAL_API_FUNCTION( rtMaterialGetAnyHitProgram );
    NORMAL_API_FUNCTION( rtMaterialGetClosestHitProgram );
    NORMAL_API_FUNCTION( rtMaterialGetContext );
    NORMAL_API_FUNCTION( rtMaterialGetVariable );
    NORMAL_API_FUNCTION( rtMaterialGetVariableCount );
    NORMAL_API_FUNCTION( rtMaterialQueryVariable );
    NORMAL_API_FUNCTION( rtMaterialRemoveVariable );
    NORMAL_API_FUNCTION( rtMaterialSetAnyHitProgram );
    NORMAL_API_FUNCTION( rtMaterialSetClosestHitProgram );
    NORMAL_API_FUNCTION( rtMaterialValidate );
    api->rtPostProcessingStageCreateBuiltin = rtPostProcessingStageCreateBuiltinInternal;
    NORMAL_API_FUNCTION( rtPostProcessingStageDeclareVariable );
    NORMAL_API_FUNCTION( rtPostProcessingStageDestroy );
    NORMAL_API_FUNCTION( rtPostProcessingStageGetContext );
    NORMAL_API_FUNCTION( rtPostProcessingStageQueryVariable );
    NORMAL_API_FUNCTION( rtPostProcessingStageGetVariableCount );
    NORMAL_API_FUNCTION( rtPostProcessingStageGetVariable );
    NORMAL_API_FUNCTION( rtProgramCreateFromPTXFile );
    NORMAL_API_FUNCTION( rtProgramCreateFromPTXString );
    NORMAL_API_FUNCTION( rtProgramDeclareVariable );
    NORMAL_API_FUNCTION( rtProgramDestroy );
    NORMAL_API_FUNCTION( rtProgramGetContext );
    NORMAL_API_FUNCTION( rtProgramGetId );
    NORMAL_API_FUNCTION( rtProgramGetVariable );
    NORMAL_API_FUNCTION( rtProgramGetVariableCount );
    NORMAL_API_FUNCTION( rtProgramQueryVariable );
    NORMAL_API_FUNCTION( rtProgramRemoveVariable );
    NORMAL_API_FUNCTION( rtProgramValidate );
    DEPRECATED_API_FUNCTION( rtRemoteDeviceCreate );
    DEPRECATED_API_FUNCTION( rtRemoteDeviceDestroy );
    DEPRECATED_API_FUNCTION( rtRemoteDeviceGetAttribute );
    DEPRECATED_API_FUNCTION( rtRemoteDeviceRelease );
    DEPRECATED_API_FUNCTION( rtRemoteDeviceReserve );
    NORMAL_API_FUNCTION( rtSelectorCreate );
    NORMAL_API_FUNCTION( rtSelectorDeclareVariable );
    NORMAL_API_FUNCTION( rtSelectorDestroy );
    NORMAL_API_FUNCTION( rtSelectorGetChild );
    NORMAL_API_FUNCTION( rtSelectorGetChildCount );
    NORMAL_API_FUNCTION( rtSelectorGetChildType );
    NORMAL_API_FUNCTION( rtSelectorGetContext );
    NORMAL_API_FUNCTION( rtSelectorGetVariable );
    NORMAL_API_FUNCTION( rtSelectorGetVariableCount );
    NORMAL_API_FUNCTION( rtSelectorGetVisitProgram );
    NORMAL_API_FUNCTION( rtSelectorQueryVariable );
    NORMAL_API_FUNCTION( rtSelectorRemoveVariable );
    NORMAL_API_FUNCTION( rtSelectorSetChild );
    NORMAL_API_FUNCTION( rtSelectorSetChildCount );
    NORMAL_API_FUNCTION( rtSelectorSetVisitProgram );
    NORMAL_API_FUNCTION( rtSelectorValidate );
    NORMAL_API_FUNCTION( rtTextureSamplerCreate );
    NORMAL_API_FUNCTION( rtTextureSamplerCreateFromGLImage );
    NORMAL_API_FUNCTION( rtTextureSamplerDestroy );
    NORMAL_API_FUNCTION( rtTextureSamplerGLRegister );
    NORMAL_API_FUNCTION( rtTextureSamplerGLUnregister );
    NORMAL_API_FUNCTION( rtTextureSamplerGetArraySize );
    NORMAL_API_FUNCTION( rtTextureSamplerGetBuffer );
    NORMAL_API_FUNCTION( rtTextureSamplerGetContext );
    NORMAL_API_FUNCTION( rtTextureSamplerGetFilteringModes );
    NORMAL_API_FUNCTION( rtTextureSamplerGetGLImageId );
    NORMAL_API_FUNCTION( rtTextureSamplerGetId );
    NORMAL_API_FUNCTION( rtTextureSamplerGetIndexingMode );
    NORMAL_API_FUNCTION( rtTextureSamplerGetMaxAnisotropy );
    NORMAL_API_FUNCTION( rtTextureSamplerGetMipLevelClamp );
    NORMAL_API_FUNCTION( rtTextureSamplerGetMipLevelBias );
    NORMAL_API_FUNCTION( rtTextureSamplerGetMipLevelCount );
    NORMAL_API_FUNCTION( rtTextureSamplerGetReadMode );
    NORMAL_API_FUNCTION( rtTextureSamplerGetWrapMode );
    NORMAL_API_FUNCTION( rtTextureSamplerSetArraySize );
    NORMAL_API_FUNCTION( rtTextureSamplerSetBuffer );
    NORMAL_API_FUNCTION( rtTextureSamplerSetFilteringModes );
    NORMAL_API_FUNCTION( rtTextureSamplerSetIndexingMode );
    NORMAL_API_FUNCTION( rtTextureSamplerSetMaxAnisotropy );
    NORMAL_API_FUNCTION( rtTextureSamplerSetMipLevelClamp );
    NORMAL_API_FUNCTION( rtTextureSamplerSetMipLevelBias );
    NORMAL_API_FUNCTION( rtTextureSamplerSetMipLevelCount );
    NORMAL_API_FUNCTION( rtTextureSamplerSetReadMode );
    NORMAL_API_FUNCTION( rtTextureSamplerSetWrapMode );
    NORMAL_API_FUNCTION( rtTextureSamplerValidate );
    NORMAL_API_FUNCTION( rtTransformCreate );
    NORMAL_API_FUNCTION( rtTransformDestroy );
    NORMAL_API_FUNCTION( rtTransformGetChild );
    NORMAL_API_FUNCTION( rtTransformGetChildType );
    NORMAL_API_FUNCTION( rtTransformGetContext );
    NORMAL_API_FUNCTION( rtTransformGetMatrix );
    NORMAL_API_FUNCTION( rtTransformSetChild );
    NORMAL_API_FUNCTION( rtTransformSetMatrix );
    NORMAL_API_FUNCTION( rtTransformSetMotionRange );
    NORMAL_API_FUNCTION( rtTransformGetMotionRange );
    NORMAL_API_FUNCTION( rtTransformSetMotionBorderMode );
    NORMAL_API_FUNCTION( rtTransformGetMotionBorderMode );
    NORMAL_API_FUNCTION( rtTransformSetMotionKeys );
    NORMAL_API_FUNCTION( rtTransformGetMotionKeyCount );
    NORMAL_API_FUNCTION( rtTransformGetMotionKeyType );
    NORMAL_API_FUNCTION( rtTransformGetMotionKeys );
    NORMAL_API_FUNCTION( rtTransformValidate );
    NORMAL_API_FUNCTION( rtVariableGet1f );
    NORMAL_API_FUNCTION( rtVariableGet1fv );
    NORMAL_API_FUNCTION( rtVariableGet1i );
    NORMAL_API_FUNCTION( rtVariableGet1iv );
    NORMAL_API_FUNCTION( rtVariableGet1ui );
    NORMAL_API_FUNCTION( rtVariableGet1uiv );
    NORMAL_API_FUNCTION( rtVariableGet1ll );
    NORMAL_API_FUNCTION( rtVariableGet1llv );
    NORMAL_API_FUNCTION( rtVariableGet1ull );
    NORMAL_API_FUNCTION( rtVariableGet1ullv );
    NORMAL_API_FUNCTION( rtVariableGet2f );
    NORMAL_API_FUNCTION( rtVariableGet2fv );
    NORMAL_API_FUNCTION( rtVariableGet2i );
    NORMAL_API_FUNCTION( rtVariableGet2iv );
    NORMAL_API_FUNCTION( rtVariableGet2ui );
    NORMAL_API_FUNCTION( rtVariableGet2uiv );
    NORMAL_API_FUNCTION( rtVariableGet2ll );
    NORMAL_API_FUNCTION( rtVariableGet2llv );
    NORMAL_API_FUNCTION( rtVariableGet2ull );
    NORMAL_API_FUNCTION( rtVariableGet2ullv );
    NORMAL_API_FUNCTION( rtVariableGet3f );
    NORMAL_API_FUNCTION( rtVariableGet3fv );
    NORMAL_API_FUNCTION( rtVariableGet3i );
    NORMAL_API_FUNCTION( rtVariableGet3iv );
    NORMAL_API_FUNCTION( rtVariableGet3ui );
    NORMAL_API_FUNCTION( rtVariableGet3uiv );
    NORMAL_API_FUNCTION( rtVariableGet3ll );
    NORMAL_API_FUNCTION( rtVariableGet3llv );
    NORMAL_API_FUNCTION( rtVariableGet3ull );
    NORMAL_API_FUNCTION( rtVariableGet3ullv );
    NORMAL_API_FUNCTION( rtVariableGet4f );
    NORMAL_API_FUNCTION( rtVariableGet4fv );
    NORMAL_API_FUNCTION( rtVariableGet4i );
    NORMAL_API_FUNCTION( rtVariableGet4iv );
    NORMAL_API_FUNCTION( rtVariableGet4ui );
    NORMAL_API_FUNCTION( rtVariableGet4uiv );
    NORMAL_API_FUNCTION( rtVariableGet4ll );
    NORMAL_API_FUNCTION( rtVariableGet4llv );
    NORMAL_API_FUNCTION( rtVariableGet4ull );
    NORMAL_API_FUNCTION( rtVariableGet4ullv );
    NORMAL_API_FUNCTION( rtVariableGetAnnotation );
    NORMAL_API_FUNCTION( rtVariableGetContext );
    NORMAL_API_FUNCTION( rtVariableGetMatrix2x2fv );
    NORMAL_API_FUNCTION( rtVariableGetMatrix2x3fv );
    NORMAL_API_FUNCTION( rtVariableGetMatrix2x4fv );
    NORMAL_API_FUNCTION( rtVariableGetMatrix3x2fv );
    NORMAL_API_FUNCTION( rtVariableGetMatrix3x3fv );
    NORMAL_API_FUNCTION( rtVariableGetMatrix3x4fv );
    NORMAL_API_FUNCTION( rtVariableGetMatrix4x2fv );
    NORMAL_API_FUNCTION( rtVariableGetMatrix4x3fv );
    NORMAL_API_FUNCTION( rtVariableGetMatrix4x4fv );
    NORMAL_API_FUNCTION( rtVariableGetName );
    NORMAL_API_FUNCTION( rtVariableGetObject );
    NORMAL_API_FUNCTION( rtVariableGetSize );
    NORMAL_API_FUNCTION( rtVariableGetType );
    NORMAL_API_FUNCTION( rtVariableGetUserData );
    NORMAL_API_FUNCTION( rtVariableSet1f );
    NORMAL_API_FUNCTION( rtVariableSet1fv );
    NORMAL_API_FUNCTION( rtVariableSet1i );
    NORMAL_API_FUNCTION( rtVariableSet1iv );
    NORMAL_API_FUNCTION( rtVariableSet1ui );
    NORMAL_API_FUNCTION( rtVariableSet1uiv );
    NORMAL_API_FUNCTION( rtVariableSet1ll );
    NORMAL_API_FUNCTION( rtVariableSet1llv );
    NORMAL_API_FUNCTION( rtVariableSet1ull );
    NORMAL_API_FUNCTION( rtVariableSet1ullv );
    NORMAL_API_FUNCTION( rtVariableSet2f );
    NORMAL_API_FUNCTION( rtVariableSet2fv );
    NORMAL_API_FUNCTION( rtVariableSet2i );
    NORMAL_API_FUNCTION( rtVariableSet2iv );
    NORMAL_API_FUNCTION( rtVariableSet2ui );
    NORMAL_API_FUNCTION( rtVariableSet2uiv );
    NORMAL_API_FUNCTION( rtVariableSet2ll );
    NORMAL_API_FUNCTION( rtVariableSet2llv );
    NORMAL_API_FUNCTION( rtVariableSet2ull );
    NORMAL_API_FUNCTION( rtVariableSet2ullv );
    NORMAL_API_FUNCTION( rtVariableSet3f );
    NORMAL_API_FUNCTION( rtVariableSet3fv );
    NORMAL_API_FUNCTION( rtVariableSet3i );
    NORMAL_API_FUNCTION( rtVariableSet3iv );
    NORMAL_API_FUNCTION( rtVariableSet3ui );
    NORMAL_API_FUNCTION( rtVariableSet3uiv );
    NORMAL_API_FUNCTION( rtVariableSet3ll );
    NORMAL_API_FUNCTION( rtVariableSet3llv );
    NORMAL_API_FUNCTION( rtVariableSet3ull );
    NORMAL_API_FUNCTION( rtVariableSet3ullv );
    NORMAL_API_FUNCTION( rtVariableSet4f );
    NORMAL_API_FUNCTION( rtVariableSet4fv );
    NORMAL_API_FUNCTION( rtVariableSet4i );
    NORMAL_API_FUNCTION( rtVariableSet4iv );
    NORMAL_API_FUNCTION( rtVariableSet4ui );
    NORMAL_API_FUNCTION( rtVariableSet4uiv );
    NORMAL_API_FUNCTION( rtVariableSet4ll );
    NORMAL_API_FUNCTION( rtVariableSet4llv );
    NORMAL_API_FUNCTION( rtVariableSet4ull );
    NORMAL_API_FUNCTION( rtVariableSet4ullv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix2x2fv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix2x3fv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix2x4fv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix3x2fv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix3x3fv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix3x4fv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix4x2fv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix4x3fv );
    NORMAL_API_FUNCTION( rtVariableSetMatrix4x4fv );
    NORMAL_API_FUNCTION( rtVariableSetObject );
    NORMAL_API_FUNCTION( rtVariableSetUserData );

#ifdef _WIN32  // windows-only section
    NORMAL_API_FUNCTION( rtDeviceGetWGLDevice );
    NORMAL_API_FUNCTION( rtBufferCreateFromD3D10Resource );
    NORMAL_API_FUNCTION( rtBufferCreateFromD3D11Resource );
    NORMAL_API_FUNCTION( rtBufferCreateFromD3D9Resource );
    NORMAL_API_FUNCTION( rtBufferD3D10Register );
    NORMAL_API_FUNCTION( rtBufferD3D10Unregister );
    NORMAL_API_FUNCTION( rtBufferD3D11Register );
    NORMAL_API_FUNCTION( rtBufferD3D11Unregister );
    NORMAL_API_FUNCTION( rtBufferD3D9Register );
    NORMAL_API_FUNCTION( rtBufferD3D9Unregister );
    NORMAL_API_FUNCTION( rtBufferGetD3D10Resource );
    NORMAL_API_FUNCTION( rtBufferGetD3D11Resource );
    NORMAL_API_FUNCTION( rtBufferGetD3D9Resource );
    NORMAL_API_FUNCTION( rtContextSetD3D10Device );
    NORMAL_API_FUNCTION( rtContextSetD3D11Device );
    NORMAL_API_FUNCTION( rtContextSetD3D9Device );
    NORMAL_API_FUNCTION( rtDeviceGetD3D9Device );
    NORMAL_API_FUNCTION( rtDeviceGetD3D10Device );
    NORMAL_API_FUNCTION( rtDeviceGetD3D11Device );
    NORMAL_API_FUNCTION( rtTextureSamplerCreateFromD3D10Resource );
    NORMAL_API_FUNCTION( rtTextureSamplerCreateFromD3D11Resource );
    NORMAL_API_FUNCTION( rtTextureSamplerCreateFromD3D9Resource );
    NORMAL_API_FUNCTION( rtTextureSamplerD3D10Register );
    NORMAL_API_FUNCTION( rtTextureSamplerD3D10Unregister );
    NORMAL_API_FUNCTION( rtTextureSamplerD3D11Register );
    NORMAL_API_FUNCTION( rtTextureSamplerD3D11Unregister );
    NORMAL_API_FUNCTION( rtTextureSamplerD3D9Register );
    NORMAL_API_FUNCTION( rtTextureSamplerD3D9Unregister );
    NORMAL_API_FUNCTION( rtTextureSamplerGetD3D10Resource );
    NORMAL_API_FUNCTION( rtTextureSamplerGetD3D11Resource );
    NORMAL_API_FUNCTION( rtTextureSamplerGetD3D9Resource );
#endif  // _WIN32

    if( abiVersion >= 4 )
    {
        NORMAL_API_FUNCTION( rtProgramCreateFromPTXFiles );
        NORMAL_API_FUNCTION( rtProgramCreateFromPTXStrings );
    }
    if( abiVersion >= 5 )
    {
        NORMAL_API_FUNCTION( rtGeometryInstanceGetGeometryTriangles );
        NORMAL_API_FUNCTION( rtGeometryInstanceSetGeometryTriangles );
        NORMAL_API_FUNCTION( rtGeometryTrianglesCreate );
        NORMAL_API_FUNCTION( rtGeometryTrianglesDestroy );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetContext );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetPrimitiveIndexOffset );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetPrimitiveIndexOffset );
        // REMOVED with LWOPTIX_ABI_VERSION 7 (LWOPTIX_ABI_VERSION 5 got never officially released, but make a basic attempt to be backward compatible)
        api->rtGeometryTrianglesSetIndexedTrianglesDeprecated = rtGeometryTrianglesSetIndexedTriangles_backCompatible;
        api->rtGeometryTrianglesSetTrianglesDeprecated        = rtGeometryTrianglesSetTriangles_backCompatible;

        // To remove backward compatibility, use this instead.
        // api->rtGeometryTrianglesSetIndexedTriangles     = (void*)( rtDeprecatedApiFunction );
        // api->rtGeometryTrianglesSetTriangles            = (void*)( rtDeprecatedApiFunction );
        NORMAL_API_FUNCTION( rtGeometryTrianglesValidate );
    }
    if( abiVersion >= 6 )
    {
        NORMAL_API_FUNCTION( rtLog );
    }
    if( abiVersion >= 7 )
    {
        NORMAL_API_FUNCTION( rtContextGetMaxCallableProgramDepth );
        NORMAL_API_FUNCTION( rtContextGetMaxTraceDepth );
        NORMAL_API_FUNCTION( rtContextSetMaxCallableProgramDepth );
        NORMAL_API_FUNCTION( rtContextSetMaxTraceDepth );
    }
    if( abiVersion >= 8 )
    {
        NORMAL_API_FUNCTION( rtProgramCallsiteSetPotentialCallees );
    }
    if( abiVersion >= 9 )
    {
        NORMAL_API_FUNCTION( rtOverridesOtherVersion );
        NORMAL_API_FUNCTION( rtGetBuildVersion );
    }
    if( abiVersion >= 10 )
    {
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetAttributeProgram );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetAttributeProgram );
        NORMAL_API_FUNCTION( rtGeometryTrianglesDeclareVariable );
        NORMAL_API_FUNCTION( rtGeometryTrianglesQueryVariable );
        NORMAL_API_FUNCTION( rtGeometryTrianglesRemoveVariable );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetVariableCount );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetVariable );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetPreTransformMatrix );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetPreTransformMatrix );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetPrimitiveCount );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetPrimitiveCount );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetTriangleIndices );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetVertices );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetMotiolwertices );
        // see comment in lwoptix.h
        api->rtGeometryTrianglesSetMotiolwerticesMultiBufferDeprecated = rtGeometryTrianglesSetMotiolwerticesMultiBuffer_backCompatible;
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetMotionSteps );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetMotionSteps );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetMotionRange );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetMotionRange );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetMotionBorderMode );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetMotionBorderMode );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetBuildFlags );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetMaterialCount );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetMaterialCount );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetMaterialIndices );
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetFlagsPerMaterial );
        NORMAL_API_FUNCTION( rtGeometryTrianglesGetFlagsPerMaterial );
        NORMAL_API_FUNCTION( rtGroupSetVisibilityMask );
        NORMAL_API_FUNCTION( rtGroupGetVisibilityMask );
        NORMAL_API_FUNCTION( rtGeometryGroupSetFlags );
        NORMAL_API_FUNCTION( rtGeometryGroupGetFlags );
        NORMAL_API_FUNCTION( rtGeometryGroupSetVisibilityMask );
        NORMAL_API_FUNCTION( rtGeometryGroupGetVisibilityMask );
        NORMAL_API_FUNCTION( rtGeometrySetFlags );
        NORMAL_API_FUNCTION( rtGeometryGetFlags );
    }

    // VCA functions deprecated in abiVersion >= 11; no new functions added
    if( abiVersion >= 12 )
    {
        NORMAL_API_FUNCTION( rtGeometryTrianglesSetMotiolwerticesMultiBuffer );
    }

    // abiVersion >= 13: Changes to device functions, no new functions added. See comment in lwoptix.h

    // Asynchronous launches
    if( abiVersion >= 14 )
    {
        NORMAL_API_FUNCTION( rtCommandListSetLwdaStream );
        NORMAL_API_FUNCTION( rtCommandListGetLwdaStream );
        NORMAL_API_FUNCTION( rtCommandListAppendLaunch1D );
        NORMAL_API_FUNCTION( rtCommandListAppendLaunch3D );
        NORMAL_API_FUNCTION( rtCommandListSetDevices );
        NORMAL_API_FUNCTION( rtCommandListGetDevices );
        NORMAL_API_FUNCTION( rtCommandListGetDeviceCount );
    }

    // Demand-load buffers and textures
    if( abiVersion >= 15 )
    {
        NORMAL_API_FUNCTION( rtBufferCreateFromCallback );
    }

    // Clone a program
    if( abiVersion >= 16 )
    {
        NORMAL_API_FUNCTION( rtProgramCreateFromProgram );
    }

    // Backwards compatibility with demand load implementation changes.
    if( abiVersion < 17 )
    {
        api->rtContextCreate = rtContextCreateABI16;
    }
    else if( abiVersion < 18 )
    {
        api->rtContextCreate = rtContextCreateABI17;
    }
    else
    {
        NORMAL_API_FUNCTION( rtContextCreate );
    }

#undef NORMAL_API_FUNCTION
#undef DEPRECATED_API_FUNCTION

    return RT_SUCCESS;
}
