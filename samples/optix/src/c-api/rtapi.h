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

#pragma once
#include <o6/optix.h>
#include <optix_lwda_interop.h>
#include <optix_gl_interop.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <optix_d3d10_interop.h>
#include <optix_d3d11_interop.h>
#include <optix_d3d9_interop.h>
#include <windows.h>
#endif

//
// Absolutely every API entry point should be reflected here, including the ones that
// are not in optix.h (e.g. interop), are windows only, or private.
//
// Please keep the sections sorted. Also, please keep formatting exactly uniform, since
// that makes it easier to parse the list with a script.
//

// clang-format off
RTresult _rtAccelerationCreate( RTcontext context_api, RTacceleration* acceleration );
RTresult _rtAccelerationDestroy( RTacceleration acceleration_api );
RTresult _rtAccelerationGetBuilder( RTacceleration acceleration_api, const char** return_string );
RTresult _rtAccelerationGetContext( RTacceleration acceleration_api, RTcontext* c );
RTresult _rtAccelerationGetData( RTacceleration acceleration_api, void* data );
RTresult _rtAccelerationGetDataSize( RTacceleration acceleration_api, RTsize* size );
RTresult _rtAccelerationGetProperty( RTacceleration acceleration_api, const char* name, const char** return_string );
RTresult _rtAccelerationGetTraverser( RTacceleration acceleration_api, const char** return_string );
RTresult _rtAccelerationIsDirty( RTacceleration acceleration_api, int* dirty );
RTresult _rtAccelerationMarkDirty( RTacceleration acceleration_api );
RTresult _rtAccelerationSetBuilder( RTacceleration acceleration_api, const char* builder );
RTresult _rtAccelerationSetData( RTacceleration acceleration_api, const void* data, RTsize size );
RTresult _rtAccelerationSetProperty( RTacceleration acceleration_api, const char* name, const char* value );
RTresult _rtAccelerationSetTraverser( RTacceleration acceleration_api, const char* traverser );
RTresult _rtAcceleratiolwalidate( RTacceleration acceleration_api );
RTresult _rtBufferCreate( RTcontext context_api, unsigned int type, RTbuffer* buffer );
RTresult _rtBufferCreateForLWDA( RTcontext context_api, unsigned int type, RTbuffer* buffer );
RTresult _rtBufferCreateFromGLBO( RTcontext context_api, unsigned int type, unsigned int gl_id, RTbuffer* buffer );
RTresult _rtBufferCreateFromCallback( RTcontext context_api, unsigned int bufferDesc, RTbuffercallback callback, void* callbackContext, RTbuffer* buffer );
RTresult _rtBufferDestroy( RTbuffer buffer_api );
RTresult _rtBufferGLRegister( RTbuffer buffer_api );
RTresult _rtBufferGLUnregister( RTbuffer buffer_api );
RTresult _rtBufferGetContext( RTbuffer buffer_api, RTcontext* c );
RTresult _rtBufferGetDevicePointer( RTbuffer buffer_api, int optix_device_ordinal, void** device_pointer );
RTresult _rtBufferGetDimensionality( RTbuffer buffer_api, unsigned int* dimensionality );
RTresult _rtBufferGetElementSize( RTbuffer buffer_api, RTsize* size_of_element );
RTresult _rtBufferGetFormat( RTbuffer buffer_api, RTformat* format );
RTresult _rtBufferGetGLBOId( RTbuffer buffer_api, unsigned int* gl_id );
RTresult _rtBufferGetId( RTbuffer buffer_api, int* buffer_id );
RTresult _rtBufferGetMipLevelCount( RTbuffer buffer_api, unsigned int* level );
RTresult _rtBufferGetMipLevelSize1D( RTbuffer buffer_api, unsigned int level, RTsize* width );
RTresult _rtBufferGetMipLevelSize2D( RTbuffer buffer_api, unsigned int level, RTsize* width, RTsize* height );
RTresult _rtBufferGetMipLevelSize3D( RTbuffer buffer_api, unsigned int level, RTsize* width, RTsize* height, RTsize* depth );
RTresult _rtBufferGetSize1D( RTbuffer buffer_api, RTsize* width );
RTresult _rtBufferGetSize2D( RTbuffer buffer_api, RTsize* width, RTsize* height );
RTresult _rtBufferGetSize3D( RTbuffer buffer_api, RTsize* width, RTsize* height, RTsize* depth );
RTresult _rtBufferGetSizev( RTbuffer buffer_api, unsigned int maxdim, RTsize* outdims );
RTresult _rtBufferMap( RTbuffer buffer_api, void** user_pointer );
RTresult _rtBufferMapEx( RTbuffer buffer, unsigned int map_flags, unsigned int level, void* user_owned, void** optix_owned );
RTresult _rtBufferMarkDirty( RTbuffer buffer_api );
RTresult _rtBufferSetDevicePointer( RTbuffer buffer_api, int optix_device_ordinal, void* device_pointer );
RTresult _rtBufferSetElementSize( RTbuffer buffer_api, RTsize size_of_element );
RTresult _rtBufferSetFormat( RTbuffer buffer_api, RTformat type );
RTresult _rtBufferSetMipLevelCount( RTbuffer buffer_api, unsigned int levels );
RTresult _rtBufferSetSize1D( RTbuffer buffer_api, RTsize width );
RTresult _rtBufferSetSize2D( RTbuffer buffer_api, RTsize width, RTsize height );
RTresult _rtBufferSetSize3D( RTbuffer buffer_api, RTsize width, RTsize height, RTsize depth );
RTresult _rtBufferSetSizev( RTbuffer buffer_api, unsigned int dimensionality, const RTsize* indims );
RTresult _rtBufferUnmap( RTbuffer buffer_api );
RTresult _rtBufferUnmapEx( RTbuffer buffer_api, unsigned int level );
RTresult _rtBufferValidate( RTbuffer buffer_api );
RTresult _rtBufferGetProgressiveUpdateReady( RTbuffer buffer_api, int* ready, unsigned int* subframe_count, unsigned int* max_subframes );
RTresult _rtBufferBindProgressiveStream( RTbuffer buffer_api, RTbuffer source );
RTresult _rtBufferSetAttribute( RTbuffer buffer, RTbufferattribute attrib, RTsize size, const void* p );
RTresult _rtBufferGetAttribute( RTbuffer buffer, RTbufferattribute attrib, RTsize size, void* p );
RTresult _rtCommandListCreate( RTcontext context_api, RTcommandlist* list );
RTresult _rtCommandListDestroy( RTcommandlist list_api );
RTresult _rtCommandListAppendPostprocessingStage( RTcommandlist list_api, RTpostprocessingstage stage_api, RTsize launch_width, RTsize launch_height );
RTresult _rtCommandListAppendLaunch1D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width );
RTresult _rtCommandListAppendLaunch2D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width, RTsize launch_height );
RTresult _rtCommandListAppendLaunch3D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width, RTsize launch_height, RTsize launch_depth );
RTresult _rtCommandListSetDevices( RTcommandlist list_api, unsigned int count, const int* devices );
RTresult _rtCommandListGetDevices( RTcommandlist list_api, int* devices );
RTresult _rtCommandListGetDeviceCount( RTcommandlist list_api, unsigned int* count );
RTresult _rtCommandListFinalize( RTcommandlist list_api );
RTresult _rtCommandListExelwte( RTcommandlist list_api );
RTresult _rtCommandListSetLwdaStream( RTcommandlist list_api, void* stream );
RTresult _rtCommandListGetLwdaStream( RTcommandlist list_api, void** stream );
RTresult _rtCommandListGetContext( RTcommandlist list_api, RTcontext* c );
RTresult _rtContextCompile( RTcontext context_api );
RTresult _rtContextCreateABI16( RTcontext* context );
RTresult _rtContextCreateABI17( RTcontext* context );
RTresult _rtContextCreateABI18( RTcontext* context );
RTresult _rtContextDeclareVariable( RTcontext context_api, const char* name, RTvariable* v );
RTresult _rtContextDestroy( RTcontext context_api );
RTresult _rtContextSetAttribute( RTcontext context_api, RTcontextattribute attrib, RTsize size, const void* p );
RTresult _rtContextGetAttribute( RTcontext context_api, RTcontextattribute attrib, RTsize size, void* p );
RTresult _rtContextGetBufferFromId( RTcontext context_api, int buffer_id, RTbuffer* buffer );
RTresult _rtContextGetDeviceCount( RTcontext context_api, unsigned int* count );
RTresult _rtContextGetDevices( RTcontext context_api, int* devices );
RTresult _rtContextGetEntryPointCount( RTcontext context_api, unsigned int* num_entry_points );
void _rtContextGetErrorString( RTcontext context_api, RTresult code, const char** return_string );
RTresult _rtContextGetExceptionEnabled( RTcontext context_api, RTexception exception, int* enabled );
RTresult _rtContextGetExceptionProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram* program );
RTresult _rtContextGetMaxCallableProgramDepth( RTcontext context_api, unsigned int* max_depth );
RTresult _rtContextGetMaxTraceDepth( RTcontext context_api, unsigned int* max_depth );
RTresult _rtContextGetMissProgram( RTcontext context_api, unsigned int ray_type_index, RTprogram* program );
RTresult _rtContextGetPrintBufferSize( RTcontext context_api, RTsize* buffer_size_bytes );
RTresult _rtContextGetPrintEnabled( RTcontext context_api, int* enabled );
RTresult _rtContextGetPrintLaunchIndex( RTcontext context_api, int* x, int* y, int* z );
RTresult _rtContextGetProgramFromId( RTcontext context_api, int program_id, RTprogram* program );
RTresult _rtContextGetRayGenerationProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram* program );
RTresult _rtContextGetRayTypeCount( RTcontext context_api, unsigned int* num_ray_types );
RTresult _rtContextGetRunningState( RTcontext context_api, int* running );
RTresult _rtContextGetStackSize( RTcontext context_api, RTsize* stack_size_bytes );
RTresult _rtContextGetTextureSamplerFromId( RTcontext context_api, int sampler_id, RTtexturesampler* sampler );
RTresult _rtContextGetVariable( RTcontext context_api, unsigned int index, RTvariable* v );
RTresult _rtContextGetVariableCount( RTcontext context_api, unsigned int* c );
RTresult _rtContextLaunch1D( RTcontext context_api, unsigned int entry_point_index, RTsize image_width );
RTresult _rtContextLaunch2D( RTcontext context_api, unsigned int entry_point_index, RTsize image_width, RTsize image_height );
RTresult _rtContextLaunch3D( RTcontext context_api, unsigned int entry_point_index, RTsize image_width, RTsize image_height, RTsize image_depth );
RTresult _rtContextLaunchProgressive2D( RTcontext context_api, unsigned int entry_point_index, RTsize image_width, RTsize image_height, unsigned int max_subframes );
RTresult _rtContextStopProgressive( RTcontext context_api );
RTresult _rtContextQueryVariable( RTcontext context_api, const char* name, RTvariable* v );
RTresult _rtContextRemoveVariable( RTcontext context_api, RTvariable v_api );
RTresult _rtContextSetDevices( RTcontext context_api, unsigned int count, const int* devices );
RTresult _rtContextSetEntryPointCount( RTcontext context_api, unsigned int num_entry_points );
RTresult _rtContextSetExceptionEnabled( RTcontext context_api, RTexception exception, int enabled );
RTresult _rtContextSetExceptionProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram program_api );
RTresult _rtContextSetMaxCallableProgramDepth( RTcontext context_api, unsigned int max_depth );
RTresult _rtContextSetMaxTraceDepth( RTcontext context_api, unsigned int max_depth );
RTresult _rtContextSetMissProgram( RTcontext context_api, unsigned int ray_type_index, RTprogram program_api );
RTresult _rtContextSetPrintBufferSize( RTcontext context_api, RTsize buffer_size_bytes );
RTresult _rtContextSetPrintEnabled( RTcontext context_api, int enabled );
RTresult _rtContextSetPrintLaunchIndex( RTcontext context_api, int x, int y, int z );
RTresult _rtContextSetRayGenerationProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram program_api );
RTresult _rtContextSetRayTypeCount( RTcontext context_api, unsigned int num_ray_types );
RTresult _rtContextSetStackSize( RTcontext context_api, RTsize stack_size_bytes );
RTresult _rtContextSetTimeoutCallback( RTcontext context_api, RTtimeoutcallback callback, double seconds );
RTresult _rtContextSetUsageReportCallback( RTcontext context_api, RTusagereportcallback callback, int verbosity, void* cbdata );
RTresult _rtContextValidate( RTcontext context_api );
RTresult _rtDeviceGetAttribute( int ordinal, RTdeviceattribute attrib, RTsize size, void* p );
RTresult _rtDeviceGetDeviceCount( unsigned int* count );
RTresult _rtGeometryCreate( RTcontext context_api, RTgeometry* geometry );
RTresult _rtGeometryDeclareVariable( RTgeometry geometry_api, const char* name, RTvariable* v );
RTresult _rtGeometryDestroy( RTgeometry geometry_api );
RTresult _rtGeometryGetBoundingBoxProgram( RTgeometry geometry_api, RTprogram* program );
RTresult _rtGeometryGetContext( RTgeometry geometry_api, RTcontext* c );
RTresult _rtGeometryGetIntersectionProgram( RTgeometry geometry_api, RTprogram* program );
RTresult _rtGeometryGetPrimitiveCount( RTgeometry geometry_api, unsigned int* num_primitives );
RTresult _rtGeometryGetPrimitiveIndexOffset( RTgeometry geometry_api, unsigned int* index_offset );
RTresult _rtGeometryGetVariable( RTgeometry geometry_api, unsigned int index, RTvariable* v );
RTresult _rtGeometryGetVariableCount( RTgeometry geometry_api, unsigned int* c );
RTresult _rtGeometryGroupCreate( RTcontext context_api, RTgeometrygroup* geometrygroup );
RTresult _rtGeometryGroupDestroy( RTgeometrygroup geometrygroup_api );
RTresult _rtGeometryGroupGetAcceleration( RTgeometrygroup geometrygroup_api, RTacceleration* acceleration );
RTresult _rtGeometryGroupGetChild( RTgeometrygroup geometrygroup_api, unsigned int index, RTgeometryinstance* geometryinstance );
RTresult _rtGeometryGroupGetChildCount( RTgeometrygroup geometrygroup_api, unsigned int* count );
RTresult _rtGeometryGroupGetContext( RTgeometrygroup geometrygroup_api, RTcontext* c );
RTresult _rtGeometryGroupSetAcceleration( RTgeometrygroup geometrygroup_api, RTacceleration acceleration_api );
RTresult _rtGeometryGroupSetChild( RTgeometrygroup geometrygroup_api, unsigned int index, RTgeometryinstance geometryinstance_api );
RTresult _rtGeometryGroupSetChildCount( RTgeometrygroup geometrygroup_api, unsigned int count );
RTresult _rtGeometryGroupValidate( RTgeometrygroup geometrygroup_api );
RTresult _rtGeometryInstanceCreate( RTcontext context_api, RTgeometryinstance* geometryinstance );
RTresult _rtGeometryInstanceDeclareVariable( RTgeometryinstance gi_api, const char* name, RTvariable* v );
RTresult _rtGeometryInstanceDestroy( RTgeometryinstance geometryinstance_api );
RTresult _rtGeometryInstanceGetContext( RTgeometryinstance gi_api, RTcontext* c );
RTresult _rtGeometryInstanceGetGeometry( RTgeometryinstance gi_api, RTgeometry* geo );
RTresult _rtGeometryInstanceGetGeometryTriangles( RTgeometryinstance gi_api, RTgeometrytriangles* geo );
RTresult _rtGeometryInstanceGetMaterial( RTgeometryinstance gi_api, unsigned int material_idx, RTmaterial* mat );
RTresult _rtGeometryInstanceGetMaterialCount( RTgeometryinstance gi_api, unsigned int* num_materials );
RTresult _rtGeometryInstanceGetVariable( RTgeometryinstance gi_api, unsigned int index, RTvariable* v );
RTresult _rtGeometryInstanceGetVariableCount( RTgeometryinstance gi_api, unsigned int* c );
RTresult _rtGeometryInstanceQueryVariable( RTgeometryinstance gi_api, const char* name, RTvariable* v );
RTresult _rtGeometryInstanceRemoveVariable( RTgeometryinstance gi_api, RTvariable v_api );
RTresult _rtGeometryInstanceSetGeometry( RTgeometryinstance gi_api, RTgeometry geo_api );
RTresult _rtGeometryInstanceSetGeometryTriangles( RTgeometryinstance gi_api, RTgeometrytriangles geo_api );
RTresult _rtGeometryInstanceSetMaterial( RTgeometryinstance gi_api, unsigned int material_idx, RTmaterial mat_api );
RTresult _rtGeometryInstanceSetMaterialCount( RTgeometryinstance gi_api, unsigned int num_materials );
RTresult _rtGeometryInstanceValidate( RTgeometryinstance geometryinstance_api );
RTresult _rtGeometryIsDirty( RTgeometry geometry_api, int* dirty );
RTresult _rtGeometryMarkDirty( RTgeometry geometry_api );
RTresult _rtGeometryQueryVariable( RTgeometry geometry_api, const char* name, RTvariable* v );
RTresult _rtGeometryRemoveVariable( RTgeometry geometry_api, RTvariable v_api );
RTresult _rtGeometrySetBoundingBoxProgram( RTgeometry geometry_api, RTprogram program_api );
RTresult _rtGeometrySetIntersectionProgram( RTgeometry geometry_api, RTprogram program_api );
RTresult _rtGeometrySetPrimitiveCount( RTgeometry geometry_api, unsigned int num_primitives );
RTresult _rtGeometrySetPrimitiveIndexOffset( RTgeometry geometry_api, unsigned int index_offset );
RTresult _rtGeometrySetMotionRange( RTgeometry geometry_api, float timeBegin, float timeEnd );
RTresult _rtGeometryGetMotionRange( RTgeometry geometry_api, float* timeBegin, float* timeEnd );
RTresult _rtGeometrySetMotionBorderMode( RTgeometry geometry_api, RTmotionbordermode beginMode, RTmotionbordermode endMode );
RTresult _rtGeometryGetMotionBorderMode( RTgeometry geometry_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode );
RTresult _rtGeometrySetMotionSteps( RTgeometry geometry_api, unsigned int n );
RTresult _rtGeometryGetMotionSteps( RTgeometry geometry_api, unsigned int* n );
RTresult _rtGeometryValidate( RTgeometry geometry_api );
RTresult _rtGeometryTrianglesCreate( RTcontext context_api, RTgeometrytriangles* geometrytriangles );
RTresult _rtGeometryTrianglesDestroy( RTgeometrytriangles geometrytriangles_api );
RTresult _rtGeometryTrianglesGetContext( RTgeometrytriangles geometrytriangles_api, RTcontext* context );
RTresult _rtGeometryTrianglesGetPrimitiveIndexOffset( RTgeometrytriangles geometrytriangles_api, unsigned int* index_offset );
RTresult _rtGeometryTrianglesSetPrimitiveIndexOffset( RTgeometrytriangles geometrytriangles_api, unsigned int index_offset );
RTresult _rtGeometryTrianglesSetPreTransformMatrix( RTgeometrytriangles geometrytriangles_api, int transpose, const float* matrix );
RTresult _rtGeometryTrianglesGetPreTransformMatrix( RTgeometrytriangles geometrytriangles_api, int transpose, float* matrix );
RTresult _rtGeometryTrianglesSetPrimitiveCount( RTgeometrytriangles geometrytriangles_api, unsigned int num_triangles );
RTresult _rtGeometryTrianglesGetPrimitiveCount( RTgeometrytriangles geometrytriangles_api, unsigned int* num_triangles );
RTresult _rtGeometryTrianglesSetTriangleIndices(  RTgeometrytriangles geometrytriangles_api,
                                                  RTbuffer            index_buffer,
                                                  RTsize              index_buffer_byte_offset,
                                                  RTsize              tri_indices_byte_stride,
                                                  RTformat            tri_indices_format );
RTresult _rtGeometryTrianglesSetVertices( RTgeometrytriangles geometrytriangles_api,
                                          unsigned int        num_vertices,
                                          RTbuffer            vertex_buffer,
                                          RTsize              vertex_buffer_byte_offset,
                                          RTsize              vertex_byte_stride,
                                          RTformat            position_format );
RTresult _rtGeometryTrianglesSetMotiolwertices( RTgeometrytriangles geometrytriangles_api,
                                                unsigned int        num_vertices,
                                                RTbuffer            vertex_buffer,
                                                RTsize              vertex_buffer_byte_offset,
                                                RTsize              vertex_byte_stride,
                                                RTsize              vertex_motion_step_byte_stride,
                                                RTformat            position_format );
RTresult _rtGeometryTrianglesSetMotiolwerticesMultiBuffer( RTgeometrytriangles geometrytriangles_api,
                                                           unsigned int        num_vertices,
                                                           RTbuffer*           vertex_buffers,
                                                           unsigned int        vertex_buffer_count,
                                                           RTsize              vertex_buffer_byte_offset,
                                                           RTsize              vertex_byte_stride,
                                                           RTformat            position_format );
RTresult _rtGeometryTrianglesSetMotionSteps( RTgeometrytriangles geometrytriangles_api, unsigned int num_motion_steps );
RTresult _rtGeometryTrianglesGetMotionSteps( RTgeometrytriangles geometrytriangles_api, unsigned int* num_motion_steps );
RTresult _rtGeometryTrianglesSetMotionRange( RTgeometrytriangles geometrytriangles_api, float timeBegin, float timeEnd );
RTresult _rtGeometryTrianglesGetMotionRange( RTgeometrytriangles geometrytriangles_api, float* timeBegin, float* timeEnd );
RTresult _rtGeometryTrianglesSetMotionBorderMode( RTgeometrytriangles geometrytriangles_api, RTmotionbordermode beginMode, RTmotionbordermode endMode );
RTresult _rtGeometryTrianglesGetMotionBorderMode( RTgeometrytriangles geometrytriangles_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode );
RTresult _rtGeometryTrianglesSetBuildFlags( RTgeometrytriangles geometrytriangles_api, RTgeometrybuildflags build_flags );
RTresult _rtGeometryTrianglesGetMaterialCount( RTgeometrytriangles geometrytriangles_api, unsigned int* num_materials );
RTresult _rtGeometryTrianglesSetMaterialCount( RTgeometrytriangles geometrytriangles_api, unsigned int num_materials );
RTresult _rtGeometryTrianglesSetMaterialIndices( RTgeometrytriangles geometrytriangles_api,
                                                 RTbuffer            material_index_buffer_api,
                                                 RTsize              material_index_buffer_byte_offset,
                                                 RTsize              material_index_byte_stride,
                                                 RTformat            material_index_format );
RTresult _rtGroupSetVisibilityMask( RTgroup group, RTvisibilitymask mask );
RTresult _rtGroupGetVisibilityMask( RTgroup group, RTvisibilitymask* mask );
RTresult _rtGeometryGroupSetFlags( RTgeometrygroup group, RTinstanceflags flags );
RTresult _rtGeometryGroupGetFlags( RTgeometrygroup group, RTinstanceflags* flags );
RTresult _rtGeometryGroupSetVisibilityMask( RTgeometrygroup group, RTvisibilitymask mask );
RTresult _rtGeometryGroupGetVisibilityMask( RTgeometrygroup group, RTvisibilitymask* mask );
RTresult _rtGeometrySetFlags( RTgeometry geometry, RTgeometryflags flags );
RTresult _rtGeometryGetFlags( RTgeometry geometry, RTgeometryflags* flags );
RTresult _rtGeometryTrianglesSetFlagsPerMaterial( RTgeometrytriangles geometrytriangles_api, unsigned int material_index, RTgeometryflags flags );
RTresult _rtGeometryTrianglesGetFlagsPerMaterial( RTgeometrytriangles geometrytriangles_api, unsigned int material_index, RTgeometryflags* flags );
RTresult _rtGeometryTrianglesValidate( RTgeometrytriangles geometrytriangles_api );
RTresult _rtGeometryTrianglesSetAttributeProgram( RTgeometrytriangles geometrytriangles_api, RTprogram program_api );
RTresult _rtGeometryTrianglesGetAttributeProgram( RTgeometrytriangles geometrytriangles_api, RTprogram* program_api );
RTresult _rtGeometryTrianglesDeclareVariable( RTgeometrytriangles geometrytriangles_api, const char* name, RTvariable* v );
RTresult _rtGeometryTrianglesQueryVariable( RTgeometrytriangles geometrytriangles_api, const char* name, RTvariable* v );
RTresult _rtGeometryTrianglesRemoveVariable( RTgeometrytriangles geometrytriangles_api, RTvariable v_api );
RTresult _rtGeometryTrianglesGetVariableCount( RTgeometrytriangles geometrytriangles_api, unsigned int* count );
RTresult _rtGeometryTrianglesGetVariable( RTgeometrytriangles geometrytriangles_api, unsigned int index, RTvariable* v );
RTresult _rtGetBuildVersion( const char** result );
RTresult _rtGetVersion( unsigned int* version );
RTresult _rtGlobalSetAttribute( RTglobalattribute attrib, RTsize size, const void* p );
RTresult _rtGlobalGetAttribute( RTglobalattribute attrib, RTsize size, void* p );
RTresult _rtGroupCreate( RTcontext context_api, RTgroup* group );
RTresult _rtGroupDestroy( RTgroup group_api );
RTresult _rtGroupGetAcceleration( RTgroup group_api, RTacceleration* acceleration );
RTresult _rtGroupGetChild( RTgroup group_api, unsigned int index, RTobject* child );
RTresult _rtGroupGetChildCount( RTgroup group_api, unsigned int* count );
RTresult _rtGroupGetChildType( RTgroup group_api, unsigned int index, RTobjecttype* type );
RTresult _rtGroupGetContext( RTgroup group_api, RTcontext* c );
RTresult _rtGroupSetAcceleration( RTgroup group_api, RTacceleration acceleration_api );
RTresult _rtGroupSetChild( RTgroup group_api, unsigned int index, RTobject child );
RTresult _rtGroupSetChildCount( RTgroup group_api, unsigned int count );
RTresult _rtGroupValidate( RTgroup group_api );
RTresult _rtMaterialCreate( RTcontext context_api, RTmaterial* material );
RTresult _rtMaterialDeclareVariable( RTmaterial material_api, const char* name, RTvariable* v );
RTresult _rtMaterialDestroy( RTmaterial material_api );
RTresult _rtMaterialGetAnyHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram* program );
RTresult _rtMaterialGetClosestHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram* program );
RTresult _rtMaterialGetContext( RTmaterial material_api, RTcontext* c );
RTresult _rtMaterialGetVariable( RTmaterial material_api, unsigned int index, RTvariable* v );
RTresult _rtMaterialGetVariableCount( RTmaterial material_api, unsigned int* c );
RTresult _rtMaterialQueryVariable( RTmaterial material_api, const char* name, RTvariable* v );
RTresult _rtMaterialRemoveVariable( RTmaterial material_api, RTvariable v_api );
RTresult _rtMaterialSetAnyHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram program_api );
RTresult _rtMaterialSetClosestHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram program_api );
RTresult _rtMaterialValidate( RTmaterial material_api );
RTresult _rtOverridesOtherVersion( const char* otherVersion, int* result );
RTresult _rtPostProcessingStageCreateBuiltin( RTcontext context_api, const char* builtin_name, void* denoiser, void* ssim_predictor, RTpostprocessingstage* stage_api );
RTresult _rtPostProcessingStageDeclareVariable( RTpostprocessingstage stage_api, const char* name, RTvariable* v );
RTresult _rtPostProcessingStageDestroy( RTpostprocessingstage stage_api );
RTresult _rtPostProcessingStageGetContext( RTpostprocessingstage stage_api, RTcontext* c );
RTresult _rtPostProcessingStageQueryVariable( RTpostprocessingstage stage_api, const char* name, RTvariable* variable );
RTresult _rtPostProcessingStageGetVariableCount( RTpostprocessingstage stage_api, unsigned int* count );
RTresult _rtPostProcessingStageGetVariable( RTpostprocessingstage stage_api, unsigned int index, RTvariable* variable );
RTresult _rtProgramCreateFromPTXFile( RTcontext context_api, const char* filename, const char* program_name, RTprogram* program );
RTresult _rtProgramCreateFromPTXFiles( RTcontext context_api, unsigned int n, const char** filenames, const char* program_name, RTprogram* program );
RTresult _rtProgramCreateFromPTXString( RTcontext context_api, const char* ptx, const char* program_name, RTprogram* program );
RTresult _rtProgramCreateFromPTXStrings( RTcontext context_api, unsigned int n, const char** ptx_strings, const char* program_name, RTprogram* program );
RTresult _rtProgramCreateFromProgram( RTcontext context_api, RTprogram program_in_api, RTprogram *program_out );
RTresult _rtProgramDeclareVariable( RTprogram program_api, const char* name, RTvariable* v );
RTresult _rtProgramDestroy( RTprogram program_api );
RTresult _rtProgramGetContext( RTprogram program_api, RTcontext* c );
RTresult _rtProgramGetId( RTprogram program_api, int* program_id );
RTresult _rtProgramGetVariable( RTprogram program_api, unsigned int index, RTvariable* v );
RTresult _rtProgramGetVariableCount( RTprogram program_api, unsigned int* c );
RTresult _rtProgramQueryVariable( RTprogram program_api, const char* name, RTvariable* v );
RTresult _rtProgramRemoveVariable( RTprogram program_api, RTvariable v_api );
RTresult _rtProgramValidate( RTprogram program_api );
RTresult _rtProgramCallsiteSetPotentialCallees( RTprogram program_api, const char* name, const int* ids, int numIds );
RTresult _rtSelectorCreate( RTcontext context_api, RTselector* selector );
RTresult _rtSelectorDeclareVariable( RTselector selector_api, const char* name, RTvariable* v );
RTresult _rtSelectorDestroy( RTselector selector_api );
RTresult _rtSelectorGetChild( RTselector selector_api, unsigned int index, RTobject* child );
RTresult _rtSelectorGetChildCount( RTselector selector_api, unsigned int* count );
RTresult _rtSelectorGetChildType( RTselector selector_api, unsigned int index, RTobjecttype* type );
RTresult _rtSelectorGetContext( RTselector selector_api, RTcontext* c );
RTresult _rtSelectorGetVariable( RTselector selector_api, unsigned int index, RTvariable* v );
RTresult _rtSelectorGetVariableCount( RTselector selector_api, unsigned int* c );
RTresult _rtSelectorGetVisitProgram( RTselector selector_api, RTprogram* program );
RTresult _rtSelectorQueryVariable( RTselector selector_api, const char* name, RTvariable* v );
RTresult _rtSelectorRemoveVariable( RTselector selector_api, RTvariable v_api );
RTresult _rtSelectorSetChild( RTselector selector_api, unsigned int index, RTobject child );
RTresult _rtSelectorSetChildCount( RTselector selector_api, unsigned int count );
RTresult _rtSelectorSetVisitProgram( RTselector selector_api, RTprogram program_api );
RTresult _rtSelectorValidate( RTselector selector_api );
RTresult _rtSetLibraryVariant( bool isLibraryFromSdk );
RTresult _rtSupportsLwrrentDriver();
RTresult _rtTextureSamplerCreate( RTcontext context_api, RTtexturesampler* textureSampler );
RTresult _rtTextureSamplerCreateFromGLImage( RTcontext context_api, unsigned int gl_id, RTgltarget target, RTtexturesampler* textureSampler );
RTresult _rtTextureSamplerDestroy( RTtexturesampler textureSampler_api );
RTresult _rtTextureSamplerGLRegister( RTtexturesampler textureSampler_api );
RTresult _rtTextureSamplerGLUnregister( RTtexturesampler textureSampler_api );
RTresult _rtTextureSamplerGetArraySize( RTtexturesampler textureSampler_api, unsigned int* deprecated );
RTresult _rtTextureSamplerGetBuffer( RTtexturesampler textureSampler_api, unsigned int deprecated0, unsigned int deprecated1, RTbuffer* buffer );
RTresult _rtTextureSamplerGetContext( RTtexturesampler textureSampler_api, RTcontext* c );
RTresult _rtTextureSamplerGetFilteringModes( RTtexturesampler textureSampler_api, RTfiltermode* minFilter, RTfiltermode* magFilter, RTfiltermode* mipFilter );
RTresult _rtTextureSamplerGetGLImageId( RTtexturesampler textureSampler_api, unsigned int* gl_id );
RTresult _rtTextureSamplerGetId( RTtexturesampler textureSampler_api, int* texture_id );
RTresult _rtTextureSamplerGetIndexingMode( RTtexturesampler textureSampler_api, RTtextureindexmode* indexmode );
RTresult _rtTextureSamplerGetMaxAnisotropy( RTtexturesampler textureSampler_api, float* maxAnisotropy );
RTresult _rtTextureSamplerGetMipLevelClamp( RTtexturesampler textureSampler_api, float* minLevel, float* maxLevel );
RTresult _rtTextureSamplerGetMipLevelBias( RTtexturesampler textureSampler_api, float* bias );
RTresult _rtTextureSamplerGetMipLevelCount( RTtexturesampler textureSampler_api, unsigned int* deprecated );
RTresult _rtTextureSamplerGetReadMode( RTtexturesampler textureSampler_api, RTtexturereadmode* readmode );
RTresult _rtTextureSamplerGetWrapMode( RTtexturesampler textureSampler_api, unsigned int dim, RTwrapmode* wm );
RTresult _rtTextureSamplerSetArraySize( RTtexturesampler textureSampler_api, unsigned int deprecated );
RTresult _rtTextureSamplerSetBuffer( RTtexturesampler textureSampler_api, unsigned int deprecated0, unsigned int deprecated1, RTbuffer buffer_api );
RTresult _rtTextureSamplerSetFilteringModes( RTtexturesampler textureSampler_api, RTfiltermode minFilter, RTfiltermode magFilter, RTfiltermode mipFilter );
RTresult _rtTextureSamplerSetIndexingMode( RTtexturesampler textureSampler_api, RTtextureindexmode indexmode );
RTresult _rtTextureSamplerSetMaxAnisotropy( RTtexturesampler textureSampler_api, float maxAnisotropy );
RTresult _rtTextureSamplerSetMipLevelClamp( RTtexturesampler textureSampler_api, float minLevel, float maxLevel );
RTresult _rtTextureSamplerSetMipLevelBias( RTtexturesampler textureSampler_api, float bias );
RTresult _rtTextureSamplerSetMipLevelCount( RTtexturesampler textureSampler_api, unsigned int deprecated );
RTresult _rtTextureSamplerSetReadMode( RTtexturesampler textureSampler_api, RTtexturereadmode readmode );
RTresult _rtTextureSamplerSetWrapMode( RTtexturesampler textureSampler_api, unsigned int dim, RTwrapmode wm );
RTresult _rtTextureSamplerValidate( RTtexturesampler textureSampler_api );
RTresult _rtTransformCreate( RTcontext context_api, RTtransform* transform );
RTresult _rtTransformDestroy( RTtransform transform_api );
RTresult _rtTransformGetChild( RTtransform transform_api, RTobject* child );
RTresult _rtTransformGetChildType( RTtransform transform_api, RTobjecttype* type );
RTresult _rtTransformGetContext( RTtransform transform_api, RTcontext* c );
RTresult _rtTransformGetMatrix( RTtransform transform_api, int transpose, float* matrix, float* ilwerse_matrix );
RTresult _rtTransformSetChild( RTtransform transform_api, RTobject child );
RTresult _rtTransformSetMatrix( RTtransform transform_api, int transpose, const float* matrix, const float* ilwerse_matrix );
RTresult _rtTransformSetMotionRange( RTtransform transform_api, float timeBegin, float timeEnd );
RTresult _rtTransformGetMotionRange( RTtransform transform_api, float* timeBegin, float* timeEnd );
RTresult _rtTransformSetMotionBorderMode( RTtransform transform_api, RTmotionbordermode beginMode, RTmotionbordermode endMode );
RTresult _rtTransformGetMotionBorderMode( RTtransform transform_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode );
RTresult _rtTransformSetMotionKeys( RTtransform transform_api, unsigned int n, RTmotionkeytype type, const float* keys );
RTresult _rtTransformGetMotionKeyCount( RTtransform transform_api, unsigned int* n );
RTresult _rtTransformGetMotionKeyType( RTtransform transform_api, RTmotionkeytype* type );
RTresult _rtTransformGetMotionKeys( RTtransform transform_api, float* keys );
RTresult _rtTransformValidate( RTtransform transform_api );
RTresult _rtVariableGet1f( RTvariable v, float* f1 );
RTresult _rtVariableGet1fv( RTvariable v, float* f );
RTresult _rtVariableGet1i( RTvariable v, int* i1 );
RTresult _rtVariableGet1iv( RTvariable v, int* i );
RTresult _rtVariableGet1ui( RTvariable v, unsigned int* ull1 );
RTresult _rtVariableGet1uiv( RTvariable v, unsigned int* ull );
RTresult _rtVariableGet1ll( RTvariable v, long long* ll1 );
RTresult _rtVariableGet1llv( RTvariable v, long long* ll );
RTresult _rtVariableGet1ull( RTvariable v, unsigned long long* ull1 );
RTresult _rtVariableGet1ullv( RTvariable v, unsigned long long* ull );
RTresult _rtVariableGet2f( RTvariable v, float* f1, float* f2 );
RTresult _rtVariableGet2fv( RTvariable v, float* f );
RTresult _rtVariableGet2i( RTvariable v, int* i1, int* i2 );
RTresult _rtVariableGet2iv( RTvariable v, int* i );
RTresult _rtVariableGet2ui( RTvariable v, unsigned int* ull1, unsigned int* ull2 );
RTresult _rtVariableGet2uiv( RTvariable v, unsigned int* ull );
RTresult _rtVariableGet2ll( RTvariable v, long long* ll1, long long* ll2 );
RTresult _rtVariableGet2llv( RTvariable v, long long* ll );
RTresult _rtVariableGet2ull( RTvariable v, unsigned long long* ull1, unsigned long long* ull2 );
RTresult _rtVariableGet2ullv( RTvariable v, unsigned long long* ull );
RTresult _rtVariableGet3f( RTvariable v, float* f1, float* f2, float* f3 );
RTresult _rtVariableGet3fv( RTvariable v, float* f );
RTresult _rtVariableGet3i( RTvariable v, int* i1, int* i2, int* i3 );
RTresult _rtVariableGet3iv( RTvariable v, int* i );
RTresult _rtVariableGet3ui( RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3 );
RTresult _rtVariableGet3uiv( RTvariable v, unsigned int* u );
RTresult _rtVariableGet3ll( RTvariable v, long long* ll1, long long* ll2, long long* ll3 );
RTresult _rtVariableGet3llv( RTvariable v, long long* ll );
RTresult _rtVariableGet3ull( RTvariable v, unsigned long long* ull1, unsigned long long* ull2, unsigned long long* ull3 );
RTresult _rtVariableGet3ullv( RTvariable v, unsigned long long* ull );
RTresult _rtVariableGet4f( RTvariable v, float* f1, float* f2, float* f3, float* f4 );
RTresult _rtVariableGet4fv( RTvariable v, float* f );
RTresult _rtVariableGet4i( RTvariable v, int* i1, int* i2, int* i3, int* i4 );
RTresult _rtVariableGet4iv( RTvariable v, int* i );
RTresult _rtVariableGet4ui( RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3, unsigned int* u4 );
RTresult _rtVariableGet4uiv( RTvariable v, unsigned int* u );
RTresult _rtVariableGet4ll( RTvariable v, long long* ll1, long long* ll2, long long* ll3, long long* ll4 );
RTresult _rtVariableGet4llv( RTvariable v, long long* ll );
RTresult _rtVariableGet4ull( RTvariable v, unsigned long long* ull1, unsigned long long* ull2, unsigned long long* ull3, unsigned long long* ull4 );
RTresult _rtVariableGet4ullv( RTvariable v, unsigned long long* ull );
RTresult _rtVariableGetAnnotation( RTvariable v, const char** annotation_return );
RTresult _rtVariableGetContext( RTvariable v, RTcontext* context );
RTresult _rtVariableGetMatrix2x2fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetMatrix2x3fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetMatrix2x4fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetMatrix3x2fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetMatrix3x3fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetMatrix3x4fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetMatrix4x2fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetMatrix4x3fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetMatrix4x4fv( RTvariable v, int transpose, float* m );
RTresult _rtVariableGetName( RTvariable v, const char** name_return );
RTresult _rtVariableGetObject( RTvariable v, RTobject* object );
RTresult _rtVariableGetSize( RTvariable v, RTsize* size );
RTresult _rtVariableGetType( RTvariable v, RTobjecttype* type_return );
RTresult _rtVariableGetUserData( RTvariable v, RTsize size, void* ptr );
RTresult _rtVariableSet1f( RTvariable v, float f1 );
RTresult _rtVariableSet1fv( RTvariable v, const float* f );
RTresult _rtVariableSet1i( RTvariable v, int i1 );
RTresult _rtVariableSet1iv( RTvariable v, const int* i );
RTresult _rtVariableSet1ui( RTvariable v, unsigned int u1 );
RTresult _rtVariableSet1uiv( RTvariable v, const unsigned int* u );
RTresult _rtVariableSet1ll( RTvariable v, long long i1 );
RTresult _rtVariableSet1llv( RTvariable v, const long long* i );
RTresult _rtVariableSet1ull( RTvariable v, unsigned long long u1 );
RTresult _rtVariableSet1ullv( RTvariable v, const unsigned long long* u );
RTresult _rtVariableSet2f( RTvariable v, float f1, float f2 );
RTresult _rtVariableSet2fv( RTvariable v, const float* f );
RTresult _rtVariableSet2i( RTvariable v, int i1, int i2 );
RTresult _rtVariableSet2iv( RTvariable v, const int* i );
RTresult _rtVariableSet2ui( RTvariable v, unsigned int u1, unsigned int u2 );
RTresult _rtVariableSet2uiv( RTvariable v, const unsigned int* u );
RTresult _rtVariableSet2ll( RTvariable v, long long i1, long long i2 );
RTresult _rtVariableSet2llv( RTvariable v, const long long* i );
RTresult _rtVariableSet2ull( RTvariable v, unsigned long long u1, unsigned long long u2 );
RTresult _rtVariableSet2ullv( RTvariable v, const unsigned long long* u );
RTresult _rtVariableSet3f( RTvariable v, float f1, float f2, float f3 );
RTresult _rtVariableSet3fv( RTvariable v, const float* f );
RTresult _rtVariableSet3i( RTvariable v, int i1, int i2, int i3 );
RTresult _rtVariableSet3iv( RTvariable v, const int* i );
RTresult _rtVariableSet3ui( RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3 );
RTresult _rtVariableSet3uiv( RTvariable v, const unsigned int* u );
RTresult _rtVariableSet3ll( RTvariable v, long long i1, long long i2, long long i3 );
RTresult _rtVariableSet3llv( RTvariable v, const long long* i );
RTresult _rtVariableSet3ull( RTvariable v, unsigned long long u1, unsigned long long u2, unsigned long long u3 );
RTresult _rtVariableSet3ullv( RTvariable v, const unsigned long long* u );
RTresult _rtVariableSet4f( RTvariable v, float f1, float f2, float f3, float f4 );
RTresult _rtVariableSet4fv( RTvariable v, const float* f );
RTresult _rtVariableSet4i( RTvariable v, int i1, int i2, int i3, int i4 );
RTresult _rtVariableSet4iv( RTvariable v, const int* i );
RTresult _rtVariableSet4ui( RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4 );
RTresult _rtVariableSet4uiv( RTvariable v, const unsigned int* u );
RTresult _rtVariableSet4ll( RTvariable v, long long i1, long long i2, long long i3, long long i4 );
RTresult _rtVariableSet4llv( RTvariable v, const long long* i );
RTresult _rtVariableSet4ull( RTvariable v, unsigned long long u1, unsigned long long u2, unsigned long long u3, unsigned long long u4 );
RTresult _rtVariableSet4ullv( RTvariable v, const unsigned long long* u );
RTresult _rtVariableSetMatrix2x2fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetMatrix2x3fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetMatrix2x4fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetMatrix3x2fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetMatrix3x3fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetMatrix3x4fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetMatrix4x2fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetMatrix4x3fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetMatrix4x4fv( RTvariable v, int transpose, const float* m );
RTresult _rtVariableSetObject( RTvariable v, RTobject object );
RTresult _rtVariableSetUserData( RTvariable v, RTsize size, const void* ptr );
#ifdef _WIN32  // windows-only section
RTresult _rtDeviceGetWGLDevice( int* device, HGPULW hGpu );
RTresult _rtBufferCreateFromD3D10Resource( RTcontext context_api, unsigned int type, ID3D10Resource* pResource, RTbuffer* buffer );
RTresult _rtBufferCreateFromD3D11Resource( RTcontext context_api, unsigned int type, ID3D11Resource* pResource, RTbuffer* buffer );
RTresult _rtBufferCreateFromD3D9Resource( RTcontext context_api, unsigned int type, IDirect3DResource9* pResource, RTbuffer* buffer );
RTresult _rtBufferD3D10Register( RTbuffer buffer );
RTresult _rtBufferD3D10Unregister( RTbuffer buffer );
RTresult _rtBufferD3D11Register( RTbuffer buffer );
RTresult _rtBufferD3D11Unregister( RTbuffer buffer );
RTresult _rtBufferD3D9Register( RTbuffer buffer );
RTresult _rtBufferD3D9Unregister( RTbuffer buffer );
RTresult _rtBufferGetD3D10Resource( RTbuffer buffer_api, ID3D10Resource** pResource );
RTresult _rtBufferGetD3D11Resource( RTbuffer buffer_api, ID3D11Resource** pResource );
RTresult _rtBufferGetD3D9Resource( RTbuffer buffer_api, IDirect3DResource9** pResource );
RTresult _rtContextSetD3D10Device( RTcontext context_api, ID3D10Device* matchingDevice );
RTresult _rtContextSetD3D11Device( RTcontext context_api, ID3D11Device* matchingDevice );
RTresult _rtContextSetD3D9Device( RTcontext context_api, IDirect3DDevice9* matchingDevice );
RTresult _rtDeviceGetD3D9Device( int* device, const char* pszAdapterName );
RTresult _rtDeviceGetD3D10Device( int* device, IDXGIAdapter* pAdapter );
RTresult _rtDeviceGetD3D11Device( int* device, IDXGIAdapter* pAdapter );
RTresult _rtTextureSamplerCreateFromD3D10Resource( RTcontext context_api, ID3D10Resource* pResource, RTtexturesampler* textureSampler );
RTresult _rtTextureSamplerCreateFromD3D11Resource( RTcontext context_api, ID3D11Resource* pResource, RTtexturesampler* textureSampler );
RTresult _rtTextureSamplerCreateFromD3D9Resource( RTcontext context_api, IDirect3DResource9* pResource, RTtexturesampler* textureSampler );
RTresult _rtTextureSamplerD3D10Register( RTtexturesampler textureSampler );
RTresult _rtTextureSamplerD3D10Unregister( RTtexturesampler textureSampler );
RTresult _rtTextureSamplerD3D11Register( RTtexturesampler textureSampler );
RTresult _rtTextureSamplerD3D11Unregister( RTtexturesampler textureSampler );
RTresult _rtTextureSamplerD3D9Register( RTtexturesampler textureSampler );
RTresult _rtTextureSamplerD3D9Unregister( RTtexturesampler textureSampler );
RTresult _rtTextureSamplerGetD3D10Resource( RTtexturesampler textureSampler_api, ID3D10Resource** pResource );
RTresult _rtTextureSamplerGetD3D11Resource( RTtexturesampler textureSampler_api, ID3D11Resource** pResource );
RTresult _rtTextureSamplerGetD3D9Resource( RTtexturesampler textureSampler_api, IDirect3DResource9** pResource );
#endif
// clang-format on
