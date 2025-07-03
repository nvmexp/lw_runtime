/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
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

#ifdef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#include "../optix.h"
#else // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#include "../o6/optix.h"
#endif // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD

/* Define the version of the ABI between the wrapper and lwoptix. Note that this needs to be changed even if the actual
   API is still the same but something is changed which makes things incompatible. */
#define LWOPTIX_ABI_VERSION 18

typedef RTresult ( *PRTACCELERATIONCREATE )( RTcontext context_api, RTacceleration* acceleration );
typedef RTresult ( *PRTACCELERATIONDESTROY )( RTacceleration acceleration_api );
typedef RTresult ( *PRTACCELERATIONGETBUILDER )( RTacceleration acceleration_api, const char** return_string );
typedef RTresult ( *PRTACCELERATIONGETCONTEXT )( RTacceleration acceleration_api, RTcontext* c );
typedef RTresult ( *PRTACCELERATIONGETDATA )( RTacceleration acceleration_api, void* data );
typedef RTresult ( *PRTACCELERATIONGETDATASIZE )( RTacceleration acceleration_api, RTsize* size );
typedef RTresult ( *PRTACCELERATIONGETPROPERTY )( RTacceleration acceleration_api, const char* name, const char** return_string );
typedef RTresult ( *PRTACCELERATIONGETTRAVERSER )( RTacceleration acceleration_api, const char** return_string );
typedef RTresult ( *PRTACCELERATIONISDIRTY )( RTacceleration acceleration_api, int* dirty );
typedef RTresult ( *PRTACCELERATIONMARKDIRTY )( RTacceleration acceleration_api );
typedef RTresult ( *PRTACCELERATIONSETBUILDER )( RTacceleration acceleration_api, const char* builder );
typedef RTresult ( *PRTACCELERATIONSETDATA )( RTacceleration acceleration_api, const void* data, RTsize size );
typedef RTresult ( *PRTACCELERATIONSETPROPERTY )( RTacceleration acceleration_api, const char* name, const char* value );
typedef RTresult ( *PRTACCELERATIONSETTRAVERSER )( RTacceleration acceleration_api, const char* traverser );
typedef RTresult ( *PRTACCELERATIOLWALIDATE )( RTacceleration acceleration_api );
typedef RTresult ( *PRTBUFFERCREATE )( RTcontext context_api, unsigned int type, RTbuffer* buffer );
typedef RTresult ( *PRTBUFFERCREATEFORLWDA )( RTcontext context_api, unsigned int type, RTbuffer* buffer );
typedef RTresult ( *PRTBUFFERCREATEFROMGLBO )( RTcontext context_api, unsigned int type, unsigned int gl_id, RTbuffer* buffer );
typedef RTresult ( *PRTBUFFERCREATEFROMCALLBACK ) ( RTcontext context_api, unsigned int bufferDesc, RTbuffercallback callback, void* callbackContext, RTbuffer* buffer );
typedef RTresult ( *PRTBUFFERDESTROY )( RTbuffer buffer_api );
typedef RTresult ( *PRTBUFFERGLREGISTER )( RTbuffer buffer_api );
typedef RTresult ( *PRTBUFFERGLUNREGISTER )( RTbuffer buffer_api );
typedef RTresult ( *PRTBUFFERGETCONTEXT )( RTbuffer buffer_api, RTcontext* c );
typedef RTresult ( *PRTBUFFERGETDEVICEPOINTER )( RTbuffer buffer_api, int optix_device_ordinal, void** device_pointer );
typedef RTresult ( *PRTBUFFERGETDIMENSIONALITY )( RTbuffer buffer_api, unsigned int* dimensionality );
typedef RTresult ( *PRTBUFFERGETELEMENTSIZE )( RTbuffer buffer_api, RTsize* size_of_element );
typedef RTresult ( *PRTBUFFERGETFORMAT )( RTbuffer buffer_api, RTformat* format );
typedef RTresult ( *PRTBUFFERGETGLBOID )( RTbuffer buffer_api, unsigned int* gl_id );
typedef RTresult ( *PRTBUFFERGETID )( RTbuffer buffer_api, int* buffer_id );
typedef RTresult ( *PRTBUFFERGETMIPLEVELCOUNT )( RTbuffer buffer_api, unsigned int* level );
typedef RTresult ( *PRTBUFFERGETMIPLEVELSIZE1D )( RTbuffer buffer_api, unsigned int level, RTsize* width );
typedef RTresult ( *PRTBUFFERGETMIPLEVELSIZE2D )( RTbuffer buffer_api, unsigned int level, RTsize* width, RTsize* height );
typedef RTresult ( *PRTBUFFERGETMIPLEVELSIZE3D )( RTbuffer buffer_api, unsigned int level, RTsize* width, RTsize* height, RTsize* depth );
typedef RTresult ( *PRTBUFFERGETSIZE1D )( RTbuffer buffer_api, RTsize* width );
typedef RTresult ( *PRTBUFFERGETSIZE2D )( RTbuffer buffer_api, RTsize* width, RTsize* height );
typedef RTresult ( *PRTBUFFERGETSIZE3D )( RTbuffer buffer_api, RTsize* width, RTsize* height, RTsize* depth );
typedef RTresult ( *PRTBUFFERGETSIZEV )( RTbuffer buffer_api, unsigned int maxdim, RTsize* outdims );
typedef RTresult ( *PRTBUFFERMAP )( RTbuffer buffer_api, void** user_pointer );
typedef RTresult ( *PRTBUFFERMAPEX )( RTbuffer buffer, unsigned int map_flags, unsigned int level, void* user_owned, void** optix_owned );
typedef RTresult ( *PRTBUFFERMARKDIRTY )( RTbuffer buffer_api );
typedef RTresult ( *PRTBUFFERSETDEVICEPOINTER )( RTbuffer buffer_api, int optix_device_ordinal, void* device_pointer );
typedef RTresult ( *PRTBUFFERSETELEMENTSIZE )( RTbuffer buffer_api, RTsize size_of_element );
typedef RTresult ( *PRTBUFFERSETFORMAT )( RTbuffer buffer_api, RTformat type );
typedef RTresult ( *PRTBUFFERSETMIPLEVELCOUNT )( RTbuffer buffer_api, unsigned int levels );
typedef RTresult ( *PRTBUFFERSETSIZE1D )( RTbuffer buffer_api, RTsize width );
typedef RTresult ( *PRTBUFFERSETSIZE2D )( RTbuffer buffer_api, RTsize width, RTsize height );
typedef RTresult ( *PRTBUFFERSETSIZE3D )( RTbuffer buffer_api, RTsize width, RTsize height, RTsize depth );
typedef RTresult ( *PRTBUFFERSETSIZEV )( RTbuffer buffer_api, unsigned int dimensionality, const RTsize* indims );
typedef RTresult ( *PRTBUFFERUNMAP )( RTbuffer buffer_api );
typedef RTresult ( *PRTBUFFERUNMAPEX )( RTbuffer buffer_api, unsigned int level );
typedef RTresult ( *PRTBUFFERVALIDATE )( RTbuffer buffer_api );
typedef RTresult ( *PRTBUFFERGETPROGRESSIVEUPDATEREADY )( RTbuffer buffer_api, int* ready, unsigned int* subframe_count, unsigned int* max_subframes );
typedef RTresult ( *PRTBUFFERBINDPROGRESSIVESTREAM )( RTbuffer buffer_api, RTbuffer source );
typedef RTresult ( *PRTBUFFERSETATTRIBUTE )( RTbuffer buffer, RTbufferattribute attrib, RTsize size, const void* p );
typedef RTresult ( *PRTBUFFERGETATTRIBUTE )( RTbuffer buffer, RTbufferattribute attrib, RTsize size, void* p );
typedef RTresult ( *PRTCOMMANDLISTCREATE )( RTcontext context_api, RTcommandlist* list );
typedef RTresult ( *PRTCOMMANDLISTDESTROY )( RTcommandlist list_api );
typedef RTresult ( *PRTCOMMANDLISTAPPENDPOSTPROCESSINGSTAGE )( RTcommandlist         list_api,
                                                               RTpostprocessingstage stage_api,
                                                               RTsize                launch_width,
                                                               RTsize                launch_height );
typedef RTresult ( *PRTCOMMANDLISTAPPENDLAUNCH1D )( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width );
typedef RTresult ( *PRTCOMMANDLISTAPPENDLAUNCH2D )( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width, RTsize launch_height );
typedef RTresult ( *PRTCOMMANDLISTAPPENDLAUNCH3D )( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width, RTsize launch_height, RTsize launch_depth );
typedef RTresult ( *PRTCOMMANDLISTSETDEVICES )(RTcommandlist list_api, unsigned int count, const int* devices);
typedef RTresult ( *PRTCOMMANDLISTGETDEVICES )(RTcommandlist list_api, int* devices);
typedef RTresult ( *PRTCOMMANDLISTGETDEVICECOUNT )(RTcommandlist list_api, unsigned int* count);
typedef RTresult ( *PRTCOMMANDLISTFINALIZE )(RTcommandlist list_api);
typedef RTresult ( *PRTCOMMANDLISTEXELWTE )(RTcommandlist list_api);
typedef RTresult ( *PRTCOMMANDLISTSETLWDASTREAM )(RTcommandlist list_api, void* stream);
typedef RTresult ( *PRTCOMMANDLISTGETLWDASTREAM )(RTcommandlist list_api, void** stream);
typedef RTresult ( *PRTCOMMANDLISTGETCONTEXT )( RTcommandlist list_api, RTcontext* c );
typedef RTresult ( *PRTCONTEXTCOMPILE )( RTcontext context_api );
typedef RTresult ( *PRTCONTEXTCREATE )( RTcontext* context );
typedef RTresult ( *PRTCONTEXTDECLAREVARIABLE )( RTcontext context_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTCONTEXTDESTROY )( RTcontext context_api );
typedef RTresult ( *PRTCONTEXTSETATTRIBUTE )( RTcontext context_api, RTcontextattribute attrib, RTsize size, const void* p );
typedef RTresult ( *PRTCONTEXTGETATTRIBUTE )( RTcontext context_api, RTcontextattribute attrib, RTsize size, void* p );
typedef RTresult ( *PRTCONTEXTGETBUFFERFROMID )( RTcontext context_api, int buffer_id, RTbuffer* buffer );
typedef RTresult ( *PRTCONTEXTGETDEVICECOUNT )( RTcontext context_api, unsigned int* count );
typedef RTresult ( *PRTCONTEXTGETDEVICES )( RTcontext context_api, int* devices );
typedef RTresult ( *PRTCONTEXTGETENTRYPOINTCOUNT )( RTcontext context_api, unsigned int* num_entry_points );
typedef void ( *PRTCONTEXTGETERRORSTRING )( RTcontext context_api, RTresult code, const char** return_string );
typedef RTresult ( *PRTCONTEXTGETEXCEPTIONENABLED )( RTcontext context_api, RTexception exception, int* enabled );
typedef RTresult ( *PRTCONTEXTGETEXCEPTIONPROGRAM )( RTcontext context_api, unsigned int entry_point_index, RTprogram* program );
typedef RTresult ( *PRTCONTEXTGETMAXCALLABLEPROGRAMDEPTH )( RTcontext context_api, unsigned int* max_depth );
typedef RTresult ( *PRTCONTEXTGETMAXTRACEDEPTH )( RTcontext context_api, unsigned int* max_depth );
typedef RTresult ( *PRTCONTEXTGETMISSPROGRAM )( RTcontext context_api, unsigned int ray_type_index, RTprogram* program );
typedef RTresult ( *PRTCONTEXTGETPRINTBUFFERSIZE )( RTcontext context_api, RTsize* buffer_size_bytes );
typedef RTresult ( *PRTCONTEXTGETPRINTENABLED )( RTcontext context_api, int* enabled );
typedef RTresult ( *PRTCONTEXTGETPRINTLAUNCHINDEX )( RTcontext context_api, int* x, int* y, int* z );
typedef RTresult ( *PRTCONTEXTGETPROGRAMFROMID )( RTcontext context_api, int program_id, RTprogram* program );
typedef RTresult ( *PRTCONTEXTGETRAYGENERATIONPROGRAM )( RTcontext context_api, unsigned int entry_point_index, RTprogram* program );
typedef RTresult ( *PRTCONTEXTGETRAYTYPECOUNT )( RTcontext context_api, unsigned int* num_ray_types );
typedef RTresult ( *PRTCONTEXTGETRUNNINGSTATE )( RTcontext context_api, int* running );
typedef RTresult ( *PRTCONTEXTGETSTACKSIZE )( RTcontext context_api, RTsize* stack_size_bytes );
typedef RTresult ( *PRTCONTEXTGETTEXTURESAMPLERFROMID )( RTcontext context_api, int sampler_id, RTtexturesampler* sampler );
typedef RTresult ( *PRTCONTEXTGETVARIABLE )( RTcontext context_api, unsigned int index, RTvariable* v );
typedef RTresult ( *PRTCONTEXTGETVARIABLECOUNT )( RTcontext context_api, unsigned int* c );
typedef RTresult ( *PRTCONTEXTLAUNCH1D )( RTcontext context_api, unsigned int entry_point_index, RTsize image_width );
typedef RTresult ( *PRTCONTEXTLAUNCH2D )( RTcontext context_api, unsigned int entry_point_index, RTsize image_width, RTsize image_height );
typedef RTresult ( *PRTCONTEXTLAUNCH3D )( RTcontext context_api, unsigned int entry_point_index, RTsize image_width, RTsize image_height, RTsize image_depth );
typedef RTresult ( *PRTCONTEXTLAUNCHPROGRESSIVE2D )( RTcontext    context_api,
                                                     unsigned int entry_point_index,
                                                     RTsize       image_width,
                                                     RTsize       image_height,
                                                     unsigned int max_subframes );
typedef RTresult ( *PRTCONTEXTSTOPPROGRESSIVE )( RTcontext context_api );
typedef RTresult ( *PRTCONTEXTQUERYVARIABLE )( RTcontext context_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTCONTEXTREMOVEVARIABLE )( RTcontext context_api, RTvariable v_api );
typedef RTresult ( *PRTCONTEXTSETDEVICES )( RTcontext context_api, unsigned int count, const int* devices );
typedef RTresult ( *PRTCONTEXTSETENTRYPOINTCOUNT )( RTcontext context_api, unsigned int num_entry_points );
typedef RTresult ( *PRTCONTEXTSETEXCEPTIONENABLED )( RTcontext context_api, RTexception exception, int enabled );
typedef RTresult ( *PRTCONTEXTSETEXCEPTIONPROGRAM )( RTcontext context_api, unsigned int entry_point_index, RTprogram program_api );
typedef RTresult ( *PRTCONTEXTSETMAXCALLABLEPROGRAMDEPTH )( RTcontext context_api, unsigned int max_depth );
typedef RTresult ( *PRTCONTEXTSETMAXTRACEDEPTH )( RTcontext context_api, unsigned int max_depth );
typedef RTresult ( *PRTCONTEXTSETMISSPROGRAM )( RTcontext context_api, unsigned int ray_type_index, RTprogram program_api );
typedef RTresult ( *PRTCONTEXTSETPRINTBUFFERSIZE )( RTcontext context_api, RTsize buffer_size_bytes );
typedef RTresult ( *PRTCONTEXTSETPRINTENABLED )( RTcontext context_api, int enabled );
typedef RTresult ( *PRTCONTEXTSETPRINTLAUNCHINDEX )( RTcontext context_api, int x, int y, int z );
typedef RTresult ( *PRTCONTEXTSETRAYGENERATIONPROGRAM )( RTcontext context_api, unsigned int entry_point_index, RTprogram program_api );
typedef RTresult ( *PRTCONTEXTSETRAYTYPECOUNT )( RTcontext context_api, unsigned int num_ray_types );
typedef RTresult ( *PRTCONTEXTSETSTACKSIZE )( RTcontext context_api, RTsize stack_size_bytes );
typedef RTresult ( *PRTCONTEXTSETTIMEOUTCALLBACK )( RTcontext context_api, RTtimeoutcallback callback, double seconds );
typedef RTresult ( *PRTCONTEXTSETUSAGEREPORTCALLBACK )( RTcontext context_api, RTusagereportcallback callback, int verbosity, void* cbdata );
typedef RTresult ( *PRTCONTEXTVALIDATE )( RTcontext context_api );
typedef RTresult ( *PRTDEVICEGETATTRIBUTE )( int ordinal, RTdeviceattribute attrib, RTsize size, void* p );
typedef RTresult ( *PRTDEVICEGETDEVICECOUNT )( unsigned int* count );
typedef RTresult ( *PRTGEOMETRYCREATE )( RTcontext context_api, RTgeometry* geometry );
typedef RTresult ( *PRTGEOMETRYDECLAREVARIABLE )( RTgeometry geometry_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTGEOMETRYDESTROY )( RTgeometry geometry_api );
typedef RTresult ( *PRTGEOMETRYGETBOUNDINGBOXPROGRAM )( RTgeometry geometry_api, RTprogram* program );
typedef RTresult ( *PRTGEOMETRYGETCONTEXT )( RTgeometry geometry_api, RTcontext* c );
typedef RTresult ( *PRTGEOMETRYGETINTERSECTIONPROGRAM )( RTgeometry geometry_api, RTprogram* program );
typedef RTresult ( *PRTGEOMETRYGETPRIMITIVECOUNT )( RTgeometry geometry_api, unsigned int* num_primitives );
typedef RTresult ( *PRTGEOMETRYGETPRIMITIVEINDEXOFFSET )( RTgeometry geometry_api, unsigned int* index_offset );
typedef RTresult ( *PRTGEOMETRYGETVARIABLE )( RTgeometry geometry_api, unsigned int index, RTvariable* v );
typedef RTresult ( *PRTGEOMETRYGETVARIABLECOUNT )( RTgeometry geometry_api, unsigned int* c );
typedef RTresult ( *PRTGEOMETRYGROUPCREATE )( RTcontext context_api, RTgeometrygroup* geometrygroup );
typedef RTresult ( *PRTGEOMETRYGROUPDESTROY )( RTgeometrygroup geometrygroup_api );
typedef RTresult ( *PRTGEOMETRYGROUPGETACCELERATION )( RTgeometrygroup geometrygroup_api, RTacceleration* acceleration );
typedef RTresult ( *PRTGEOMETRYGROUPGETCHILD )( RTgeometrygroup geometrygroup_api, unsigned int index, RTgeometryinstance* geometryinstance );
typedef RTresult ( *PRTGEOMETRYGROUPGETCHILDCOUNT )( RTgeometrygroup geometrygroup_api, unsigned int* count );
typedef RTresult ( *PRTGEOMETRYGROUPGETCONTEXT )( RTgeometrygroup geometrygroup_api, RTcontext* c );
typedef RTresult ( *PRTGEOMETRYGROUPSETACCELERATION )( RTgeometrygroup geometrygroup_api, RTacceleration acceleration_api );
typedef RTresult ( *PRTGEOMETRYGROUPSETCHILD )( RTgeometrygroup geometrygroup_api, unsigned int index, RTgeometryinstance geometryinstance_api );
typedef RTresult ( *PRTGEOMETRYGROUPSETCHILDCOUNT )( RTgeometrygroup geometrygroup_api, unsigned int count );
typedef RTresult ( *PRTGEOMETRYGROUPVALIDATE )( RTgeometrygroup geometrygroup_api );
typedef RTresult ( *PRTGEOMETRYINSTANCECREATE )( RTcontext context_api, RTgeometryinstance* geometryinstance );
typedef RTresult ( *PRTGEOMETRYINSTANCEDECLAREVARIABLE )( RTgeometryinstance gi_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTGEOMETRYINSTANCEDESTROY )( RTgeometryinstance geometryinstance_api );
typedef RTresult ( *PRTGEOMETRYINSTANCEGETCONTEXT )( RTgeometryinstance gi_api, RTcontext* c );
typedef RTresult ( *PRTGEOMETRYINSTANCEGETGEOMETRY )( RTgeometryinstance gi_api, RTgeometry* geo );
typedef RTresult ( *PRTGEOMETRYINSTANCEGETGEOMETRYTRIANGLES )( RTgeometryinstance gi_api, RTgeometrytriangles* geo );
typedef RTresult ( *PRTGEOMETRYINSTANCEGETMATERIAL )( RTgeometryinstance gi_api, unsigned int material_idx, RTmaterial* mat );
typedef RTresult ( *PRTGEOMETRYINSTANCEGETMATERIALCOUNT )( RTgeometryinstance gi_api, unsigned int* num_materials );
typedef RTresult ( *PRTGEOMETRYINSTANCEGETVARIABLE )( RTgeometryinstance gi_api, unsigned int index, RTvariable* v );
typedef RTresult ( *PRTGEOMETRYINSTANCEGETVARIABLECOUNT )( RTgeometryinstance gi_api, unsigned int* c );
typedef RTresult ( *PRTGEOMETRYINSTANCEQUERYVARIABLE )( RTgeometryinstance gi_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTGEOMETRYINSTANCEREMOVEVARIABLE )( RTgeometryinstance gi_api, RTvariable v_api );
typedef RTresult ( *PRTGEOMETRYINSTANCESETGEOMETRY )( RTgeometryinstance gi_api, RTgeometry geo_api );
typedef RTresult ( *PRTGEOMETRYINSTANCESETGEOMETRYTRIANGLES )( RTgeometryinstance gi_api, RTgeometrytriangles geo_api );
typedef RTresult ( *PRTGEOMETRYINSTANCESETMATERIAL )( RTgeometryinstance gi_api, unsigned int material_idx, RTmaterial mat_api );
typedef RTresult ( *PRTGEOMETRYINSTANCESETMATERIALCOUNT )( RTgeometryinstance gi_api, unsigned int num_materials );
typedef RTresult ( *PRTGEOMETRYINSTANCEVALIDATE )( RTgeometryinstance geometryinstance_api );
typedef RTresult ( *PRTGEOMETRYISDIRTY )( RTgeometry geometry_api, int* dirty );
typedef RTresult ( *PRTGEOMETRYMARKDIRTY )( RTgeometry geometry_api );
typedef RTresult ( *PRTGEOMETRYQUERYVARIABLE )( RTgeometry geometry_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTGEOMETRYREMOVEVARIABLE )( RTgeometry geometry_api, RTvariable v_api );
typedef RTresult ( *PRTGEOMETRYSETBOUNDINGBOXPROGRAM )( RTgeometry geometry_api, RTprogram program_api );
typedef RTresult ( *PRTGEOMETRYSETINTERSECTIONPROGRAM )( RTgeometry geometry_api, RTprogram program_api );
typedef RTresult ( *PRTGEOMETRYSETPRIMITIVECOUNT )( RTgeometry geometry_api, unsigned int num_primitives );
typedef RTresult ( *PRTGEOMETRYSETPRIMITIVEINDEXOFFSET )( RTgeometry geometry_api, unsigned int index_offset );
typedef RTresult ( *PRTGEOMETRYSETMOTIONRANGE )( RTgeometry geometry_api, float timeBegin, float timeEnd );
typedef RTresult ( *PRTGEOMETRYGETMOTIONRANGE )( RTgeometry geometry_api, float* timeBegin, float* timeEnd );
typedef RTresult ( *PRTGEOMETRYSETMOTIONBORDERMODE )( RTgeometry geometry_api, RTmotionbordermode beginMode, RTmotionbordermode endMode );
typedef RTresult ( *PRTGEOMETRYGETMOTIONBORDERMODE )( RTgeometry geometry_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode );
typedef RTresult ( *PRTGEOMETRYSETMOTIONSTEPS )( RTgeometry geometry_api, unsigned int n );
typedef RTresult ( *PRTGEOMETRYGETMOTIONSTEPS )( RTgeometry geometry_api, unsigned int* n );
typedef RTresult ( *PRTGEOMETRYVALIDATE )( RTgeometry geometry_api );
typedef RTresult ( *PRTGEOMETRYTRIANGLESCREATE )( RTcontext context_api, RTgeometrytriangles* geometrytriangles );
typedef RTresult ( *PRTGEOMETRYTRIANGLESDESTROY )( RTgeometrytriangles geometrytriangles_api );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETCONTEXT )( RTgeometrytriangles geometrytriangles_api, RTcontext* context );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETPRIMITIVEINDEXOFFSET )( RTgeometrytriangles geometrytriangles_api,
                                                                   unsigned int*       index_offset );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETPRIMITIVEINDEXOFFSET )( RTgeometrytriangles geometrytriangles_api, unsigned int index_offset );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETPRETRANSFORMMATRIX )( RTgeometrytriangles geometrytriangles_api,
                                                                 int                 transpose,
                                                                 const float*        matrix );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETPRETRANSFORMMATRIX )( RTgeometrytriangles geometrytriangles_api, int transpose, float* matrix );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETPRIMITIVECOUNT )( RTgeometrytriangles geometrytriangles_api, unsigned int num_triangles );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETPRIMITIVECOUNT )( RTgeometrytriangles geometrytriangles_api, unsigned int* num_triangles );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETINDEXEDTRIANGLESDEPRECATED )( RTgeometrytriangles  geometrytriangles_api,
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
                                                                         RTgeometrybuildflags build_flags );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETTRIANGLESDEPRECATED )( RTgeometrytriangles  geometrytriangles_api,
                                                                  unsigned int         num_triangles,
                                                                  RTbuffer             vertex_buffer_api,
                                                                  RTsize               vertex_buffer_byte_offset,
                                                                  RTsize               vertex_byte_stride,
                                                                  RTformat             position_format,
                                                                  RTgeometrybuildflags build_flags );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETTRIANGLESINDICES )( RTgeometrytriangles geometrytriangles_api,
                                                               RTbuffer            index_buffer_api,
                                                               RTsize              index_buffer_byte_offset,
                                                               RTsize              tri_indices_byte_stride,
                                                               RTformat            tri_indices_format );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETVERTICES )( RTgeometrytriangles geometrytriangles_api,
                                                       unsigned int        num_vertices,
                                                       RTbuffer            vertex_buffer_api,
                                                       RTsize              vertex_buffer_byte_offset,
                                                       RTsize              vertex_byte_stride,
                                                       RTformat            position_format );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETMOTIOLWERTICES )( RTgeometrytriangles geometrytriangles_api,
                                                             unsigned int        num_vertices,
                                                             RTbuffer            vertex_buffer_api,
                                                             RTsize              vertex_buffer_byte_offset,
                                                             RTsize              vertex_byte_stride,
                                                             RTsize              vertex_motion_step_byte_stride,
                                                             RTformat            position_format );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETMOTIOLWERTICESMULTIBUFFERDEPRECATED )( RTgeometrytriangles geometrytriangles_api,
                                                                                  unsigned int        num_vertices,
                                                                                  RTbuffer*           vertex_buffers,
                                                                                  RTsize   vertex_buffer_byte_offset,
                                                                                  RTsize   vertex_byte_stride,
                                                                                  RTformat position_format );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETMOTIOLWERTICESMULTIBUFFER )( RTgeometrytriangles geometrytriangles_api,
                                                                        unsigned int        num_vertices,
                                                                        RTbuffer*           vertex_buffers,
                                                                        unsigned int        vertex_buffer_count,
                                                                        RTsize              vertex_buffer_byte_offset,
                                                                        RTsize              vertex_byte_stride,
                                                                        RTformat            position_format );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETMOTIONSTEPS )( RTgeometrytriangles geometrytriangles_api, unsigned int num_motion_steps );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETMOTIONSTEPS )( RTgeometrytriangles geometrytriangles_api, unsigned int* num_motion_steps );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETMOTIONRANGE )( RTgeometrytriangles geometrytriangles_api, float timeBegin, float timeEnd );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETMOTIONRANGE )( RTgeometrytriangles geometrytriangles_api, float* timeBegin, float* timeEnd );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETMOTIONBORDERMODE )( RTgeometrytriangles geometrytriangles_api,
                                                               RTmotionbordermode  beginMode,
                                                               RTmotionbordermode  endMode );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETMOTIONBORDERMODE )( RTgeometrytriangles geometrytriangles_api,
                                                               RTmotionbordermode* beginMode,
                                                               RTmotionbordermode* endMode );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETBUILDFLAGS )( RTgeometrytriangles geometrytriangles, RTgeometrybuildflags build_flags );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETMATERIALCOUNT )( RTgeometrytriangles geometrytriangles, unsigned int num_materials );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETMATERIALCOUNT )( RTgeometrytriangles geometrytriangles, unsigned int* num_materials );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETMATERIALINDICES )( RTgeometrytriangles geometrytriangles,
                                                              RTbuffer            material_index_buffer,
                                                              RTsize              material_index_buffer_byte_offset,
                                                              RTsize              material_index_byte_stride,
                                                              RTformat            material_index_format );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETFLAGSPERMATERIAL )( RTgeometrytriangles geometrytriangles,
                                                               unsigned int        material_index,
                                                               RTgeometryflags     flags );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETFLAGSPERMATERIAL )( RTgeometrytriangles geometrytriangles,
                                                               unsigned int        material_index,
                                                               RTgeometryflags*    flags );
typedef RTresult ( *PRTGEOMETRYTRIANGLESVALIDATE )( RTgeometrytriangles geometrytriangles_api );
typedef RTresult ( *PRTGEOMETRYTRIANGLESSETATTRIBUTEPROGRAM )( RTgeometrytriangles geometrytriangles_api, RTprogram program );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETATTRIBUTEPROGRAM )( RTgeometrytriangles geometrytriangles_api, RTprogram* program );
typedef RTresult ( *PRTGEOMETRYTRIANGLESDECLAREVARIABLE )( RTgeometrytriangles geometrytriangles, const char* name, RTvariable* v );
typedef RTresult ( *PRTGEOMETRYTRIANGLESQUERYVARIABLE )( RTgeometrytriangles geometrytriangles, const char* name, RTvariable* v );
typedef RTresult ( *PRTGEOMETRYTRIANGLESREMOVEVARIABLE )( RTgeometrytriangles geometrytriangles, RTvariable v );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETVARIABLECOUNT )( RTgeometrytriangles geometrytriangles, unsigned int* count );
typedef RTresult ( *PRTGEOMETRYTRIANGLESGETVARIABLE )( RTgeometrytriangles geometrytriangles, unsigned int index, RTvariable* v );
typedef RTresult ( *PRTGETVERSION )( unsigned int* version );
typedef RTresult ( *PRTGLOBALSETATTRIBUTE )( RTglobalattribute attrib, RTsize size, const void* p );
typedef RTresult ( *PRTGLOBALGETATTRIBUTE )( RTglobalattribute attrib, RTsize size, void* p );
typedef RTresult ( *PRTGROUPCREATE )( RTcontext context_api, RTgroup* group );
typedef RTresult ( *PRTGROUPDESTROY )( RTgroup group_api );
typedef RTresult ( *PRTGROUPGETACCELERATION )( RTgroup group_api, RTacceleration* acceleration );
typedef RTresult ( *PRTGROUPGETCHILD )( RTgroup group_api, unsigned int index, RTobject* child );
typedef RTresult ( *PRTGROUPGETCHILDCOUNT )( RTgroup group_api, unsigned int* count );
typedef RTresult ( *PRTGROUPGETCHILDTYPE )( RTgroup group_api, unsigned int index, RTobjecttype* type );
typedef RTresult ( *PRTGROUPGETCONTEXT )( RTgroup group_api, RTcontext* c );
typedef RTresult ( *PRTGROUPSETACCELERATION )( RTgroup group_api, RTacceleration acceleration_api );
typedef RTresult ( *PRTGROUPSETCHILD )( RTgroup group_api, unsigned int index, RTobject child );
typedef RTresult ( *PRTGROUPSETCHILDCOUNT )( RTgroup group_api, unsigned int count );
typedef RTresult ( *PRTGROUPVALIDATE )( RTgroup group_api );
typedef RTresult ( *PRTGROUPSETVISIBLITYMASK )( RTgroup group, RTvisibilitymask visibility_mask );
typedef RTresult ( *PRTGROUPGETVISIBLITYMASK )( RTgroup group, RTvisibilitymask* visibility_mask );
typedef RTresult ( *PRTGEOMETRYGROUPSETFLAGS )( RTgeometrygroup geometrygroup, RTinstanceflags instance_flags );
typedef RTresult ( *PRTGEOMETRYGROUPGETFLAGS )( RTgeometrygroup geometrygroup, RTinstanceflags* instance_flags );
typedef RTresult ( *PRTGEOMETRYGROUPSETVISIBILITYMASK )( RTgeometrygroup geometrygroup, RTvisibilitymask mask );
typedef RTresult ( *PRTGEOMETRYGROUPGETVISIBILITYMASK )( RTgeometrygroup geometrygroup, RTvisibilitymask* mask );
typedef RTresult ( *PRTGEOMETRYSETFLAGS )( RTgeometry geometry, RTgeometryflags flags );
typedef RTresult ( *PRTGEOMETRYGETFLAGS )( RTgeometry geometry, RTgeometryflags* flags );
typedef RTresult ( *PRTMATERIALCREATE )( RTcontext context_api, RTmaterial* material );
typedef RTresult ( *PRTMATERIALDECLAREVARIABLE )( RTmaterial material_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTMATERIALDESTROY )( RTmaterial material_api );
typedef RTresult ( *PRTMATERIALGETANYHITPROGRAM )( RTmaterial material_api, unsigned int ray_type_index, RTprogram* program );
typedef RTresult ( *PRTMATERIALGETCLOSESTHITPROGRAM )( RTmaterial material_api, unsigned int ray_type_index, RTprogram* program );
typedef RTresult ( *PRTMATERIALGETCONTEXT )( RTmaterial material_api, RTcontext* c );
typedef RTresult ( *PRTMATERIALGETVARIABLE )( RTmaterial material_api, unsigned int index, RTvariable* v );
typedef RTresult ( *PRTMATERIALGETVARIABLECOUNT )( RTmaterial material_api, unsigned int* c );
typedef RTresult ( *PRTMATERIALQUERYVARIABLE )( RTmaterial material_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTMATERIALREMOVEVARIABLE )( RTmaterial material_api, RTvariable v_api );
typedef RTresult ( *PRTMATERIALSETANYHITPROGRAM )( RTmaterial material_api, unsigned int ray_type_index, RTprogram program_api );
typedef RTresult ( *PRTMATERIALSETCLOSESTHITPROGRAM )( RTmaterial material_api, unsigned int ray_type_index, RTprogram program_api );
typedef RTresult ( *PRTMATERIALVALIDATE )( RTmaterial material_api );
typedef RTresult ( *PRTPOSTPROCESSINGSTAGECREATEBUILTIN )( RTcontext              context_api,
                                                           const char*            builtin_name,
                                                           void*                  denoiser,
                                                           void*                  ssim_predictor,
                                                           RTpostprocessingstage* stage_api );
typedef RTresult ( *PRTPOSTPROCESSINGSTAGEDECLAREVARIABLE )( RTpostprocessingstage stage_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTPOSTPROCESSINGSTAGEDESTROY )( RTpostprocessingstage stage_api );
typedef RTresult ( *PRTPOSTPROCESSINGSTAGEGETCONTEXT )( RTpostprocessingstage stage_api, RTcontext* c );
typedef RTresult ( *PRTPOSTPROCESSINGSTAGEQUERYVARIABLE )( RTpostprocessingstage stage_api, const char* name, RTvariable* variable );
typedef RTresult ( *PRTPOSTPROCESSINGSTAGEGETVARIABLECOUNT )( RTpostprocessingstage stage_api, unsigned int* count );
typedef RTresult ( *PRTPOSTPROCESSINGSTAGEGETVARIABLE )( RTpostprocessingstage stage_api, unsigned int index, RTvariable* variable );
typedef RTresult ( *PRTPROGRAMCREATEFROMPTXFILE )( RTcontext context_api, const char* filename, const char* program_name, RTprogram* program );
typedef RTresult ( *PRTPROGRAMCREATEFROMPTXFILES )( RTcontext    context_api,
                                                    unsigned int n,
                                                    const char** filenames,
                                                    const char*  program_name,
                                                    RTprogram*   program );
typedef RTresult ( *PRTPROGRAMCREATEFROMPTXSTRING )( RTcontext context_api, const char* ptx, const char* program_name, RTprogram* program );
typedef RTresult ( *PRTPROGRAMCREATEFROMPTXSTRINGS )( RTcontext    context_api,
                                                      unsigned int n,
                                                      const char** ptx_strings,
                                                      const char*  program_name,
                                                      RTprogram*   program );
typedef RTresult ( *PRTPROGRAMCREATEFROMPROGRAM )( RTcontext  context_api,
                                                   RTprogram  program_in,
                                                   RTprogram* program_out );
typedef RTresult ( *PRTPROGRAMDECLAREVARIABLE )( RTprogram program_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTPROGRAMDESTROY )( RTprogram program_api );
typedef RTresult ( *PRTPROGRAMGETCONTEXT )( RTprogram program_api, RTcontext* c );
typedef RTresult ( *PRTPROGRAMGETID )( RTprogram program_api, int* program_id );
typedef RTresult ( *PRTPROGRAMGETVARIABLE )( RTprogram program_api, unsigned int index, RTvariable* v );
typedef RTresult ( *PRTPROGRAMGETVARIABLECOUNT )( RTprogram program_api, unsigned int* c );
typedef RTresult ( *PRTPROGRAMQUERYVARIABLE )( RTprogram program_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTPROGRAMREMOVEVARIABLE )( RTprogram program_api, RTvariable v_api );
typedef RTresult ( *PRTPROGRAMVALIDATE )( RTprogram program_api );
typedef RTresult ( *PRTPROGRAMCALLSITESETPOTENTIALCALLEES )( RTprogram program_api, const char* csName, const int* calleeIds, int numIds );
typedef RTresult ( *PRTSELECTORCREATE )( RTcontext context_api, RTselector* selector );
typedef RTresult ( *PRTSELECTORDECLAREVARIABLE )( RTselector selector_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTSELECTORDESTROY )( RTselector selector_api );
typedef RTresult ( *PRTSELECTORGETCHILD )( RTselector selector_api, unsigned int index, RTobject* child );
typedef RTresult ( *PRTSELECTORGETCHILDCOUNT )( RTselector selector_api, unsigned int* count );
typedef RTresult ( *PRTSELECTORGETCHILDTYPE )( RTselector selector_api, unsigned int index, RTobjecttype* type );
typedef RTresult ( *PRTSELECTORGETCONTEXT )( RTselector selector_api, RTcontext* c );
typedef RTresult ( *PRTSELECTORGETVARIABLE )( RTselector selector_api, unsigned int index, RTvariable* v );
typedef RTresult ( *PRTSELECTORGETVARIABLECOUNT )( RTselector selector_api, unsigned int* c );
typedef RTresult ( *PRTSELECTORGETVISITPROGRAM )( RTselector selector_api, RTprogram* program );
typedef RTresult ( *PRTSELECTORQUERYVARIABLE )( RTselector selector_api, const char* name, RTvariable* v );
typedef RTresult ( *PRTSELECTORREMOVEVARIABLE )( RTselector selector_api, RTvariable v_api );
typedef RTresult ( *PRTSELECTORSETCHILD )( RTselector selector_api, unsigned int index, RTobject child );
typedef RTresult ( *PRTSELECTORSETCHILDCOUNT )( RTselector selector_api, unsigned int count );
typedef RTresult ( *PRTSELECTORSETVISITPROGRAM )( RTselector selector_api, RTprogram program_api );
typedef RTresult ( *PRTSELECTORVALIDATE )( RTselector selector_api );

typedef RTresult ( *PRTTEXTURESAMPLERCREATE )( RTcontext context_api, RTtexturesampler* textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERCREATEFROMGLIMAGE )( RTcontext context_api, unsigned int gl_id, RTgltarget target, RTtexturesampler* textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERDESTROY )( RTtexturesampler textureSampler_api );
typedef RTresult ( *PRTTEXTURESAMPLERGLREGISTER )( RTtexturesampler textureSampler_api );
typedef RTresult ( *PRTTEXTURESAMPLERGLUNREGISTER )( RTtexturesampler textureSampler_api );
typedef RTresult ( *PRTTEXTURESAMPLERGETARRAYSIZE )( RTtexturesampler textureSampler_api, unsigned int* deprecated );
typedef RTresult ( *PRTTEXTURESAMPLERGETBUFFER )( RTtexturesampler textureSampler_api,
                                                  unsigned int     deprecated0,
                                                  unsigned int     deprecated1,
                                                  RTbuffer*        buffer );
typedef RTresult ( *PRTTEXTURESAMPLERGETCONTEXT )( RTtexturesampler textureSampler_api, RTcontext* c );
typedef RTresult ( *PRTTEXTURESAMPLERGETFILTERINGMODES )( RTtexturesampler textureSampler_api,
                                                          RTfiltermode*    minFilter,
                                                          RTfiltermode*    magFilter,
                                                          RTfiltermode*    mipFilter );
typedef RTresult ( *PRTTEXTURESAMPLERGETGLIMAGEID )( RTtexturesampler textureSampler_api, unsigned int* gl_id );
typedef RTresult ( *PRTTEXTURESAMPLERGETID )( RTtexturesampler textureSampler_api, int* texture_id );
typedef RTresult ( *PRTTEXTURESAMPLERGETINDEXINGMODE )( RTtexturesampler textureSampler_api, RTtextureindexmode* indexmode );
typedef RTresult ( *PRTTEXTURESAMPLERGETMAXANISOTROPY )( RTtexturesampler textureSampler_api, float* maxAnisotropy );
typedef RTresult ( *PRTTEXTURESAMPLERGETMIPLEVELCLAMP )( RTtexturesampler textureSampler_api, float* minLevel, float* maxLevel );
typedef RTresult ( *PRTTEXTURESAMPLERGETMIPLEVELBIAS )( RTtexturesampler textureSampler_api, float* bias );
typedef RTresult ( *PRTTEXTURESAMPLERGETMIPLEVELCOUNT )( RTtexturesampler textureSampler_api, unsigned int* deprecated );
typedef RTresult ( *PRTTEXTURESAMPLERGETREADMODE )( RTtexturesampler textureSampler_api, RTtexturereadmode* readmode );
typedef RTresult ( *PRTTEXTURESAMPLERGETWRAPMODE )( RTtexturesampler textureSampler_api, unsigned int dim, RTwrapmode* wm );
typedef RTresult ( *PRTTEXTURESAMPLERSETARRAYSIZE )( RTtexturesampler textureSampler_api, unsigned int deprecated );
typedef RTresult ( *PRTTEXTURESAMPLERSETBUFFER )( RTtexturesampler textureSampler_api,
                                                  unsigned int     deprecated0,
                                                  unsigned int     deprecated1,
                                                  RTbuffer         buffer_api );
typedef RTresult ( *PRTTEXTURESAMPLERSETFILTERINGMODES )( RTtexturesampler textureSampler_api,
                                                          RTfiltermode     minFilter,
                                                          RTfiltermode     magFilter,
                                                          RTfiltermode     mipFilter );
typedef RTresult ( *PRTTEXTURESAMPLERSETINDEXINGMODE )( RTtexturesampler textureSampler_api, RTtextureindexmode indexmode );
typedef RTresult ( *PRTTEXTURESAMPLERSETMAXANISOTROPY )( RTtexturesampler textureSampler_api, float maxAnisotropy );
typedef RTresult ( *PRTTEXTURESAMPLERSETMIPLEVELCLAMP )( RTtexturesampler textureSampler_api, float minLevel, float maxLevel );
typedef RTresult ( *PRTTEXTURESAMPLERSETMIPLEVELBIAS )( RTtexturesampler textureSampler_api, float bias );
typedef RTresult ( *PRTTEXTURESAMPLERSETMIPLEVELCOUNT )( RTtexturesampler textureSampler_api, unsigned int deprecated );
typedef RTresult ( *PRTTEXTURESAMPLERSETREADMODE )( RTtexturesampler textureSampler_api, RTtexturereadmode readmode );
typedef RTresult ( *PRTTEXTURESAMPLERSETWRAPMODE )( RTtexturesampler textureSampler_api, unsigned int dim, RTwrapmode wm );
typedef RTresult ( *PRTTEXTURESAMPLERVALIDATE )( RTtexturesampler textureSampler_api );
typedef RTresult ( *PRTTRANSFORMCREATE )( RTcontext context_api, RTtransform* transform );
typedef RTresult ( *PRTTRANSFORMDESTROY )( RTtransform transform_api );
typedef RTresult ( *PRTTRANSFORMGETCHILD )( RTtransform transform_api, RTobject* child );
typedef RTresult ( *PRTTRANSFORMGETCHILDTYPE )( RTtransform transform_api, RTobjecttype* type );
typedef RTresult ( *PRTTRANSFORMGETCONTEXT )( RTtransform transform_api, RTcontext* c );
typedef RTresult ( *PRTTRANSFORMGETMATRIX )( RTtransform transform_api, int transpose, float* matrix, float* ilwerse_matrix );
typedef RTresult ( *PRTTRANSFORMSETCHILD )( RTtransform transform_api, RTobject child );
typedef RTresult ( *PRTTRANSFORMSETMATRIX )( RTtransform transform_api, int transpose, const float* matrix, const float* ilwerse_matrix );
typedef RTresult ( *PRTTRANSFORMSETMOTIONRANGE )( RTtransform transform_api, float timeBegin, float timeEnd );
typedef RTresult ( *PRTTRANSFORMGETMOTIONRANGE )( RTtransform transform_api, float* timeBegin, float* timeEnd );
typedef RTresult ( *PRTTRANSFORMSETMOTIONBORDERMODE )( RTtransform transform_api, RTmotionbordermode beginMode, RTmotionbordermode endMode );
typedef RTresult ( *PRTTRANSFORMGETMOTIONBORDERMODE )( RTtransform transform_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode );
typedef RTresult ( *PRTTRANSFORMSETMOTIONKEYS )( RTtransform transform_api, unsigned int n, RTmotionkeytype type, const float* keys );
typedef RTresult ( *PRTTRANSFORMGETMOTIONKEYCOUNT )( RTtransform transform_api, unsigned int* n );
typedef RTresult ( *PRTTRANSFORMGETMOTIONKEYTYPE )( RTtransform transform_api, RTmotionkeytype* type );
typedef RTresult ( *PRTTRANSFORMGETMOTIONKEYS )( RTtransform transform_api, float* keys );
typedef RTresult ( *PRTTRANSFORMVALIDATE )( RTtransform transform_api );
typedef RTresult ( *PRTVARIABLEGET1F )( RTvariable v, float* f1 );
typedef RTresult ( *PRTVARIABLEGET1FV )( RTvariable v, float* f );
typedef RTresult ( *PRTVARIABLEGET1I )( RTvariable v, int* i1 );
typedef RTresult ( *PRTVARIABLEGET1IV )( RTvariable v, int* i );
typedef RTresult ( *PRTVARIABLEGET1UI )( RTvariable v, unsigned int* ull1 );
typedef RTresult ( *PRTVARIABLEGET1UIV )( RTvariable v, unsigned int* ull );
typedef RTresult ( *PRTVARIABLEGET1LL )( RTvariable v, long long* ll1 );
typedef RTresult ( *PRTVARIABLEGET1LLV )( RTvariable v, long long* ll );
typedef RTresult ( *PRTVARIABLEGET1ULL )( RTvariable v, unsigned long long* ull1 );
typedef RTresult ( *PRTVARIABLEGET1ULLV )( RTvariable v, unsigned long long* ull );
typedef RTresult ( *PRTVARIABLEGET2F )( RTvariable v, float* f1, float* f2 );
typedef RTresult ( *PRTVARIABLEGET2FV )( RTvariable v, float* f );
typedef RTresult ( *PRTVARIABLEGET2I )( RTvariable v, int* i1, int* i2 );
typedef RTresult ( *PRTVARIABLEGET2IV )( RTvariable v, int* i );
typedef RTresult ( *PRTVARIABLEGET2UI )( RTvariable v, unsigned int* ull1, unsigned int* ull2 );
typedef RTresult ( *PRTVARIABLEGET2UIV )( RTvariable v, unsigned int* ull );
typedef RTresult ( *PRTVARIABLEGET2LL )( RTvariable v, long long* ll1, long long* ll2 );
typedef RTresult ( *PRTVARIABLEGET2LLV )( RTvariable v, long long* ll );
typedef RTresult ( *PRTVARIABLEGET2ULL )( RTvariable v, unsigned long long* ull1, unsigned long long* ull2 );
typedef RTresult ( *PRTVARIABLEGET2ULLV )( RTvariable v, unsigned long long* ull );
typedef RTresult ( *PRTVARIABLEGET3F )( RTvariable v, float* f1, float* f2, float* f3 );
typedef RTresult ( *PRTVARIABLEGET3FV )( RTvariable v, float* f );
typedef RTresult ( *PRTVARIABLEGET3I )( RTvariable v, int* i1, int* i2, int* i3 );
typedef RTresult ( *PRTVARIABLEGET3IV )( RTvariable v, int* i );
typedef RTresult ( *PRTVARIABLEGET3UI )( RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3 );
typedef RTresult ( *PRTVARIABLEGET3UIV )( RTvariable v, unsigned int* u );
typedef RTresult ( *PRTVARIABLEGET3LL )( RTvariable v, long long* ll1, long long* ll2, long long* ll3 );
typedef RTresult ( *PRTVARIABLEGET3LLV )( RTvariable v, long long* ll );
typedef RTresult ( *PRTVARIABLEGET3ULL )( RTvariable v, unsigned long long* ull1, unsigned long long* ull2, unsigned long long* ull3 );
typedef RTresult ( *PRTVARIABLEGET3ULLV )( RTvariable v, unsigned long long* ull );
typedef RTresult ( *PRTVARIABLEGET4F )( RTvariable v, float* f1, float* f2, float* f3, float* f4 );
typedef RTresult ( *PRTVARIABLEGET4FV )( RTvariable v, float* f );
typedef RTresult ( *PRTVARIABLEGET4I )( RTvariable v, int* i1, int* i2, int* i3, int* i4 );
typedef RTresult ( *PRTVARIABLEGET4IV )( RTvariable v, int* i );
typedef RTresult ( *PRTVARIABLEGET4UI )( RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3, unsigned int* u4 );
typedef RTresult ( *PRTVARIABLEGET4UIV )( RTvariable v, unsigned int* u );
typedef RTresult ( *PRTVARIABLEGET4LL )( RTvariable v, long long* ll1, long long* ll2, long long* ll3, long long* ll4 );
typedef RTresult ( *PRTVARIABLEGET4LLV )( RTvariable v, long long* ll );
typedef RTresult ( *PRTVARIABLEGET4ULL )( RTvariable          v,
                                          unsigned long long* ull1,
                                          unsigned long long* ull2,
                                          unsigned long long* ull3,
                                          unsigned long long* ull4 );
typedef RTresult ( *PRTVARIABLEGET4ULLV )( RTvariable v, unsigned long long* ull );
typedef RTresult ( *PRTVARIABLEGETANNOTATION )( RTvariable v, const char** annotation_return );
typedef RTresult ( *PRTVARIABLEGETCONTEXT )( RTvariable v, RTcontext* context );
typedef RTresult ( *PRTVARIABLEGETMATRIX2X2FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETMATRIX2X3FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETMATRIX2X4FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETMATRIX3X2FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETMATRIX3X3FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETMATRIX3X4FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETMATRIX4X2FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETMATRIX4X3FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETMATRIX4X4FV )( RTvariable v, int transpose, float* m );
typedef RTresult ( *PRTVARIABLEGETNAME )( RTvariable v, const char** name_return );
typedef RTresult ( *PRTVARIABLEGETOBJECT )( RTvariable v, RTobject* object );
typedef RTresult ( *PRTVARIABLEGETSIZE )( RTvariable v, RTsize* size );
typedef RTresult ( *PRTVARIABLEGETTYPE )( RTvariable v, RTobjecttype* type_return );
typedef RTresult ( *PRTVARIABLEGETUSERDATA )( RTvariable v, RTsize size, void* ptr );
typedef RTresult ( *PRTVARIABLESET1F )( RTvariable v, float f1 );
typedef RTresult ( *PRTVARIABLESET1FV )( RTvariable v, const float* f );
typedef RTresult ( *PRTVARIABLESET1I )( RTvariable v, int i1 );
typedef RTresult ( *PRTVARIABLESET1IV )( RTvariable v, const int* i );
typedef RTresult ( *PRTVARIABLESET1UI )( RTvariable v, unsigned int u1 );
typedef RTresult ( *PRTVARIABLESET1UIV )( RTvariable v, const unsigned int* u );
typedef RTresult ( *PRTVARIABLESET1LL )( RTvariable v, long long i1 );
typedef RTresult ( *PRTVARIABLESET1LLV )( RTvariable v, const long long* i );
typedef RTresult ( *PRTVARIABLESET1ULL )( RTvariable v, unsigned long long u1 );
typedef RTresult ( *PRTVARIABLESET1ULLV )( RTvariable v, const unsigned long long* u );
typedef RTresult ( *PRTVARIABLESET2F )( RTvariable v, float f1, float f2 );
typedef RTresult ( *PRTVARIABLESET2FV )( RTvariable v, const float* f );
typedef RTresult ( *PRTVARIABLESET2I )( RTvariable v, int i1, int i2 );
typedef RTresult ( *PRTVARIABLESET2IV )( RTvariable v, const int* i );
typedef RTresult ( *PRTVARIABLESET2UI )( RTvariable v, unsigned int u1, unsigned int u2 );
typedef RTresult ( *PRTVARIABLESET2UIV )( RTvariable v, const unsigned int* u );
typedef RTresult ( *PRTVARIABLESET2LL )( RTvariable v, long long i1, long long i2 );
typedef RTresult ( *PRTVARIABLESET2LLV )( RTvariable v, const long long* i );
typedef RTresult ( *PRTVARIABLESET2ULL )( RTvariable v, unsigned long long u1, unsigned long long u2 );
typedef RTresult ( *PRTVARIABLESET2ULLV )( RTvariable v, const unsigned long long* u );
typedef RTresult ( *PRTVARIABLESET3F )( RTvariable v, float f1, float f2, float f3 );
typedef RTresult ( *PRTVARIABLESET3FV )( RTvariable v, const float* f );
typedef RTresult ( *PRTVARIABLESET3I )( RTvariable v, int i1, int i2, int i3 );
typedef RTresult ( *PRTVARIABLESET3IV )( RTvariable v, const int* i );
typedef RTresult ( *PRTVARIABLESET3UI )( RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3 );
typedef RTresult ( *PRTVARIABLESET3UIV )( RTvariable v, const unsigned int* u );
typedef RTresult ( *PRTVARIABLESET3LL )( RTvariable v, long long i1, long long i2, long long i3 );
typedef RTresult ( *PRTVARIABLESET3LLV )( RTvariable v, const long long* i );
typedef RTresult ( *PRTVARIABLESET3ULL )( RTvariable v, unsigned long long u1, unsigned long long u2, unsigned long long u3 );
typedef RTresult ( *PRTVARIABLESET3ULLV )( RTvariable v, const unsigned long long* u );
typedef RTresult ( *PRTVARIABLESET4F )( RTvariable v, float f1, float f2, float f3, float f4 );
typedef RTresult ( *PRTVARIABLESET4FV )( RTvariable v, const float* f );
typedef RTresult ( *PRTVARIABLESET4I )( RTvariable v, int i1, int i2, int i3, int i4 );
typedef RTresult ( *PRTVARIABLESET4IV )( RTvariable v, const int* i );
typedef RTresult ( *PRTVARIABLESET4UI )( RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4 );
typedef RTresult ( *PRTVARIABLESET4UIV )( RTvariable v, const unsigned int* u );
typedef RTresult ( *PRTVARIABLESET4LL )( RTvariable v, long long i1, long long i2, long long i3, long long i4 );
typedef RTresult ( *PRTVARIABLESET4LLV )( RTvariable v, const long long* i );
typedef RTresult ( *PRTVARIABLESET4ULL )( RTvariable v, unsigned long long u1, unsigned long long u2, unsigned long long u3, unsigned long long u4 );
typedef RTresult ( *PRTVARIABLESET4ULLV )( RTvariable v, const unsigned long long* u );
typedef RTresult ( *PRTVARIABLESETMATRIX2X2FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETMATRIX2X3FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETMATRIX2X4FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETMATRIX3X2FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETMATRIX3X3FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETMATRIX3X4FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETMATRIX4X2FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETMATRIX4X3FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETMATRIX4X4FV )( RTvariable v, int transpose, const float* m );
typedef RTresult ( *PRTVARIABLESETOBJECT )( RTvariable v, RTobject object );
typedef RTresult ( *PRTVARIABLESETUSERDATA )( RTvariable v, RTsize size, const void* ptr );
typedef RTresult ( *PRTLOG )( int level, const char* msg );
typedef RTresult ( *PRTGETBUILDVERSION )( const char** result );
typedef RTresult ( *PRTOVERRIDESOTHERVERSION )( const char* otherVersion, int* result );

typedef RTresult ( *PRTDEPRECATEDAPIFUNCTION )();

#ifdef _WIN32  // windows-only section
typedef struct IDirect3DDevice9   IDirect3DDevice9;
typedef struct IDirect3DResource9 IDirect3DResource9;
typedef struct IDXGIAdapter       IDXGIAdapter;
typedef struct ID3D10Device       ID3D10Device;
typedef struct ID3D10Resource     ID3D10Resource;
typedef struct IDXGIAdapter       IDXGIAdapter;
typedef struct ID3D11Device       ID3D11Device;
typedef struct ID3D11Resource     ID3D11Resource;

#if !defined( WGL_LW_gpu_affinity )
typedef void* HGPULW;
#endif

typedef RTresult ( *PRTDEVICEGETWGLDEVICE )( int* device, HGPULW hGpu );
typedef RTresult ( *PRTBUFFERCREATEFROMD3D10RESOURCE )( RTcontext context_api, unsigned int type, ID3D10Resource* pResource, RTbuffer* buffer );
typedef RTresult ( *PRTBUFFERCREATEFROMD3D11RESOURCE )( RTcontext context_api, unsigned int type, ID3D11Resource* pResource, RTbuffer* buffer );
typedef RTresult ( *PRTBUFFERCREATEFROMD3D9RESOURCE )( RTcontext context_api, unsigned int type, IDirect3DResource9* pResource, RTbuffer* buffer );
typedef RTresult ( *PRTBUFFERD3D10REGISTER )( RTbuffer buffer );
typedef RTresult ( *PRTBUFFERD3D10UNREGISTER )( RTbuffer buffer );
typedef RTresult ( *PRTBUFFERD3D11REGISTER )( RTbuffer buffer );
typedef RTresult ( *PRTBUFFERD3D11UNREGISTER )( RTbuffer buffer );
typedef RTresult ( *PRTBUFFERD3D9REGISTER )( RTbuffer buffer );
typedef RTresult ( *PRTBUFFERD3D9UNREGISTER )( RTbuffer buffer );
typedef RTresult ( *PRTBUFFERGETD3D10RESOURCE )( RTbuffer buffer_api, ID3D10Resource** pResource );
typedef RTresult ( *PRTBUFFERGETD3D11RESOURCE )( RTbuffer buffer_api, ID3D11Resource** pResource );
typedef RTresult ( *PRTBUFFERGETD3D9RESOURCE )( RTbuffer buffer_api, IDirect3DResource9** pResource );
typedef RTresult ( *PRTCONTEXTSETD3D10DEVICE )( RTcontext context_api, ID3D10Device* matchingDevice );
typedef RTresult ( *PRTCONTEXTSETD3D11DEVICE )( RTcontext context_api, ID3D11Device* matchingDevice );
typedef RTresult ( *PRTCONTEXTSETD3D9DEVICE )( RTcontext context_api, IDirect3DDevice9* matchingDevice );
typedef RTresult ( *PRTDEVICEGETD3D9DEVICE )( int* device, const char* pszAdapterName );
typedef RTresult ( *PRTDEVICEGETD3D10DEVICE )( int* device, IDXGIAdapter* pAdapter );
typedef RTresult ( *PRTDEVICEGETD3D11DEVICE )( int* device, IDXGIAdapter* pAdapter );
typedef RTresult ( *PRTTEXTURESAMPLERCREATEFROMD3D10RESOURCE )( RTcontext         context_api,
                                                                ID3D10Resource*   pResource,
                                                                RTtexturesampler* textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERCREATEFROMD3D11RESOURCE )( RTcontext         context_api,
                                                                ID3D11Resource*   pResource,
                                                                RTtexturesampler* textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERCREATEFROMD3D9RESOURCE )( RTcontext           context_api,
                                                               IDirect3DResource9* pResource,
                                                               RTtexturesampler*   textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERD3D10REGISTER )( RTtexturesampler textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERD3D10UNREGISTER )( RTtexturesampler textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERD3D11REGISTER )( RTtexturesampler textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERD3D11UNREGISTER )( RTtexturesampler textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERD3D9REGISTER )( RTtexturesampler textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERD3D9UNREGISTER )( RTtexturesampler textureSampler );
typedef RTresult ( *PRTTEXTURESAMPLERGETD3D10RESOURCE )( RTtexturesampler textureSampler_api, ID3D10Resource** pResource );
typedef RTresult ( *PRTTEXTURESAMPLERGETD3D11RESOURCE )( RTtexturesampler textureSampler_api, ID3D11Resource** pResource );
typedef RTresult ( *PRTTEXTURESAMPLERGETD3D9RESOURCE )( RTtexturesampler textureSampler_api, IDirect3DResource9** pResource );
#endif  // _WIN32

// IMPORTANT: new entries to this struct should go at the end

struct OptiXAPI
{
    unsigned int                            version;
    PRTACCELERATIONCREATE                   rtAccelerationCreate;
    PRTACCELERATIONDESTROY                  rtAccelerationDestroy;
    PRTACCELERATIONGETBUILDER               rtAccelerationGetBuilder;
    PRTACCELERATIONGETCONTEXT               rtAccelerationGetContext;
    PRTACCELERATIONGETDATA                  rtAccelerationGetData;
    PRTACCELERATIONGETDATASIZE              rtAccelerationGetDataSize;
    PRTACCELERATIONGETPROPERTY              rtAccelerationGetProperty;
    PRTACCELERATIONGETTRAVERSER             rtAccelerationGetTraverser;
    PRTACCELERATIONISDIRTY                  rtAccelerationIsDirty;
    PRTACCELERATIONMARKDIRTY                rtAccelerationMarkDirty;
    PRTACCELERATIONSETBUILDER               rtAccelerationSetBuilder;
    PRTACCELERATIONSETDATA                  rtAccelerationSetData;
    PRTACCELERATIONSETPROPERTY              rtAccelerationSetProperty;
    PRTACCELERATIONSETTRAVERSER             rtAccelerationSetTraverser;
    PRTACCELERATIOLWALIDATE                 rtAcceleratiolwalidate;
    PRTBUFFERCREATE                         rtBufferCreate;
    PRTBUFFERCREATEFORLWDA                  rtBufferCreateForLWDA;
    PRTBUFFERCREATEFROMGLBO                 rtBufferCreateFromGLBO;
    PRTBUFFERDESTROY                        rtBufferDestroy;
    PRTBUFFERGLREGISTER                     rtBufferGLRegister;
    PRTBUFFERGLUNREGISTER                   rtBufferGLUnregister;
    PRTBUFFERGETCONTEXT                     rtBufferGetContext;
    PRTBUFFERGETDEVICEPOINTER               rtBufferGetDevicePointer;
    PRTBUFFERGETDIMENSIONALITY              rtBufferGetDimensionality;
    PRTBUFFERGETELEMENTSIZE                 rtBufferGetElementSize;
    PRTBUFFERGETFORMAT                      rtBufferGetFormat;
    PRTBUFFERGETGLBOID                      rtBufferGetGLBOId;
    PRTBUFFERGETID                          rtBufferGetId;
    PRTBUFFERGETMIPLEVELCOUNT               rtBufferGetMipLevelCount;
    PRTBUFFERGETMIPLEVELSIZE1D              rtBufferGetMipLevelSize1D;
    PRTBUFFERGETMIPLEVELSIZE2D              rtBufferGetMipLevelSize2D;
    PRTBUFFERGETMIPLEVELSIZE3D              rtBufferGetMipLevelSize3D;
    PRTBUFFERGETSIZE1D                      rtBufferGetSize1D;
    PRTBUFFERGETSIZE2D                      rtBufferGetSize2D;
    PRTBUFFERGETSIZE3D                      rtBufferGetSize3D;
    PRTBUFFERGETSIZEV                       rtBufferGetSizev;
    PRTBUFFERMAP                            rtBufferMap;
    PRTBUFFERMAPEX                          rtBufferMapEx;
    PRTBUFFERMARKDIRTY                      rtBufferMarkDirty;
    PRTBUFFERSETDEVICEPOINTER               rtBufferSetDevicePointer;
    PRTBUFFERSETELEMENTSIZE                 rtBufferSetElementSize;
    PRTBUFFERSETFORMAT                      rtBufferSetFormat;
    PRTBUFFERSETMIPLEVELCOUNT               rtBufferSetMipLevelCount;
    PRTBUFFERSETSIZE1D                      rtBufferSetSize1D;
    PRTBUFFERSETSIZE2D                      rtBufferSetSize2D;
    PRTBUFFERSETSIZE3D                      rtBufferSetSize3D;
    PRTBUFFERSETSIZEV                       rtBufferSetSizev;
    PRTBUFFERUNMAP                          rtBufferUnmap;
    PRTBUFFERUNMAPEX                        rtBufferUnmapEx;
    PRTBUFFERVALIDATE                       rtBufferValidate;
    PRTBUFFERGETPROGRESSIVEUPDATEREADY      rtBufferGetProgressiveUpdateReady;
    PRTBUFFERBINDPROGRESSIVESTREAM          rtBufferBindProgressiveStream;
    PRTBUFFERSETATTRIBUTE                   rtBufferSetAttribute;
    PRTBUFFERGETATTRIBUTE                   rtBufferGetAttribute;
    PRTCOMMANDLISTCREATE                    rtCommandListCreate;
    PRTCOMMANDLISTDESTROY                   rtCommandListDestroy;
    PRTCOMMANDLISTAPPENDPOSTPROCESSINGSTAGE rtCommandListAppendPostprocessingStage;
    PRTCOMMANDLISTAPPENDLAUNCH2D            rtCommandListAppendLaunch2D;
    PRTCOMMANDLISTFINALIZE                  rtCommandListFinalize;
    PRTCOMMANDLISTEXELWTE                   rtCommandListExelwte;
    PRTCOMMANDLISTGETCONTEXT                rtCommandListGetContext;
    PRTCONTEXTCOMPILE                       rtContextCompile;
    PRTCONTEXTCREATE                        rtContextCreate;
    PRTCONTEXTDECLAREVARIABLE               rtContextDeclareVariable;
    PRTCONTEXTDESTROY                       rtContextDestroy;
    PRTCONTEXTSETATTRIBUTE                  rtContextSetAttribute;
    PRTCONTEXTGETATTRIBUTE                  rtContextGetAttribute;
    PRTCONTEXTGETBUFFERFROMID               rtContextGetBufferFromId;
    PRTCONTEXTGETDEVICECOUNT                rtContextGetDeviceCount;
    PRTCONTEXTGETDEVICES                    rtContextGetDevices;
    PRTCONTEXTGETENTRYPOINTCOUNT            rtContextGetEntryPointCount;
    PRTCONTEXTGETERRORSTRING                rtContextGetErrorString;
    PRTCONTEXTGETEXCEPTIONENABLED           rtContextGetExceptionEnabled;
    PRTCONTEXTGETEXCEPTIONPROGRAM           rtContextGetExceptionProgram;
    PRTCONTEXTGETMISSPROGRAM                rtContextGetMissProgram;
    PRTCONTEXTGETPRINTBUFFERSIZE            rtContextGetPrintBufferSize;
    PRTCONTEXTGETPRINTENABLED               rtContextGetPrintEnabled;
    PRTCONTEXTGETPRINTLAUNCHINDEX           rtContextGetPrintLaunchIndex;
    PRTCONTEXTGETPROGRAMFROMID              rtContextGetProgramFromId;
    PRTCONTEXTGETRAYGENERATIONPROGRAM       rtContextGetRayGenerationProgram;
    PRTCONTEXTGETRAYTYPECOUNT               rtContextGetRayTypeCount;
    PRTCONTEXTGETRUNNINGSTATE               rtContextGetRunningState;
    PRTCONTEXTGETSTACKSIZE                  rtContextGetStackSize;
    PRTCONTEXTGETTEXTURESAMPLERFROMID       rtContextGetTextureSamplerFromId;
    PRTCONTEXTGETVARIABLE                   rtContextGetVariable;
    PRTCONTEXTGETVARIABLECOUNT              rtContextGetVariableCount;
    PRTCONTEXTLAUNCH1D                      rtContextLaunch1D;
    PRTCONTEXTLAUNCH2D                      rtContextLaunch2D;
    PRTCONTEXTLAUNCH3D                      rtContextLaunch3D;
    PRTCONTEXTLAUNCHPROGRESSIVE2D           rtContextLaunchProgressive2D;
    PRTCONTEXTSTOPPROGRESSIVE               rtContextStopProgressive;
    PRTCONTEXTQUERYVARIABLE                 rtContextQueryVariable;
    PRTCONTEXTREMOVEVARIABLE                rtContextRemoveVariable;
    PRTCONTEXTSETDEVICES                    rtContextSetDevices;
    PRTCONTEXTSETENTRYPOINTCOUNT            rtContextSetEntryPointCount;
    PRTCONTEXTSETEXCEPTIONENABLED           rtContextSetExceptionEnabled;
    PRTCONTEXTSETEXCEPTIONPROGRAM           rtContextSetExceptionProgram;
    PRTCONTEXTSETMISSPROGRAM                rtContextSetMissProgram;
    PRTCONTEXTSETPRINTBUFFERSIZE            rtContextSetPrintBufferSize;
    PRTCONTEXTSETPRINTENABLED               rtContextSetPrintEnabled;
    PRTCONTEXTSETPRINTLAUNCHINDEX           rtContextSetPrintLaunchIndex;
    PRTCONTEXTSETRAYGENERATIONPROGRAM       rtContextSetRayGenerationProgram;
    PRTCONTEXTSETRAYTYPECOUNT               rtContextSetRayTypeCount;
    PRTDEPRECATEDAPIFUNCTION                rtContextSetRemoteDevice;
    PRTCONTEXTSETSTACKSIZE                  rtContextSetStackSize;
    PRTCONTEXTSETTIMEOUTCALLBACK            rtContextSetTimeoutCallback;
    PRTCONTEXTSETUSAGEREPORTCALLBACK        rtContextSetUsageReportCallback;
    PRTCONTEXTVALIDATE                      rtContextValidate;
    PRTDEVICEGETATTRIBUTE                   rtDeviceGetAttribute;
    PRTDEVICEGETDEVICECOUNT                 rtDeviceGetDeviceCount;
    PRTGEOMETRYCREATE                       rtGeometryCreate;
    PRTGEOMETRYDECLAREVARIABLE              rtGeometryDeclareVariable;
    PRTGEOMETRYDESTROY                      rtGeometryDestroy;
    PRTGEOMETRYGETBOUNDINGBOXPROGRAM        rtGeometryGetBoundingBoxProgram;
    PRTGEOMETRYGETCONTEXT                   rtGeometryGetContext;
    PRTGEOMETRYGETINTERSECTIONPROGRAM       rtGeometryGetIntersectionProgram;
    PRTGEOMETRYGETPRIMITIVECOUNT            rtGeometryGetPrimitiveCount;
    PRTGEOMETRYGETPRIMITIVEINDEXOFFSET      rtGeometryGetPrimitiveIndexOffset;
    PRTGEOMETRYGETVARIABLE                  rtGeometryGetVariable;
    PRTGEOMETRYGETVARIABLECOUNT             rtGeometryGetVariableCount;
    PRTGEOMETRYGROUPCREATE                  rtGeometryGroupCreate;
    PRTGEOMETRYGROUPDESTROY                 rtGeometryGroupDestroy;
    PRTGEOMETRYGROUPGETACCELERATION         rtGeometryGroupGetAcceleration;
    PRTGEOMETRYGROUPGETCHILD                rtGeometryGroupGetChild;
    PRTGEOMETRYGROUPGETCHILDCOUNT           rtGeometryGroupGetChildCount;
    PRTGEOMETRYGROUPGETCONTEXT              rtGeometryGroupGetContext;
    PRTGEOMETRYGROUPSETACCELERATION         rtGeometryGroupSetAcceleration;
    PRTGEOMETRYGROUPSETCHILD                rtGeometryGroupSetChild;
    PRTGEOMETRYGROUPSETCHILDCOUNT           rtGeometryGroupSetChildCount;
    PRTGEOMETRYGROUPVALIDATE                rtGeometryGroupValidate;
    PRTGEOMETRYINSTANCECREATE               rtGeometryInstanceCreate;
    PRTGEOMETRYINSTANCEDECLAREVARIABLE      rtGeometryInstanceDeclareVariable;
    PRTGEOMETRYINSTANCEDESTROY              rtGeometryInstanceDestroy;
    PRTGEOMETRYINSTANCEGETCONTEXT           rtGeometryInstanceGetContext;
    PRTGEOMETRYINSTANCEGETGEOMETRY          rtGeometryInstanceGetGeometry;
    PRTGEOMETRYINSTANCEGETMATERIAL          rtGeometryInstanceGetMaterial;
    PRTGEOMETRYINSTANCEGETMATERIALCOUNT     rtGeometryInstanceGetMaterialCount;
    PRTGEOMETRYINSTANCEGETVARIABLE          rtGeometryInstanceGetVariable;
    PRTGEOMETRYINSTANCEGETVARIABLECOUNT     rtGeometryInstanceGetVariableCount;
    PRTGEOMETRYINSTANCEQUERYVARIABLE        rtGeometryInstanceQueryVariable;
    PRTGEOMETRYINSTANCEREMOVEVARIABLE       rtGeometryInstanceRemoveVariable;
    PRTGEOMETRYINSTANCESETGEOMETRY          rtGeometryInstanceSetGeometry;
    PRTGEOMETRYINSTANCESETMATERIAL          rtGeometryInstanceSetMaterial;
    PRTGEOMETRYINSTANCESETMATERIALCOUNT     rtGeometryInstanceSetMaterialCount;
    PRTGEOMETRYINSTANCEVALIDATE             rtGeometryInstanceValidate;
    PRTGEOMETRYISDIRTY                      rtGeometryIsDirty;
    PRTGEOMETRYMARKDIRTY                    rtGeometryMarkDirty;
    PRTGEOMETRYQUERYVARIABLE                rtGeometryQueryVariable;
    PRTGEOMETRYREMOVEVARIABLE               rtGeometryRemoveVariable;
    PRTGEOMETRYSETBOUNDINGBOXPROGRAM        rtGeometrySetBoundingBoxProgram;
    PRTGEOMETRYSETINTERSECTIONPROGRAM       rtGeometrySetIntersectionProgram;
    PRTGEOMETRYSETPRIMITIVECOUNT            rtGeometrySetPrimitiveCount;
    PRTGEOMETRYSETPRIMITIVEINDEXOFFSET      rtGeometrySetPrimitiveIndexOffset;
    PRTGEOMETRYSETMOTIONRANGE               rtGeometrySetMotionRange;
    PRTGEOMETRYGETMOTIONRANGE               rtGeometryGetMotionRange;
    PRTGEOMETRYSETMOTIONBORDERMODE          rtGeometrySetMotionBorderMode;
    PRTGEOMETRYGETMOTIONBORDERMODE          rtGeometryGetMotionBorderMode;
    PRTGEOMETRYSETMOTIONSTEPS               rtGeometrySetMotionSteps;
    PRTGEOMETRYGETMOTIONSTEPS               rtGeometryGetMotionSteps;
    PRTGEOMETRYVALIDATE                     rtGeometryValidate;
    PRTGETVERSION                           rtGetVersion;
    PRTGLOBALSETATTRIBUTE                   rtGlobalSetAttribute;
    PRTGLOBALGETATTRIBUTE                   rtGlobalGetAttribute;
    PRTGROUPCREATE                          rtGroupCreate;
    PRTGROUPDESTROY                         rtGroupDestroy;
    PRTGROUPGETACCELERATION                 rtGroupGetAcceleration;
    PRTGROUPGETCHILD                        rtGroupGetChild;
    PRTGROUPGETCHILDCOUNT                   rtGroupGetChildCount;
    PRTGROUPGETCHILDTYPE                    rtGroupGetChildType;
    PRTGROUPGETCONTEXT                      rtGroupGetContext;
    PRTGROUPSETACCELERATION                 rtGroupSetAcceleration;
    PRTGROUPSETCHILD                        rtGroupSetChild;
    PRTGROUPSETCHILDCOUNT                   rtGroupSetChildCount;
    PRTGROUPVALIDATE                        rtGroupValidate;
    PRTMATERIALCREATE                       rtMaterialCreate;
    PRTMATERIALDECLAREVARIABLE              rtMaterialDeclareVariable;
    PRTMATERIALDESTROY                      rtMaterialDestroy;
    PRTMATERIALGETANYHITPROGRAM             rtMaterialGetAnyHitProgram;
    PRTMATERIALGETCLOSESTHITPROGRAM         rtMaterialGetClosestHitProgram;
    PRTMATERIALGETCONTEXT                   rtMaterialGetContext;
    PRTMATERIALGETVARIABLE                  rtMaterialGetVariable;
    PRTMATERIALGETVARIABLECOUNT             rtMaterialGetVariableCount;
    PRTMATERIALQUERYVARIABLE                rtMaterialQueryVariable;
    PRTMATERIALREMOVEVARIABLE               rtMaterialRemoveVariable;
    PRTMATERIALSETANYHITPROGRAM             rtMaterialSetAnyHitProgram;
    PRTMATERIALSETCLOSESTHITPROGRAM         rtMaterialSetClosestHitProgram;
    PRTMATERIALVALIDATE                     rtMaterialValidate;
    PRTPOSTPROCESSINGSTAGECREATEBUILTIN     rtPostProcessingStageCreateBuiltin;
    PRTPOSTPROCESSINGSTAGEDECLAREVARIABLE   rtPostProcessingStageDeclareVariable;
    PRTPOSTPROCESSINGSTAGEDESTROY           rtPostProcessingStageDestroy;
    PRTPOSTPROCESSINGSTAGEGETCONTEXT        rtPostProcessingStageGetContext;
    PRTPOSTPROCESSINGSTAGEQUERYVARIABLE     rtPostProcessingStageQueryVariable;
    PRTPOSTPROCESSINGSTAGEGETVARIABLECOUNT  rtPostProcessingStageGetVariableCount;
    PRTPOSTPROCESSINGSTAGEGETVARIABLE       rtPostProcessingStageGetVariable;
    PRTPROGRAMCREATEFROMPTXFILE             rtProgramCreateFromPTXFile;
    PRTPROGRAMCREATEFROMPTXSTRING           rtProgramCreateFromPTXString;
    PRTPROGRAMDECLAREVARIABLE               rtProgramDeclareVariable;
    PRTPROGRAMDESTROY                       rtProgramDestroy;
    PRTPROGRAMGETCONTEXT                    rtProgramGetContext;
    PRTPROGRAMGETID                         rtProgramGetId;
    PRTPROGRAMGETVARIABLE                   rtProgramGetVariable;
    PRTPROGRAMGETVARIABLECOUNT              rtProgramGetVariableCount;
    PRTPROGRAMQUERYVARIABLE                 rtProgramQueryVariable;
    PRTPROGRAMREMOVEVARIABLE                rtProgramRemoveVariable;
    PRTPROGRAMVALIDATE                      rtProgramValidate;
    PRTDEPRECATEDAPIFUNCTION                rtRemoteDeviceCreate;
    PRTDEPRECATEDAPIFUNCTION                rtRemoteDeviceDestroy;
    PRTDEPRECATEDAPIFUNCTION                rtRemoteDeviceGetAttribute;
    PRTDEPRECATEDAPIFUNCTION                rtRemoteDeviceRelease;
    PRTDEPRECATEDAPIFUNCTION                rtRemoteDeviceReserve;
    PRTSELECTORCREATE                       rtSelectorCreate;
    PRTSELECTORDECLAREVARIABLE              rtSelectorDeclareVariable;
    PRTSELECTORDESTROY                      rtSelectorDestroy;
    PRTSELECTORGETCHILD                     rtSelectorGetChild;
    PRTSELECTORGETCHILDCOUNT                rtSelectorGetChildCount;
    PRTSELECTORGETCHILDTYPE                 rtSelectorGetChildType;
    PRTSELECTORGETCONTEXT                   rtSelectorGetContext;
    PRTSELECTORGETVARIABLE                  rtSelectorGetVariable;
    PRTSELECTORGETVARIABLECOUNT             rtSelectorGetVariableCount;
    PRTSELECTORGETVISITPROGRAM              rtSelectorGetVisitProgram;
    PRTSELECTORQUERYVARIABLE                rtSelectorQueryVariable;
    PRTSELECTORREMOVEVARIABLE               rtSelectorRemoveVariable;
    PRTSELECTORSETCHILD                     rtSelectorSetChild;
    PRTSELECTORSETCHILDCOUNT                rtSelectorSetChildCount;
    PRTSELECTORSETVISITPROGRAM              rtSelectorSetVisitProgram;
    PRTSELECTORVALIDATE                     rtSelectorValidate;
    PRTTEXTURESAMPLERCREATE                 rtTextureSamplerCreate;
    PRTTEXTURESAMPLERCREATEFROMGLIMAGE      rtTextureSamplerCreateFromGLImage;
    PRTTEXTURESAMPLERDESTROY                rtTextureSamplerDestroy;
    PRTTEXTURESAMPLERGLREGISTER             rtTextureSamplerGLRegister;
    PRTTEXTURESAMPLERGLUNREGISTER           rtTextureSamplerGLUnregister;
    PRTTEXTURESAMPLERGETARRAYSIZE           rtTextureSamplerGetArraySize;
    PRTTEXTURESAMPLERGETBUFFER              rtTextureSamplerGetBuffer;
    PRTTEXTURESAMPLERGETCONTEXT             rtTextureSamplerGetContext;
    PRTTEXTURESAMPLERGETFILTERINGMODES      rtTextureSamplerGetFilteringModes;
    PRTTEXTURESAMPLERGETGLIMAGEID           rtTextureSamplerGetGLImageId;
    PRTTEXTURESAMPLERGETID                  rtTextureSamplerGetId;
    PRTTEXTURESAMPLERGETINDEXINGMODE        rtTextureSamplerGetIndexingMode;
    PRTTEXTURESAMPLERGETMAXANISOTROPY       rtTextureSamplerGetMaxAnisotropy;
    PRTTEXTURESAMPLERGETMIPLEVELCLAMP       rtTextureSamplerGetMipLevelClamp;
    PRTTEXTURESAMPLERGETMIPLEVELBIAS        rtTextureSamplerGetMipLevelBias;
    PRTTEXTURESAMPLERGETMIPLEVELCOUNT       rtTextureSamplerGetMipLevelCount;
    PRTTEXTURESAMPLERGETREADMODE            rtTextureSamplerGetReadMode;
    PRTTEXTURESAMPLERGETWRAPMODE            rtTextureSamplerGetWrapMode;
    PRTTEXTURESAMPLERSETARRAYSIZE           rtTextureSamplerSetArraySize;
    PRTTEXTURESAMPLERSETBUFFER              rtTextureSamplerSetBuffer;
    PRTTEXTURESAMPLERSETFILTERINGMODES      rtTextureSamplerSetFilteringModes;
    PRTTEXTURESAMPLERSETINDEXINGMODE        rtTextureSamplerSetIndexingMode;
    PRTTEXTURESAMPLERSETMAXANISOTROPY       rtTextureSamplerSetMaxAnisotropy;
    PRTTEXTURESAMPLERSETMIPLEVELCLAMP       rtTextureSamplerSetMipLevelClamp;
    PRTTEXTURESAMPLERSETMIPLEVELBIAS        rtTextureSamplerSetMipLevelBias;
    PRTTEXTURESAMPLERSETMIPLEVELCOUNT       rtTextureSamplerSetMipLevelCount;
    PRTTEXTURESAMPLERSETREADMODE            rtTextureSamplerSetReadMode;
    PRTTEXTURESAMPLERSETWRAPMODE            rtTextureSamplerSetWrapMode;
    PRTTEXTURESAMPLERVALIDATE               rtTextureSamplerValidate;
    PRTTRANSFORMCREATE                      rtTransformCreate;
    PRTTRANSFORMDESTROY                     rtTransformDestroy;
    PRTTRANSFORMGETCHILD                    rtTransformGetChild;
    PRTTRANSFORMGETCHILDTYPE                rtTransformGetChildType;
    PRTTRANSFORMGETCONTEXT                  rtTransformGetContext;
    PRTTRANSFORMGETMATRIX                   rtTransformGetMatrix;
    PRTTRANSFORMSETCHILD                    rtTransformSetChild;
    PRTTRANSFORMSETMATRIX                   rtTransformSetMatrix;
    PRTTRANSFORMSETMOTIONRANGE              rtTransformSetMotionRange;
    PRTTRANSFORMGETMOTIONRANGE              rtTransformGetMotionRange;
    PRTTRANSFORMSETMOTIONBORDERMODE         rtTransformSetMotionBorderMode;
    PRTTRANSFORMGETMOTIONBORDERMODE         rtTransformGetMotionBorderMode;
    PRTTRANSFORMSETMOTIONKEYS               rtTransformSetMotionKeys;
    PRTTRANSFORMGETMOTIONKEYCOUNT           rtTransformGetMotionKeyCount;
    PRTTRANSFORMGETMOTIONKEYTYPE            rtTransformGetMotionKeyType;
    PRTTRANSFORMGETMOTIONKEYS               rtTransformGetMotionKeys;
    PRTTRANSFORMVALIDATE                    rtTransformValidate;
    PRTVARIABLEGET1F                        rtVariableGet1f;
    PRTVARIABLEGET1FV                       rtVariableGet1fv;
    PRTVARIABLEGET1I                        rtVariableGet1i;
    PRTVARIABLEGET1IV                       rtVariableGet1iv;
    PRTVARIABLEGET1UI                       rtVariableGet1ui;
    PRTVARIABLEGET1UIV                      rtVariableGet1uiv;
    PRTVARIABLEGET1LL                       rtVariableGet1ll;
    PRTVARIABLEGET1LLV                      rtVariableGet1llv;
    PRTVARIABLEGET1ULL                      rtVariableGet1ull;
    PRTVARIABLEGET1ULLV                     rtVariableGet1ullv;
    PRTVARIABLEGET2F                        rtVariableGet2f;
    PRTVARIABLEGET2FV                       rtVariableGet2fv;
    PRTVARIABLEGET2I                        rtVariableGet2i;
    PRTVARIABLEGET2IV                       rtVariableGet2iv;
    PRTVARIABLEGET2UI                       rtVariableGet2ui;
    PRTVARIABLEGET2UIV                      rtVariableGet2uiv;
    PRTVARIABLEGET2LL                       rtVariableGet2ll;
    PRTVARIABLEGET2LLV                      rtVariableGet2llv;
    PRTVARIABLEGET2ULL                      rtVariableGet2ull;
    PRTVARIABLEGET2ULLV                     rtVariableGet2ullv;
    PRTVARIABLEGET3F                        rtVariableGet3f;
    PRTVARIABLEGET3FV                       rtVariableGet3fv;
    PRTVARIABLEGET3I                        rtVariableGet3i;
    PRTVARIABLEGET3IV                       rtVariableGet3iv;
    PRTVARIABLEGET3UI                       rtVariableGet3ui;
    PRTVARIABLEGET3UIV                      rtVariableGet3uiv;
    PRTVARIABLEGET3LL                       rtVariableGet3ll;
    PRTVARIABLEGET3LLV                      rtVariableGet3llv;
    PRTVARIABLEGET3ULL                      rtVariableGet3ull;
    PRTVARIABLEGET3ULLV                     rtVariableGet3ullv;
    PRTVARIABLEGET4F                        rtVariableGet4f;
    PRTVARIABLEGET4FV                       rtVariableGet4fv;
    PRTVARIABLEGET4I                        rtVariableGet4i;
    PRTVARIABLEGET4IV                       rtVariableGet4iv;
    PRTVARIABLEGET4UI                       rtVariableGet4ui;
    PRTVARIABLEGET4UIV                      rtVariableGet4uiv;
    PRTVARIABLEGET4LL                       rtVariableGet4ll;
    PRTVARIABLEGET4LLV                      rtVariableGet4llv;
    PRTVARIABLEGET4ULL                      rtVariableGet4ull;
    PRTVARIABLEGET4ULLV                     rtVariableGet4ullv;
    PRTVARIABLEGETANNOTATION                rtVariableGetAnnotation;
    PRTVARIABLEGETCONTEXT                   rtVariableGetContext;
    PRTVARIABLEGETMATRIX2X2FV               rtVariableGetMatrix2x2fv;
    PRTVARIABLEGETMATRIX2X3FV               rtVariableGetMatrix2x3fv;
    PRTVARIABLEGETMATRIX2X4FV               rtVariableGetMatrix2x4fv;
    PRTVARIABLEGETMATRIX3X2FV               rtVariableGetMatrix3x2fv;
    PRTVARIABLEGETMATRIX3X3FV               rtVariableGetMatrix3x3fv;
    PRTVARIABLEGETMATRIX3X4FV               rtVariableGetMatrix3x4fv;
    PRTVARIABLEGETMATRIX4X2FV               rtVariableGetMatrix4x2fv;
    PRTVARIABLEGETMATRIX4X3FV               rtVariableGetMatrix4x3fv;
    PRTVARIABLEGETMATRIX4X4FV               rtVariableGetMatrix4x4fv;
    PRTVARIABLEGETNAME                      rtVariableGetName;
    PRTVARIABLEGETOBJECT                    rtVariableGetObject;
    PRTVARIABLEGETSIZE                      rtVariableGetSize;
    PRTVARIABLEGETTYPE                      rtVariableGetType;
    PRTVARIABLEGETUSERDATA                  rtVariableGetUserData;
    PRTVARIABLESET1F                        rtVariableSet1f;
    PRTVARIABLESET1FV                       rtVariableSet1fv;
    PRTVARIABLESET1I                        rtVariableSet1i;
    PRTVARIABLESET1IV                       rtVariableSet1iv;
    PRTVARIABLESET1UI                       rtVariableSet1ui;
    PRTVARIABLESET1UIV                      rtVariableSet1uiv;
    PRTVARIABLESET1LL                       rtVariableSet1ll;
    PRTVARIABLESET1LLV                      rtVariableSet1llv;
    PRTVARIABLESET1ULL                      rtVariableSet1ull;
    PRTVARIABLESET1ULLV                     rtVariableSet1ullv;
    PRTVARIABLESET2F                        rtVariableSet2f;
    PRTVARIABLESET2FV                       rtVariableSet2fv;
    PRTVARIABLESET2I                        rtVariableSet2i;
    PRTVARIABLESET2IV                       rtVariableSet2iv;
    PRTVARIABLESET2UI                       rtVariableSet2ui;
    PRTVARIABLESET2UIV                      rtVariableSet2uiv;
    PRTVARIABLESET2LL                       rtVariableSet2ll;
    PRTVARIABLESET2LLV                      rtVariableSet2llv;
    PRTVARIABLESET2ULL                      rtVariableSet2ull;
    PRTVARIABLESET2ULLV                     rtVariableSet2ullv;
    PRTVARIABLESET3F                        rtVariableSet3f;
    PRTVARIABLESET3FV                       rtVariableSet3fv;
    PRTVARIABLESET3I                        rtVariableSet3i;
    PRTVARIABLESET3IV                       rtVariableSet3iv;
    PRTVARIABLESET3UI                       rtVariableSet3ui;
    PRTVARIABLESET3UIV                      rtVariableSet3uiv;
    PRTVARIABLESET3LL                       rtVariableSet3ll;
    PRTVARIABLESET3LLV                      rtVariableSet3llv;
    PRTVARIABLESET3ULL                      rtVariableSet3ull;
    PRTVARIABLESET3ULLV                     rtVariableSet3ullv;
    PRTVARIABLESET4F                        rtVariableSet4f;
    PRTVARIABLESET4FV                       rtVariableSet4fv;
    PRTVARIABLESET4I                        rtVariableSet4i;
    PRTVARIABLESET4IV                       rtVariableSet4iv;
    PRTVARIABLESET4UI                       rtVariableSet4ui;
    PRTVARIABLESET4UIV                      rtVariableSet4uiv;
    PRTVARIABLESET4LL                       rtVariableSet4ll;
    PRTVARIABLESET4LLV                      rtVariableSet4llv;
    PRTVARIABLESET4ULL                      rtVariableSet4ull;
    PRTVARIABLESET4ULLV                     rtVariableSet4ullv;
    PRTVARIABLESETMATRIX2X2FV               rtVariableSetMatrix2x2fv;
    PRTVARIABLESETMATRIX2X3FV               rtVariableSetMatrix2x3fv;
    PRTVARIABLESETMATRIX2X4FV               rtVariableSetMatrix2x4fv;
    PRTVARIABLESETMATRIX3X2FV               rtVariableSetMatrix3x2fv;
    PRTVARIABLESETMATRIX3X3FV               rtVariableSetMatrix3x3fv;
    PRTVARIABLESETMATRIX3X4FV               rtVariableSetMatrix3x4fv;
    PRTVARIABLESETMATRIX4X2FV               rtVariableSetMatrix4x2fv;
    PRTVARIABLESETMATRIX4X3FV               rtVariableSetMatrix4x3fv;
    PRTVARIABLESETMATRIX4X4FV               rtVariableSetMatrix4x4fv;
    PRTVARIABLESETOBJECT                    rtVariableSetObject;
    PRTVARIABLESETUSERDATA                  rtVariableSetUserData;

#ifdef _WIN32  // windows-only section
    PRTDEVICEGETWGLDEVICE                    rtDeviceGetWGLDevice;
    PRTBUFFERCREATEFROMD3D10RESOURCE         rtBufferCreateFromD3D10Resource;
    PRTBUFFERCREATEFROMD3D11RESOURCE         rtBufferCreateFromD3D11Resource;
    PRTBUFFERCREATEFROMD3D9RESOURCE          rtBufferCreateFromD3D9Resource;
    PRTBUFFERD3D10REGISTER                   rtBufferD3D10Register;
    PRTBUFFERD3D10UNREGISTER                 rtBufferD3D10Unregister;
    PRTBUFFERD3D11REGISTER                   rtBufferD3D11Register;
    PRTBUFFERD3D11UNREGISTER                 rtBufferD3D11Unregister;
    PRTBUFFERD3D9REGISTER                    rtBufferD3D9Register;
    PRTBUFFERD3D9UNREGISTER                  rtBufferD3D9Unregister;
    PRTBUFFERGETD3D10RESOURCE                rtBufferGetD3D10Resource;
    PRTBUFFERGETD3D11RESOURCE                rtBufferGetD3D11Resource;
    PRTBUFFERGETD3D9RESOURCE                 rtBufferGetD3D9Resource;
    PRTCONTEXTSETD3D10DEVICE                 rtContextSetD3D10Device;
    PRTCONTEXTSETD3D11DEVICE                 rtContextSetD3D11Device;
    PRTCONTEXTSETD3D9DEVICE                  rtContextSetD3D9Device;
    PRTDEVICEGETD3D9DEVICE                   rtDeviceGetD3D9Device;
    PRTDEVICEGETD3D10DEVICE                  rtDeviceGetD3D10Device;
    PRTDEVICEGETD3D11DEVICE                  rtDeviceGetD3D11Device;
    PRTTEXTURESAMPLERCREATEFROMD3D10RESOURCE rtTextureSamplerCreateFromD3D10Resource;
    PRTTEXTURESAMPLERCREATEFROMD3D11RESOURCE rtTextureSamplerCreateFromD3D11Resource;
    PRTTEXTURESAMPLERCREATEFROMD3D9RESOURCE  rtTextureSamplerCreateFromD3D9Resource;
    PRTTEXTURESAMPLERD3D10REGISTER           rtTextureSamplerD3D10Register;
    PRTTEXTURESAMPLERD3D10UNREGISTER         rtTextureSamplerD3D10Unregister;
    PRTTEXTURESAMPLERD3D11REGISTER           rtTextureSamplerD3D11Register;
    PRTTEXTURESAMPLERD3D11UNREGISTER         rtTextureSamplerD3D11Unregister;
    PRTTEXTURESAMPLERD3D9REGISTER            rtTextureSamplerD3D9Register;
    PRTTEXTURESAMPLERD3D9UNREGISTER          rtTextureSamplerD3D9Unregister;
    PRTTEXTURESAMPLERGETD3D10RESOURCE        rtTextureSamplerGetD3D10Resource;
    PRTTEXTURESAMPLERGETD3D11RESOURCE        rtTextureSamplerGetD3D11Resource;
    PRTTEXTURESAMPLERGETD3D9RESOURCE         rtTextureSamplerGetD3D9Resource;
#endif  // _WIN32

    // LWOPTIX_ABI_VERSION >= 4
    PRTPROGRAMCREATEFROMPTXSTRINGS rtProgramCreateFromPTXStrings;
    PRTPROGRAMCREATEFROMPTXFILES   rtProgramCreateFromPTXFiles;

    // LWOPTIX_ABI_VERSION >= 5
    PRTGEOMETRYINSTANCEGETGEOMETRYTRIANGLES     rtGeometryInstanceGetGeometryTriangles;
    PRTGEOMETRYINSTANCESETGEOMETRYTRIANGLES     rtGeometryInstanceSetGeometryTriangles;
    PRTGEOMETRYTRIANGLESCREATE                  rtGeometryTrianglesCreate;
    PRTGEOMETRYTRIANGLESDESTROY                 rtGeometryTrianglesDestroy;
    PRTGEOMETRYTRIANGLESGETCONTEXT              rtGeometryTrianglesGetContext;
    PRTGEOMETRYTRIANGLESGETPRIMITIVEINDEXOFFSET rtGeometryTrianglesGetPrimitiveIndexOffset;
    PRTGEOMETRYTRIANGLESSETPRIMITIVEINDEXOFFSET rtGeometryTrianglesSetPrimitiveIndexOffset;

    // NOTE: Version 5 was never publicly released, and we've deprecated a couple of the unreleased functions.
    PRTGEOMETRYTRIANGLESSETINDEXEDTRIANGLESDEPRECATED rtGeometryTrianglesSetIndexedTrianglesDeprecated;  // Deprecated with LWOPTIX_ABI_VERSION 7
    PRTGEOMETRYTRIANGLESSETTRIANGLESDEPRECATED rtGeometryTrianglesSetTrianglesDeprecated;  // Deprecated with LWOPTIX_ABI_VERSION 7

    PRTGEOMETRYTRIANGLESVALIDATE rtGeometryTrianglesValidate;

    // LWOPTIX_ABI_VERSION >= 6
    PRTLOG rtLog;  // note that this isn't a public API function, but in internal one

    // LWOPTIX_ABI_VERSION >= 7
    PRTCONTEXTGETMAXCALLABLEPROGRAMDEPTH rtContextGetMaxCallableProgramDepth;
    PRTCONTEXTSETMAXCALLABLEPROGRAMDEPTH rtContextSetMaxCallableProgramDepth;
    PRTCONTEXTGETMAXTRACEDEPTH           rtContextGetMaxTraceDepth;
    PRTCONTEXTSETMAXTRACEDEPTH           rtContextSetMaxTraceDepth;

    // LWOPTIX_ABI_VERSION >= 8
    PRTPROGRAMCALLSITESETPOTENTIALCALLEES rtProgramCallsiteSetPotentialCallees;

    // LWOPTIX_ABI_VERSION >= 9
    PRTGETBUILDVERSION       rtGetBuildVersion;
    PRTOVERRIDESOTHERVERSION rtOverridesOtherVersion;

    // LWOPTIX_ABI_VERSION >= 10
    PRTGEOMETRYTRIANGLESSETATTRIBUTEPROGRAM   rtGeometryTrianglesSetAttributeProgram;
    PRTGEOMETRYTRIANGLESGETATTRIBUTEPROGRAM   rtGeometryTrianglesGetAttributeProgram;
    PRTGEOMETRYTRIANGLESDECLAREVARIABLE       rtGeometryTrianglesDeclareVariable;
    PRTGEOMETRYTRIANGLESQUERYVARIABLE         rtGeometryTrianglesQueryVariable;
    PRTGEOMETRYTRIANGLESREMOVEVARIABLE        rtGeometryTrianglesRemoveVariable;
    PRTGEOMETRYTRIANGLESGETVARIABLECOUNT      rtGeometryTrianglesGetVariableCount;
    PRTGEOMETRYTRIANGLESGETVARIABLE           rtGeometryTrianglesGetVariable;
    PRTGEOMETRYTRIANGLESSETPRETRANSFORMMATRIX rtGeometryTrianglesSetPreTransformMatrix;
    PRTGEOMETRYTRIANGLESGETPRETRANSFORMMATRIX rtGeometryTrianglesGetPreTransformMatrix;
    PRTGEOMETRYTRIANGLESSETPRIMITIVECOUNT     rtGeometryTrianglesSetPrimitiveCount;
    PRTGEOMETRYTRIANGLESGETPRIMITIVECOUNT     rtGeometryTrianglesGetPrimitiveCount;
    PRTGEOMETRYTRIANGLESSETTRIANGLESINDICES   rtGeometryTrianglesSetTriangleIndices;
    PRTGEOMETRYTRIANGLESSETVERTICES           rtGeometryTrianglesSetVertices;
    PRTGEOMETRYTRIANGLESSETMOTIOLWERTICES     rtGeometryTrianglesSetMotiolwertices;
    // This function got fixed, the first version was introduced only internally and was not shared with anyone outside lwpu.
    // While the member must not be (re)moved, backwards compatibility is not an issue and can be dropped in the future if desired.
    PRTGEOMETRYTRIANGLESSETMOTIOLWERTICESMULTIBUFFERDEPRECATED rtGeometryTrianglesSetMotiolwerticesMultiBufferDeprecated;
    PRTGEOMETRYTRIANGLESSETMOTIONSTEPS                         rtGeometryTrianglesSetMotionSteps;
    PRTGEOMETRYTRIANGLESGETMOTIONSTEPS                         rtGeometryTrianglesGetMotionSteps;
    PRTGEOMETRYTRIANGLESSETMOTIONRANGE                         rtGeometryTrianglesSetMotionRange;
    PRTGEOMETRYTRIANGLESGETMOTIONRANGE                         rtGeometryTrianglesGetMotionRange;
    PRTGEOMETRYTRIANGLESSETMOTIONBORDERMODE                    rtGeometryTrianglesSetMotionBorderMode;
    PRTGEOMETRYTRIANGLESGETMOTIONBORDERMODE                    rtGeometryTrianglesGetMotionBorderMode;
    PRTGEOMETRYTRIANGLESSETBUILDFLAGS                          rtGeometryTrianglesSetBuildFlags;
    PRTGEOMETRYTRIANGLESGETMATERIALCOUNT                       rtGeometryTrianglesGetMaterialCount;
    PRTGEOMETRYTRIANGLESSETMATERIALCOUNT                       rtGeometryTrianglesSetMaterialCount;
    PRTGEOMETRYTRIANGLESSETMATERIALINDICES                     rtGeometryTrianglesSetMaterialIndices;
    PRTGEOMETRYTRIANGLESSETFLAGSPERMATERIAL                    rtGeometryTrianglesSetFlagsPerMaterial;
    PRTGEOMETRYTRIANGLESGETFLAGSPERMATERIAL                    rtGeometryTrianglesGetFlagsPerMaterial;
    PRTGROUPSETVISIBLITYMASK                                   rtGroupSetVisibilityMask;
    PRTGROUPGETVISIBLITYMASK                                   rtGroupGetVisibilityMask;
    PRTGEOMETRYGROUPSETFLAGS                                   rtGeometryGroupSetFlags;
    PRTGEOMETRYGROUPGETFLAGS                                   rtGeometryGroupGetFlags;
    PRTGEOMETRYGROUPSETVISIBILITYMASK                          rtGeometryGroupSetVisibilityMask;
    PRTGEOMETRYGROUPGETVISIBILITYMASK                          rtGeometryGroupGetVisibilityMask;
    PRTGEOMETRYSETFLAGS                                        rtGeometrySetFlags;
    PRTGEOMETRYGETFLAGS                                        rtGeometryGetFlags;

    // LWOPTIX_ABI_VERSION >= 12
    PRTGEOMETRYTRIANGLESSETMOTIOLWERTICESMULTIBUFFER rtGeometryTrianglesSetMotiolwerticesMultiBuffer;

    // LWOPTIX_ABI_VERSION >= 13
    // Bumped ABI version to reflect changes in the Device ABI: Expose getter to ray flags, ray mask, rtcore's instance index.
    // No changes in the wrapper <-> lwoptix ABI.

    // LWOPTIX_ABI_VERSION >= 14
    PRTCOMMANDLISTSETLWDASTREAM                      rtCommandListSetLwdaStream;
    PRTCOMMANDLISTGETLWDASTREAM                      rtCommandListGetLwdaStream;
    PRTCOMMANDLISTAPPENDLAUNCH1D                     rtCommandListAppendLaunch1D;
    PRTCOMMANDLISTAPPENDLAUNCH3D                     rtCommandListAppendLaunch3D;
    PRTCOMMANDLISTSETDEVICES                         rtCommandListSetDevices;
    PRTCOMMANDLISTGETDEVICES                         rtCommandListGetDevices;
    PRTCOMMANDLISTGETDEVICECOUNT                     rtCommandListGetDeviceCount;

    // LWOPTIX_ABI_VERSION >= 15
    PRTBUFFERCREATEFROMCALLBACK                      rtBufferCreateFromCallback;

    // LWOPTIX_ABI_VERSION >= 16
    PRTPROGRAMCREATEFROMPROGRAM                      rtProgramCreateFromProgram;
};
typedef struct OptiXAPI OptiXAPI_t;

typedef enum {
    /// Flag that indicates whether this DLL is loaded as part of the SDK or from the driver. This
    /// affects the implementation of rtOverridesOtherVersion() and the name of the rtcore DLL.
    /// Option value: 1 bool
    RT_OPTIX_OPTION_FROM_SDK = 0,
    /// Flag that indicates whether rtGetSymbolTable() should skip the check for the minimum
    /// required driver version.
    RT_OPTIX_OPTION_SKIP_DRIVER_VERSION_CHECK = 1,
} RToptixoptions;

typedef RTresult ( *PRTGETSYMBOLTABLE )( unsigned int    version,
                                         unsigned int    numOptions,
                                         RToptixoptions* options,
                                         void**          option_values,
                                         OptiXAPI_t*     symbol_table );

typedef void* ( *PRTSSIMPREDICTORFACTORY )();
