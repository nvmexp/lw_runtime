
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

/// @file
/// @author LWPU Corporation
/// @brief  OptiX public API header
///
/// OptiX types include file -- defines types and enums used by the API.
/// For the math library routines include optix_math.h

#if !defined( __OPTIX_INCLUDE_INTERNAL_HEADERS__ )
#error("optix_7_types.h is an internal header file and must not be used directly.  Please use optix_types.h, optix_host.h, optix_device.h or optix.h instead.")
#endif

#ifndef __optix_optix_7_types_h__
#define __optix_optix_7_types_h__

#if !defined(__LWDACC_RTC__)
#include <stddef.h> /* for size_t */
#endif

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#include <g_lwconfig.h>
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD


/// \defgroup optix_types Types
/// \brief OptiX Types

/** \addtogroup optix_types
@{
*/

// This typedef should match the one in lwca.h in order to avoid compilation errors.
#if defined(_WIN64) || defined(__LP64__)
/// LWCA device pointer
typedef unsigned long long LWdeviceptr;
#else
/// LWCA device pointer
typedef unsigned int LWdeviceptr;
#endif

/// Opaque type representing a device context
typedef struct OptixDeviceContext_t* OptixDeviceContext;

/// Opaque type representing a module
typedef struct OptixModule_t* OptixModule;

/// Opaque type representing a program group
typedef struct OptixProgramGroup_t* OptixProgramGroup;

/// Opaque type representing a pipeline
typedef struct OptixPipeline_t* OptixPipeline;

/// Opaque type representing a denoiser instance
typedef struct OptixDenoiser_t* OptixDenoiser;

/// Opaque type representing a work task
typedef struct OptixTask_t* OptixTask;

/// Traversable handle
typedef unsigned long long OptixTraversableHandle;

/// Visibility mask
typedef unsigned int OptixVisibilityMask;

/// Size of the SBT record headers.
#define OPTIX_SBT_RECORD_HEADER_SIZE ( (size_t)32 )

/// Alignment requirement for device pointers in OptixShaderBindingTable.
#define OPTIX_SBT_RECORD_ALIGNMENT 16ull

/// Alignment requirement for output and temporay buffers for acceleration structures.
#define OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT 128ull

/// Alignment requirement for OptixBuildInputInstanceArray::instances.
#define OPTIX_INSTANCE_BYTE_ALIGNMENT 16ull

/// Alignment requirement for OptixBuildInputLwstomPrimitiveArray::aabbBuffers
#define OPTIX_AABB_BUFFER_BYTE_ALIGNMENT 8ull

/// Alignment requirement for OptixBuildInputTriangleArray::preTransform
#define OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT 16ull

/// Alignment requirement for OptixStaticTransform, OptixMatrixMotionTransform, OptixSRTMotionTransform.
#define OPTIX_TRANSFORM_BYTE_ALIGNMENT 64ull

/// Maximum number of registers allowed. Defaults to no explicit limit.
#define OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT 0

/// Maximum number of payload types allowed.
#define OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_TYPE_COUNT 8

/// Maximum number of payload values allowed.
#define OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT 32

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
/// Special index to indicate that a triangle in the BVH build doesn't have an associated VM,
/// and that it should revert to one of the four possible states for the full triangle.
#define OPTIX_VISIBILITY_MAP_SPECIAL_INDEX_FULLY_TRANSPARENT          ( -1 )
#define OPTIX_VISIBILITY_MAP_SPECIAL_INDEX_FULLY_OPAQUE               ( -2 )
#define OPTIX_VISIBILITY_MAP_SPECIAL_INDEX_FULLY_UNKNOWN_TRANSPARENT  ( -3 )
#define OPTIX_VISIBILITY_MAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE       ( -4 )

/// Alignment requirement for VM array buffers
#define OPTIX_VISIBILITY_MAP_ARRAY_BUFFER_BYTE_ALIGNMENT 128ull

/// Alignment requirement for OptixBuildInputInstanceArray::vmArrayReplacements
#define OPTIX_INSTANCE_VM_REPLACEMENTS_BYTE_ALIGNMENT 8ull

/// Maximum subdivision level for VMs
#define OPTIX_VISIBILITY_MAP_MAX_SUBDIVISION_LEVEL 12

/// Maximum subdivision level for DMMs
#define OPTIX_DISPLACED_MICROMESH_MAX_SUBDIVISION_LEVEL 5

/// Alignment requirement for DMM array buffers
#define OPTIX_DISPLACED_MICROMESH_ARRAY_BUFFER_BYTE_ALIGNMENT 128ull
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD

/// Result codes returned from API functions
///
/// All host side API functions return OptixResult with the exception of optixGetErrorName
/// and optixGetErrorString.  When successful OPTIX_SUCCESS is returned.  All return codes
/// except for OPTIX_SUCCESS should be assumed to be errors as opposed to a warning.
///
/// \see #optixGetErrorName(), #optixGetErrorString()
typedef enum OptixResult
{
    OPTIX_SUCCESS                               = 0,
    OPTIX_ERROR_ILWALID_VALUE                   = 7001,
    OPTIX_ERROR_HOST_OUT_OF_MEMORY              = 7002,
    OPTIX_ERROR_ILWALID_OPERATION               = 7003,
    OPTIX_ERROR_FILE_IO_ERROR                   = 7004,
    OPTIX_ERROR_ILWALID_FILE_FORMAT             = 7005,
    OPTIX_ERROR_DISK_CACHE_ILWALID_PATH         = 7010,
    OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR     = 7011,
    OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR       = 7012,
    OPTIX_ERROR_DISK_CACHE_ILWALID_DATA         = 7013,
    OPTIX_ERROR_LAUNCH_FAILURE                  = 7050,
    OPTIX_ERROR_ILWALID_DEVICE_CONTEXT          = 7051,
    OPTIX_ERROR_LWDA_NOT_INITIALIZED            = 7052,
    OPTIX_ERROR_VALIDATION_FAILURE              = 7053,
    OPTIX_ERROR_ILWALID_PTX                     = 7200,
    OPTIX_ERROR_ILWALID_LAUNCH_PARAMETER        = 7201,
    OPTIX_ERROR_ILWALID_PAYLOAD_ACCESS          = 7202,
    OPTIX_ERROR_ILWALID_ATTRIBUTE_ACCESS        = 7203,
    OPTIX_ERROR_ILWALID_FUNCTION_USE            = 7204,
    OPTIX_ERROR_ILWALID_FUNCTION_ARGUMENTS      = 7205,
    OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY = 7250,
    OPTIX_ERROR_PIPELINE_LINK_ERROR             = 7251,
    OPTIX_ERROR_ILLEGAL_DURING_TASK_EXELWTE     = 7270,
    OPTIX_ERROR_INTERNAL_COMPILER_ERROR         = 7299,
    OPTIX_ERROR_DENOISER_MODEL_NOT_SET          = 7300,
    OPTIX_ERROR_DENOISER_NOT_INITIALIZED        = 7301,
    OPTIX_ERROR_ACCEL_NOT_COMPATIBLE            = 7400,
    OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH           = 7500,
    OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED  = 7501,
    OPTIX_ERROR_PAYLOAD_TYPE_ID_ILWALID         = 7502,
    OPTIX_ERROR_NOT_SUPPORTED                   = 7800,
    OPTIX_ERROR_UNSUPPORTED_ABI_VERSION         = 7801,
    OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH    = 7802,
    OPTIX_ERROR_ILWALID_ENTRY_FUNCTION_OPTIONS  = 7803,
    OPTIX_ERROR_LIBRARY_NOT_FOUND               = 7804,
    OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND          = 7805,
    OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE          = 7806,
    OPTIX_ERROR_DEVICE_OUT_OF_MEMORY            = 7807,
    OPTIX_ERROR_LWDA_ERROR                      = 7900,
    OPTIX_ERROR_INTERNAL_ERROR                  = 7990,
    OPTIX_ERROR_UNKNOWN                         = 7999,
} OptixResult;

/// Parameters used for #optixDeviceContextGetProperty()
///
/// \see #optixDeviceContextGetProperty()
typedef enum OptixDeviceProperty
{
    /// Maximum value for OptixPipelineLinkOptions::maxTraceDepth. sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH = 0x2001,

    /// Maximum value to pass into optixPipelineSetStackSize for parameter
    /// maxTraversableGraphDepth.v sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH = 0x2002,

    /// The maximum number of primitives (over all build inputs) as input to a single
    /// Geometry Acceleration Structure (GAS). sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS = 0x2003,

    /// The maximum number of instances (over all build inputs) as input to a single
    /// Instance Acceleration Structure (IAS). sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS = 0x2004,

    /// The RT core version supported by the device (0 for no support, 10 for version
    /// 1.0). sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_RTCORE_VERSION = 0x2005,

    /// The maximum value for #OptixInstance::instanceId. sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID = 0x2006,

    /// The number of bits available for the #OptixInstance::visibilityMask.
    /// Higher bits must be set to zero. sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK = 0x2007,

    /// The maximum number of instances that can be added to a single Instance
    /// Acceleration Structure (IAS). sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS = 0x2008,

    /// The maximum value for #OptixInstance::sbtOffset. sizeof( unsigned int )
    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET = 0x2009,
} OptixDeviceProperty;

/// Type of the callback function used for log messages.
///
/// \param[in] level      The log level indicates the severity of the message. See below for
///                       possible values.
/// \param[in] tag        A terse message category description (e.g., 'SCENE STAT').
/// \param[in] message    Null terminated log message (without newline at the end).
/// \param[in] cbdata     Callback data that was provided with the callback pointer.
///
/// It is the users responsibility to ensure thread safety within this function.
///
/// The following log levels are defined.
///
///   0   disable   Setting the callback level will disable all messages.  The callback
///                 function will not be called in this case.
///   1   fatal     A non-recoverable error. The context and/or OptiX itself might no longer
///                 be in a usable state.
///   2   error     A recoverable error, e.g., when passing invalid call parameters.
///   3   warning   Hints that OptiX might not behave exactly as requested by the user or
///                 may perform slower than expected.
///   4   print     Status or progress messages.
///
/// Higher levels might occur.
///
/// \see #optixDeviceContextSetLogCallback(), #OptixDeviceContextOptions
typedef void ( *OptixLogCallback )( unsigned int level, const char* tag, const char* message, void* cbdata );

/// Validation mode settings.
///
/// When enabled, certain device code utilities will be enabled to provide as good debug and
/// error checking facilities as possible.
///
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
/// Lwrrently, when not OFF all builtin debug exceptions are enabled and each thrown builtin exception
/// will hard-stop program exelwtion at the end of the exception program run, both for the default
/// or user-provided exception programs.
/// If really needed this could be fine-tuned like eg
/// - OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_DEBUG_EXCEPTIONS
/// - ...
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// \see #optixDeviceContextCreate()
typedef enum OptixDeviceContextValidationMode
{
    OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF = 0,
    OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL = 0xFFFFFFFF
} OptixDeviceContextValidationMode;

/// Parameters used for #optixDeviceContextCreate()
///
/// \see #optixDeviceContextCreate()
typedef struct OptixDeviceContextOptions
{
    /// Function pointer used when OptiX wishes to generate messages
    OptixLogCallback logCallbackFunction;
    /// Pointer stored and passed to logCallbackFunction when a message is generated
    void* logCallbackData;
    /// Maximum callback level to generate message for (see #OptixLogCallback)
    int logCallbackLevel;
    /// Validation mode of context.
    OptixDeviceContextValidationMode validationMode;
} OptixDeviceContextOptions;

/// Flags used by #OptixBuildInputTriangleArray::flags
/// and #OptixBuildInput::flag
/// and #OptixBuildInputLwstomPrimitiveArray::flags
typedef enum OptixGeometryFlags
{
    /// No flags set
    OPTIX_GEOMETRY_FLAG_NONE = 0,

    /// Disables the invocation of the anyhit program.
    /// Can be overridden by OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT and OPTIX_RAY_FLAG_ENFORCE_ANYHIT.
    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT = 1u << 0,

    /// If set, an intersection with the primitive will trigger one and only one
    /// invocation of the anyhit program.  Otherwise, the anyhit program may be ilwoked
    /// more than once.
    OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL = 1u << 1,

    /// Prevent triangles from getting lwlled due to their orientation.
    /// Effectively ignores ray flags
    /// OPTIX_RAY_FLAG_LWLL_BACK_FACING_TRIANGLES and OPTIX_RAY_FLAG_LWLL_FRONT_FACING_TRIANGLES.
    OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_LWLLING = 1u << 2,

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    /// This flag is set if the application intends to swap out the VM array via the instance.
    /// Only one VM array per GAS may be replaced, and it must have been built with the 'mutable' flag.
    /// If multiple geometries set this flag, they must all reference the same VM array.
    OPTIX_GEOMETRY_FLAG_REPLACEABLE_VM_ARRAY = 1u << 3,
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
} OptixGeometryFlags;

/// Legacy type: A subset of the hit kinds for built-in primitive intersections.
/// It is preferred to use optixGetPrimitiveType(), together with
/// optixIsFrontFaceHit() or optixIsBackFaceHit().
///
/// \see #optixGetHitKind()
typedef enum OptixHitKind
{
    /// Ray hit the triangle on the front face
    OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE = 0xFE,
    /// Ray hit the triangle on the back face
    OPTIX_HIT_KIND_TRIANGLE_BACK_FACE = 0xFF
} OptixHitKind;

/// Format of indices used int #OptixBuildInputTriangleArray::indexFormat.
typedef enum OptixIndicesFormat
{
    /// No indices, this format must only be used in combination with triangle soups, i.e., numIndexTriplets must be zero
    OPTIX_INDICES_FORMAT_NONE = 0,
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    // Not lwrrently supported, but reserve the enum anyway.
    /// Three bytes
    OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3 = 0x2101,
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    /// Three shorts
    OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 = 0x2102,
    /// Three ints
    OPTIX_INDICES_FORMAT_UNSIGNED_INT3 = 0x2103
} OptixIndicesFormat;

/// Format of vertices used in #OptixBuildInputTriangleArray::vertexFormat.
typedef enum OptixVertexFormat
{
    OPTIX_VERTEX_FORMAT_NONE      = 0,       ///< No vertices
    OPTIX_VERTEX_FORMAT_FLOAT3    = 0x2121,  ///< Vertices are represented by three floats
    OPTIX_VERTEX_FORMAT_FLOAT2    = 0x2122,  ///< Vertices are represented by two floats
    OPTIX_VERTEX_FORMAT_HALF3     = 0x2123,  ///< Vertices are represented by three halfs
    OPTIX_VERTEX_FORMAT_HALF2     = 0x2124,  ///< Vertices are represented by two halfs
    OPTIX_VERTEX_FORMAT_SNORM16_3 = 0x2125,
    OPTIX_VERTEX_FORMAT_SNORM16_2 = 0x2126
} OptixVertexFormat;

/// Format of transform used in #OptixBuildInputTriangleArray::transformFormat.
typedef enum OptixTransformFormat
{
    OPTIX_TRANSFORM_FORMAT_NONE           = 0,       ///< no transform, default for zero initialization
    OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12 = 0x21E1,  ///< 3x4 row major affine matrix
} OptixTransformFormat;

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

typedef enum OptixDisplacementBiasAndScaleFormat
{
    OPTIX_DISPLACEMENT_BIAS_AND_SCALE_FORMAT_FLOAT = 0,
    OPTIX_DISPLACEMENT_BIAS_AND_SCALE_FORMAT_HALF  = 1,
} OptixDisplacementBiasAndScaleFormat;

typedef enum OptixDisplacementVectorFormat
{
    OPTIX_MICROMESH_DISPLACEMENT_VECTOR_FORMAT_HALF3 = 0,
} OptixDisplacementVectorFormat;

/// Specifies whether to use a 2- or 4-state VM format.
typedef enum OptixVisibilityMapFormat
{
    /// invalid format
    OPTIX_VISIBILITY_MAP_FORMAT_NONE = 0,
    /// 0: Transparent, 1: Opaque
    OPTIX_VISIBILITY_MAP_FORMAT_2_STATE = 1,
    /// 0: Transparent, 1: Opaque, 2: Unknown-Transparent, 3: Unknown-Opaque
    OPTIX_VISIBILITY_MAP_FORMAT_4_STATE = 2,
} OptixVisibilityMapFormat;

typedef struct OptixBuildInputVisibilityMap
{
    /// Device pointer to a visibility map array used by this build input array.
    LWdeviceptr  visibilityMapArray;

    /// int16 or int32 buffer specifying which VM index to use for each triangle.
    /// Instead of an actual index, one of the special values
    /// OPTIX_VISIBILITY_MAP_SPECIAL_INDEX_(FULLY_TRANSPARENT | FULLY_OPAQUE | FULLY_UNKNOWN_TRANSPARENT | FULLY_UNKNOWN_OPAQUE)
    /// can be used to indicate that there is no VM for this particular triangle but the triangle is in a uniform state and
    /// the selected behavior is applied to the entire triangle.
    LWdeviceptr  indexBuffer;

    /// VM index buffer stride
    unsigned int indexStrideInBytes;

    /// 2 or 4 (16 or 32 bit)
    unsigned int indexSizeInBytes;

    /// Constant offset to non-negative VM indices
    unsigned int baseLocation;
} OptixBuildInputVisibilityMap;

/// DMM input data format.
typedef enum OptixDisplacedMicromeshFormat
{
    OPTIX_DISPLACED_MICROMESH_FORMAT_NONE                = 0,
    OPTIX_DISPLACED_MICROMESH_FORMAT_64_TRIS_64_BYTES    = 1,
    OPTIX_DISPLACED_MICROMESH_FORMAT_256_TRIS_128_BYTES  = 2,
    OPTIX_DISPLACED_MICROMESH_FORMAT_1024_TRIS_128_BYTES = 3,
} OptixDisplacedMicromeshFormat;

/// Flags defining behavior of DMMs in a DMM array.
typedef enum OptixDisplacedMicromeshFlags
{
    OPTIX_DISPLACED_MICROMESH_FLAG_NONE = 0,

    OPTIX_DISPLACED_MICROMESH_FLAG_PREFER_FAST_TRACE = 1 << 0,
    OPTIX_DISPLACED_MICROMESH_FLAG_PREFER_FAST_BUILD = 1 << 1,

} OptixDisplacedMicromeshFlags;

typedef enum OptixDisplacementPrimitiveFlags
{
    /// The edge v0..v1 is decimated: after subdivision the number of microtriangles on that edge is halved (i.e. the neighboring primitive can have a lower subdivision level without introducing cracks)
    OPTIX_DISPLACED_MICROMESH_PRIMITIVE_FLAG_DECIMATE_01 = 1 << 0,
    /// The edge v1..v2 -"-
    OPTIX_DISPLACED_MICROMESH_PRIMITIVE_FLAG_DECIMATE_12 = 1 << 1,
    /// The edge v2..v0 -"-
    OPTIX_DISPLACED_MICROMESH_PRIMITIVE_FLAG_DECIMATE_20 = 1 << 2,
} OptixDisplacementPrimitiveFlags;

typedef struct OptixDisplacedMicromeshDesc
{
    /// Block is located at inputBuffer + byteOffset
    unsigned int    byteOffset;
    /// Number of micro-triangles is 4^level. Valid levels are [0, 5]
    unsigned short  subdivisionLevel;
    /// Format (OptixDisplacedMicromeshFormat)
    unsigned short  format;
} OptixDisplacedMicromeshDesc;

/// DMM usage count.
typedef struct OptixDisplacedMicromeshUsageCount
{
    /// Number of DMMs with the format and subdivision level specified in this struct.
    unsigned int                  count;
    /// Number of micro-triangles is 4^level. Valid levels are [0, 5]
    unsigned int                  subdivisionLevel;
    /// DMM format.
    OptixDisplacedMicromeshFormat format;
} OptixDisplacedMicromeshUsageCount;

/// Inputs to DMM array construction.
typedef struct OptixDisplacedMicromeshArrayBuildInput
{
    /// Applies to all DMMs in array.
    OptixDisplacedMicromeshFlags             flags;
    /// 128B aligned base pointer for raw DMM input data.*/
    LWdeviceptr                              inputBuffer;
    /// One OptixDisplacedMicromeshDesc entry per DMM.
    LWdeviceptr                              perDmmDescBuffer;
    /// Dmm desc stride. A stride of 0 indices a minimal stride matching the size of the underlying type
    unsigned int                             perDmmDescStrideInBytes;
    /// Number of OptixDisplacedMicromeshUsageCount entries.
    unsigned int                             numDmmUsageCounts;
    /// Histogram over DMMs for input format and subdiv combinations.
    const OptixDisplacedMicromeshUsageCount* dmmUsageCounts;

} OptixDisplacedMicromeshArrayBuildInput;

/// Optional displacement part of a triangle array input
typedef struct OptixBuildInputDisplacement
{
    /// Address to a displaced micromesh array used by this build input array. Set to NULL to disable VMs for this input.
    LWdeviceptr displacedMicromeshArray;
    /// int16 or int32 buffer specifying which DMM index to use for each triangle. Only valid if displacedMicromeshArray != NULL.
    LWdeviceptr displacedMicromeshIndexBuffer;
    /// Per-vertex displacement directions.
    LWdeviceptr displacementVectorBuffer;
    /// Opyional per-vertex offset along displacement direction and displacement vector scale.
    LWdeviceptr displacementBiasAndScaleBuffer;
    /// Optional per-triangle flags, uint8_t per triangle, possible values defined in enum OptixDisplacementPrimitiveFlags
    LWdeviceptr primitiveFlagBuffer;

    /// Constant offset to DMM indices
    unsigned int                  baseLocation;

    /// DMM index buffer stride
    unsigned int                  displacedMicromeshIndexStrideInBytes;
    /// 2 or 4 (16 or 32 bit)
    unsigned int                  displacedMicromeshIndexSizeInBytes;

    /// Format of displacement vectors
    OptixDisplacementVectorFormat displacementVectorFormat;
    /// Stride between displacement vectors
    unsigned int                  displacementVectorStrideInBytes;

    /// Format of base vertex bias and vector scale
    OptixDisplacementBiasAndScaleFormat displacementBiasAndScaleFormat;
    /// Stride between base vertex bias entries
    unsigned int                  displacementBiasAndScaleStrideInBytes;

    /// Stride in bytes for primitiveFlags
    unsigned int                  primitiveFlagsStrideInBytes;

    /// Number of OptixDisplacedMicromeshUsageCount entries.
    unsigned int                             numDmmUsageCounts;
    /// Histogram over DMMs for input format and subdiv combinations.
    const OptixDisplacedMicromeshUsageCount* dmmUsageCounts;

} OptixBuildInputDisplacement;


#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD

/// Triangle inputs
///
/// \see #OptixBuildInput::triangleArray
typedef struct OptixBuildInputTriangleArray
{
    /// Points to host array of device pointers, one per motion step. Host array size must match the number of
    /// motion keys as set in #OptixMotionOptions (or an array of size 1 if OptixMotionOptions::numKeys is set
    /// to 0 or 1). Each per motion key device pointer must point to an array of vertices of the
    /// triangles in the format as described by vertexFormat. The minimum alignment must match the natural
    /// alignment of the type as specified in the vertexFormat, i.e., for OPTIX_VERTEX_FORMAT_FLOATX 4-byte,
    /// for all others a 2-byte alignment. However, an 16-byte stride (and buffer alignment) is recommended for
    /// vertices of format OPTIX_VERTEX_FORMAT_FLOAT3 for GAS build performance.
    const LWdeviceptr* vertexBuffers;

    /// Number of vertices in each of buffer in OptixBuildInputTriangleArray::vertexBuffers.
    unsigned int numVertices;

    /// \see #OptixVertexFormat
    OptixVertexFormat vertexFormat;

    /// Stride between vertices. If set to zero, vertices are assumed to be tightly
    /// packed and stride is inferred from vertexFormat.
    unsigned int vertexStrideInBytes;

    /// Optional pointer to array of 16 or 32-bit int triplets, one triplet per triangle.
    /// The minimum alignment must match the natural alignment of the type as specified in the indexFormat, i.e.,
    /// for OPTIX_INDICES_FORMAT_UNSIGNED_INT3 4-byte and for OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 a 2-byte alignment.
    LWdeviceptr indexBuffer;

    /// Size of array in OptixBuildInputTriangleArray::indexBuffer. For build, needs to be zero if indexBuffer is \c nullptr.
    unsigned int numIndexTriplets;

    /// \see #OptixIndicesFormat
    OptixIndicesFormat indexFormat;

    /// Stride between triplets of indices. If set to zero, indices are assumed to be tightly
    /// packed and stride is inferred from indexFormat.
    unsigned int indexStrideInBytes;

    /// Optional pointer to array of floats
    /// representing a 3x4 row major affine
    /// transformation matrix. This pointer must be a multiple of OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT
    LWdeviceptr preTransform;

    /// Array of flags, to specify flags per sbt record,
    /// combinations of OptixGeometryFlags describing the
    /// primitive behavior, size must match numSbtRecords
    const unsigned int* flags;

    /// Number of sbt records available to the sbt index offset override.
    unsigned int numSbtRecords;

    /// Device pointer to per-primitive local sbt index offset buffer. May be NULL.
    /// Every entry must be in range [0,numSbtRecords-1].
    /// Size needs to be the number of primitives.
    LWdeviceptr sbtIndexOffsetBuffer;

    /// Size of type of the sbt index offset. Needs to be 0, 1, 2 or 4 (8, 16 or 32 bit).
    unsigned int sbtIndexOffsetSizeInBytes;

    /// Stride between the index offsets. If set to zero, the offsets are assumed to be tightly
    /// packed and the stride matches the size of the type (sbtIndexOffsetSizeInBytes).
    unsigned int sbtIndexOffsetStrideInBytes;

    /// Primitive index bias, applied in optixGetPrimitiveIndex().
    /// Sum of primitiveIndexOffset and number of triangles must not overflow 32bits.
    unsigned int primitiveIndexOffset;

#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    // introduced in ABI 24
    // appended to previous struct
#endif // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    /// \see #OptixTransformFormat
    OptixTransformFormat transformFormat;

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    /// Optional visibility map inputs. Set to NULL to disable for this input.
    const OptixBuildInputVisibilityMap* visibilityMap;
    /// Optional displacement inputs. Set to NULL to disable for this input.
    const OptixBuildInputDisplacement* displacement;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD

} OptixBuildInputTriangleArray;

/// Builtin primitive types
///
typedef enum OptixPrimitiveType
{
    /// Custom primitive.
    OPTIX_PRIMITIVE_TYPE_LWSTOM                        = 0x2500,
    /// B-spline lwrve of degree 2 with cirlwlar cross-section.
    OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE       = 0x2501,
    /// B-spline lwrve of degree 3 with cirlwlar cross-section.
    OPTIX_PRIMITIVE_TYPE_ROUND_LWBIC_BSPLINE           = 0x2502,
    /// Piecewise linear lwrve with cirlwlar cross-section.
    OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR                  = 0x2503,
    /// CatmullRom lwrve with cirlwlar cross-section.
    OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM              = 0x2504,
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_RIBBONS
    /// OPTIX_PRIMITIVE_TYPE_RIBBON                    = 0x2505,
#endif // OPTIX_OPTIONAL_FEATURE_OPTIX7_RIBBONS
    OPTIX_PRIMITIVE_TYPE_SPHERE                        = 0x2506,
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    // OPTIX_PRIMITIVE_TYPE_QUAD                       = 0x2530,
#endif // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    /// Triangle.
    OPTIX_PRIMITIVE_TYPE_TRIANGLE                      = 0x2531,
} OptixPrimitiveType;

/// Builtin flags may be bitwise combined.
///
/// \see #OptixPipelineCompileOptions::usesPrimitiveTypeFlags
typedef enum OptixPrimitiveTypeFlags
{
    /// Custom primitive.
    OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM                  = 1 << 0,
    /// B-spline lwrve of degree 2 with cirlwlar cross-section.
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE = 1 << 1,
    /// B-spline lwrve of degree 3 with cirlwlar cross-section.
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LWBIC_BSPLINE     = 1 << 2,
    /// Piecewise linear lwrve with cirlwlar cross-section.
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR            = 1 << 3,
    /// CatmullRom lwrve with cirlwlar cross-section.
    OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM        = 1 << 4,
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_RIBBONS
    OPTIX_PRIMITIVE_TYPE_FLAGS_RIBBON                  = 1 << 5,
#endif // OPTIX_OPTIONAL_FEATURE_OPTIX7_RIBBONS
    OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE                  = 1 << 6,
    /// Triangle.
    OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE                = 1 << 31,
} OptixPrimitiveTypeFlags;

/// Lwrve end cap types, for non-linear lwrves
///
typedef enum OptixLwrveEndcapFlags
{
    /// Default end caps. Round end caps for linear, no end caps for quadratic/cubic.
    OPTIX_LWRVE_ENDCAP_DEFAULT                        = 0,
    /// Flat end caps at both ends of quadratic/cubic lwrve segments. Not valid for linear.
    OPTIX_LWRVE_ENDCAP_ON                             = 1 << 0,
} OptixLwrveEndcapFlags;

/// Lwrve inputs
///
/// A lwrve is a swept surface defined by a 3D spline lwrve and a varying width (radius). A lwrve (or "strand") of
/// degree d (3=cubic, 2=quadratic, 1=linear) is represented by N > d vertices and N width values, and comprises N - d segments.
/// Each segment is defined by d+1 conselwtive vertices. Each lwrve may have a different number of vertices.
///
/// OptiX describes the lwrve array as a list of lwrve segments. The primitive id is the segment number.
/// It is the user's responsibility to maintain a mapping between lwrves and lwrve segments.
/// Each index buffer entry i = indexBuffer[primid] specifies the start of a lwrve segment,
/// represented by d+1 conselwtive vertices in the vertex buffer,
/// and d+1 conselwtive widths in the width buffer. Width is interpolated the same
/// way vertices are interpolated, that is, using the lwrve basis.
///
/// Each lwrves build input has only one SBT record.
/// To create lwrves with different materials in the same BVH, use multiple build inputs.
///
/// \see #OptixBuildInput::lwrveArray
typedef struct OptixBuildInputLwrveArray
{
    /// Lwrve degree and basis
    /// \see #OptixPrimitiveType
    OptixPrimitiveType lwrveType;
    /// Number of primitives. Each primitive is a polynomial lwrve segment.
    unsigned int numPrimitives;

    /// Pointer to host array of device pointers, one per motion step. Host array size must match number of
    /// motion keys as set in #OptixMotionOptions (or an array of size 1 if OptixMotionOptions::numKeys is set
    /// to 1). Each per-motion-key device pointer must point to an array of floats (the vertices of the
    /// lwrves).
    const LWdeviceptr* vertexBuffers;
    /// Number of vertices in each buffer in vertexBuffers.
    unsigned int numVertices;
    /// Stride between vertices. If set to zero, vertices are assumed to be tightly
    /// packed and stride is sizeof( float3 ).
    unsigned int vertexStrideInBytes;

    /// Parallel to vertexBuffers: a device pointer per motion step, each with numVertices float values,
    /// specifying the lwrve width (radius) corresponding to each vertex.
    const LWdeviceptr* widthBuffers;
    /// Stride between widths. If set to zero, widths are assumed to be tightly
    /// packed and stride is sizeof( float ).
    unsigned int widthStrideInBytes;

    /// Reserved for future use.
    const LWdeviceptr* normalBuffers;
    /// Reserved for future use.
    unsigned int normalStrideInBytes;

    /// Device pointer to array of unsigned ints, one per lwrve segment.
    /// This buffer is required (unlike for OptixBuildInputTriangleArray).
    /// Each index is the start of degree+1 conselwtive vertices in vertexBuffers,
    /// and corresponding widths in widthBuffers and normals in normalBuffers.
    /// These define a single segment. Size of array is numPrimitives.
    LWdeviceptr indexBuffer;
    /// Stride between indices. If set to zero, indices are assumed to be tightly
    /// packed and stride is sizeof( unsigned int ).
    unsigned int indexStrideInBytes;

    /// Combination of OptixGeometryFlags describing the
    /// primitive behavior.
    unsigned int flag;

    /// Primitive index bias, applied in optixGetPrimitiveIndex().
    /// Sum of primitiveIndexOffset and number of primitives must not overflow 32bits.
    unsigned int primitiveIndexOffset;

    /// End cap flags, see OptixLwrveEndcapFlags
    unsigned int endcapFlags;
} OptixBuildInputLwrveArray;

/// Sphere inputs
///
/// A sphere is defined by a center point and a radius.
/// Each center point is represented by a vertex in the vertex buffer.
/// There is either a single radius for all spheres, or the radii are represented by entries in the radius buffer.
///
/// The vertex buffers and radius buffers point to a host array of device pointers, one per motion step.
/// Host array size must match the number of motion keys as set in #OptixMotionOptions (or an array of size 1 if OptixMotionOptions::numKeys is set
/// to 0 or 1). Each per motion key device pointer must point to an array of vertices corresponding to the center points of the spheres, or
/// an array of 1 or N radii. Format OPTIX_VERTEX_FORMAT_FLOAT3 is used for vertices, OPTIX_VERTEX_FORMAT_FLOAT for radii.
///
/// \see #OptixBuildInput::sphereArray
typedef struct OptixBuildInputSphereArray
{
  /// Pointer to host array of device pointers, one per motion step. Host array size must match number of
  /// motion keys as set in #OptixMotionOptions (or an array of size 1 if OptixMotionOptions::numKeys is set
  /// to 1). Each per-motion-key device pointer must point to an array of floats (the center points of 
  /// the spheres). 
  const LWdeviceptr* vertexBuffers;

  /// Stride between vertices. If set to zero, vertices are assumed to be tightly
  /// packed and stride is sizeof( float3 ).
  unsigned int vertexStrideInBytes;
  /// Number of vertices in each buffer in vertexBuffers.
  unsigned int numVertices;

  /// Parallel to vertexBuffers: a device pointer per motion step, each with numRadii float values,
  /// specifying the sphere radius corresponding to each vertex.
  const LWdeviceptr* radiusBuffers;
  /// Stride between radii. If set to zero, widths are assumed to be tightly
  /// packed and stride is sizeof( float ).
  unsigned int radiusStrideInBytes;
  /// Boolean value indicating whether a single radius per radius buffer is used,
  /// or the number of radii in radiusBuffers equals numVertices.
  int singleRadius;

  /// Array of flags, to specify flags per sbt record,
  /// combinations of OptixGeometryFlags describing the
  /// primitive behavior, size must match numSbtRecords
  const unsigned int* flags;

  /// Number of sbt records available to the sbt index offset override.
  unsigned int numSbtRecords;
  /// Device pointer to per-primitive local sbt index offset buffer. May be NULL.
  /// Every entry must be in range [0,numSbtRecords-1].
  /// Size needs to be the number of primitives.
  LWdeviceptr sbtIndexOffsetBuffer;
  /// Size of type of the sbt index offset. Needs to be 0, 1, 2 or 4 (8, 16 or 32 bit).
  unsigned int sbtIndexOffsetSizeInBytes;
  /// Stride between the sbt index offsets. If set to zero, the offsets are assumed to be tightly
  /// packed and the stride matches the size of the type (sbtIndexOffsetSizeInBytes).
  unsigned int sbtIndexOffsetStrideInBytes;

  /// Primitive index bias, applied in optixGetPrimitiveIndex().
  /// Sum of primitiveIndexOffset and number of primitives must not overflow 32bits.
  unsigned int primitiveIndexOffset;
} OptixBuildInputSphereArray;

/// AABB inputs
typedef struct OptixAabb
{
    float minX;  ///< Lower extent in X direction.
    float minY;  ///< Lower extent in Y direction.
    float minZ;  ///< Lower extent in Z direction.
    float maxX;  ///< Upper extent in X direction.
    float maxY;  ///< Upper extent in Y direction.
    float maxZ;  ///< Upper extent in Z direction.
} OptixAabb;

/// Custom primitive inputs
///
/// \see #OptixBuildInput::lwstomPrimitiveArray
typedef struct OptixBuildInputLwstomPrimitiveArray
{
    /// Points to host array of device pointers to AABBs (type OptixAabb), one per motion step.
    /// Host array size must match number of motion keys as set in OptixMotionOptions (or an array of size 1
    /// if OptixMotionOptions::numKeys is set to 1).
    /// Each device pointer must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.
    const LWdeviceptr* aabbBuffers;

    /// Number of primitives in each buffer (i.e., per motion step) in
    /// #OptixBuildInputLwstomPrimitiveArray::aabbBuffers.
    unsigned int numPrimitives;

    /// Stride between AABBs (per motion key). If set to zero, the aabbs are assumed to be tightly
    /// packed and the stride is assumed to be sizeof( OptixAabb ).
    /// If non-zero, the value must be a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT.
    unsigned int strideInBytes;

    /// Array of flags, to specify flags per sbt record,
    /// combinations of OptixGeometryFlags describing the
    /// primitive behavior, size must match numSbtRecords
    const unsigned int* flags;

    /// Number of sbt records available to the sbt index offset override.
    unsigned int numSbtRecords;

    /// Device pointer to per-primitive local sbt index offset buffer. May be NULL.
    /// Every entry must be in range [0,numSbtRecords-1].
    /// Size needs to be the number of primitives.
    LWdeviceptr sbtIndexOffsetBuffer;

    /// Size of type of the sbt index offset. Needs to be 0, 1, 2 or 4 (8, 16 or 32 bit).
    unsigned int sbtIndexOffsetSizeInBytes;

    /// Stride between the index offsets. If set to zero, the offsets are assumed to be tightly
    /// packed and the stride matches the size of the type (sbtIndexOffsetSizeInBytes).
    unsigned int sbtIndexOffsetStrideInBytes;

    /// Primitive index bias, applied in optixGetPrimitiveIndex().
    /// Sum of primitiveIndexOffset and number of primitive must not overflow 32bits.
    unsigned int primitiveIndexOffset;
} OptixBuildInputLwstomPrimitiveArray;

/// Instance and instance pointer inputs
///
/// \see #OptixBuildInput::instanceArray
typedef struct OptixBuildInputInstanceArray
{
    /// If OptixBuildInput::type is OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS instances and
    /// aabbs should be interpreted as arrays of pointers instead of arrays of structs.
    ///
    /// This pointer must be a multiple of OPTIX_INSTANCE_BYTE_ALIGNMENT if
    /// OptixBuildInput::type is OPTIX_BUILD_INPUT_TYPE_INSTANCES. The array elements must
    /// be a multiple of OPTIX_INSTANCE_BYTE_ALIGNMENT if OptixBuildInput::type is
    /// OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS.
    LWdeviceptr instances;

    /// Number of elements in #OptixBuildInputInstanceArray::instances.
    unsigned int numInstances;
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    // For future consideration: These fields were removed in OPTIX_ABI 43 which needs
    // to be taken into account if fields are added in the future.
    //LWdeviceptr aabbs;
    //unsigned int numAabbs;

    //The following is added in OPTIX_ABI 54
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    /// Only valid for OPTIX_BUILD_INPUT_TYPE_INSTANCE
    /// Defines the stride between instances. A stride of 0 indicates a tight packing, i.e.,
    /// stride = sizeof( OptixInstance )
    unsigned int instanceStride;

    /// Device pointer to an array of replacement pointers to visibility map arrays (array of objects (pointer objects)).
    /// The size of the array must match the number of instances provided with this build input.
    /// Each pointer in the array corresponds to an instance and replaces the VM array used by geometries built
    /// with the OPTIX_GEOMETRY_FLAG_REPLACEABLE_VM_ARRAY flag.
    LWdeviceptr  vmArrayReplacements;

    /// Stride between vmArray replacement pointers in array vmArrayReplacements. A stride of 0 indicates a tight packing.
    unsigned int vmReplacementsStride;

#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
} OptixBuildInputInstanceArray;

/// Enum to distinguish the different build input types.
///
/// \see #OptixBuildInput::type
typedef enum OptixBuildInputType
{
    /// Triangle inputs. \see #OptixBuildInputTriangleArray
    OPTIX_BUILD_INPUT_TYPE_TRIANGLES = 0x2141,
    /// Custom primitive inputs. \see #OptixBuildInputLwstomPrimitiveArray
    OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES = 0x2142,
    /// Instance inputs. \see #OptixBuildInputInstanceArray
    OPTIX_BUILD_INPUT_TYPE_INSTANCES = 0x2143,
    /// Instance pointer inputs. \see #OptixBuildInputInstanceArray
    OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS = 0x2144,
    /// Lwrve inputs. \see #OptixBuildInputLwrveArray
    OPTIX_BUILD_INPUT_TYPE_LWRVES = 0x2145,
    /// Sphere inputs. \see #OptixBuildInputSphereArray
    OPTIX_BUILD_INPUT_TYPE_SPHERES = 0x2146
} OptixBuildInputType;

/// Build inputs.
///
/// All of them support motion and the size of the data arrays needs to match the number of motion steps
///
/// \see #optixAccelComputeMemoryUsage(), #optixAccelBuild()
typedef struct OptixBuildInput
{
    /// The type of the build input.
    OptixBuildInputType type;

    union
    {
        /// Triangle inputs.
        OptixBuildInputTriangleArray triangleArray;
        /// Lwrve inputs.
        OptixBuildInputLwrveArray lwrveArray;
        /// Sphere inputs.
        OptixBuildInputSphereArray sphereArray;
        /// Custom primitive inputs.
        OptixBuildInputLwstomPrimitiveArray lwstomPrimitiveArray;
        /// Instance and instance pointer inputs.
        OptixBuildInputInstanceArray instanceArray;
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
        // TODO Why is the padding so huge? Why is there padding at all?
        // Why is there an explicit padding, but not for other types like  OptixProgramGroupDesc?
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
        char pad[1024];
    };
} OptixBuildInput;

#ifndef OPTIX_OPTIONAL_FEATURE_OPTIX7_32_BIT_SUPPORT
// Some 32-bit tools use this header. This static_assert fails for them because
// the default enum size is 4 bytes, rather than 8, under 32-bit compilers.
// This #ifndef allows them to disable the static assert.

// TODO Define a static assert for C/pre-C++-11
#if defined( __cplusplus ) && __cplusplus >= 201103L
static_assert( sizeof( OptixBuildInput ) == 8 + 1024, "OptixBuildInput has wrong size" );
#endif
#endif // OPTIX_OPTIONAL_FEATURE_OPTIX7_32_BIT_SUPPORT

/// Flags set on the #OptixInstance::flags.
///
/// These can be or'ed together to combine multiple flags.
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// These flags are interpreted on the device by rtcore, and should mirror the
/// RtcInstanceFlags.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef enum OptixInstanceFlags
{
    /// No special flag set
    OPTIX_INSTANCE_FLAG_NONE = 0,

    /// Prevent triangles from getting lwlled due to their orientation.
    /// Effectively ignores ray flags
    /// OPTIX_RAY_FLAG_LWLL_BACK_FACING_TRIANGLES and OPTIX_RAY_FLAG_LWLL_FRONT_FACING_TRIANGLES.
    OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_LWLLING = 1u << 0,

    /// Flip triangle orientation.
    /// This affects front/backface lwlling as well as the reported face in case of a hit.
    OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING = 1u << 1,

    /// Disable anyhit programs for all geometries of the instance.
    /// Can be overridden by OPTIX_RAY_FLAG_ENFORCE_ANYHIT.
    /// This flag is mutually exclusive with OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT.
    OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT = 1u << 2,

    /// Enables anyhit programs for all geometries of the instance.
    /// Overrides OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    /// Can be overridden by OPTIX_RAY_FLAG_DISABLE_ANYHIT.
    /// This flag is mutually exclusive with OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT.
    OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT = 1u << 3,

#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    /// Deprecated flag to disable the instance transformation.
    /// This flag never actually worked.
    /// If there where users they must have been setting the instance transform to identity,
    /// so the broken flag didn't cause any visible issues.
    /// WARNING: take care that any new flag overlapping with this deprecated flag won't trigger bugs in user code using an older SDK!
    /// OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM = 1u << 6,
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    /// Force 4-state VMs to behave as 2-state VMs during traversal.
    OPTIX_INSTANCE_FLAG_FORCE_VM_2_STATE           = 1u << 4,
    /// Don't perform VM query for this instance. GAS must be built with ALLOW_DISABLE_VMS for this to be valid.
    /// This flag overrides FORCE_VM_2_STATE instance and ray flags.
    OPTIX_INSTANCE_FLAG_DISABLE_VMS                = 1u << 5,
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD

} OptixInstanceFlags;

/// Instances
///
/// \see #OptixBuildInputInstanceArray::instances
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// This struct is interpreted on the device by rtcore, and should mirror the
/// RtcFatInstance.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef struct OptixInstance
{
    /// affine object-to-world transformation as 3x4 matrix in row-major layout
    float transform[12];

    /// Application supplied ID. The maximal ID can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID.
    unsigned int instanceId;

    /// SBT record offset.  Will only be used for instances of geometry acceleration structure (GAS) objects.
    /// Needs to be set to 0 for instances of instance acceleration structure (IAS) objects. The maximal SBT offset
    /// can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_SBT_OFFSET.
    unsigned int sbtOffset;

    /// Visibility mask. If rayMask & instanceMask == 0 the instance is lwlled. The number of available bits can be
    /// queried using OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK.
    unsigned int visibilityMask;

    /// Any combination of OptixInstanceFlags is allowed.
    unsigned int flags;

    /// Set with an OptixTraversableHandle.
    OptixTraversableHandle traversableHandle;

    /// round up to 80-byte, to ensure 16-byte alignment
    unsigned int pad[2];
} OptixInstance;

/// Builder Options
///
/// Used for #OptixAccelBuildOptions::buildFlags. Can be or'ed together.
typedef enum OptixBuildFlags
{
    /// No special flags set.
    OPTIX_BUILD_FLAG_NONE = 0,

    /// Allow updating the build with new vertex positions with subsequent calls to
    /// optixAccelBuild.
    OPTIX_BUILD_FLAG_ALLOW_UPDATE = 1u << 0,

    OPTIX_BUILD_FLAG_ALLOW_COMPACTION = 1u << 1,

    OPTIX_BUILD_FLAG_PREFER_FAST_TRACE = 1u << 2,

    OPTIX_BUILD_FLAG_PREFER_FAST_BUILD = 1u << 3,

    /// Allow random access to build input vertices
    /// See optixGetTriangleVertexData
    ///     optixGetLinearLwrveVertexData
    ///     optixGetQuadraticBSplineVertexData
    ///     optixGetLwbicBSplineVertexData
    ///     optixGetCatmullRomVertexData
    ///     optixGetSphereData
    OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS = 1u << 4,

    /// Allow random access to instances
    /// See optixGetInstanceTraversableFromIAS
    OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS = 1u << 5,

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    /// Support updating the VM array and VM indices on refits.
    /// May increase AS size and may have a small negative impact on traversal performance.
    /// If this flag is absent, all VM inputs must remain unchanged between the initial AS builds and their subsequent refits.
    OPTIX_BUILD_FLAG_ALLOW_VM_UPDATE = 1u << 6,

    /// If enabled, any instances referencing this GAS are allowed to disable the VM test through the DISABLE_VMS flag instance flag.
    /// Note that the GAS will not be optimized for the attached VM Arrays if this flag is set,
    /// which may result in reduced traversal performance.
    OPTIX_BUILD_FLAG_ALLOW_DISABLE_VMS = 1u << 7,

    /// Support updating the DMM array and DMM indices on refits.
    OPTIX_BUILD_FLAG_ALLOW_DMM_UPDATE = 1u << 8,

    /// Needs to be set for IAS builds if any referenced GAS has DMMs.
    OPTIX_BUILD_FLAG_IAS_REFERENCES_GAS_WITH_DMM = 1u << 9,

    /// FIXME, this flag should be moved to flags in the instance build input type
    OPTIX_BUILD_FLAG_IAS_USES_VM_REPLACEMENTS = 1u << 10,
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
} OptixBuildFlags;


#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
/// Flags defining behavior of VMs in a VM array.
typedef enum OptixVisibilityMapFlags
{
    OPTIX_VISIBILITY_MAP_FLAG_NONE              = 0,
    OPTIX_VISIBILITY_MAP_FLAG_PREFER_FAST_TRACE = 1 << 0,
    OPTIX_VISIBILITY_MAP_FLAG_PREFER_FAST_BUILD = 1 << 1,
} OptixVisibilityMapFlags;

/// VM descriptor.
typedef struct OptixVisibilityMapDesc
{
    /// Byte offset to VM
    unsigned int  byteOffset;
    /// Number of micro-triangles is 4^level. Valid levels are [0, 12]
    unsigned short subdivisionLevel;
    /// OptixVisibilityMapFormat
    unsigned short format;
} OptixVisibilityMapDesc;

/// VM usage count.
typedef struct OptixVisibilityMapUsageCount
{
    /// Number of VMs with the format and subdivision level specified in this struct.
    unsigned int            count;
    /// Number of micro-triangles is 4^level. Valid levels are [0, 12]
    unsigned int            subdivisionLevel;
    /// VM format.
    OptixVisibilityMapFormat format;
} OptixVisibilityMapUsageCount;

/// Inputs to VM array construction.
typedef struct OptixVisibilityMapArrayBuildInput
{
    /// Applies to all VMs in array.
    OptixVisibilityMapFlags flags;

    /// 128B aligned base pointer for raw VM input data.
    LWdeviceptr inputBuffer;

    /// One OptixVisibilityMapDesc entry per VM.
    LWdeviceptr perVmDescBuffer;

    /// Stride between OptixVisibilityMapDescs in perVmDescBuffer.
    /// If set to zero, the visibility map descriptors are assumed to be tightly packed and the stride is assumed to be sizeof( OptixVisibilityMapDesc ).
    unsigned int perVmDescStrideInBytes;

    /// Number of OptixVisibilityMapUsageCount entries.
    unsigned int numVmUsageCounts;
    /// Histogram over VMs for input format and subdiv combinations.
    const OptixVisibilityMapUsageCount* vmUsageCounts;
} OptixVisibilityMapArrayBuildInput;

/// Use to select which OptixEmittedMicromeshProperty* to output from optixMicromeshEmitProperties.
typedef enum OptixMicromeshPropertyType
{
    /// Vm or DMM array size in bytes.
    OPTIX_MICROMESH_PROPERTY_TYPE_LWRRENT_SIZE = 0,
} OptixMicromeshPropertyType;

/// Specifies a type and output destination for emitted VM/DMM post-build properties.
typedef struct OptixMicromeshEmitDesc
{
    /// Output buffer for the properties
    LWdeviceptr result;
    /// Type of information requested
    OptixMicromeshPropertyType type;
} OptixMicromeshEmitDesc;

/// Conservative memory requirements for building a VM/DMM array
typedef struct OptixMicromeshBufferSizes
{
    size_t outputSizeInBytes;
} OptixMicromeshBufferSizes;

/// Buffer inputs for VM/DMM array builds.
typedef struct OptixMicromeshBuffers
{
    /// VM/DMM array output buffer buffer.
    LWdeviceptr output;
    /// Input/output buffer size (~0ULL if the size is unknown. Used for validation only)
    size_t   outputSizeInBytes;

} OptixMicromeshBuffers;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD


/// Enum to specify the acceleration build operation.
///
/// Used in OptixAccelBuildOptions, which is then passed to optixAccelBuild and
/// optixAccelComputeMemoryUsage, this enum indicates whether to do a build or an update
/// of the acceleration structure.
///
/// Acceleration structure updates utilize the same acceleration structure, but with
/// updated bounds.  Updates are typically much faster than builds, however, large
/// perturbations can degrade the quality of the acceleration structure.
///
/// \see #optixAccelComputeMemoryUsage(), #optixAccelBuild(), #OptixAccelBuildOptions
typedef enum OptixBuildOperation
{
    /// Perform a full build operation
    OPTIX_BUILD_OPERATION_BUILD = 0x2161,
    /// Perform an update using new bounds
    OPTIX_BUILD_OPERATION_UPDATE = 0x2162,
} OptixBuildOperation;

/// Enum to specify motion flags.
///
/// \see #OptixMotionOptions::flags.
typedef enum OptixMotionFlags
{
    OPTIX_MOTION_FLAG_NONE         = 0,
    OPTIX_MOTION_FLAG_START_VANISH = 1u << 0,
    OPTIX_MOTION_FLAG_END_VANISH   = 1u << 1
} OptixMotionFlags;

/// Motion options
///
/// \see #OptixAccelBuildOptions::motionOptions, #OptixMatrixMotionTransform::motionOptions,
///      #OptixSRTMotionTransform::motionOptions
typedef struct OptixMotionOptions
{
    /// If numKeys > 1, motion is enabled. timeBegin,
    /// timeEnd and flags are all ignored when motion is disabled.
    unsigned short numKeys;

    /// Combinations of #OptixMotionFlags
    unsigned short flags;

    /// Point in time where motion starts.
    float timeBegin;

    /// Point in time where motion ends.
    float timeEnd;
} OptixMotionOptions;

/// Build options for acceleration structures.
///
/// \see #optixAccelComputeMemoryUsage(), #optixAccelBuild()
typedef struct OptixAccelBuildOptions
{
    /// Combinations of OptixBuildFlags
    unsigned int buildFlags;

    /// If OPTIX_BUILD_OPERATION_UPDATE the output buffer is assumed to contain the result
    /// of a full build with OPTIX_BUILD_FLAG_ALLOW_UPDATE set and using the same number of
    /// primitives.  It is updated incrementally to reflect the current position of the
    /// primitives.
#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    /// If a BLAS has been built with OPTIX_BUILD_FLAG_ALLOW_VM_UPDATE, new VM arrays
    /// and VM indices may be provided to the refit.
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
    OptixBuildOperation operation;

    /// Options for motion.
    OptixMotionOptions motionOptions;
} OptixAccelBuildOptions;

/// Struct for querying builder allocation requirements.
///
/// Once queried the sizes should be used to allocate device memory of at least these sizes.
///
/// \see #optixAccelComputeMemoryUsage()
typedef struct OptixAccelBufferSizes
{
    /// The size in bytes required for the outputBuffer parameter to optixAccelBuild when
    /// doing a build (OPTIX_BUILD_OPERATION_BUILD).
    size_t outputSizeInBytes;

    /// The size in bytes required for the tempBuffer paramter to optixAccelBuild when
    /// doing a build (OPTIX_BUILD_OPERATION_BUILD).
    size_t tempSizeInBytes;

    /// The size in bytes required for the tempBuffer parameter to optixAccelBuild
    /// when doing an update (OPTIX_BUILD_OPERATION_UPDATE).  This value can be different
    /// than tempSizeInBytes used for a full build.  Only non-zero if
    /// OPTIX_BUILD_FLAG_ALLOW_UPDATE flag is set in OptixAccelBuildOptions.
    size_t tempUpdateSizeInBytes;
} OptixAccelBufferSizes;

/// Properties which can be emitted during acceleration structure build.
///
/// \see #OptixAccelEmitDesc::type.
typedef enum OptixAccelPropertyType
{
    /// Size of a compacted acceleration structure. The device pointer points to a uint64.
    OPTIX_PROPERTY_TYPE_COMPACTED_SIZE = 0x2181,

    /// OptixAabb * numMotionSteps
    OPTIX_PROPERTY_TYPE_AABBS = 0x2182,
} OptixAccelPropertyType;

/// Specifies a type and output destination for emitted post-build properties.
///
/// \see #optixAccelBuild()
typedef struct OptixAccelEmitDesc
{
    /// Output buffer for the properties
    LWdeviceptr result;

    /// Requested property
    OptixAccelPropertyType type;
} OptixAccelEmitDesc;

/// Used to store information related to relocation of acceleration structures.
///
/// \see #optixAccelGetRelocationInfo(), #optixAccelCheckRelocationCompatibility(), #optixAccelRelocate()
typedef struct OptixAccelRelocationInfo
{
    /// Opaque data, used internally, should not be modified
    unsigned long long info[4];
} OptixAccelRelocationInfo;

/// Static transform
///
/// The device address of instances of this type must be a multiple of OPTIX_TRANSFORM_BYTE_ALIGNMENT.
///
/// \see #optixColwertPointerToTraversableHandle()
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// This struct is interpreted on the device by rtcore, and should mirror
/// RtcTravStaticTransform.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef struct OptixStaticTransform
{
    /// The traversable transformed by this transformation
    OptixTraversableHandle child;

    /// Padding to make the transformations 16 byte aligned
    unsigned int pad[2];

    /// Affine object-to-world transformation as 3x4 matrix in row-major layout
    float transform[12];

    /// Affine world-to-object transformation as 3x4 matrix in row-major layout
    /// Must be the ilwerse of the transform matrix
    float ilwTransform[12];
} OptixStaticTransform;

/// Represents a matrix motion transformation.
///
/// The device address of instances of this type must be a multiple of OPTIX_TRANSFORM_BYTE_ALIGNMENT.
///
/// This struct, as defined here, handles only N=2 motion keys due to the fixed array length of its transform member.
/// The following example shows how to create instances for an arbitrary number N of motion keys:
///
/// \code
/// float matrixData[N][12];
/// ... // setup matrixData
///
/// size_t transformSizeInBytes = sizeof( OptixMatrixMotionTransform ) + ( N-2 ) * 12 * sizeof( float );
/// OptixMatrixMotionTransform* matrixMoptionTransform = (OptixMatrixMotionTransform*) malloc( transformSizeInBytes );
/// memset( matrixMoptionTransform, 0, transformSizeInBytes );
///
/// ... // setup other members of matrixMoptionTransform
/// matrixMoptionTransform->motionOptions.numKeys/// = N;
/// memcpy( matrixMoptionTransform->transform, matrixData, N * 12 * sizeof( float ) );
///
/// ... // copy matrixMoptionTransform to device memory
/// free( matrixMoptionTransform )
/// \endcode
///
/// \see #optixColwertPointerToTraversableHandle()
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// This struct is interpreted on the device by rtcore, and should mirror
/// RtcTravMatrixMotionTransform.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef struct OptixMatrixMotionTransform
{
    /// The traversable that is transformed by this transformation
    OptixTraversableHandle child;

    /// The motion options for this transformation
    OptixMotionOptions motionOptions;

    /// Padding to make the transformation 16 byte aligned
    unsigned int pad[3];

    /// Affine object-to-world transformation as 3x4 matrix in row-major layout
    float transform[2][12];
} OptixMatrixMotionTransform;

/// Represents an SRT transformation.
///
/// An SRT transformation can represent a smooth rotation with fewer motion keys than a matrix transformation. Each
/// motion key is constructed from elements taken from a matrix S, a quaternion R, and a translation T.
///
/// The scaling matrix
/// \f$S = \begin{bmatrix} sx & a & b & pvx \\ 0 & sy & c & pvy \\ 0 & 0  & sz & pvz \end{bmatrix}\f$
//      [ sx   a   b  pvx ]
//  S = [  0  sy   c  pvy ]
//      [  0   0  sz  pvz ]
/// defines an affine transformation that can include scale, shear, and a translation.
/// The translation allows to define the pivot point for the subsequent rotation.
///
/// The quaternion R = [ qx, qy, qz, qw ] describes a rotation  with angular component qw = cos(theta/2) and other
/// components [ qx, qy, qz ] = sin(theta/2) * [ ax, ay, az ] where the axis [ ax, ay, az ] is normalized.
///
/// The translation matrix
/// \f$T = \begin{bmatrix} 1 & 0 & 0 & tx \\ 0 & 1 & 0 & ty \\ 0 & 0 & 1 & tz \end{bmatrix}\f$
//      [  1  0  0 tx ]
//  T = [  0  1  0 ty ]
//      [  0  0  1 tz ]
/// defines another translation that is applied after the rotation. Typically, this translation includes
/// the ilwerse translation from the matrix S to reverse the translation for the pivot point for R.
///
/// To obtain the effective transformation at time t, the elements of the components of S, R, and T will be interpolated
/// linearly. The components are then multiplied to obtain the combined transformation C = T * R * S. The transformation
/// C is the effective object-to-world transformations at time t, and C^(-1) is the effective world-to-object
/// transformation at time t.
///
/// \see #OptixSRTMotionTransform::srtData, #optixColwertPointerToTraversableHandle()
typedef struct OptixSRTData
{
    /// \name Parameters describing the SRT transformation
    /// @{
    float sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz;
    /// @}
} OptixSRTData;

// TODO Define a static assert for C/pre-C++-11
#if defined( __cplusplus ) && __cplusplus >= 201103L
static_assert( sizeof( OptixSRTData ) == 16 * 4, "OptixSRTData has wrong size" );
#endif

/// Represents an SRT motion transformation.
///
/// The device address of instances of this type must be a multiple of OPTIX_TRANSFORM_BYTE_ALIGNMENT.
///
/// This struct, as defined here, handles only N=2 motion keys due to the fixed array length of its srtData member.
/// The following example shows how to create instances for an arbitrary number N of motion keys:
///
/// \code
/// OptixSRTData srtData[N];
/// ... // setup srtData
///
/// size_t transformSizeInBytes = sizeof( OptixSRTMotionTransform ) + ( N-2 ) * sizeof( OptixSRTData );
/// OptixSRTMotionTransform* srtMotionTransform = (OptixSRTMotionTransform*) malloc( transformSizeInBytes );
/// memset( srtMotionTransform, 0, transformSizeInBytes );
///
/// ... // setup other members of srtMotionTransform
/// srtMotionTransform->motionOptions.numKeys   = N;
/// memcpy( srtMotionTransform->srtData, srtData, N * sizeof( OptixSRTData ) );
///
/// ... // copy srtMotionTransform to device memory
/// free( srtMotionTransform )
/// \endcode
///
/// \see #optixColwertPointerToTraversableHandle()
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// This struct is interpreted on the device by rtcore, and should mirror
/// RtcTravSRTMotionTransform.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef struct OptixSRTMotionTransform
{
    /// The traversable transformed by this transformation
    OptixTraversableHandle child;

    /// The motion options for this transformation
    OptixMotionOptions motionOptions;

    /// Padding to make the SRT data 16 byte aligned
    unsigned int pad[3];

    /// The actual SRT data describing the transformation
    OptixSRTData srtData[2];
} OptixSRTMotionTransform;

// TODO Define a static assert for C/pre-C++-11
#if defined( __cplusplus ) && __cplusplus >= 201103L
static_assert( sizeof( OptixSRTMotionTransform ) == 8 + 12 + 12 + 2 * 16 * 4, "OptixSRTMotionTransform has wrong size" );
#endif

/// Traversable Handles
///
/// \see #optixColwertPointerToTraversableHandle()
typedef enum OptixTraversableType
{
    /// Static transforms. \see #OptixStaticTransform
    OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM = 0x21C1,
    /// Matrix motion transform. \see #OptixMatrixMotionTransform
    OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM = 0x21C2,
    /// SRT motion transform. \see #OptixSRTMotionTransform
    OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM = 0x21C3,
} OptixTraversableType;

/// Pixel formats used by the denoiser.
///
/// \see #OptixImage2D::format
typedef enum OptixPixelFormat
{
    OPTIX_PIXEL_FORMAT_HALF2  = 0x2207,               ///< two halfs, XY
    OPTIX_PIXEL_FORMAT_HALF3  = 0x2201,               ///< three halfs, RGB
    OPTIX_PIXEL_FORMAT_HALF4  = 0x2202,               ///< four halfs, RGBA
    OPTIX_PIXEL_FORMAT_FLOAT2 = 0x2208,               ///< two floats, XY
    OPTIX_PIXEL_FORMAT_FLOAT3 = 0x2203,               ///< three floats, RGB
    OPTIX_PIXEL_FORMAT_FLOAT4 = 0x2204,               ///< four floats, RGBA
    OPTIX_PIXEL_FORMAT_UCHAR3 = 0x2205,               ///< three unsigned chars, RGB
    OPTIX_PIXEL_FORMAT_UCHAR4 = 0x2206,               ///< four unsigned chars, RGBA
    OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER = 0x2209, ///< internal format
} OptixPixelFormat;

/// Image descriptor used by the denoiser.
///
/// \see #optixDenoiserIlwoke(), #optixDenoiserComputeIntensity()
typedef struct OptixImage2D
{
    /// Pointer to the actual pixel data.
    LWdeviceptr data;
    /// Width of the image (in pixels)
    unsigned int width;
    /// Height of the image (in pixels)
    unsigned int height;
    /// Stride between subsequent rows of the image (in bytes).
    unsigned int rowStrideInBytes;
    /// Stride between subsequent pixels of the image (in bytes).
    /// If set to 0, dense packing (no gaps) is assumed.
    /// For pixel format OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER it must be set to
    /// at least OptixDenoiserSizes::internalGuideLayerSizeInBytes.
    unsigned int pixelStrideInBytes;
    /// Pixel format.
    OptixPixelFormat format;
} OptixImage2D;

/// Model kind used by the denoiser.
///
/// \see #optixDenoiserCreate
typedef enum OptixDenoiserModelKind
{
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    /// obsolete, removed in ABI44
    /// Use the model provided by the associated pointer.  See the programming guide for a
    /// description of how to format the data.
    /// OPTIX_DENOISER_MODEL_KIND_USER = 0x2321,
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    /// Use the built-in model appropriate for low dynamic range input.
    OPTIX_DENOISER_MODEL_KIND_LDR = 0x2322,

    /// Use the built-in model appropriate for high dynamic range input.
    OPTIX_DENOISER_MODEL_KIND_HDR = 0x2323,

    /// Use the built-in model appropriate for high dynamic range input and support for AOVs
    OPTIX_DENOISER_MODEL_KIND_AOV = 0x2324,

    /// Use the built-in model appropriate for high dynamic range input, temporally stable
    OPTIX_DENOISER_MODEL_KIND_TEMPORAL = 0x2325,

    /// Use the built-in model appropriate for high dynamic range input and support for AOVs, temporally stable
    OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV = 0x2326,

    /// Use the built-in model appropriate for high dynamic range input and support for AOVs, upscaling 2x
    OPTIX_DENOISER_MODEL_KIND_UPSCALE2X = 0x2327,
} OptixDenoiserModelKind;

/// Options used by the denoiser
///
/// \see #optixDenoiserCreate()
typedef struct OptixDenoiserOptions
{
    // if nonzero, albedo image must be given in OptixDenoiserGuideLayer
    unsigned int guideAlbedo;

    // if nonzero, normal image must be given in OptixDenoiserGuideLayer
    unsigned int guideNormal;
} OptixDenoiserOptions;

/// Guide layer for the denoiser
///
/// \see #optixDenoiserIlwoke()
typedef struct OptixDenoiserGuideLayer
{
    // albedo/bsdf image
    OptixImage2D  albedo;

    // normal vector image (2d or 3d pixel format)
    OptixImage2D  normal;

    // 2d flow image, pixel flow from previous to current frame for each pixel
    OptixImage2D  flow;

    OptixImage2D  previousOutputInternalGuideLayer;
    OptixImage2D  outputInternalGuideLayer;
} OptixDenoiserGuideLayer;

/// Input/Output layers for the denoiser
///
/// \see #optixDenoiserIlwoke()
typedef struct OptixDenoiserLayer
{
    // input image (beauty or AOV)
    OptixImage2D  input;

    // denoised output image from previous frame if temporal model kind selected
    OptixImage2D  previousOutput;

    // denoised output for given input
    OptixImage2D  output;
} OptixDenoiserLayer;

/// Various parameters used by the denoiser
///
/// \see #optixDenoiserIlwoke()
/// \see #optixDenoiserComputeIntensity()
/// \see #optixDenoiserComputeAverageColor()
typedef enum OptixDenoiserAlphaMode
{
    /// Copy alpha (if present) from input layer, no denoising.
    OPTIX_DENOISER_ALPHA_MODE_COPY = 0,

    /// Denoise alpha separately. With AOV model kinds, treat alpha like an AOV.
    OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV = 1,

    /// With AOV model kinds, full denoise pass with alpha.
    /// This is slower than OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV.
    OPTIX_DENOISER_ALPHA_MODE_FULL_DENOISE_PASS = 2
} OptixDenoiserAlphaMode;
typedef struct OptixDenoiserParams
{
    /// alpha denoise mode
    OptixDenoiserAlphaMode denoiseAlpha;

    /// average log intensity of input image (default null pointer). points to a single float.
    /// with the default (null pointer) denoised results will not be optimal for very dark or
    /// bright input images.
    LWdeviceptr  hdrIntensity;

    /// blend factor.
    /// If set to 0 the output is 100% of the denoised input. If set to 1, the output is 100% of
    /// the unmodified input. Values between 0 and 1 will linearly interpolate between the denoised
    /// and unmodified input.
    float        blendFactor;

    /// this parameter is used when the OPTIX_DENOISER_MODEL_KIND_AOV model kind is set.
    /// average log color of input image, separate for RGB channels (default null pointer).
    /// points to three floats. with the default (null pointer) denoised results will not be
    /// optimal.
    LWdeviceptr  hdrAverageColor;
} OptixDenoiserParams;

/// Various sizes related to the denoiser.
///
/// \see #optixDenoiserComputeMemoryResources()
typedef struct OptixDenoiserSizes
{
    /// Size of state memory passed to #optixDenoiserSetup, #optixDenoiserIlwoke.
    size_t stateSizeInBytes;

    /// Size of scratch memory passed to #optixDenoiserSetup, #optixDenoiserIlwoke.
    /// Overlap added to dimensions passed to #optixDenoiserComputeMemoryResources.
    size_t withOverlapScratchSizeInBytes;

    /// Size of scratch memory passed to #optixDenoiserSetup, #optixDenoiserIlwoke.
    /// No overlap added.
    size_t withoutOverlapScratchSizeInBytes;

    /// Overlap on all four tile sides.
    unsigned int overlapWindowSizeInPixels;

    /// Size of scratch memory passed to #optixDenoiserComputeAverageColor.
    /// The size is independent of the tile/image resolution.
    size_t computeAverageColorSizeInBytes;

    /// Size of scratch memory passed to #optixDenoiserComputeIntensity.
    /// The size is independent of the tile/image resolution.
    size_t computeIntensitySizeInBytes;

    /// Number of bytes for each pixel in internal guide layers.
    size_t internalGuideLayerSizeInBytes;
} OptixDenoiserSizes;

/// Ray flags passed to the device function #optixTrace().  These affect the behavior of
/// traversal per invocation.
///
/// \see #optixTrace()
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// These flags are interpreted on the device by rtcore, and should mirror
/// RtcRayFlags.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef enum OptixRayFlags
{
    /// No change from the behavior configured for the individual AS.
    OPTIX_RAY_FLAG_NONE = 0u,

    /// Disables anyhit programs for the ray.
    /// Overrides OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT.
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_ENFORCE_ANYHIT,
    /// OPTIX_RAY_FLAG_LWLL_DISABLED_ANYHIT, OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT.
    OPTIX_RAY_FLAG_DISABLE_ANYHIT = 1u << 0,

    /// Forces anyhit program exelwtion for the ray.
    /// Overrides OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT as well as OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT.
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_DISABLE_ANYHIT,
    /// OPTIX_RAY_FLAG_LWLL_DISABLED_ANYHIT, OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT.
    OPTIX_RAY_FLAG_ENFORCE_ANYHIT = 1u << 1,

    /// Terminates the ray after the first hit and exelwtes
    /// the closesthit program of that hit.
    OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1u << 2,

    /// Disables closesthit programs for the ray, but still exelwtes miss program in case of a miss.
    OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT = 1u << 3,

    /// Do not intersect triangle back faces
    /// (respects a possible face change due to instance flag
    /// OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING).
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_LWLL_FRONT_FACING_TRIANGLES.
    OPTIX_RAY_FLAG_LWLL_BACK_FACING_TRIANGLES = 1u << 4,

    /// Do not intersect triangle front faces
    /// (respects a possible face change due to instance flag
    /// OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING).
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_LWLL_BACK_FACING_TRIANGLES.
    OPTIX_RAY_FLAG_LWLL_FRONT_FACING_TRIANGLES = 1u << 5,

    /// Do not intersect geometry which disables anyhit programs
    /// (due to setting geometry flag OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT or
    /// instance flag OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT).
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT,
    /// OPTIX_RAY_FLAG_ENFORCE_ANYHIT, OPTIX_RAY_FLAG_DISABLE_ANYHIT.
    OPTIX_RAY_FLAG_LWLL_DISABLED_ANYHIT = 1u << 6,

    /// Do not intersect geometry which have an enabled anyhit program
    /// (due to not setting geometry flag OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT or
    /// setting instance flag OPTIX_INSTANCE_FLAG_ENFORCE_ANYHIT).
    /// This flag is mutually exclusive with OPTIX_RAY_FLAG_LWLL_DISABLED_ANYHIT,
    /// OPTIX_RAY_FLAG_ENFORCE_ANYHIT, OPTIX_RAY_FLAG_DISABLE_ANYHIT.
    OPTIX_RAY_FLAG_LWLL_ENFORCED_ANYHIT = 1u << 7,

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    OPTIX_RAY_FLAG_FORCE_VM_2_STATE = 1u << 10,
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
} OptixRayFlags;

/// Transform
///
/// OptixTransformType is used by the device function #optixGetTransformTypeFromHandle() to
/// determine the type of the OptixTraversableHandle returned from
/// optixGetTransformListHandle().
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// The values of this enum are used on the device by rtcore, and should mirror
/// RtcTransformType.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef enum OptixTransformType
{
    OPTIX_TRANSFORM_TYPE_NONE                    = 0, ///< Not a transformation
    OPTIX_TRANSFORM_TYPE_STATIC_TRANSFORM        = 1, ///< \see #OptixStaticTransform
    OPTIX_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM = 2, ///< \see #OptixMatrixMotionTransform
    OPTIX_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM    = 3, ///< \see #OptixSRTMotionTransform
    OPTIX_TRANSFORM_TYPE_INSTANCE                = 4, ///< \see #OptixInstance
} OptixTransformType;

/// Specifies the set of valid traversable graphs that may be
/// passed to invocation of #optixTrace(). Flags may be bitwise combined.
typedef enum OptixTraversableGraphFlags
{
    ///  Used to signal that any traversable graphs is valid.
    ///  This flag is mutually exclusive with all other flags.
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY = 0,

    ///  Used to signal that a traversable graph of a single Geometry Acceleration
    ///  Structure (GAS) without any transforms is valid. This flag may be combined with
    ///  other flags except for OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY.
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS = 1u << 0,

    ///  Used to signal that a traversable graph of a single Instance Acceleration
    ///  Structure (IAS) directly connected to Geometry Acceleration Structure (GAS)
    ///  traversables without transform traversables in between is valid.  This flag may
    ///  be combined with other flags except for OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY.
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING = 1u << 1,
} OptixTraversableGraphFlags;

/// Optimization levels
///
/// \see #OptixModuleCompileOptions::optLevel
typedef enum OptixCompileOptimizationLevel
{
    /// Default is to run all optimizations
    OPTIX_COMPILE_OPTIMIZATION_DEFAULT = 0,
    /// No optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 = 0x2340,
    /// Some optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_1 = 0x2341,
    /// Most optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_2 = 0x2342,
    /// All optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 = 0x2343,
} OptixCompileOptimizationLevel;

/// Debug levels
///
/// \see #OptixModuleCompileOptions::debugLevel
typedef enum OptixCompileDebugLevel
{
    /// Default lwrrently is minimal
    OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT  = 0,
    /// No debug information
    OPTIX_COMPILE_DEBUG_LEVEL_NONE     = 0x2350,
    /// Generate information that does not impact performance.
    /// Note this replaces OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO.
    OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL  = 0x2351,
    /// Generate some debug information with slight performance cost
    OPTIX_COMPILE_DEBUG_LEVEL_MODERATE = 0x2353,
    /// Generate full debug information
    OPTIX_COMPILE_DEBUG_LEVEL_FULL     = 0x2352,
} OptixCompileDebugLevel;

/// Module compilation state.
///
/// \see #optixModuleGetCompilationState(), #optixModuleCreateFromPTXWithTasks()
typedef enum OptixModuleCompileState
{
    /// No OptixTask objects have started
    OPTIX_MODULE_COMPILE_STATE_NOT_STARTED       = 0x2360,

    /// Started, but not all OptixTask objects have completed. No detected failures.
    OPTIX_MODULE_COMPILE_STATE_STARTED           = 0x2361,

    /// Not all OptixTask objects have completed, but at least one has failed.
    OPTIX_MODULE_COMPILE_STATE_IMPENDING_FAILURE = 0x2362,

    /// All OptixTask objects have completed, and at least one has failed
    OPTIX_MODULE_COMPILE_STATE_FAILED            = 0x2363,

    /// All OptixTask objects have completed. The OptixModule is ready to be used.
    OPTIX_MODULE_COMPILE_STATE_COMPLETED         = 0x2364,
} OptixModuleCompileState;



/// Struct for specifying specializations for pipelineParams as specified in
/// OptixPipelineCompileOptions::pipelineLaunchParamsVariableName.
///
/// The bound values are supposed to represent a constant value in the
/// pipelineParams. OptiX will attempt to locate all loads from the pipelineParams and
/// correlate them to the appropriate bound value, but there are cases where OptiX cannot
/// safely or reliably do this. For example if the pointer to the pipelineParams is passed
/// as an argument to a non-inline function or the offset of the load to the
/// pipelineParams cannot be statically determined (e.g. accessed in a loop). No module
/// should rely on the value being specialized in order to work correctly.  The values in
/// the pipelineParams specified on optixLaunch should match the bound value. If
/// validation mode is enabled on the context, OptiX will verify that the bound values
/// specified matches the values in pipelineParams specified to optixLaunch.
///
/// These values are compiled in to the module as constants. Once the constants are
/// inserted into the code, an optimization pass will be run that will attempt to
/// propagate the consants and remove unreachable code.
///
/// If caching is enabled, changes in these values will result in newly compiled modules.
///
/// The pipelineParamOffset and sizeInBytes must be within the bounds of the
/// pipelineParams variable. OPTIX_ERROR_ILWALID_VALUE will be returned from
/// optixModuleCreateFromPTX otherwise.
///
/// If more than one bound value overlaps or the size of a bound value is equal to 0,
/// an OPTIX_ERROR_ILWALID_VALUE will be returned from optixModuleCreateFromPTX.
///
/// The same set of bound values do not need to be used for all modules in a pipeline, but
/// overlapping values between modules must have the same value.
/// OPTIX_ERROR_ILWALID_VALUE will be returned from optixPipelineCreate otherwise.
///
/// \see #OptixModuleCompileOptions
typedef struct OptixModuleCompileBoundValueEntry {
    size_t pipelineParamOffsetInBytes;
    size_t sizeInBytes;
    const void* boundValuePtr;
    const char* annotation; // optional string to display, set to 0 if unused.  If unused,
                            // OptiX will report the annotation as "No annotation"
} OptixModuleCompileBoundValueEntry;

/// Payload type identifiers.
typedef enum OptixPayloadTypeID {
    OPTIX_PAYLOAD_TYPE_DEFAULT = 0,
    OPTIX_PAYLOAD_TYPE_ID_0 = (1 << 0u),
    OPTIX_PAYLOAD_TYPE_ID_1 = (1 << 1u),
    OPTIX_PAYLOAD_TYPE_ID_2 = (1 << 2u),
    OPTIX_PAYLOAD_TYPE_ID_3 = (1 << 3u),
    OPTIX_PAYLOAD_TYPE_ID_4 = (1 << 4u),
    OPTIX_PAYLOAD_TYPE_ID_5 = (1 << 5u),
    OPTIX_PAYLOAD_TYPE_ID_6 = (1 << 6u),
    OPTIX_PAYLOAD_TYPE_ID_7 = (1 << 7u)
} OptixPayloadTypeID;

/// Semantic flags for a single payload word.
///
/// Used to specify the semantics of a payload word per shader type.
/// "read":  Shader of this type may read the payload word.
/// "write": Shader of this type may write the payload word.
///
/// "trace_caller_write": Shaders may consume the value of the payload word passed to optixTrace by the caller.
/// "trace_caller_read": The caller to optixTrace may read the payload word after the call to optixTrace.
///
/// Semantics can be bitwise combined.
/// Combining "read" and "write" is equivalent to specifying "read_write".
/// A payload needs to be writable by the caller or at least one shader type.
/// A payload needs to be readable by the caller or at least one shader type after a being writable.
typedef enum OptixPayloadSemantics
{
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_NONE       = 0,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ       = 1u << 0,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE      = 2u << 0,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE = 3u << 0,

    OPTIX_PAYLOAD_SEMANTICS_CH_NONE                 = 0,
    OPTIX_PAYLOAD_SEMANTICS_CH_READ                 = 1u << 2,
    OPTIX_PAYLOAD_SEMANTICS_CH_WRITE                = 2u << 2,
    OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE           = 3u << 2,

    OPTIX_PAYLOAD_SEMANTICS_MS_NONE                 = 0,
    OPTIX_PAYLOAD_SEMANTICS_MS_READ                 = 1u << 4,
    OPTIX_PAYLOAD_SEMANTICS_MS_WRITE                = 2u << 4,
    OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE           = 3u << 4,

    OPTIX_PAYLOAD_SEMANTICS_AH_NONE                 = 0,
    OPTIX_PAYLOAD_SEMANTICS_AH_READ                 = 1u << 6,
    OPTIX_PAYLOAD_SEMANTICS_AH_WRITE                = 2u << 6,
    OPTIX_PAYLOAD_SEMANTICS_AH_READ_WRITE           = 3u << 6,

    OPTIX_PAYLOAD_SEMANTICS_IS_NONE                 = 0,
    OPTIX_PAYLOAD_SEMANTICS_IS_READ                 = 1u << 8,
    OPTIX_PAYLOAD_SEMANTICS_IS_WRITE                = 2u << 8,
    OPTIX_PAYLOAD_SEMANTICS_IS_READ_WRITE           = 3u << 8,
} OptixPayloadSemantics;

/// Specifies a single payload type
typedef struct OptixPayloadType
{
    /// The number of 32b words the payload of this type holds
    unsigned int numPayloadValues;

    /// Points to host array of payload word semantics, size must match numPayloadValues
    const unsigned int *payloadSemantics;
} OptixPayloadType;

/// Compilation options for module
///
/// \see #optixModuleCreateFromPTX()
typedef struct OptixModuleCompileOptions
{
    /// Maximum number of registers allowed when compiling to SASS.
    /// Set to 0 for no explicit limit. May vary within a pipeline.
    int maxRegisterCount;

    /// Optimization level. May vary within a pipeline.
    OptixCompileOptimizationLevel optLevel;

    /// Generate debug information.
    OptixCompileDebugLevel debugLevel;

    /// Ingored if numBoundValues is set to 0
    const OptixModuleCompileBoundValueEntry* boundValues;

    /// set to 0 if unused
    unsigned int numBoundValues;

    /// The number of different payload types available for compilation.
    /// Must be zero if OptixPipelineCompileOptions::numPayloadValues is not zero.
    unsigned int numPayloadTypes;

    /// Points to host array of payload type definitions, size must match numPayloadTypes
    OptixPayloadType *payloadTypes;

} OptixModuleCompileOptions;

/// Distinguishes different kinds of program groups.
typedef enum OptixProgramGroupKind
{
    /// Program group containing a raygen (RG) program
    /// \see #OptixProgramGroupSingleModule, #OptixProgramGroupDesc::raygen
    OPTIX_PROGRAM_GROUP_KIND_RAYGEN = 0x2421,

    /// Program group containing a miss (MS) program
    /// \see #OptixProgramGroupSingleModule, #OptixProgramGroupDesc::miss
    OPTIX_PROGRAM_GROUP_KIND_MISS = 0x2422,

    /// Program group containing an exception (EX) program
    /// \see OptixProgramGroupHitgroup, #OptixProgramGroupDesc::exception
    OPTIX_PROGRAM_GROUP_KIND_EXCEPTION = 0x2423,

    /// Program group containing an intersection (IS), any hit (AH), and/or closest hit (CH) program
    /// \see #OptixProgramGroupSingleModule, #OptixProgramGroupDesc::hitgroup
    OPTIX_PROGRAM_GROUP_KIND_HITGROUP = 0x2424,

    /// Program group containing a direct (DC) or continuation (CC) callable program
    /// \see OptixProgramGroupCallables, #OptixProgramGroupDesc::callables
    OPTIX_PROGRAM_GROUP_KIND_CALLABLES = 0x2425
} OptixProgramGroupKind;

/// Flags for program groups
typedef enum OptixProgramGroupFlags
{
    /// Lwrrently there are no flags
    OPTIX_PROGRAM_GROUP_FLAGS_NONE = 0
} OptixProgramGroupFlags;

/// Program group representing a single module.
///
/// Used for raygen, miss, and exception programs. In case of raygen and exception programs, module and entry
/// function name need to be valid. For miss programs, module and entry function name might both be \c nullptr.
///
/// \see #OptixProgramGroupDesc::raygen, #OptixProgramGroupDesc::miss, #OptixProgramGroupDesc::exception
typedef struct OptixProgramGroupSingleModule
{
    /// Module holding single program.
    OptixModule module;
    /// Entry function name of the single program.
    const char* entryFunctionName;
} OptixProgramGroupSingleModule;

/// Program group representing the hitgroup.
///
/// For each of the three program types, module and entry function name might both be \c nullptr.
///
/// \see #OptixProgramGroupDesc::hitgroup
typedef struct OptixProgramGroupHitgroup
{
    /// Module holding the closest hit (CH) program.
    OptixModule moduleCH;
    /// Entry function name of the closest hit (CH) program.
    const char* entryFunctionNameCH;
    /// Module holding the any hit (AH) program.
    OptixModule moduleAH;
    /// Entry function name of the any hit (AH) program.
    const char* entryFunctionNameAH;
    /// Module holding the intersection (Is) program.
    OptixModule moduleIS;
    /// Entry function name of the intersection (IS) program.
    const char* entryFunctionNameIS;
} OptixProgramGroupHitgroup;

/// Program group representing callables.
///
/// Module and entry function name need to be valid for at least one of the two callables.
///
/// \see ##OptixProgramGroupDesc::callables
typedef struct OptixProgramGroupCallables
{
    /// Module holding the direct callable (DC) program.
    OptixModule moduleDC;
    /// Entry function name of the direct callable (DC) program.
    const char* entryFunctionNameDC;
    /// Module holding the continuation callable (CC) program.
    OptixModule moduleCC;
    /// Entry function name of the continuation callable (CC) program.
    const char* entryFunctionNameCC;
} OptixProgramGroupCallables;

/// Descriptor for program groups.
typedef struct OptixProgramGroupDesc
{
    /// The kind of program group.
    OptixProgramGroupKind kind;

    /// See #OptixProgramGroupFlags
    unsigned int flags;

    union
    {
        /// \see #OPTIX_PROGRAM_GROUP_KIND_RAYGEN
        OptixProgramGroupSingleModule raygen;
        /// \see #OPTIX_PROGRAM_GROUP_KIND_MISS
        OptixProgramGroupSingleModule miss;
        /// \see #OPTIX_PROGRAM_GROUP_KIND_EXCEPTION
        OptixProgramGroupSingleModule exception;
        /// \see #OPTIX_PROGRAM_GROUP_KIND_CALLABLES
        OptixProgramGroupCallables callables;
        /// \see #OPTIX_PROGRAM_GROUP_KIND_HITGROUP
        OptixProgramGroupHitgroup hitgroup;
    };
} OptixProgramGroupDesc;

/// Program group options
///
/// \see #optixProgramGroupCreate()
typedef struct OptixProgramGroupOptions
{
    /// Specifies the payload type of this program group.
    /// All programs in the group must support the payload type
    /// (Program support for a type is specified by calling
    /// \see #optixSetPayloadTypes or otherwise all types specified in
    /// \see #OptixModuleCompileOptions are supported).
    /// If a program is not available for the requested payload type,
    /// optixProgramGroupCreate returns OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH.
    /// If the payloadType is left zero, a unique type is deduced.
    /// The payload type can be uniquely deduced if there is exactly one payload type
    /// for which all programs in the group are available.
    /// If the payload type could not be deduced uniquely
    /// optixProgramGroupCreate returns OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED.
    OptixPayloadType* payloadType;
} OptixProgramGroupOptions;

/// The following values are used to indicate which exception was thrown.
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// These flags are interpreted on the device by rtcore, and should mirror
/// RtcExceptionCodes.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef enum OptixExceptionCodes
{
    /// Stack overflow of the continuation stack.
    /// no exception details.
    OPTIX_EXCEPTION_CODE_STACK_OVERFLOW = -1,

    /// The trace depth is exceeded.
    /// no exception details.
    OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED = -2,

    /// The traversal depth is exceeded.
    /// Exception details:
    ///     optixGetTransformListSize()
    ///     optixGetTransformListHandle()
    OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED = -3,

    /// Traversal encountered an invalid traversable type.
    /// Exception details:
    ///     optixGetTransformListSize()
    ///     optixGetTransformListHandle()
    ///     optixGetExceptionIlwalidTraversable()
    OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_TRAVERSABLE = -5,

    /// The miss SBT record index is out of bounds
    /// A miss SBT record index is valid within the range [0, OptixShaderBindingTable::missRecordCount) (See optixLaunch)
    /// Exception details:
    ///     optixGetExceptionIlwalidSbtOffset()
    OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_MISS_SBT = -6,

    /// The traversal hit SBT record index out of bounds.
    ///
    /// A traversal hit SBT record index is valid within the range [0, OptixShaderBindingTable::hitgroupRecordCount) (See optixLaunch)
    /// The following formula relates the
    //      sbt-index (See optixGetExceptionIlwalidSbtOffset),
    //      sbt-instance-offset (See OptixInstance::sbtOffset),
    ///     sbt-geometry-acceleration-structure-index (See optixGetSbtGASIndex),
    ///     sbt-stride-from-trace-call and sbt-offset-from-trace-call (See optixTrace)
    ///
    /// sbt-index = sbt-instance-offset + (sbt-geometry-acceleration-structure-index * sbt-stride-from-trace-call) + sbt-offset-from-trace-call
    ///
    /// Exception details:
    ///     optixGetTransformListSize()
    ///     optixGetTransformListHandle()
    ///     optixGetExceptionIlwalidSbtOffset()
    ///     optixGetSbtGASIndex()
    OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_HIT_SBT = -7,

    /// The shader encountered an unsupported primitive type (See OptixPipelineCompileOptions::usesPrimitiveTypeFlags).
    /// no exception details.
    OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE = -8,

    /// The shader encountered a call to optixTrace with at least
    /// one of the float arguments being inf or nan, or the tmin argument is negative.
    /// Exception details:
    ///     optixGetExceptionIlwalidRay()
    OPTIX_EXCEPTION_CODE_ILWALID_RAY = -9,

    /// The shader encountered a call to either optixDirectCall or optixCallableCall
    /// where the argument count does not match the parameter count of the callable
    /// program which is called.
    /// Exception details:
    ///     optixGetExceptionParameterMismatch
    OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH = -10,

    /// The ilwoked builtin IS does not match the current GAS
    OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH = -11,

    /// Tried to call a callable program using an SBT offset that is larger
    /// than the number of passed in callable SBT records.
    /// Exception details:
    ///     optixGetExceptionIlwalidSbtOffset()
    OPTIX_EXCEPTION_CODE_CALLABLE_ILWALID_SBT = -12,

    /// Tried to call a direct callable using an SBT offset of a record that
    /// was built from a program group that did not include a direct callable.
    OPTIX_EXCEPTION_CODE_CALLABLE_NO_DC_SBT_RECORD = -13,

    /// Tried to call a continuation callable using an SBT offset of a record
    /// that was built from a program group that did not include a continuation callable.
    OPTIX_EXCEPTION_CODE_CALLABLE_NO_CC_SBT_RECORD = -14,

    /// Tried to directly traverse a single gas while single gas traversable graphs are not enabled
    ///   (see OptixTraversableGraphFlags::OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS).
    /// Exception details:
    ///     optixGetTransformListSize()
    ///     optixGetTransformListHandle()
    ///     optixGetExceptionIlwalidTraversable()
    OPTIX_EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS = -15,

    /// argument passed to an optix call is
    /// not within an acceptable range of values.
    OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_0 = -16,
    OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_1 = -17,
    OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_2 = -18,
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    /// ...
    /// reserved up to -31
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION

    /// Tried to access data on an AS without random data access support (See OptixBuildFlags).
    OPTIX_EXCEPTION_CODE_UNSUPPORTED_DATA_ACCESS = -32,

    /// The program payload type doesn't match the trace payload type.
    OPTIX_EXCEPTION_CODE_PAYLOAD_TYPE_MISMATCH = -33,
} OptixExceptionCodes;

/// Exception flags.
///
/// \see #OptixPipelineCompileOptions::exceptionFlags, #OptixExceptionCodes
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
///
/// These flags are interpreted on the device by rtcore, and should mirror
/// RtcExceptionFlags.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
typedef enum OptixExceptionFlags
{
    /// No exception are enabled.
    OPTIX_EXCEPTION_FLAG_NONE = 0,

    /// Enables exceptions check related to the continuation stack.
    OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW = 1u << 0,

    /// Enables exceptions check related to trace depth.
    OPTIX_EXCEPTION_FLAG_TRACE_DEPTH = 1u << 1,

    /// Enables user exceptions via optixThrowException(). This flag must be specified for all modules in a pipeline
    /// if any module calls optixThrowException().
    OPTIX_EXCEPTION_FLAG_USER = 1u << 2,

    /// Enables various exceptions check related to traversal.
    OPTIX_EXCEPTION_FLAG_DEBUG = 1u << 3
} OptixExceptionFlags;

/// Compilation options for all modules of a pipeline.
///
/// Similar to #OptixModuleCompileOptions, but these options here need to be equal for all modules of a pipeline.
///
/// \see #optixModuleCreateFromPTX(), #optixPipelineCreate()
typedef struct OptixPipelineCompileOptions
{
    /// Boolean value indicating whether motion blur could be used
    int usesMotionBlur;

    /// Traversable graph bitfield. See OptixTraversableGraphFlags
    unsigned int traversableGraphFlags;

    /// How much storage, in 32b words, to make available for the payload, [0..32]
    /// Must be zero if numPayloadTypes is not zero.
    int numPayloadValues;

    /// How much storage, in 32b words, to make available for the attributes. The
    /// minimum number is 2. Values below that will automatically be changed to 2. [2..8]
    int numAttributeValues;

    /// A bitmask of OptixExceptionFlags indicating which exceptions are enabled.
    unsigned int exceptionFlags;

    /// The name of the pipeline parameter variable.  If 0, no pipeline parameter
    /// will be available. This will be ignored if the launch param variable was
    /// optimized out or was not found in the modules linked to the pipeline.
    const char* pipelineLaunchParamsVariableName;

    /// Bit field enabling primitive types. See OptixPrimitiveTypeFlags.
    /// Setting to zero corresponds to enabling OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM and OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE.
    unsigned int usesPrimitiveTypeFlags;

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    /// Boolean value indicating whether visibility maps could be used
    int allowVisibilityMaps;

    /// Boolean value indicating whether displaced micromeshes could be used
    int allowDisplacedMicromeshes;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
} OptixPipelineCompileOptions;

/// Link options for a pipeline
///
/// \see #optixPipelineCreate()
typedef struct OptixPipelineLinkOptions
{
    /// Maximum trace relwrsion depth. 0 means a ray generation program can be
    /// launched, but can't trace any rays. The maximum allowed value is 31.
    unsigned int maxTraceDepth;

    /// Generate debug information.
    OptixCompileDebugLevel debugLevel;
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    // For future consideration: This field was removed which needs
    // to be taken into account if fields are added in the future.
    //int overrideUsesMotionBlur;
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    // needs to be set to enable the handling of visibility maps on Ampere and before
    int enableVisibilityMaps;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
} OptixPipelineLinkOptions;

/// Describes the shader binding table (SBT)
///
/// \see #optixLaunch()
typedef struct OptixShaderBindingTable
{
    /// Device address of the SBT record of the ray gen program to start launch at. The address must be a multiple of
    /// OPTIX_SBT_RECORD_ALIGNMENT.
    LWdeviceptr raygenRecord;

    /// Device address of the SBT record of the exception program. The address must be a multiple of
    /// OPTIX_SBT_RECORD_ALIGNMENT.
    LWdeviceptr exceptionRecord;

    /// Arrays of SBT records for miss programs. The base address and the stride must be a multiple of
    /// OPTIX_SBT_RECORD_ALIGNMENT.
    /// @{
    LWdeviceptr  missRecordBase;
    unsigned int missRecordStrideInBytes;
    unsigned int missRecordCount;
    /// @}

    /// Arrays of SBT records for hit groups. The base address and the stride must be a multiple of
    /// OPTIX_SBT_RECORD_ALIGNMENT.
    /// @{
    LWdeviceptr  hitgroupRecordBase;
    unsigned int hitgroupRecordStrideInBytes;
    unsigned int hitgroupRecordCount;
    /// @}

    /// Arrays of SBT records for callable programs. If the base address is not null, the stride and count must not be
    /// zero. If the base address is null, then the count needs to zero. The base address and the stride must be a
    /// multiple of OPTIX_SBT_RECORD_ALIGNMENT.
    /// @{
    LWdeviceptr  callablesRecordBase;
    unsigned int callablesRecordStrideInBytes;
    unsigned int callablesRecordCount;
    /// @}

} OptixShaderBindingTable;

/// Describes the stack size requirements of a program group.
///
/// \see optixProgramGroupGetStackSize()
typedef struct OptixStackSizes
{
    /// Continuation stack size of RG programs in bytes
    unsigned int cssRG;
    /// Continuation stack size of MS programs in bytes
    unsigned int cssMS;
    /// Continuation stack size of CH programs in bytes
    unsigned int cssCH;
    /// Continuation stack size of AH programs in bytes
    unsigned int cssAH;
    /// Continuation stack size of IS programs in bytes
    unsigned int cssIS;
    /// Continuation stack size of CC programs in bytes
    unsigned int cssCC;
    /// Direct stack size of DC programs in bytes
    unsigned int dssDC;

} OptixStackSizes;

/// Options that can be passed to \c optixQueryFunctionTable()
typedef enum OptixQueryFunctionTableOptions
{
    /// Placeholder (there are no options yet)
    OPTIX_QUERY_FUNCTION_TABLE_OPTION_DUMMY = 0

} OptixQueryFunctionTableOptions;

/// Type of the function \c optixQueryFunctionTable()
typedef OptixResult( OptixQueryFunctionTable_t )( int          abiId,
                                                  unsigned int numOptions,
                                                  OptixQueryFunctionTableOptions* /*optionKeys*/,
                                                  const void** /*optiolwalues*/,
                                                  void*  functionTable,
                                                  size_t sizeOfTable );

/// Specifies the options for retrieving an intersection program for a built-in primitive type.
/// The primitive type must not be OPTIX_PRIMITIVE_TYPE_LWSTOM.
///
/// \see #optixBuiltinISModuleGet()
typedef struct OptixBuiltinISOptions
{
    OptixPrimitiveType        builtinISModuleType;
    /// Boolean value indicating whether vertex motion blur is used (but not motion transform blur).
    int                       usesMotionBlur;
    /// Build flags, see OptixBuildFlags.
    unsigned int              buildFlags;
    /// End cap properties of lwrves, see OptixLwrveEndcapFlags, 0 for non-lwrve types.
    unsigned int              lwrveEndcapFlags;
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
    /// lwrveEndcapFlags should be replaced by primitiveFlags if used for additional primitive flags of any primitive type.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
} OptixBuiltinISOptions;

#if defined( __LWDACC__ )
/// Describes the ray that was passed into \c optixTrace() which caused an exception with
/// exception code OPTIX_EXCEPTION_CODE_ILWALID_RAY.
///
/// \see #optixGetExceptionIlwalidRay()
typedef struct OptixIlwalidRayExceptionDetails
{
    float3 origin;
    float3 direction;
    float  tmin;
    float  tmax;
    float  time;
} OptixIlwalidRayExceptionDetails;

/// Describes the details of a call to a callable program which caused an exception with
/// exception code OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH,
/// Note that OptiX packs the parameters into individual 32 bit values, so the number of
/// expected and passed values may not correspond to the number of arguments passed into
/// optixDirectCall or optixContinuationCall, or the number parameters in the definition
/// of the function that is called.
typedef struct OptixParameterMismatchExceptionDetails
{
    /// Number of 32 bit values expected by the callable program
    unsigned int expectedParameterCount;
    /// Number of 32 bit values that were passed to the callable program
    unsigned int passedArgumentCount;
    /// The offset of the SBT entry of the callable program relative to OptixShaderBindingTable::callablesRecordBase
    unsigned int sbtIndex;
    /// Pointer to a string that holds the name of the callable program that was called
    char*        callableName;
} OptixParameterMismatchExceptionDetails;
#endif

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
/// The mixed case identifier usage here is so that the API call
/// name is written in the same casing style as the API call itself.
/// As nested optional features are not supported, all device API calls are in.
typedef enum OptixDeviceAPICallCoverageID
{
    OPTIX_Trace_0_COVERAGE_ID = 0,
    OPTIX_Trace_1_COVERAGE_ID,
    OPTIX_Trace_2_COVERAGE_ID,
    OPTIX_Trace_3_COVERAGE_ID,
    OPTIX_Trace_4_COVERAGE_ID,
    OPTIX_Trace_5_COVERAGE_ID,
    OPTIX_Trace_6_COVERAGE_ID,
    OPTIX_Trace_7_COVERAGE_ID,
    OPTIX_Trace_8_COVERAGE_ID,
    OPTIX_Trace_9_COVERAGE_ID,
    OPTIX_Trace_10_COVERAGE_ID,
    OPTIX_Trace_11_COVERAGE_ID,
    OPTIX_Trace_12_COVERAGE_ID,
    OPTIX_Trace_13_COVERAGE_ID,
    OPTIX_Trace_14_COVERAGE_ID,
    OPTIX_Trace_15_COVERAGE_ID,
    OPTIX_Trace_16_COVERAGE_ID,
    OPTIX_Trace_17_COVERAGE_ID,
    OPTIX_Trace_18_COVERAGE_ID,
    OPTIX_Trace_19_COVERAGE_ID,
    OPTIX_Trace_20_COVERAGE_ID,
    OPTIX_Trace_21_COVERAGE_ID,
    OPTIX_Trace_22_COVERAGE_ID,
    OPTIX_Trace_23_COVERAGE_ID,
    OPTIX_Trace_24_COVERAGE_ID,
    OPTIX_Trace_25_COVERAGE_ID,
    OPTIX_Trace_26_COVERAGE_ID,
    OPTIX_Trace_27_COVERAGE_ID,
    OPTIX_Trace_28_COVERAGE_ID,
    OPTIX_Trace_29_COVERAGE_ID,
    OPTIX_Trace_30_COVERAGE_ID,
    OPTIX_Trace_31_COVERAGE_ID,
    OPTIX_Trace_32_COVERAGE_ID,
    OPTIX_Trace_Typed_0_COVERAGE_ID,
    OPTIX_Trace_Typed_1_COVERAGE_ID,
    OPTIX_Trace_Typed_2_COVERAGE_ID,
    OPTIX_Trace_Typed_3_COVERAGE_ID,
    OPTIX_Trace_Typed_4_COVERAGE_ID,
    OPTIX_Trace_Typed_5_COVERAGE_ID,
    OPTIX_Trace_Typed_6_COVERAGE_ID,
    OPTIX_Trace_Typed_7_COVERAGE_ID,
    OPTIX_Trace_Typed_8_COVERAGE_ID,
    OPTIX_Trace_Typed_9_COVERAGE_ID,
    OPTIX_Trace_Typed_10_COVERAGE_ID,
    OPTIX_Trace_Typed_11_COVERAGE_ID,
    OPTIX_Trace_Typed_12_COVERAGE_ID,
    OPTIX_Trace_Typed_13_COVERAGE_ID,
    OPTIX_Trace_Typed_14_COVERAGE_ID,
    OPTIX_Trace_Typed_15_COVERAGE_ID,
    OPTIX_Trace_Typed_16_COVERAGE_ID,
    OPTIX_Trace_Typed_17_COVERAGE_ID,
    OPTIX_Trace_Typed_18_COVERAGE_ID,
    OPTIX_Trace_Typed_19_COVERAGE_ID,
    OPTIX_Trace_Typed_20_COVERAGE_ID,
    OPTIX_Trace_Typed_21_COVERAGE_ID,
    OPTIX_Trace_Typed_22_COVERAGE_ID,
    OPTIX_Trace_Typed_23_COVERAGE_ID,
    OPTIX_Trace_Typed_24_COVERAGE_ID,
    OPTIX_Trace_Typed_25_COVERAGE_ID,
    OPTIX_Trace_Typed_26_COVERAGE_ID,
    OPTIX_Trace_Typed_27_COVERAGE_ID,
    OPTIX_Trace_Typed_28_COVERAGE_ID,
    OPTIX_Trace_Typed_29_COVERAGE_ID,
    OPTIX_Trace_Typed_30_COVERAGE_ID,
    OPTIX_Trace_Typed_31_COVERAGE_ID,
    OPTIX_Trace_Typed_32_COVERAGE_ID,
    OPTIX_GetPayload_0_COVERAGE_ID,
    OPTIX_GetPayload_1_COVERAGE_ID,
    OPTIX_GetPayload_2_COVERAGE_ID,
    OPTIX_GetPayload_3_COVERAGE_ID,
    OPTIX_GetPayload_4_COVERAGE_ID,
    OPTIX_GetPayload_5_COVERAGE_ID,
    OPTIX_GetPayload_6_COVERAGE_ID,
    OPTIX_GetPayload_7_COVERAGE_ID,
    OPTIX_GetPayload_8_COVERAGE_ID,
    OPTIX_GetPayload_9_COVERAGE_ID,
    OPTIX_GetPayload_10_COVERAGE_ID,
    OPTIX_GetPayload_11_COVERAGE_ID,
    OPTIX_GetPayload_12_COVERAGE_ID,
    OPTIX_GetPayload_13_COVERAGE_ID,
    OPTIX_GetPayload_14_COVERAGE_ID,
    OPTIX_GetPayload_15_COVERAGE_ID,
    OPTIX_GetPayload_16_COVERAGE_ID,
    OPTIX_GetPayload_17_COVERAGE_ID,
    OPTIX_GetPayload_18_COVERAGE_ID,
    OPTIX_GetPayload_19_COVERAGE_ID,
    OPTIX_GetPayload_20_COVERAGE_ID,
    OPTIX_GetPayload_21_COVERAGE_ID,
    OPTIX_GetPayload_22_COVERAGE_ID,
    OPTIX_GetPayload_23_COVERAGE_ID,
    OPTIX_GetPayload_24_COVERAGE_ID,
    OPTIX_GetPayload_25_COVERAGE_ID,
    OPTIX_GetPayload_26_COVERAGE_ID,
    OPTIX_GetPayload_27_COVERAGE_ID,
    OPTIX_GetPayload_28_COVERAGE_ID,
    OPTIX_GetPayload_29_COVERAGE_ID,
    OPTIX_GetPayload_30_COVERAGE_ID,
    OPTIX_GetPayload_31_COVERAGE_ID,
    OPTIX_SetPayload_0_COVERAGE_ID,
    OPTIX_SetPayload_1_COVERAGE_ID,
    OPTIX_SetPayload_2_COVERAGE_ID,
    OPTIX_SetPayload_3_COVERAGE_ID,
    OPTIX_SetPayload_4_COVERAGE_ID,
    OPTIX_SetPayload_5_COVERAGE_ID,
    OPTIX_SetPayload_6_COVERAGE_ID,
    OPTIX_SetPayload_7_COVERAGE_ID,
    OPTIX_SetPayload_8_COVERAGE_ID,
    OPTIX_SetPayload_9_COVERAGE_ID,
    OPTIX_SetPayload_10_COVERAGE_ID,
    OPTIX_SetPayload_11_COVERAGE_ID,
    OPTIX_SetPayload_12_COVERAGE_ID,
    OPTIX_SetPayload_13_COVERAGE_ID,
    OPTIX_SetPayload_14_COVERAGE_ID,
    OPTIX_SetPayload_15_COVERAGE_ID,
    OPTIX_SetPayload_16_COVERAGE_ID,
    OPTIX_SetPayload_17_COVERAGE_ID,
    OPTIX_SetPayload_18_COVERAGE_ID,
    OPTIX_SetPayload_19_COVERAGE_ID,
    OPTIX_SetPayload_20_COVERAGE_ID,
    OPTIX_SetPayload_21_COVERAGE_ID,
    OPTIX_SetPayload_22_COVERAGE_ID,
    OPTIX_SetPayload_23_COVERAGE_ID,
    OPTIX_SetPayload_24_COVERAGE_ID,
    OPTIX_SetPayload_25_COVERAGE_ID,
    OPTIX_SetPayload_26_COVERAGE_ID,
    OPTIX_SetPayload_27_COVERAGE_ID,
    OPTIX_SetPayload_28_COVERAGE_ID,
    OPTIX_SetPayload_29_COVERAGE_ID,
    OPTIX_SetPayload_30_COVERAGE_ID,
    OPTIX_SetPayload_31_COVERAGE_ID,
    OPTIX_SetPayloadTypes_COVERAGE_ID,
    OPTIX_UndefinedValue_COVERAGE_ID,
    OPTIX_GetWorldRayOrigin_COVERAGE_ID,
    OPTIX_GetWorldRayDirection_COVERAGE_ID,
    OPTIX_GetObjectRayOrigin_COVERAGE_ID,
    OPTIX_GetObjectRayDirection_COVERAGE_ID,
    OPTIX_GetRayTmin_COVERAGE_ID,
    OPTIX_GetRayTmax_COVERAGE_ID,
    OPTIX_GetRayTime_COVERAGE_ID,
    OPTIX_GetRayFlags_COVERAGE_ID,
    OPTIX_GetRayVisibilityMask_COVERAGE_ID,
    OPTIX_GetInstanceTraversableFromIAS_COVERAGE_ID,
    OPTIX_GetTriangleVertexData_COVERAGE_ID,
    OPTIX_GetGASTraversableHandle_COVERAGE_ID,
    OPTIX_GetGASMotionTimeBegin_COVERAGE_ID,
    OPTIX_GetGASMotionTimeEnd_COVERAGE_ID,
    OPTIX_GetGASMotionStepCount_COVERAGE_ID,
    OPTIX_GetWorldToObjectTransformMatrix_COVERAGE_ID,
    OPTIX_GetObjectToWorldTransformMatrix_COVERAGE_ID,
    OPTIX_TransformPointFromWorldToObjectSpace_COVERAGE_ID,
    OPTIX_TransformVectorFromWorldToObjectSpace_COVERAGE_ID,
    OPTIX_TransformNormalFromWorldToObjectSpace_COVERAGE_ID,
    OPTIX_TransformPointFromObjectToWorldSpace_COVERAGE_ID,
    OPTIX_TransformVectorFromObjectToWorldSpace_COVERAGE_ID,
    OPTIX_TransformNormalFromObjectToWorldSpace_COVERAGE_ID,
    OPTIX_GetTransformListSize_COVERAGE_ID,
    OPTIX_GetTransformListHandle_COVERAGE_ID,
    OPTIX_GetTransformTypeFromHandle_COVERAGE_ID,
    OPTIX_GetStaticTransformFromHandle_COVERAGE_ID,
    OPTIX_GetSRTMotionTransformFromHandle_COVERAGE_ID,
    OPTIX_GetMatrixMotionTransformFromHandle_COVERAGE_ID,
    OPTIX_GetInstanceIdFromHandle_COVERAGE_ID,
    OPTIX_GetInstanceChildFromHandle_COVERAGE_ID,
    OPTIX_GetInstanceTransformFromHandle_COVERAGE_ID,
    OPTIX_GetInstanceIlwerseTransformFromHandle_COVERAGE_ID,
    OPTIX_ReportIntersection_0_COVERAGE_ID,
    OPTIX_ReportIntersection_1_COVERAGE_ID,
    OPTIX_ReportIntersection_2_COVERAGE_ID,
    OPTIX_ReportIntersection_3_COVERAGE_ID,
    OPTIX_ReportIntersection_4_COVERAGE_ID,
    OPTIX_ReportIntersection_5_COVERAGE_ID,
    OPTIX_ReportIntersection_6_COVERAGE_ID,
    OPTIX_ReportIntersection_7_COVERAGE_ID,
    OPTIX_ReportIntersection_8_COVERAGE_ID,
    OPTIX_GetAttribute_0_COVERAGE_ID,
    OPTIX_GetAttribute_1_COVERAGE_ID,
    OPTIX_GetAttribute_2_COVERAGE_ID,
    OPTIX_GetAttribute_3_COVERAGE_ID,
    OPTIX_GetAttribute_4_COVERAGE_ID,
    OPTIX_GetAttribute_5_COVERAGE_ID,
    OPTIX_GetAttribute_6_COVERAGE_ID,
    OPTIX_GetAttribute_7_COVERAGE_ID,
    OPTIX_TerminateRay_COVERAGE_ID,
    OPTIX_IgnoreIntersection_COVERAGE_ID,
    OPTIX_GetPrimitiveIndex_COVERAGE_ID,
    OPTIX_GetSbtGASIndex_COVERAGE_ID,
    OPTIX_GetInstanceId_COVERAGE_ID,
    OPTIX_GetInstanceIndex_COVERAGE_ID,
    OPTIX_GetHitKind_COVERAGE_ID,
    OPTIX_IsTriangleHit_COVERAGE_ID,
    OPTIX_IsTriangleFrontFaceHit_COVERAGE_ID,
    OPTIX_IsTriangleBackFaceHit_COVERAGE_ID,
    OPTIX_GetTriangleBarycentrics_COVERAGE_ID,
    OPTIX_GetLaunchIndex_COVERAGE_ID,
    OPTIX_GetLaunchDimensions_COVERAGE_ID,
    OPTIX_GetSbtDataPointer_COVERAGE_ID,
    OPTIX_ThrowException_0_COVERAGE_ID,
    OPTIX_ThrowException_1_COVERAGE_ID,
    OPTIX_ThrowException_2_COVERAGE_ID,
    OPTIX_ThrowException_3_COVERAGE_ID,
    OPTIX_ThrowException_4_COVERAGE_ID,
    OPTIX_ThrowException_5_COVERAGE_ID,
    OPTIX_ThrowException_6_COVERAGE_ID,
    OPTIX_ThrowException_7_COVERAGE_ID,
    OPTIX_ThrowException_8_COVERAGE_ID,
    OPTIX_GetExceptionCode_COVERAGE_ID,
    OPTIX_GetExceptionDetail_0_COVERAGE_ID,
    OPTIX_GetExceptionDetail_1_COVERAGE_ID,
    OPTIX_GetExceptionDetail_2_COVERAGE_ID,
    OPTIX_GetExceptionDetail_3_COVERAGE_ID,
    OPTIX_GetExceptionDetail_4_COVERAGE_ID,
    OPTIX_GetExceptionDetail_5_COVERAGE_ID,
    OPTIX_GetExceptionDetail_6_COVERAGE_ID,
    OPTIX_GetExceptionDetail_7_COVERAGE_ID,
    OPTIX_GetExceptionIlwalidTraversable_COVERAGE_ID,
    OPTIX_GetExceptionIlwalidSbtOffset_COVERAGE_ID,
    OPTIX_GetExceptionIlwalidRay_COVERAGE_ID,
    OPTIX_GetExceptionParameterMismatch_COVERAGE_ID,
    OPTIX_GetExceptionLineInfo_COVERAGE_ID,
    OPTIX_DirectCall_COVERAGE_ID,
    OPTIX_ContinuationCall_COVERAGE_ID,
    OPTIX_GetLinearLwrveVertexData_COVERAGE_ID,
    OPTIX_GetQuadraticBSplineVertexData_COVERAGE_ID,
    OPTIX_GetLwbicBSplineVertexData_COVERAGE_ID,
    OPTIX_GetCatmullRomVertexData_COVERAGE_ID,
    OPTIX_GetSphereData_COVERAGE_ID,
    OPTIX_GetPrimitiveType_COVERAGE_ID,
    OPTIX_IsBackFaceHit_COVERAGE_ID,
    OPTIX_IsFrontFaceHit_COVERAGE_ID,
    OPTIX_GetPrimitiveTypeB_COVERAGE_ID,
    OPTIX_IsBackFaceHitB_COVERAGE_ID,
    OPTIX_IsFrontFaceHitB_COVERAGE_ID,
    OPTIX_GetLwrveParameter_COVERAGE_ID,
    OPTIX_Undefined_COVERAGE_ID,
} OptixDeviceAPICallCoverageID;
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

/*@}*/  // end group optix_types

#endif  // __optix_optix_7_types_h__
