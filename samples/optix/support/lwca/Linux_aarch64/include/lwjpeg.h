/*
 * Copyright 2009-2019 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

  
#ifndef LW_JPEG_HEADER
#define LW_JPEG_HEADER

#define LWJPEGAPI


#include "lwda_runtime_api.h"
#include "library_types.h"

#include "stdint.h"

#if defined(__cplusplus)
  extern "C" {
#endif

// Maximum number of channels lwjpeg decoder supports
#define LWJPEG_MAX_COMPONENT 4

// lwjpeg version information
#define LWJPEG_VER_MAJOR 11
#define LWJPEG_VER_MINOR 4
#define LWJPEG_VER_PATCH 0
#define LWJPEG_VER_BUILD 0

/* lwJPEG status enums, returned by lwJPEG API */
typedef enum
{
    LWJPEG_STATUS_SUCCESS                       = 0,
    LWJPEG_STATUS_NOT_INITIALIZED               = 1,
    LWJPEG_STATUS_ILWALID_PARAMETER             = 2,
    LWJPEG_STATUS_BAD_JPEG                      = 3,
    LWJPEG_STATUS_JPEG_NOT_SUPPORTED            = 4,
    LWJPEG_STATUS_ALLOCATOR_FAILURE             = 5,
    LWJPEG_STATUS_EXELWTION_FAILED              = 6,
    LWJPEG_STATUS_ARCH_MISMATCH                 = 7,
    LWJPEG_STATUS_INTERNAL_ERROR                = 8,
    LWJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED  = 9,
} lwjpegStatus_t;


// Enum identifies image chroma subsampling values stored inside JPEG input stream
// In the case of LWJPEG_CSS_GRAY only 1 luminance channel is encoded in JPEG input stream
// Otherwise both chroma planes are present
typedef enum
{
    LWJPEG_CSS_444 = 0,
    LWJPEG_CSS_422 = 1,
    LWJPEG_CSS_420 = 2,
    LWJPEG_CSS_440 = 3,
    LWJPEG_CSS_411 = 4,
    LWJPEG_CSS_410 = 5,
    LWJPEG_CSS_GRAY = 6,
    LWJPEG_CSS_UNKNOWN = -1
} lwjpegChromaSubsampling_t;

// Parameter of this type specifies what type of output user wants for image decoding
typedef enum
{
    // return decompressed image as it is - write planar output
    LWJPEG_OUTPUT_UNCHANGED   = 0,
    // return planar luma and chroma, assuming YCbCr colorspace
    LWJPEG_OUTPUT_YUV         = 1, 
    // return luma component only, if YCbCr colorspace, 
    // or try to colwert to grayscale,
    // writes to 1-st channel of lwjpegImage_t
    LWJPEG_OUTPUT_Y           = 2,
    // colwert to planar RGB 
    LWJPEG_OUTPUT_RGB         = 3,
    // colwert to planar BGR
    LWJPEG_OUTPUT_BGR         = 4, 
    // colwert to interleaved RGB and write to 1-st channel of lwjpegImage_t
    LWJPEG_OUTPUT_RGBI        = 5, 
    // colwert to interleaved BGR and write to 1-st channel of lwjpegImage_t
    LWJPEG_OUTPUT_BGRI        = 6,
    // maximum allowed value
    LWJPEG_OUTPUT_FORMAT_MAX  = 6  
} lwjpegOutputFormat_t;

// Parameter of this type specifies what type of input user provides for encoding
typedef enum
{
    LWJPEG_INPUT_RGB         = 3, // Input is RGB - will be colwerted to YCbCr before encoding
    LWJPEG_INPUT_BGR         = 4, // Input is RGB - will be colwerted to YCbCr before encoding
    LWJPEG_INPUT_RGBI        = 5, // Input is interleaved RGB - will be colwerted to YCbCr before encoding
    LWJPEG_INPUT_BGRI        = 6  // Input is interleaved RGB - will be colwerted to YCbCr before encoding
} lwjpegInputFormat_t;

// Implementation
// LWJPEG_BACKEND_DEFAULT    : default value
// LWJPEG_BACKEND_HYBRID     : uses CPU for Huffman decode
// LWJPEG_BACKEND_GPU_HYBRID : uses GPU assisted Huffman decode. lwjpegDecodeBatched will use GPU decoding for baseline JPEG bitstreams with
//                             interleaved scan when batch size is bigger than 100
// LWJPEG_BACKEND_HARDWARE   : supports baseline JPEG bitstream with single scan. 410 and 411 sub-samplings are not supported
typedef enum 
{
    LWJPEG_BACKEND_DEFAULT = 0,
    LWJPEG_BACKEND_HYBRID  = 1,
    LWJPEG_BACKEND_GPU_HYBRID = 2,
    LWJPEG_BACKEND_HARDWARE = 3
} lwjpegBackend_t;

// Lwrrently parseable JPEG encodings (SOF markers)
typedef enum
{
    LWJPEG_ENCODING_UNKNOWN                                 = 0x0,

    LWJPEG_ENCODING_BASELINE_DCT                            = 0xc0,
    LWJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN         = 0xc1,
    LWJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN                 = 0xc2

} lwjpegJpegEncoding_t;

typedef enum 
{
    LWJPEG_SCALE_NONE = 0, // decoded output is not scaled 
    LWJPEG_SCALE_1_BY_2 = 1, // decoded output width and height is scaled by a factor of 1/2
    LWJPEG_SCALE_1_BY_4 = 2, // decoded output width and height is scaled by a factor of 1/4
    LWJPEG_SCALE_1_BY_8 = 3, // decoded output width and height is scaled by a factor of 1/8
} lwjpegScaleFactor_t;

#define LWJPEG_FLAGS_DEFAULT 0
#define LWJPEG_FLAGS_HW_DECODE_NO_PIPELINE 1
#define LWJPEG_FLAGS_ENABLE_MEMORY_POOLS   1<<1
#define LWJPEG_FLAGS_BITSTREAM_STRICT  1<<2

// Output descriptor.
// Data that is written to planes depends on output format
typedef struct
{
    unsigned char * channel[LWJPEG_MAX_COMPONENT];
    size_t    pitch[LWJPEG_MAX_COMPONENT];
} lwjpegImage_t;

// Prototype for device memory allocation, modelled after lwdaMalloc()
typedef int (*tDevMalloc)(void**, size_t);
// Prototype for device memory release
typedef int (*tDevFree)(void*);

// Prototype for pinned memory allocation, modelled after lwdaHostAlloc()
typedef int (*tPinnedMalloc)(void**, size_t, unsigned int flags);
// Prototype for device memory release
typedef int (*tPinnedFree)(void*);

// Memory allocator using mentioned prototypes, provided to lwjpegCreateEx
// This allocator will be used for all device memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct 
{
    tDevMalloc dev_malloc;
    tDevFree dev_free;
} lwjpegDevAllocator_t;

// Pinned memory allocator using mentioned prototypes, provided to lwjpegCreate
// This allocator will be used for all pinned host memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
typedef struct 
{
    tPinnedMalloc pinned_malloc;
    tPinnedFree pinned_free;
} lwjpegPinnedAllocator_t;

// Opaque library handle identifier.
struct lwjpegHandle;
typedef struct lwjpegHandle* lwjpegHandle_t;

// Opaque jpeg decoding state handle identifier - used to store intermediate information between deccding phases
struct lwjpegJpegState;
typedef struct lwjpegJpegState* lwjpegJpegState_t;

// returns library's property values, such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
lwjpegStatus_t LWJPEGAPI lwjpegGetProperty(libraryPropertyType type, int *value);
// returns LWCA Toolkit property values that was used for building library, 
// such as MAJOR_VERSION, MINOR_VERSION or PATCH_LEVEL
lwjpegStatus_t LWJPEGAPI lwjpegGetLwdartProperty(libraryPropertyType type, int *value);

// Initalization of lwjpeg handle. This handle is used for all conselwtive calls
// IN         backend       : Backend to use. Lwrrently Default or Hybrid (which is the same at the moment) is supported.
// IN         allocator     : Pointer to lwjpegDevAllocator. If NULL - use default lwca calls (lwdaMalloc/lwdaFree)
// INT/OUT    handle        : Codec instance, use for other calls
lwjpegStatus_t LWJPEGAPI lwjpegCreate(lwjpegBackend_t backend, lwjpegDevAllocator_t *dev_allocator, lwjpegHandle_t *handle);

// Initalization of lwjpeg handle with default backend and default memory allocators.
// INT/OUT    handle        : Codec instance, use for other calls
lwjpegStatus_t LWJPEGAPI lwjpegCreateSimple(lwjpegHandle_t *handle);

// Initalization of lwjpeg handle with additional parameters. This handle is used for all conselwtive lwjpeg calls
// IN         backend       : Backend to use. Lwrrently Default or Hybrid (which is the same at the moment) is supported.
// IN         dev_allocator : Pointer to lwjpegDevAllocator. If NULL - use default lwca calls (lwdaMalloc/lwdaFree)
// IN         pinned_allocator : Pointer to lwjpegPinnedAllocator. If NULL - use default lwca calls (lwdaHostAlloc/lwdaFreeHost)
// IN         flags         : Parameters for the operation. Must be 0.
// INT/OUT    handle        : Codec instance, use for other calls
lwjpegStatus_t LWJPEGAPI lwjpegCreateEx(lwjpegBackend_t backend, 
        lwjpegDevAllocator_t *dev_allocator, 
        lwjpegPinnedAllocator_t *pinned_allocator, 
        unsigned int flags,
        lwjpegHandle_t *handle);

// Release the handle and resources.
// IN/OUT     handle: instance handle to release 
lwjpegStatus_t LWJPEGAPI lwjpegDestroy(lwjpegHandle_t handle);

// Sets padding for device memory allocations. After success on this call any device memory allocation
// would be padded to the multiple of specified number of bytes. 
// IN         padding: padding size
// IN/OUT     handle: instance handle to release 
lwjpegStatus_t LWJPEGAPI lwjpegSetDeviceMemoryPadding(size_t padding, lwjpegHandle_t handle);

// Retrieves padding for device memory allocations
// IN/OUT     padding: padding size lwrrently used in handle.
// IN/OUT     handle: instance handle to release 
lwjpegStatus_t LWJPEGAPI lwjpegGetDeviceMemoryPadding(size_t *padding, lwjpegHandle_t handle);

// Sets padding for pinned host memory allocations. After success on this call any pinned host memory allocation
// would be padded to the multiple of specified number of bytes. 
// IN         padding: padding size
// IN/OUT     handle: instance handle to release 
lwjpegStatus_t LWJPEGAPI lwjpegSetPinnedMemoryPadding(size_t padding, lwjpegHandle_t handle);

// Retrieves padding for pinned host memory allocations
// IN/OUT     padding: padding size lwrrently used in handle.
// IN/OUT     handle: instance handle to release 
lwjpegStatus_t LWJPEGAPI lwjpegGetPinnedMemoryPadding(size_t *padding, lwjpegHandle_t handle);



// Initalization of decoding state
// IN         handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
lwjpegStatus_t LWJPEGAPI lwjpegJpegStateCreate(lwjpegHandle_t handle, lwjpegJpegState_t *jpeg_handle);

// Release the jpeg image handle.
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
lwjpegStatus_t LWJPEGAPI lwjpegJpegStateDestroy(lwjpegJpegState_t jpeg_handle);
// 
// Retrieve the image info, including channel, width and height of each component, and chroma subsampling.
// If less than LWJPEG_MAX_COMPONENT channels are encoded, then zeros would be set to absent channels information
// If the image is 3-channel, all three groups are valid.
// This function is thread safe.
// IN         handle      : Library handle
// IN         data        : Pointer to the buffer containing the jpeg stream data to be decoded. 
// IN         length      : Length of the jpeg image buffer.
// OUT        nComponent  : Number of componenets of the image, lwrrently only supports 1-channel (grayscale) or 3-channel.
// OUT        subsampling : Chroma subsampling used in this JPEG, see lwjpegChromaSubsampling_t
// OUT        widths      : pointer to LWJPEG_MAX_COMPONENT of ints, returns width of each channel. 0 if channel is not encoded  
// OUT        heights     : pointer to LWJPEG_MAX_COMPONENT of ints, returns height of each channel. 0 if channel is not encoded 
lwjpegStatus_t LWJPEGAPI lwjpegGetImageInfo(
        lwjpegHandle_t handle,
        const unsigned char *data, 
        size_t length,
        int *nComponents, 
        lwjpegChromaSubsampling_t *subsampling,
        int *widths,
        int *heights);
                   

// Decodes single image. The API is back-end agnostic. It will decide on which implementation to use internally
// Destination buffers should be large enough to be able to store  output of specified format.
// For each color plane sizes could be retrieved for image using lwjpegGetImageInfo()
// and minimum required memory buffer for each plane is nPlaneHeight*nPlanePitch where nPlanePitch >= nPlaneWidth for
// planar output formats and nPlanePitch >= nPlaneWidth*nOutputComponents for interleaved output format.
// 
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Pointer to the buffer containing the jpeg image to be decoded. 
// IN         length        : Length of the jpeg image buffer.
// IN         output_format : Output data format. See lwjpegOutputFormat_t for description
// IN/OUT     destination   : Pointer to structure with information about output buffers. See lwjpegImage_t description.
// IN/OUT     stream        : LWCA stream where to submit all GPU work
// 
// \return LWJPEG_STATUS_SUCCESS if successful
lwjpegStatus_t LWJPEGAPI lwjpegDecode(
        lwjpegHandle_t handle,
        lwjpegJpegState_t jpeg_handle,
        const unsigned char *data,
        size_t length, 
        lwjpegOutputFormat_t output_format,
        lwjpegImage_t *destination,
        lwdaStream_t stream);


//////////////////////////////////////////////
/////////////// Batch decoding ///////////////
//////////////////////////////////////////////

// Resets and initizlizes batch decoder for working on the batches of specified size
// Should be called once for decoding bathes of this specific size, also use to reset failed batches
// IN/OUT     handle          : Library handle
// INT/OUT    jpeg_handle     : Decoded jpeg image state handle
// IN         batch_size      : Size of the batch
// IN         max_cpu_threads : Maximum number of CPU threads that will be processing this batch
// IN         output_format   : Output data format. Will be the same for every image in batch
//
// \return LWJPEG_STATUS_SUCCESS if successful
lwjpegStatus_t LWJPEGAPI lwjpegDecodeBatchedInitialize(
          lwjpegHandle_t handle,
          lwjpegJpegState_t jpeg_handle,
          int batch_size,
          int max_cpu_threads,
          lwjpegOutputFormat_t output_format);

// Decodes batch of images. Output buffers should be large enough to be able to store 
// outputs of specified format, see single image decoding description for details. Call to 
// lwjpegDecodeBatchedInitialize() is required prior to this call, batch size is expected to be the same as 
// parameter to this batch initialization function.
// 
// IN/OUT     handle        : Library handle
// INT/OUT    jpeg_handle   : Decoded jpeg image state handle
// IN         data          : Array of size batch_size of pointers to the input buffers containing the jpeg images to be decoded. 
// IN         lengths       : Array of size batch_size with lengths of the jpeg images' buffers in the batch.
// IN/OUT     destinations  : Array of size batch_size with pointers to structure with information about output buffers, 
// IN/OUT     stream        : LWCA stream where to submit all GPU work
// 
// \return LWJPEG_STATUS_SUCCESS if successful
lwjpegStatus_t LWJPEGAPI lwjpegDecodeBatched(
          lwjpegHandle_t handle,
          lwjpegJpegState_t jpeg_handle,
          const unsigned char *const *data,
          const size_t *lengths, 
          lwjpegImage_t *destinations,
          lwdaStream_t stream);

// Allocates the internal buffers as a pre-allocation step
// IN    handle          : Library handle
// IN    jpeg_handle     : Decoded jpeg image state handle
// IN    width   : frame width
// IN    height  : frame height
// IN    chroma_subsampling   : chroma subsampling of images to be decoded
// IN    output_format : out format

lwjpegStatus_t LWJPEGAPI lwjpegDecodeBatchedPreAllocate(
          lwjpegHandle_t handle,
          lwjpegJpegState_t jpeg_handle,
          int batch_size,
          int width,
          int height,
          lwjpegChromaSubsampling_t chroma_subsampling,
          lwjpegOutputFormat_t output_format);

/**********************************************************
*                        Compression                      *
**********************************************************/
struct lwjpegEncoderState;
typedef struct lwjpegEncoderState* lwjpegEncoderState_t;

lwjpegStatus_t LWJPEGAPI lwjpegEncoderStateCreate(
        lwjpegHandle_t handle,
        lwjpegEncoderState_t *encoder_state,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncoderStateDestroy(lwjpegEncoderState_t encoder_state);

struct lwjpegEncoderParams;
typedef struct lwjpegEncoderParams* lwjpegEncoderParams_t;

lwjpegStatus_t LWJPEGAPI lwjpegEncoderParamsCreate(
        lwjpegHandle_t handle, 
        lwjpegEncoderParams_t *encoder_params,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncoderParamsDestroy(lwjpegEncoderParams_t encoder_params);

lwjpegStatus_t LWJPEGAPI lwjpegEncoderParamsSetQuality(
        lwjpegEncoderParams_t encoder_params,
        const int quality,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncoderParamsSetEncoding(
        lwjpegEncoderParams_t encoder_params,
        lwjpegJpegEncoding_t etype,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncoderParamsSetOptimizedHuffman(
        lwjpegEncoderParams_t encoder_params,
        const int optimized,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncoderParamsSetSamplingFactors(
        lwjpegEncoderParams_t encoder_params,
        const lwjpegChromaSubsampling_t chroma_subsampling,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncodeGetBufferSize(
        lwjpegHandle_t handle,
        const lwjpegEncoderParams_t encoder_params,
        int image_width,
        int image_height,
        size_t *max_stream_length);

lwjpegStatus_t LWJPEGAPI lwjpegEncodeYUV(
        lwjpegHandle_t handle,
        lwjpegEncoderState_t encoder_state,
        const lwjpegEncoderParams_t encoder_params,
        const lwjpegImage_t *source,
        lwjpegChromaSubsampling_t chroma_subsampling, 
        int image_width,
        int image_height,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncodeImage(
        lwjpegHandle_t handle,
        lwjpegEncoderState_t encoder_state,
        const lwjpegEncoderParams_t encoder_params,
        const lwjpegImage_t *source,
        lwjpegInputFormat_t input_format, 
        int image_width,
        int image_height,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncodeRetrieveBitstreamDevice(
        lwjpegHandle_t handle,
        lwjpegEncoderState_t encoder_state,
        unsigned char *data,
        size_t *length,
        lwdaStream_t stream);

lwjpegStatus_t LWJPEGAPI lwjpegEncodeRetrieveBitstream(
        lwjpegHandle_t handle,
        lwjpegEncoderState_t encoder_state,
        unsigned char *data,
        size_t *length,
        lwdaStream_t stream);

///////////////////////////////////////////////////////////////////////////////////
// API v2 //
///////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////
// LWJPEG buffers //
///////////////////////////////////////////////////////////////////////////////////

struct lwjpegBufferPinned;
typedef struct lwjpegBufferPinned* lwjpegBufferPinned_t;

lwjpegStatus_t LWJPEGAPI lwjpegBufferPinnedCreate(lwjpegHandle_t handle, 
    lwjpegPinnedAllocator_t* pinned_allocator,
    lwjpegBufferPinned_t* buffer);

lwjpegStatus_t LWJPEGAPI lwjpegBufferPinnedDestroy(lwjpegBufferPinned_t buffer);

struct lwjpegBufferDevice;
typedef struct lwjpegBufferDevice* lwjpegBufferDevice_t;

lwjpegStatus_t LWJPEGAPI lwjpegBufferDeviceCreate(lwjpegHandle_t handle, 
    lwjpegDevAllocator_t* device_allocator, 
    lwjpegBufferDevice_t* buffer);

lwjpegStatus_t LWJPEGAPI lwjpegBufferDeviceDestroy(lwjpegBufferDevice_t buffer);

// retrieve buffer size and pointer - this allows reusing buffer when decode is not needed
lwjpegStatus_t LWJPEGAPI lwjpegBufferPinnedRetrieve(lwjpegBufferPinned_t buffer, size_t* size, void** ptr);

lwjpegStatus_t LWJPEGAPI lwjpegBufferDeviceRetrieve(lwjpegBufferDevice_t buffer, size_t* size, void** ptr);

// this allows attaching same memory buffers to different states, allowing to switch implementations
// without allocating extra memory
lwjpegStatus_t LWJPEGAPI lwjpegStateAttachPinnedBuffer(lwjpegJpegState_t decoder_state,
    lwjpegBufferPinned_t pinned_buffer);

lwjpegStatus_t LWJPEGAPI lwjpegStateAttachDeviceBuffer(lwjpegJpegState_t decoder_state,
    lwjpegBufferDevice_t device_buffer);

///////////////////////////////////////////////////////////////////////////////////
// JPEG stream parameters //
///////////////////////////////////////////////////////////////////////////////////

// handle that stores stream information - metadata, encoded image parameters, encoded stream parameters
// stores everything on CPU side. This allows us parse header separately from implementation
// and retrieve more information on the stream. Also can be used for transcoding and transfering 
// metadata to encoder
struct lwjpegJpegStream;
typedef struct lwjpegJpegStream* lwjpegJpegStream_t;

lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamCreate(
    lwjpegHandle_t handle, 
    lwjpegJpegStream_t *jpeg_stream);

lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamDestroy(lwjpegJpegStream_t jpeg_stream);

lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamParse(
    lwjpegHandle_t handle,
    const unsigned char *data, 
    size_t length,
    int save_metadata,
    int save_stream,
    lwjpegJpegStream_t jpeg_stream);

lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamParseHeader(
    lwjpegHandle_t handle,
    const unsigned char *data, 
    size_t length,
    lwjpegJpegStream_t jpeg_stream);    

lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamGetJpegEncoding(
    lwjpegJpegStream_t jpeg_stream,
    lwjpegJpegEncoding_t* jpeg_encoding);

lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamGetFrameDimensions(
    lwjpegJpegStream_t jpeg_stream,
    unsigned int* width,
    unsigned int* height);

lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamGetComponentsNum(
    lwjpegJpegStream_t jpeg_stream,
    unsigned int* components_num);

lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamGetComponentDimensions(
    lwjpegJpegStream_t jpeg_stream,
    unsigned int component,
    unsigned int* width,
    unsigned int* height);




// if encoded is 1 color component then it assumes 4:0:0 (LWJPEG_CSS_GRAY, grayscale)
// if encoded is 3 color components it tries to assign one of the known subsamplings
//   based on the components subsampling infromation
// in case sampling factors are not stadard or number of components is different 
//   it will return LWJPEG_CSS_UNKNOWN
lwjpegStatus_t LWJPEGAPI lwjpegJpegStreamGetChromaSubsampling(
    lwjpegJpegStream_t jpeg_stream,
    lwjpegChromaSubsampling_t* chroma_subsampling);

///////////////////////////////////////////////////////////////////////////////////
// Decode parameters //
///////////////////////////////////////////////////////////////////////////////////
// decode parameters structure. Used to set decode-related tweaks
struct lwjpegDecodeParams;
typedef struct lwjpegDecodeParams* lwjpegDecodeParams_t;

lwjpegStatus_t LWJPEGAPI lwjpegDecodeParamsCreate(
    lwjpegHandle_t handle, 
    lwjpegDecodeParams_t *decode_params);

lwjpegStatus_t LWJPEGAPI lwjpegDecodeParamsDestroy(lwjpegDecodeParams_t decode_params);

// set output pixel format - same value as in lwjpegDecode()
lwjpegStatus_t LWJPEGAPI lwjpegDecodeParamsSetOutputFormat(
    lwjpegDecodeParams_t decode_params,
    lwjpegOutputFormat_t output_format);

// set to desired ROI. set to (0, 0, -1, -1) to disable ROI decode (decode whole image)
lwjpegStatus_t LWJPEGAPI lwjpegDecodeParamsSetROI(
    lwjpegDecodeParams_t decode_params,
    int offset_x, int offset_y, int roi_width, int roi_height);

// set to true to allow colwersion from CMYK to RGB or YUV that follows simple subtractive scheme
lwjpegStatus_t LWJPEGAPI lwjpegDecodeParamsSetAllowCMYK(
    lwjpegDecodeParams_t decode_params,
    int allow_cmyk);

// works only with the hardware decoder backend
lwjpegStatus_t LWJPEGAPI lwjpegDecodeParamsSetScaleFactor(
    lwjpegDecodeParams_t decode_params,
    lwjpegScaleFactor_t scale_factor);

///////////////////////////////////////////////////////////////////////////////////
// Decoder helper functions //
///////////////////////////////////////////////////////////////////////////////////

struct lwjpegJpegDecoder;
typedef struct lwjpegJpegDecoder* lwjpegJpegDecoder_t;

//creates decoder implementation
lwjpegStatus_t LWJPEGAPI lwjpegDecoderCreate(lwjpegHandle_t lwjpeg_handle, 
    lwjpegBackend_t implementation, 
    lwjpegJpegDecoder_t* decoder_handle);

lwjpegStatus_t LWJPEGAPI lwjpegDecoderDestroy(lwjpegJpegDecoder_t decoder_handle);

// on return sets is_supported value to 0 if decoder is capable to handle jpeg_stream 
// with specified decode parameters
lwjpegStatus_t LWJPEGAPI lwjpegDecoderJpegSupported(lwjpegJpegDecoder_t decoder_handle, 
    lwjpegJpegStream_t jpeg_stream,
    lwjpegDecodeParams_t decode_params,
    int* is_supported);

lwjpegStatus_t LWJPEGAPI lwjpegDecodeBatchedSupported(lwjpegHandle_t handle,
    lwjpegJpegStream_t jpeg_stream,
    int* is_supported);

lwjpegStatus_t LWJPEGAPI lwjpegDecodeBatchedSupportedEx(lwjpegHandle_t handle,
    lwjpegJpegStream_t jpeg_stream,
    lwjpegDecodeParams_t decode_params,
    int* is_supported);

// creates decoder state 
lwjpegStatus_t LWJPEGAPI lwjpegDecoderStateCreate(lwjpegHandle_t lwjpeg_handle,
    lwjpegJpegDecoder_t decoder_handle,
    lwjpegJpegState_t* decoder_state);

///////////////////////////////////////////////////////////////////////////////////
// Decode functions //
///////////////////////////////////////////////////////////////////////////////////
// takes parsed jpeg as input and performs decoding
lwjpegStatus_t LWJPEGAPI lwjpegDecodeJpeg(
    lwjpegHandle_t handle,
    lwjpegJpegDecoder_t decoder,
    lwjpegJpegState_t decoder_state,
    lwjpegJpegStream_t jpeg_bitstream,
    lwjpegImage_t *destination,
    lwjpegDecodeParams_t decode_params,
    lwdaStream_t stream);


// starts decoding on host and save decode parameters to the state
lwjpegStatus_t LWJPEGAPI lwjpegDecodeJpegHost(
    lwjpegHandle_t handle,
    lwjpegJpegDecoder_t decoder,
    lwjpegJpegState_t decoder_state,
    lwjpegDecodeParams_t decode_params,
    lwjpegJpegStream_t jpeg_stream);

// hybrid stage of decoding image,  ilwolves device async calls
// note that jpeg stream is a parameter here - because we still might need copy 
// parts of bytestream to device
lwjpegStatus_t LWJPEGAPI lwjpegDecodeJpegTransferToDevice(
    lwjpegHandle_t handle,
    lwjpegJpegDecoder_t decoder,
    lwjpegJpegState_t decoder_state,
    lwjpegJpegStream_t jpeg_stream,
    lwdaStream_t stream);

// finishing async operations on the device
lwjpegStatus_t LWJPEGAPI lwjpegDecodeJpegDevice(
    lwjpegHandle_t handle,
    lwjpegJpegDecoder_t decoder,
    lwjpegJpegState_t decoder_state,
    lwjpegImage_t *destination,
    lwdaStream_t stream);


lwjpegStatus_t LWJPEGAPI lwjpegDecodeBatchedEx(
          lwjpegHandle_t handle,
          lwjpegJpegState_t jpeg_handle,
          const unsigned char *const *data,
          const size_t *lengths,
          lwjpegImage_t *destinations,
          lwjpegDecodeParams_t *decode_params,
          lwdaStream_t stream);

///////////////////////////////////////////////////////////////////////////////////
// JPEG Transcoding Functions //
///////////////////////////////////////////////////////////////////////////////////

// copies metadata (JFIF, APP, EXT, COM markers) from parsed stream
lwjpegStatus_t lwjpegEncoderParamsCopyMetadata(
	lwjpegEncoderState_t encoder_state,
    lwjpegEncoderParams_t encode_params,
    lwjpegJpegStream_t jpeg_stream,
    lwdaStream_t stream);

// copies quantization tables from parsed stream
lwjpegStatus_t lwjpegEncoderParamsCopyQuantizationTables(
    lwjpegEncoderParams_t encode_params,
    lwjpegJpegStream_t jpeg_stream,
    lwdaStream_t stream);

// copies huffman tables from parsed stream. should require same scans structure
lwjpegStatus_t lwjpegEncoderParamsCopyHuffmanTables(
    lwjpegEncoderState_t encoder_state,
    lwjpegEncoderParams_t encode_params,
    lwjpegJpegStream_t jpeg_stream,
    lwdaStream_t stream);

#if defined(__cplusplus)
  }
#endif
 
#endif /* LW_JPEG_HEADER */
