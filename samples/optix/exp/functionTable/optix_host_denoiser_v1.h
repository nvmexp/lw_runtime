/*
 * Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

extern "C" {

typedef enum OptixDenoiserInputKind_v1
{
    OPTIX_DENOISER_INPUT_RGB               = 0x2301,
    OPTIX_DENOISER_INPUT_RGB_ALBEDO        = 0x2302,
    OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL = 0x2303,
} OptixDenoiserInputKind_v1;

typedef struct OptixDenoiserOptions_v1
{
    OptixDenoiserInputKind_v1 inputKind;
} OptixDenoiserOptions_v1;

typedef enum OptixDenoiserModelKind_v1
{
    /// Use the model provided by the associated pointer.  See the programming guide for a
    /// description of how to format the data.
    OPTIX_DENOISER_MODEL_KIND_USER_v1 = 0x2321,

    /// Use the built-in model appropriate for low dynamic range input.
    OPTIX_DENOISER_MODEL_KIND_LDR_v1 = 0x2322,

    /// Use the built-in model appropriate for high dynamic range input.
    OPTIX_DENOISER_MODEL_KIND_HDR_v1 = 0x2323,

    /// Use the built-in model appropriate for high dynamic range input and support for AOVs
    OPTIX_DENOISER_MODEL_KIND_AOV_v1 = 0x2324,

    /// Use the built-in model appropriate for high dynamic range input, temporally stable
    OPTIX_DENOISER_MODEL_KIND_TEMPORAL_v1 = 0x2325
} OptixDenoiserModelKind_v1;

OptixResult optixDenoiserCreate_v1( OptixDeviceContext context,
                                    const OptixDenoiserOptions_v1* options,
                                    OptixDenoiser* denoiser );

OptixResult optixDenoiserSetModel_v1( OptixDenoiser denoiser, OptixDenoiserModelKind_v1 kind, void* data, size_t sizeInBytes );

OptixResult optixDenoiserIlwoke_v1(
                                 OptixDenoiser              denoiser,
                                 LWstream                   stream,
                                 const OptixDenoiserParams* params,
                                 LWdeviceptr                denoiserState,
                                 size_t                     denoiserStateSizeInBytes,
                                 const OptixImage2D*        inputLayers,
                                 unsigned int               numInputLayers,
                                 unsigned int               inputOffsetX,
                                 unsigned int               inputOffsetY,
                                 const OptixImage2D*        outputLayer,
                                 LWdeviceptr                scratch,
                                 size_t                     scratchSizeInBytes );
}
