//
//  Copyright (c) 2021 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <vector>

#include "CommonAsserts.h"

#include "lwca/helpers.h"

#include "test_PathTracer.h"
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include <exptest/coverageFileWriter.h>
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#include "tests/sanity/test_PathTracer_ptx_bin.h"

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenRecord   = Record<RayGenData>;
using MissRecord     = Record<MissData>;
using HitGroupRecord = Record<HitGroupData>;

//------------------------------------------------------------------------------
// Scene data
//------------------------------------------------------------------------------

// clang-format off

struct Vertex
{
    float x, y, z, pad;
};

struct Instance
{
    float transform[12];
};

const int32_t TRIANGLE_COUNT     = 32;
const int32_t TRIANGLE_MAT_COUNT = 5;
const int32_t SPHERE_COUNT       = 1;
const int32_t SPHERE_MAT_COUNT   = 1;

const std::array<Vertex, TRIANGLE_COUNT*3> g_vertices =
{ {
    // Floor  -- white lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,    0.0f,    0.0f, 0.0f },

    // Ceiling -- white lambert
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,  548.8f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    // Back wall -- white lambert
    {    0.0f,    0.0f,  559.2f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },

    // Right wall -- green lambert
    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },

    {    0.0f,    0.0f,    0.0f, 0.0f },
    {    0.0f,  548.8f,  559.2f, 0.0f },
    {    0.0f,    0.0f,  559.2f, 0.0f },

    // Left wall -- red lambert
    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,    0.0f,  559.2f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },

    {  556.0f,    0.0f,    0.0f, 0.0f },
    {  556.0f,  548.8f,  559.2f, 0.0f },
    {  556.0f,  548.8f,    0.0f, 0.0f },

    // Short block -- white lambert
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },

    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  242.0f,  165.0f,  274.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },

    {  290.0f,    0.0f,  114.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {  240.0f,    0.0f,  272.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },

    {  130.0f,    0.0f,   65.0f, 0.0f },
    {  290.0f,  165.0f,  114.0f, 0.0f },
    {  290.0f,    0.0f,  114.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },

    {   82.0f,    0.0f,  225.0f, 0.0f },
    {  130.0f,  165.0f,   65.0f, 0.0f },
    {  130.0f,    0.0f,   65.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {  240.0f,  165.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },

    {  240.0f,    0.0f,  272.0f, 0.0f },
    {   82.0f,  165.0f,  225.0f, 0.0f },
    {   82.0f,    0.0f,  225.0f, 0.0f },

    // Tall block -- white lambert
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },

    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  314.0f,  330.0f,  455.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },

    {  423.0f,    0.0f,  247.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  472.0f,    0.0f,  406.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  472.0f,  330.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },

    {  472.0f,    0.0f,  406.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  314.0f,    0.0f,  456.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  314.0f,  330.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },

    {  314.0f,    0.0f,  456.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  265.0f,    0.0f,  296.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  265.0f,  330.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },

    {  265.0f,    0.0f,  296.0f, 0.0f },
    {  423.0f,  330.0f,  247.0f, 0.0f },
    {  423.0f,    0.0f,  247.0f, 0.0f },

    // Ceiling light -- emmissive
    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },

    {  343.0f,  548.6f,  227.0f, 0.0f },
    {  213.0f,  548.6f,  332.0f, 0.0f },
    {  343.0f,  548.6f,  332.0f, 0.0f }
} };


const std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices =
{ {
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,  // Short block   -- cutout
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // Tall block    -- white lambert
    3, 3                           // Ceiling light -- emmissive
} };


const std::array<float3, TRIANGLE_MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f, 0.0f },
    {  0.0f,  0.0f, 0.0f },
    {  0.0f,  0.0f, 0.0f },
    { 15.0f, 15.0f, 5.0f },
    {  0.0f,  0.0f, 0.0f }
} };


const std::array<float3, TRIANGLE_MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.00f, 0.00f },
    { 0.70f, 0.25f, 0.00f }
} };


// NB: Some UV scaling is baked into the coordinates for the short block, since
//     the coordinates are used for the cutout texture.
const std::array<float2, TRIANGLE_COUNT* 3> g_tex_coords =
{ {
    // Floor
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Ceiling
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Back wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Right wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Left wall
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },

    // Short Block
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 8.0f },
    { 8.0f, 0.0f }, { 0.0f, 8.0f }, { 8.0f, 8.0f },

    // Tall Block
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },
    { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f },
    { 0.0f, 1.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f },

    // Ceiling light
    { 1.0f, 0.0f }, { 0.0f, 0.0f }, { 0.0f, 1.0f },
    { 1.0f, 0.0f }, { 0.0f, 1.0f }, { 1.0f, 1.0f }
} };


const Sphere g_sphere                = { 410.0f, 90.0f, 110.0f, 90.0f };
const float3 g_sphere_emission_color = { 0.0f };
const float3 g_sphere_diffuse_color  = { 0.1f, 0.2f, 0.8f };

const char expected_ascii[] =
    "                                                                                                \n"
    " .,,,,,,,,,,,,,,,,,,,,,;,,,,,,,,,,,,,,,;,;,,;,,,,,;;,,,;;;;,;;,;;,,;,,,,,,,,,,,,,,,,,,,,,,,,,,. \n"
    " ...,,,,,,,;,,,,,,,;;;,;;,;;;;;;;;;;,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,;,,,,,,... \n"
    " .....,,,;,;;;,;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;,,,,,,. \n"
    " .......,;;;;;;;;;;;;;;;;;;!;!!;;;;;!;;;!!!!;!!!;;!!!!;!;!;;;;;;;;;!;;;;;;;;;;;;;;;;;;;,,,,,,,, \n"
    " .........;;;;;;;;!;!!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!;;;;;;;;;;;;;,,,,,,,,, \n"
    " .....,,,,,,;;;;!!!!!!!!!!!!!!!oo!!oooo@@@@@@@@@@@@@@@@@@!!!!!!!!!!!!!!!!!!!!!!;;;;;,,;;;;,,,,, \n"
    " ...,,,,,,,,,,;;;!!!!!!!!!!oo!ooooooooooooooooooooooo!oo!!!!!!!!!!!!!!!!!!!!!!!;;;,;;;;;;;;;,,, \n"
    " ..,,,,,,,,,,,,,;;!!!!!!!!!!!!ooooooooooooooooooooooo!o!!!!!!!!!!!!!!!!!!!!!;;;;,;;;;!;;;;;;,,, \n"
    " ..,,,,,,,;;;,,,,.;;;;;;;!;;!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!;;;;;;;;,;;;!!!!!!;;;;;,, \n"
    " ..,,,,,;;;;;;;,,,.,;;;;;;;;;;;!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!;;!;;;;;;;;;;;,,,;;;!!!!!!!!;;;;;, \n"
    " .,,,,,,;;;;;;;;,,,;;;;;!!!!!!!!!ooooooooooooooooooooooooooo!!!!!!!!!!!;;;;;;;;;!!!oo!!!!;;;;;, \n"
    " .,,,,,;;;;;;;;;,,,;!!!!!!!!!!ooooooooo&&o&&&&&&&&&&&&&ooooooooooo!!!!!!!!!!;;;!!!ooo!!!!!;;;;, \n"
    " ,,,,,,;;;;;;;;;;,,!!!!!!!oooooooo&&&&&&&&&&&&&&&&&&&&&&&&&&&ooooooooo!!!!!!!;!!!ooooo!!!;;;;;, \n"
    " .,,,,,;;;;;;;;;;,,!!!!!oooooooo&&&&&&&&&8&8888888&88&&&&&&&&&&oooooooooo!!!!;!!!ooooo!!!!;;;;, \n"
    " .,,,,,,;;;;;;;;;;,!!!!ooooooo&&&&&&&&&8888888888888888&&&&&&&&&&&ooooooooo!!;!!oooooo!!!!;;;;, \n"
    " .,,,,,,;;;;;;;;;;,!!!!oooooo&&&&&&&&&8888888888888888888&&&&&&&&&&ooooooooo!!!!oooooo!!!!;;;;, \n"
    " ,,,,,,,;;;;;;;;;;,!!!oooooo&&&&&&&&&88888888888888888888&&&&&&&&&&&ooooooooo!!!oooooo!!!!;;;;, \n"
    " .,,,,,,;;;;;;;;;;,!!ooooooo&&&&&&&&&888888888888888888888&&&&&&&&&&ooooooooo!!!oooooo!!!!;;;;, \n"
    " .,,,,,,;;;;;;;;;;,!oooooooo&&&&&&&&&&88888888888888888888&&&&&&&&&&ooooooooo!!!ooooo!!!!!;;;;, \n"
    " .,,,,,,;;;;;;;;;;,!!ooooooo!!!!oooooo&&&&&&&&&&&&8888888&&&&&&&&&&&&oooooooo!!!!oooo!!!!!;;;;, \n"
    " ,,,,,,,,;;;;;;;;;,!!oooo!,,,;;;;;;;;;;;;!!!!!!!!o88888&&&&&&&&&&&&oooooooooo!!!oooo!!!!!!;;;;, \n"
    " .,,,,,,,;;;;;;;;;,!!!ooo!,,,;;;;;;;;;;;;;;;;;;;;o8&&&&&&&&&&&&&&&ooooooooooo!!!!ooo!!!!!;;;;;, \n"
    " .,,,,,,,,;;;;;;;;,!!!ooo!,,,,;;;;;;;;;;;;;;;;;;;o&&&&&&&&&&&&&&&&ooooooooooo!!!!o!o!!!!!;;;;;, \n"
    " ..,,,,,,,;;;;;;;;,!!!ooo;,,,,,;;;;;;;;;;;;;;;;;;o&&&&&&&&&&&&&&&o&oooooooooo!!!!!!!!!!!!;;;;;, \n"
    " .,,,,,,,,,;;;;;;,,!!!!oo;,,,,,,,;;;;;;;;;;;;;;;;o&&&&&&&&&&&&&&ooooooooooooo!!!!!!!!!!!;;;;;;, \n"
    " ..,,,,,,,,,;;;,,,,!!!!!o;,,,,,,,,;;;;;;;;;;;;;;;!&&&&&&&&&&&&ooooooooooooooo!!!!!!!!!!!;;;;;;, \n"
    " ..,,,,,,,,,,,,,,,,!!!!!!;,,,,,,,,,,;;;;;;;;;;;;;!&&&&&&&&&&&&ooooooooooooooo!!!!!!!!!!!;;;;;;, \n"
    " ...,,,,,,,,,,,,.,,!!!!!!;,,,,,,,,,,;,;;;;;;;;;;;!&&&&&&&&&&oooooooooooooooo!!!!!!!!!!!;;;;;;,, \n"
    " ....,,,,,,,,,,,..,!!!!!!;,,,,,,,,,,,,;,;;;;;;;;;!&&&&oo&ooooooooooooooooooo!!!!!!!!!!!;;;;;;,, \n"
    " ...,,,,,,,,,,....,;;;!;;,.,,,,,,,,,,,;;,,,;;;;;;!o&ooooooooooooooooooooooo!o!!!!!!!!!!;;;;;;,, \n"
    " ....,,,,,,,.......;;;;;;,,,;;;;;,,,,,,,,,;;;;;;;!oo!ooo!!ooo!!ooo!!!oo!!!!!!;!!!!!!!!;;;;;;;,, \n"
    " .....,,,,,........;,,,.,;.,,,;;;,;;,,,,,,,;;;. .;!!;;!!!;!!!;;!o!;,;!!!,.;!!;!!!!!!!!;;;;;;,,, \n"
    " .....,,,,,..........,,....,,,,..,;;,,,,,;;;;;.  ;;;.  !oo,  ;o!.   o..,;;!!!;!!!!!!!;;;;;;;,,, \n"
    " .....,,,,.........   .,,,. ...,;;,,.,,,,,;;;;.,..  ,oo;  !oo.  ,o,. ,,;.,;!!;;!!!!!!;;;;;;,,,, \n"
    " ......,,,......  ....    .,,,,.  ..,,;,,;;,,;,.,,;.;,,,,.,;;;  .!;;   ;,.;!!;;!!!!!!;;;;;,,,,, \n"
    " ......,,,...           .. .....,,,,,...,;;;;;.  ;;;.  ;oo,  ;o!.   !..,!;!!!;!!!!!!;;;;;;;,,,, \n"
    " ......,,....        ....      .;;,...,,.,;;;;,;,.  ,oo;  ;o!..,;;,..!!;..;!!;!!!!!!;;;;;;,,,,, \n"
    " ......,,....     ...     ..,,..     ,,;,,;;;;,.,.,.!...  ..;!  .,!o.  ,;;!oo!;;!!!;;;;;;;,,,,, \n"
    " .......,...         .....  ..  ,,,;;   ,;;;;;.  ;!!,  ;!!;  ;o!;   o..;!!!oooo!!!!;;;;;;;,,,,, \n"
    " ......,,...  .  .   .....     .,,...;;,,;;;;;,;,. .,oo;..!!!..!o.  ,&&!..!&&&&&o!!!;;;;;,,,,,, \n"
    " .......,............     ..,,..   ....,;;;;;;,,;.  o,..  ..;o  ..o&.  ,o!o&&&&o&oo!;;;;;;,,,,, \n"
    " ......,,.,;;;;;;,    .;!!.    ;,.;!,,!ooooooo;..!!!,. ;!!;  ;!!;.,;;,,;!!!oooooooooo!;;;;,,,,, \n"
    " ......,,!!!!!!!!!!,;,,   ,;;...  ,;!ooooooooo;o,;,,;;;;;;;..;!!;   !!!;;;;;;;;;;;;;!!!;;;;,,,, \n"
    " .....;!!!;;;;;,,,;;,,.,. ,.. ...,;;,;;!oooooo;!!,  ;!!,  ;!!;  ,!!!.  ,;;;;;;;;;;;;!!!!!;;,,,, \n"
    " ...,;;,,;,,.,;;,,,,,.,;,.,,,;,,,,,;,,;;!!!!!!;;,,,,,,,,,,,;;,..,;;;.  ;;;;,;,;;;;;;!!!!!!!;,,, \n"
    " .,;;,,,,,..,,,,,,,,,.,,,,,,,,,,,,,,;;;!!!!!!!!;;;;;;;;;;;;;,;,,,,,,,;;;;;,,,,,;;;;;;;!!!!!!!;, \n"
    " ....   .     ...               ......................,....................... ................ \n";

// clang-format on

//------------------------------------------------------------------------------

struct O7_API_PathTracer : public testing::Test
{
    void SetUp() override {}
    void TearDown() override {}
    void runTest();

    std::string ascii_output;
};


void O7_API_PathTracer::runTest()
{
    OptixDeviceContext       context;
    OptixRecordingLogger     logger;
    std::vector<LWdeviceptr> devicePointersToFree;

    char   log_string[4096];
    size_t log_string_size = 4096;

    // Initialize the Context
    {
        exptest::lwdaInitialize();
        OPTIX_CHECK( optixInit() );

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction       = &OptixRecordingLogger::callback;
        options.logCallbackData           = &logger;
        options.logCallbackLevel          = 3;
        LWcontext lwCtx                   = 0;
        OPTIX_CHECK( optixDeviceContextCreate( lwCtx, &options, &context ) );
    }

    // Build the triangle GAS
    OptixTraversableHandle triangle_gas_handle          = 0;
    LWdeviceptr            d_triangle_gas_output_buffer = 0;
    LWdeviceptr            d_vertices                   = 0;
    LWdeviceptr            d_tex_coords                 = 0;
    {
        const size_t vertices_size_in_bytes = g_vertices.size() * sizeof( Vertex );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size_in_bytes ) );
        devicePointersToFree.push_back( d_vertices );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_vertices ), g_vertices.data(), vertices_size_in_bytes, lwdaMemcpyHostToDevice ) );

        LWdeviceptr  d_mat_indices             = 0;
        const size_t mat_indices_size_in_bytes = g_mat_indices.size() * sizeof( uint32_t );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_mat_indices ), mat_indices_size_in_bytes ) );
        devicePointersToFree.push_back( d_mat_indices );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_mat_indices ), g_mat_indices.data(),
                                mat_indices_size_in_bytes, lwdaMemcpyHostToDevice ) );

        const size_t tex_coords_size_in_bytes = g_tex_coords.size() * sizeof( float2 );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_tex_coords ), tex_coords_size_in_bytes ) );
        devicePointersToFree.push_back( d_tex_coords );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_tex_coords ), g_tex_coords.data(), tex_coords_size_in_bytes,
                                lwdaMemcpyHostToDevice ) );

        uint32_t triangle_input_flags[TRIANGLE_MAT_COUNT] = {
            // One per SBT record for this build input
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            // Do not disable anyhit on the cutout material for the short block
            OPTIX_GEOMETRY_FLAG_NONE
        };

        OptixBuildInput triangle_input                           = {};
        triangle_input.type                                      = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangle_input.triangleArray.vertexFormat                = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangle_input.triangleArray.vertexStrideInBytes         = sizeof( Vertex );
        triangle_input.triangleArray.numVertices                 = static_cast<uint32_t>( g_vertices.size() );
        triangle_input.triangleArray.vertexBuffers               = &d_vertices;
        triangle_input.triangleArray.flags                       = triangle_input_flags;
        triangle_input.triangleArray.numSbtRecords               = TRIANGLE_MAT_COUNT;
        triangle_input.triangleArray.sbtIndexOffsetBuffer        = d_mat_indices;
        triangle_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof( uint32_t );
        triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof( uint32_t );

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &triangle_input, 1, &gas_buffer_sizes ) );

        LWdeviceptr d_temp_buffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );
        devicePointersToFree.push_back( d_temp_buffer );

        // non-compacted output
        LWdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8 ) );
        devicePointersToFree.push_back( d_buffer_temp_output_gas_and_compacted_size );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = ( LWdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild( context, 0, &accel_options, &triangle_input, 1, d_temp_buffer,
                                      gas_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_gas_and_compacted_size,
                                      gas_buffer_sizes.outputSizeInBytes, &triangle_gas_handle, &emitProperty, 1 ) );

        size_t compacted_gas_size;
        LWDA_CHECK( lwdaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), lwdaMemcpyDeviceToHost ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_triangle_gas_output_buffer ), compacted_gas_size ) );
            devicePointersToFree.push_back( d_triangle_gas_output_buffer );

            OPTIX_CHECK( optixAccelCompact( context, 0, triangle_gas_handle, d_triangle_gas_output_buffer,
                                            compacted_gas_size, &triangle_gas_handle ) );
        }
        else
        {
            d_triangle_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }

    // Build the sphere GAS
    OptixTraversableHandle sphere_gas_handle          = 0;
    LWdeviceptr            d_sphere_gas_output_buffer = 0;
    {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        // AABB build input
        float3    m_min = g_sphere.center - g_sphere.radius;
        float3    m_max = g_sphere.center + g_sphere.radius;
        OptixAabb aabb  = { m_min.x, m_min.y, m_min.z, m_max.x, m_max.y, m_max.z };

        LWdeviceptr d_aabb_buffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_aabb_buffer ), sizeof( OptixAabb ) ) );
        devicePointersToFree.push_back( d_aabb_buffer );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_aabb_buffer ), &aabb, sizeof( OptixAabb ), lwdaMemcpyHostToDevice ) );

        uint32_t        sphere_input_flag               = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput sphere_input                    = {};
        sphere_input.type                               = OPTIX_BUILD_INPUT_TYPE_LWSTOM_PRIMITIVES;
        sphere_input.lwstomPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
        sphere_input.lwstomPrimitiveArray.numPrimitives = 1;
        sphere_input.lwstomPrimitiveArray.flags         = &sphere_input_flag;
        sphere_input.lwstomPrimitiveArray.numSbtRecords = 1;

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &sphere_input, 1, &gas_buffer_sizes ) );

        LWdeviceptr d_temp_buffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), gas_buffer_sizes.tempSizeInBytes ) );
        devicePointersToFree.push_back( d_temp_buffer );

        // non-compacted output
        LWdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ), compactedSizeOffset + 8 ) );
        devicePointersToFree.push_back( d_buffer_temp_output_gas_and_compacted_size );

        OptixAccelEmitDesc emitProperty = {};
        emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitProperty.result = ( LWdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

        OPTIX_CHECK( optixAccelBuild( context, 0, &accel_options, &sphere_input, 1, d_temp_buffer,
                                      gas_buffer_sizes.tempSizeInBytes, d_buffer_temp_output_gas_and_compacted_size,
                                      gas_buffer_sizes.outputSizeInBytes, &sphere_gas_handle, &emitProperty, 1 ) );

        size_t compacted_gas_size;
        LWDA_CHECK( lwdaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), lwdaMemcpyDeviceToHost ) );

        if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
        {
            LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_sphere_gas_output_buffer ), compacted_gas_size ) );
            devicePointersToFree.push_back( d_sphere_gas_output_buffer );

            OPTIX_CHECK( optixAccelCompact( context, 0, sphere_gas_handle, d_sphere_gas_output_buffer,
                                            compacted_gas_size, &sphere_gas_handle ) );
        }
        else
        {
            d_sphere_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
        }
    }

    // Build the IAS
    OptixTraversableHandle ias_handle          = 0;
    LWdeviceptr            d_ias_output_buffer = 0;
    {
        LWdeviceptr d_instances;
        size_t      instance_size_in_bytes = sizeof( OptixInstance ) * 2;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_instances ), instance_size_in_bytes ) );
        devicePointersToFree.push_back( d_instances );

        OptixBuildInput instance_input = {};

        instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.instances    = d_instances;
        instance_input.instanceArray.numInstances = 2;

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags             = OPTIX_BUILD_FLAG_NONE;
        accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &instance_input, 1, &ias_buffer_sizes ) );

        LWdeviceptr d_temp_buffer;
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_temp_buffer ), ias_buffer_sizes.tempSizeInBytes ) );
        devicePointersToFree.push_back( d_temp_buffer );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_ias_output_buffer ), ias_buffer_sizes.outputSizeInBytes ) );
        devicePointersToFree.push_back( d_ias_output_buffer );

        // Use the identity matrix for the instance transform
        Instance instance = { { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 } };

        OptixInstance optix_instances[2];
        memset( optix_instances, 0, instance_size_in_bytes );

        optix_instances[0].traversableHandle = triangle_gas_handle;
        optix_instances[0].flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instances[0].instanceId        = 0;
        optix_instances[0].sbtOffset         = 0;
        optix_instances[0].visibilityMask    = 1;
        memcpy( optix_instances[0].transform, instance.transform, sizeof( float ) * 12 );

        optix_instances[1].traversableHandle = sphere_gas_handle;
        optix_instances[1].flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instances[1].instanceId        = 1;
        optix_instances[1].sbtOffset         = TRIANGLE_MAT_COUNT * RAY_TYPE_COUNT;
        optix_instances[1].visibilityMask    = 1;
        memcpy( optix_instances[1].transform, instance.transform, sizeof( float ) * 12 );

        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_instances ), &optix_instances, instance_size_in_bytes, lwdaMemcpyHostToDevice ) );

        OPTIX_CHECK( optixAccelBuild( context, 0, &accel_options, &instance_input, 1, d_temp_buffer, ias_buffer_sizes.tempSizeInBytes,
                                      d_ias_output_buffer, ias_buffer_sizes.outputSizeInBytes, &ias_handle, nullptr, 0 ) );
    }

    // Create the Module
    OptixModule                 ptx_module               = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount          = 100;

        module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipeline_compile_options.numPayloadValues      = 2;
        pipeline_compile_options.numAttributeValues    = sphere::NUM_ATTRIBUTE_VALUES;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        size_t      inputSize = optix::data::gettest_PathTracerSourceSizes()[0];
        const char* input     = optix::data::gettest_PathTracerSources()[1];

        OPTIX_CHECK( optixModuleCreateFromPTX( context, &module_compile_options, &pipeline_compile_options, input,
                                               inputSize, log_string, &log_string_size, &ptx_module ) );
    }

    // Create the Program Groups
    OptixProgramGroup raygen_prog_group    = 0;
    OptixProgramGroup radiance_miss_group  = 0;
    OptixProgramGroup occlusion_miss_group = 0;
    OptixProgramGroup radiance_hit_group   = 0;
    OptixProgramGroup occlusion_hit_group  = 0;
    {
        OptixProgramGroupOptions program_group_options = {};

        OptixProgramGroupDesc raygen_prog_group_desc    = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        OPTIX_CHECK( optixProgramGroupCreate( context, &raygen_prog_group_desc, 1, &program_group_options, log_string,
                                              &log_string_size, &raygen_prog_group ) );

        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        OPTIX_CHECK( optixProgramGroupCreate( context, &miss_prog_group_desc, 1, &program_group_options, log_string,
                                              &log_string_size, &radiance_miss_group ) );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        OPTIX_CHECK( optixProgramGroupCreate( context, &miss_prog_group_desc, 1, &program_group_options, log_string,
                                              &log_string_size, &occlusion_miss_group ) );

        OptixProgramGroupDesc hit_prog_group_desc        = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        hit_prog_group_desc.hitgroup.moduleAH            = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        hit_prog_group_desc.hitgroup.moduleIS            = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        OPTIX_CHECK( optixProgramGroupCreate( context, &hit_prog_group_desc, 1, &program_group_options, log_string,
                                              &log_string_size, &radiance_hit_group ) );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        hit_prog_group_desc.hitgroup.moduleAH            = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
        hit_prog_group_desc.hitgroup.moduleIS            = ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        OPTIX_CHECK( optixProgramGroupCreate( context, &hit_prog_group_desc, 1, &program_group_options, log_string,
                                              &log_string_size, &occlusion_hit_group ) );
    }

    // Link the Pipeline
    OptixPipeline pipeline = 0;
    {
        const uint32_t    max_trace_depth  = 2;
        OptixProgramGroup program_groups[] = { raygen_prog_group, radiance_miss_group, occlusion_miss_group,
                                               radiance_hit_group, occlusion_hit_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth            = max_trace_depth;
        pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

        OPTIX_CHECK( optixPipelineCreate( context, &pipeline_compile_options, &pipeline_link_options, program_groups,
                                          sizeof( program_groups ) / sizeof( program_groups[0] ), log_string,
                                          &log_string_size, &pipeline ) );

        OptixStackSizes stack_sizes = {};
        for( auto& prog_group : program_groups )
        {
            OPTIX_CHECK( optixUtilAclwmulateStackSizes( prog_group, &stack_sizes ) );
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth, 0, 0, &direct_callable_stack_size_from_traversal,
                                                 &direct_callable_stack_size_from_state, &continuation_stack_size ) );
        OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                direct_callable_stack_size_from_state, continuation_stack_size, 1 ) );
    }

    // Create the SBT
    OptixShaderBindingTable sbt = {};
    {
        LWdeviceptr  d_raygen_record;
        const size_t raygen_record_size = sizeof( RayGenRecord );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_raygen_record ), raygen_record_size ) );
        devicePointersToFree.push_back( d_raygen_record );

        RayGenRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
        rg_sbt.data = { 1.0f, 0.0f, 0.0f };

        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_raygen_record ), &rg_sbt, raygen_record_size, lwdaMemcpyHostToDevice ) );

        LWdeviceptr  d_miss_records;
        const size_t miss_record_size = sizeof( MissRecord );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_miss_records ), miss_record_size * RAY_TYPE_COUNT ) );
        devicePointersToFree.push_back( d_miss_records );

        MissRecord ms_sbt[2];
        OPTIX_CHECK( optixSbtRecordPackHeader( radiance_miss_group, &ms_sbt[0] ) );
        ms_sbt[0].data = { 0.0f, 0.0f, 0.0f };
        OPTIX_CHECK( optixSbtRecordPackHeader( occlusion_miss_group, &ms_sbt[1] ) );
        ms_sbt[1].data = { 0.0f, 0.0f, 0.0f };

        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_miss_records ), ms_sbt, miss_record_size * RAY_TYPE_COUNT,
                                lwdaMemcpyHostToDevice ) );

        LWdeviceptr  d_hitgroup_records;
        const size_t hitgroup_record_size = sizeof( HitGroupRecord );
        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_hitgroup_records ),
                                hitgroup_record_size * ( RAY_TYPE_COUNT * ( TRIANGLE_MAT_COUNT + SPHERE_MAT_COUNT ) ) ) );
        devicePointersToFree.push_back( d_hitgroup_records );

        HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * ( TRIANGLE_MAT_COUNT + SPHERE_MAT_COUNT )];

        // Set up the HitGroupRecords for the triangle materials
        for( int i = 0; i < TRIANGLE_MAT_COUNT; ++i )
        {
            {
                const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

                OPTIX_CHECK( optixSbtRecordPackHeader( radiance_hit_group, &hitgroup_records[sbt_idx] ) );
                hitgroup_records[sbt_idx].data.emission_color = g_emission_colors[i];
                hitgroup_records[sbt_idx].data.diffuse_color  = g_diffuse_colors[i];
                hitgroup_records[sbt_idx].data.vertices       = reinterpret_cast<float4*>( d_vertices );
                hitgroup_records[sbt_idx].data.tex_coords     = reinterpret_cast<float2*>( d_tex_coords );
            }

            {
                const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
                memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

                OPTIX_CHECK( optixSbtRecordPackHeader( occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
                hitgroup_records[sbt_idx].data.vertices   = reinterpret_cast<float4*>( d_vertices );
                hitgroup_records[sbt_idx].data.tex_coords = reinterpret_cast<float2*>( d_tex_coords );
            }
        }

        // Set up the HitGroupRecords for the sphere material
        {
            const int sbt_idx = TRIANGLE_MAT_COUNT * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for sphere material

            OPTIX_CHECK( optixSbtRecordPackHeader( radiance_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[sbt_idx].data.emission_color = g_sphere_emission_color;
            hitgroup_records[sbt_idx].data.diffuse_color  = g_sphere_diffuse_color;
            hitgroup_records[sbt_idx].data.sphere         = g_sphere;
        }

        {
            const int sbt_idx = TRIANGLE_MAT_COUNT * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for sphere material
            memset( &hitgroup_records[sbt_idx], 0, hitgroup_record_size );

            OPTIX_CHECK( optixSbtRecordPackHeader( occlusion_hit_group, &hitgroup_records[sbt_idx] ) );
            hitgroup_records[sbt_idx].data.sphere = g_sphere;
        }

        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_hitgroup_records ), hitgroup_records,
                                hitgroup_record_size * ( RAY_TYPE_COUNT * ( TRIANGLE_MAT_COUNT + SPHERE_MAT_COUNT ) ),
                                lwdaMemcpyHostToDevice ) );

        sbt.raygenRecord                = d_raygen_record;
        sbt.missRecordBase              = d_miss_records;
        sbt.missRecordStrideInBytes     = static_cast<uint32_t>( miss_record_size );
        sbt.missRecordCount             = RAY_TYPE_COUNT;
        sbt.hitgroupRecordBase          = d_hitgroup_records;
        sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
        sbt.hitgroupRecordCount         = RAY_TYPE_COUNT * ( TRIANGLE_MAT_COUNT + SPHERE_MAT_COUNT );
    }

    // Init the launch params
    Params  params   = {};
    Params* d_params = nullptr;
    {
        params.height = 96;
        params.width  = 96;

        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &params.frame_buffer ), params.width * params.height * sizeof( uchar4 ) ) );
        devicePointersToFree.push_back( (LWdeviceptr)params.frame_buffer );

        params.samples_per_launch = 1024;

        params.light.emission = make_float3( 15.0f, 15.0f, 5.0f );
        params.light.corner   = make_float3( 343.0f, 548.5f, 227.0f );
        params.light.v1       = make_float3( 0.0f, 0.0f, 105.0f );
        params.light.v2       = make_float3( -130.0f, 0.0f, 0.0f );
        params.light.normal   = normalize( cross( params.light.v1, params.light.v2 ) );

        // Camera parameters derived from the optixLwtouts_exp sample with a square aspect ratio
        params.eye = make_float3( 278.0f, 273.0f, -900.0f );
        params.U   = make_float3( -387.817566f, 0.0f, 0.0f );
        params.V   = make_float3( 0.0f, 387.817566f, -0.0f );
        params.W   = make_float3( 0.0f, 0.0f, 1230.0f );

        params.handle = ias_handle;

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
        SETUP_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

        LWDA_CHECK( lwdaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( Params ) ) );
        devicePointersToFree.push_back( (LWdeviceptr)d_params );
        LWDA_CHECK( lwdaMemcpy( reinterpret_cast<void*>( d_params ), &params, sizeof( Params ), lwdaMemcpyHostToDevice ) );
    }

    // Launch
    {
        LWstream stream;
        LWDA_CHECK( lwdaStreamCreate( &stream ) );
        OPTIX_CHECK( optixLaunch( pipeline, stream, reinterpret_cast<LWdeviceptr>( d_params ), sizeof( Params ), &sbt,
                                  params.width, params.height, 1 ) );
        LWDA_SYNC_CHECK();

        // Copy the output back to the host
        std::vector<uchar4> buffer;
        buffer.resize( params.width * params.height );
        LWDA_CHECK( lwdaMemcpy( static_cast<void*>( buffer.data() ), params.frame_buffer,
                                params.width * params.height * sizeof( uchar4 ), lwdaMemcpyDeviceToHost ) );

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
        ANALYZE_API_CALL_COVERAGE( params.covered );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

        // Output the image as ASCII art
        std::ostringstream ascii_out;
        char               density_map[] = { ' ', '.', ',', ';', '!', 'o', '&', '8', '#', '@' };
        for( unsigned int y = 0; y < params.height; y += 2 )
        {
            uchar4* row = (uchar4*)buffer.data() + ( ( params.height - y - 1 ) * params.width );
            for( unsigned int x = 0; x < params.width; ++x )
            {
                uchar4 ucolor = row[x];
                float3 color  = make_float3( static_cast<float>( ucolor.x ), static_cast<float>( ucolor.y ),
                                            static_cast<float>( ucolor.z ) )
                               / make_float3( 256.0f );
                float lum = color.x * 0.3f + color.y * 0.6f + color.z * 0.1f;
                ascii_out << density_map[static_cast<int>( lum * 10 )];
            }
            ascii_out << "\n";
        }
        ascii_output = ascii_out.str();
    }

    // Cleanup
    {
        OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
        OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( radiance_miss_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( occlusion_miss_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( radiance_hit_group ) );
        OPTIX_CHECK( optixProgramGroupDestroy( occlusion_hit_group ) );
        OPTIX_CHECK( optixModuleDestroy( ptx_module ) );

        for( auto d : devicePointersToFree )
            LWDA_CHECK( lwdaFree( (void*)d ) );

        OPTIX_CHECK( optixDeviceContextDestroy( context ) );
    }
}


TEST_F( O7_API_PathTracer, CheckOutput )
{
    runTest();

    // Check the output against the reference
    std::string expected_output( expected_ascii );
    if( ascii_output != expected_output )
    {
        std::cout << "Expected:\n" << expected_output << std::endl;
        std::cout << "Actual:\n" << ascii_output << std::endl;
    }

    // This should never fail, but I'll leave it as a bounds check on the
    // string sizes
    ASSERT_EQ( expected_output.size(), ascii_output.size() );

    // The image should be well colwerged at 1024 SPP, but the quantization when
    // colwerting to ASCII could make the test overly sensitive to small
    // changes. So we will allow some small number of pixels to differ before
    // reporting failure.
    size_t error_count = 0;
    for( size_t idx = 0; idx < ascii_output.size(); ++idx )
        error_count += ascii_output[idx] != expected_output[idx];

    EXPECT_LT( error_count, 100 );
}
