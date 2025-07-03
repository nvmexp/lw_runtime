/*
 * Copyright (c) 2015-2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// File: tests.h

// LWOG_TEST_PATH_MASK_IMMED_ONLY:  Colwenience macro to specify that only
// immediate-mode tests should be performed.
#define LWOG_TEST_PATH_MASK_IMMED_ONLY          LWOG_TEST_PATH_MASK_BASE

#define DBGTST   LWOG_TEST_PATH_MASK_DEBUG_TEST   // test is old and superseded by a better test (not regressed)

// Indicate that a test should be skipped by default (e.g., if it crashes in automated testing)
// Overridden by the -unskip command-line parameter.
#if defined(LW_HOS)
#   define SKIP_HOS     LWOG_TEST_PATH_MASK_SKIP
#   define SKIP_LINUX   0
#   define SKIP_WINDOWS 0
#elif defined(LW_LINUX)
#   define SKIP_HOS     0
#   define SKIP_LINUX   LWOG_TEST_PATH_MASK_SKIP
#   define SKIP_WINDOWS 0
#elif defined(_WIN32)
#   define SKIP_HOS     0
#   define SKIP_LINUX   0
#   define SKIP_WINDOWS LWOG_TEST_PATH_MASK_SKIP
#endif

//////////////////////////////////////////////////////////////////////////

BEGIN_TEST_GROUP(TESTS_LWN)
#if !defined(TEST_GROUP_ENUMERATE_GROUPS)

TEST_DESC_I(lwn_01tri,                              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_01tri_cpp,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_04compute,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_alloc_pools,                        LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_alloc_pools_flush,                  LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_alloc_programs,                     LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_alloc_queues_4,                     LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_alloc_queues_8,                     LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_alloc_queues_32,                    LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_alphatest,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_bindless,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_bindless_cpp,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_command,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_command_cpp,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_command_transient,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_command_transient_cpp,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_debug,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_debug_cpp,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_queue,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_queue_cpp,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_geometry,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_geometry_cpp,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_multisample,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_multisample_cpp,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_tess_tcs,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_tess_tcs_cpp,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_tess_tesonly,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_basic_tess_tesonly_cpp,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_bgr5,                               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_bigtex,                             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_bind_deferred,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_bind_deferred_fastpath,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_blend,                              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_blend_srgb,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_blend_multisample,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)

#define MKTEST(suff, num, overlap, premult)                                         \
TEST_DESC_I(lwn_blendadv_##num##suff,               LWOG_TEST_PATH_MASK_IMMED_ONLY)

#define MKTESTP(suff, num, overlap)     \
  MKTEST(suff, num, overlap, false)     \
  MKTEST(suff##p, num, overlap, true)

#define MKTESTO(num)                                    \
  MKTESTP(c, num, BlendAdvancedOverlap::CONJOINT)       \
  MKTESTP(d, num, BlendAdvancedOverlap::DISJOINT)       \
  MKTESTP(u, num, BlendAdvancedOverlap::UNCORRELATED)

MKTESTO(0)
MKTESTO(1)
MKTESTO(2)
MKTESTO(3)
MKTESTO(4)
// Skip test set #5.
MKTESTO(6)
MKTESTO(7)
MKTESTO(8)

#undef MKTEST
#undef MKTESTP
#undef MKTESTO

#define C2CTEST(__aa__)                                                        \
    TEST_DESC_I(lwn_c2c_##__aa__##xaa, LWOG_TEST_PATH_MASK_IMMED_ONLY )        \
    TEST_DESC_I(lwn_c2c_##__aa__##xaa_early, LWOG_TEST_PATH_MASK_IMMED_ONLY)   \
    TEST_DESC_I(lwn_c2c_##__aa__##xaa_tir, LWOG_TEST_PATH_MASK_IMMED_ONLY)     \
    TEST_DESC_I(lwn_c2c_##__aa__##xaa_a2c, LWOG_TEST_PATH_MASK_IMMED_ONLY)     \
    TEST_DESC_I(lwn_c2c_##__aa__##xaa_c2, LWOG_TEST_PATH_MASK_IMMED_ONLY)      \
    TEST_DESC_I(lwn_c2c_##__aa__##xaa_m, LWOG_TEST_PATH_MASK_IMMED_ONLY)

C2CTEST(2)
C2CTEST(4)
C2CTEST(8)

#undef C2CTEST

TEST_DESC_I(lwn_clear_buffer,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_clear_texture,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_clear_texture_rgba8_msaa2,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_clear_texture_rgba8_msaa4,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_clear_texture_rgba8_msaa8,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_clear_tex_overwrite,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Not really a test; this is just to verify client allocator for mempools
TEST_DESC_I(lwn_client_allocator,                   LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_client_allocator_expand,            LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_cmdbuf_add_memory,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_cmdbuf_callcopy,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_cmdbuf_callcopy_fastpath,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_cmdbuf_disable_oom,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)

#define COLOR_REDUCTION(cls) \
    TEST_DESC_I(lwn_color_reduction_ ## cls, LWOG_TEST_PATH_MASK_IMMED_ONLY)

COLOR_REDUCTION(unorm8)
COLOR_REDUCTION(unorm10)
COLOR_REDUCTION(unorm16)
COLOR_REDUCTION(srgb8)
COLOR_REDUCTION(fp16)
COLOR_REDUCTION(fp11)

#undef COLOR_REDUCTION

TEST_DESC_I(lwn_compare_objects,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_compute_indirect,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_compression_zbc,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_compression_gradient,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_conditional_render,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)

#define CONSRAST(N,MODENAME,MODE)    \
    TEST_DESC_I(lwn_conservative_raster_##MODENAME##_##N##_msdisable, LWOG_TEST_PATH_MASK_IMMED_ONLY) \
    TEST_DESC_I(lwn_conservative_raster_##MODENAME##_##N##_msenable, LWOG_TEST_PATH_MASK_IMMED_ONLY) \
    TEST_DESC_I(lwn_conservative_raster_##MODENAME##_##N##_dilate25, LWOG_TEST_PATH_MASK_IMMED_ONLY) \
    TEST_DESC_I(lwn_conservative_raster_##MODENAME##_##N##_dilate50, LWOG_TEST_PATH_MASK_IMMED_ONLY) \
    TEST_DESC_I(lwn_conservative_raster_##MODENAME##_##N##_dilate75, LWOG_TEST_PATH_MASK_IMMED_ONLY) \

#define CONSRAST2(N)    \
    CONSRAST(N,tri,TRIANGLES)    \
    CONSRAST(N,line,LINES)       \
    CONSRAST(N,point,POINTS)     \

CONSRAST2(0)
CONSRAST2(4)

TEST_DESC_I(lwn_copy_compressible_off,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_compressible_on,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image_1d,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image_linear_2d,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image_linear_rectangle,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image_mipmaps,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image_mirror,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image_ms_2x,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image_ms_4x,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_image_ms_8x,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_texture_limits,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_copy_texture_stride,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_counters,                           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_counters_zlwll,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_counters_zlwll_no_zlwll,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
#if defined(LWNTEST_LWDA_ENABLED)
TEST_DESC_I(lwn_lwda_buffer_interop,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Setting texture interop tests as debug tests dues to Bug 3315968
TEST_DESC_I(lwn_lwda_tex2d_n2c_interop,             LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_lwda_tex2d_c2n_interop,             LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
#endif
TEST_DESC_I(lwn_dbounds,                            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_depth_range,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_depth_stencil_s8,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_depth_stencil_d16,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_depth_stencil_d24,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_depth_stencil_d32f,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_depth_stencil_d24s8,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_depth_stencil_d32fs8,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Skipped for Jira LWN-77
TEST_DESC_I(lwn_debug_level0,                       LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_debug_level1,                       LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_debug_level2,                       LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_debug_level3,                       LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_debug_level4,                       LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_debug_walk,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_depth_write_war,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_debuglabel,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)

#define DIPTEST(N)                                                                   \
    TEST_DESC_I(lwn_decompz##N,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)  \
    TEST_DESC_I(lwn_decompzscissor##N,              LWOG_TEST_PATH_MASK_IMMED_ONLY)

DIPTEST(0)
DIPTEST(2)
DIPTEST(4)
DIPTEST(8)

#undef DIPTEST

TEST_DESC_I(lwn_defaults,                           LWOG_TEST_PATH_MASK_IMMED_ONLY)
#if defined(LW_TEGRA)
TEST_DESC_I(lwn_disable_both_compression,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
#endif

#define DOWNSAMPLE(__EXTENSION__)                                                      \
    TEST_DESC_I(lwn_##__EXTENSION__,        LWOG_TEST_PATH_MASK_IMMED_ONLY)  \
    TEST_DESC_I(lwn_##__EXTENSION__##_view, LWOG_TEST_PATH_MASK_IMMED_ONLY) 

DOWNSAMPLE(downsample)
DOWNSAMPLE(downsample_many)
DOWNSAMPLE(downsample_queues)

#undef DOWNSAMPLE

TEST_DESC_I(lwn_draw_basic,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_draw_basic_spirv,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_draw_texture,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_draw_texture_shdz,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_event,                              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_fastpath,                           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_force_sample_shading,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_formats_copy_tex_colwert,           LWOG_TEST_PATH_MASK_IMMED_ONLY)

#define FORMATS_TEST(__test_name__)                                                 \
    TEST_DESC_I(__test_name__,                      LWOG_TEST_PATH_MASK_IMMED_ONLY) \
    TEST_DESC_I(__test_name__##_coherent,           LWOG_TEST_PATH_MASK_IMMED_ONLY)

FORMATS_TEST(lwn_formats_tex_1d)
FORMATS_TEST(lwn_formats_tex_1d_array)
FORMATS_TEST(lwn_formats_tex_2d)
FORMATS_TEST(lwn_formats_tex_2d_array)
FORMATS_TEST(lwn_formats_tex_3d)
FORMATS_TEST(lwn_formats_tex_buffer)
FORMATS_TEST(lwn_formats_tex_lwbe)
FORMATS_TEST(lwn_formats_tex_lwbe_array)
FORMATS_TEST(lwn_formats_tex_rectangle)

#undef FORMATS_TEST
TEST_DESC_I(lwn_formats_vertex,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_fs_interlock,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_getters,                            LWOG_TEST_PATH_MASK_IMMED_ONLY)
#if !defined(LW_LINUX)
// LWN/GL interop not supported on L4T
TEST_DESC_I(lwn_gl_interop,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gl_sync_interop,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
#endif
TEST_DESC_I(lwn_glsl_builtins,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glsl_bindings_image,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glsl_bindings_ssbo,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glsl_bindings_texture,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glsl_bindings_ubo,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glsl_bindings_unused_single,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glsl_bindings_unused_array,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glsl_derivative_control,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glsl_initialized_array,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_glslc,                              LWOG_TEST_PATH_MASK_IMMED_ONLY)

#if defined(_WIN32) && !defined(LW_MODS)
TEST_DESC_I(lwn_glslc_include,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
#endif

TEST_DESC_I(lwn_gs_lwbe_single,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gs_lwbe_inst,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gs_lwbe_pass,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gs_multires_inst,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gs_multires_inst_tess,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gs_multires_pass,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gs_multires_pass_tess,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gs_multires_single,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_gs_multires_single_tess,            LWOG_TEST_PATH_MASK_IMMED_ONLY)

TEST_DESC_I(lwn_hlsl,                               LWOG_TEST_PATH_MASK_IMMED_ONLY)

TEST_DESC_I(lwn_image_basic,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_basic_discard,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_basic_th,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_binding,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_bindless_basic,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_bindless_basic_th,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_bindless_compute,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_bindless_compute_th,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_compute,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_compute_th,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_swizzle,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_swizzle_th,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_targets_load,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_targets_store,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_targets_query,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_image_targets_texload,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
#if defined(_WIN32)
TEST_DESC_I(lwn_internal_shader_pool,               LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
#endif
// Not really a test; this is just a helper to test texture decompression in the other lwn_formats_* tests.
// TEST_DESC_I(lwn_formats_gen_data,                   LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
// Skipped for Jira LWN-377
TEST_DESC_I(lwn_logicop,                            LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_lowp_mediump_mixed,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Skipped for Jira LWN-77 / LWN-78
TEST_DESC_I(lwn_mempool,                            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_mempool_external,                   LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_mempool_mapping,                    LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_mempool_virtual,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_mempool_virtual_mix,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_mempool_virtual_mix_compr,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Skipped for Jira LWN-372
TEST_DESC_I(lwn_mt_texture_creation_rgba8,          LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_mt_texture_creation_rgba8_comp,     LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)

TEST_DESC_I(lwn_multi_draw_indirect_count,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_multibind_compute,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_multisample_a2cov,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_multisample_a2cov_nd,               LWOG_TEST_PATH_MASK_IMMED_ONLY)

TEST_DESC_I(lwn_natvis_c,                           LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
TEST_DESC_I(lwn_natvis_cpp,                         LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)

TEST_DESC_I(lwn_polygon_offset,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_pool_bufaddr,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_post_depth_coverage,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Skipped for Jira LWN-143
TEST_DESC_I(lwn_present_interval,                   LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX | SKIP_WINDOWS)
TEST_DESC_I(lwn_present_interval_full,              LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
// Skipped for Jira LWN-372
TEST_DESC_I(lwn_programs_mt_2,                      LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_programs_mt_8,                      LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_programs_mt_glslc_2,                LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_programs_mt_glslc_8,                LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_programs_mt_glslc_16,               LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_programs_mt_glslc_32,               LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_programs_mt_glslc_64,               LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_programs_mt_glslc_spec_2,           LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_programs_mt_glslc_spec_8,           LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_provoking_vertex,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_psprite_ll,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_psprite_ul,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
#if defined(HAS_DEVTOOLS) && defined(LW_TEGRA)
TEST_DESC_I(lwn_pte,                                LWOG_TEST_PATH_MASK_IMMED_ONLY)
#endif
TEST_DESC_I(lwn_query,                              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_raster_points,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_raster_lines,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_raster_triangles,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Skipped for Jira LWN-77
TEST_DESC_I(lwn_raw_storage_class_switch,           LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
#if defined(LW_TEGRA)
// Skipped for Jira LWN-81
TEST_DESC_I(lwn_robustness_ctxsw_timeout,           LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_robustness_mmu_fault,               LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_robustness_preempt_timeout,         LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_robustness_pbdma_exception,         LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_robustness_gr_exception,            LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
#endif
TEST_DESC_I(lwn_rt_attachments,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sample_control_interpolant,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sample_control_per_pixel,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sample_control_sampleid,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sample_control_samplemask,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sample_control_samplepos,           LWOG_TEST_PATH_MASK_IMMED_ONLY)


#define SAMPLELOC(num) \
    TEST_DESC_I(lwn_sample_location_fbo##num##_pattern16, LWOG_TEST_PATH_MASK_IMMED_ONLY) \
    TEST_DESC_I(lwn_sample_location_fbo##num##_regular16, LWOG_TEST_PATH_MASK_IMMED_ONLY) \
    TEST_DESC_I(lwn_sample_location_fbo##num##_face,      LWOG_TEST_PATH_MASK_IMMED_ONLY)

SAMPLELOC(2)
SAMPLELOC(4)
SAMPLELOC(8)

TEST_DESC_I(lwn_sample_location_window_pattern16,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sample_location_window_regular16,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sample_location_window_face,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sample_location_fbo8_pattern16_msdisable, LWOG_TEST_PATH_MASK_IMMED_ONLY)

#undef SAMPLELOC

TEST_DESC_I(lwn_sampler_anisotropy,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_border,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_filter,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_lodbias,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_lodclamp,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_minmax,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_querylod,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_wrap,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_wrap_noobj,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_wrapmix,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sampler_wrapnearest,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_separate_sampler,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_scratch,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_scratch_debug,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_scratch_throttle,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_ballot,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_vote,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_uint_add,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_uint_mul,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_uint_min,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_uint_max,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_uint_and,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_uint_or,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_uint_xor,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_int_add,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_int_mul,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_int_min,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_int_max,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_int_and,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_int_or,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_int_xor,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_float_add,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_float_mul,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_float_min,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subgroup_float_max,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_subroutine_basic,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_specialization,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_spirv,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
#if defined(HAS_DEVTOOLS)
TEST_DESC_I(lwn_shader_tracking_graphics,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_shader_tracking_compute,            LWOG_TEST_PATH_MASK_IMMED_ONLY)
#endif
TEST_DESC_I(lwn_shadowmap_2d,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_snapbiasx,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_snapbiasy,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Skipped for Jira LWN-77 / LWN-78
TEST_DESC_I(lwn_sparse_compressed,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_compressed_compressible_pool,LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_2d_f0,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_2d_f1,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_2da_f0,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_2da_f1,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_3d_f0,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_3d_f1,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_compr_2d_f0,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_compr_2d_f1,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_compr_2da_f0,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_compr_2da_f1,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_compr_3d_f0,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_compr_3d_f1,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_texture_linear,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_sparse_tile_size,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_ssbo_basic,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_ssbo_binding,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_ssbo_compute,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_ssbo_global_basic,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_ssbo_global_compute,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_ssbo_extended_atomics,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_subtile_override,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tess_basic,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tess_even,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tess_odd,                           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tess_tcsparams,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tess_tcsparams_sep,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texbind_random,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_binding,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_lwbe_seamless,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
#if defined(LW_TEGRA)
TEST_DESC_I(lwn_texture_etc,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
#endif
// Skipped for Jira LWN-78
TEST_DESC_I(lwn_texture_get_address,                LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_texture_packager,                   LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_packager_astc,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_packager_astc_srgb,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Skipped for Jira LWN-377
TEST_DESC_I(lwn_texture_pitch,                      LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_texture_pool,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_pool_noncoherent,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_rt_compress,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_storagelayout,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_storagesize,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_swizzle,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_views,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_views_single_layer,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_view_render_targets,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_view_multiple_render_targets, LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_view_multiple_render_targets_disabled, LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_view_format_swizzle,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_2d,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_2d_array,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_2d_array_mipmap,  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_2d_mipmap,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_3d,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_3d_mipmap,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_array,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_array_mipmap,       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_array_mipmap_multi, LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_array_multi,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_mipmap,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_mipmap_multi,       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_multi,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_2d_stride,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_2d_array_stride,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_2d_array_mipmap_stride,  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_2d_mipmap_stride,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_3d_stride,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_3d_mipmap_stride,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_stride,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_array_stride,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_array_mipmap_stride,       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_array_mipmap_multi_stride, LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_array_multi_stride,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_mipmap_stride,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_mipmap_multi_stride,       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_lwbemap_multi_stride,              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_texture_wr_texels_texture_view,     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tiled_cache_entry_points,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tiled_cache_off,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tiled_cache_on,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_tiled_cache_state,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_timestamp,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)

#if defined(LW_TEGRA)
TEST_DESC_I(lwn_timestamp_colwersion,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
#else
// This test depends on reliable timing.
// On NX, we expect performance to be reliable, but on Windows, external factors can cause
// the test to fail intermittently.
TEST_DESC_I(lwn_timestamp_colwersion,               LWOG_TEST_PATH_MASK_IMMED_ONLY | DBGTST)
#endif

#define TIR_TEST(N, M)                                                        \
    TEST_DESC_I(lwn_tir_blend_rs ## N ## x_aa ## M ## x, LWOG_TEST_PATH_MASK_IMMED_ONLY) \
    TEST_DESC_I(lwn_tir_blend_rs ## N ## x_aa ## M ## x_table, LWOG_TEST_PATH_MASK_IMMED_ONLY)

// N > M only
TIR_TEST(2, 1)
TIR_TEST(4, 1)
TIR_TEST(8, 1)
TIR_TEST(4, 2)
TIR_TEST(8, 2)
TIR_TEST(8, 4)

#undef TIR_TEST

TEST_DESC_I(lwn_trilinopt,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_ubo_binding,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_ubo_zerosize,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_unbound_id,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_unfinalized_queues,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_uniformbuffer,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_util_datatypes,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_vertex_attrib,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_vertex_streams,                     LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_vertexbuffer_binding,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
#if defined(LW_HOS)
TEST_DESC_I(lwn_vic_interop,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
#endif
TEST_DESC_I(lwn_view_offset,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_viewports,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
// Skipped for Jira LWN-144
TEST_DESC_I(lwn_window,                             LWOG_TEST_PATH_MASK_IMMED_ONLY | SKIP_LINUX)
TEST_DESC_I(lwn_xfb_basic,                          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_xfb_binding,                        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zbc,                                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zbc1,                               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zbc2,                               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zbc3,                               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zbc4,                               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll,                              LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_subreg,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d16,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d24,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d24s8,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d32f,                 LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d32fs8,               LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d16_subreg,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d24_subreg,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d24s8_subreg,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d32f_subreg,          LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_layered_d32fs8_subreg,        LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_msaa_4,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_msaa_4_subreg,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_msaa_8,                       LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_msaa_8_subreg,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_rt_change,                    LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_rt_change_subreg,             LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_stencil,                      LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_zalias_change,                LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_zalias_change_subreg,         LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_zdup_change,                  LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_zdup_change_subreg,           LWOG_TEST_PATH_MASK_IMMED_ONLY)
TEST_DESC_I(lwn_zlwll_zf32,                         LWOG_TEST_PATH_MASK_IMMED_ONLY)

#endif // !defined(TEST_GROUP_ENUMERATE_GROUPS)
END_TEST_GROUP(TESTS_LWN)
