/*
 * Copyright 2014-2021  LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to LWPU ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and conditions
 * of a form of LWPU software license agreement.
 *
 * LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#pragma once
#include <stdint.h>
#include "lwperf_RealtimeMetrics.h"


namespace lw { namespace metrics { namespace gm20b {


    namespace RawMetricIdx
    {
        enum Enum
        {
            gpmsd__input_quads_3d                                           , //      0
            prop__earlyz_killed_samples                                     , //      1
            tgb__output_verts_complex_boundary                              , //      2
            lts__mccif_read_request_latency_320                             , //      3
            gpmsd__input_quads                                              , //      4
            lts__t_requests_zrd_ni_condrd                                   , //      5
            crop__read_subpackets                                           , //      6
            vpc__clip_cycles_active                                         , //      7
            sm__cycles_active_3d_vtg                                        , //      8
            mpc__isbe_allocations_beta                                      , //      9
            crop__input_samples_part1                                       , //     10
            lts__t_sectors_hit_niso_wr                                      , //     11
            l1tex__m_write_sectors_surface_red                              , //     12
            lts__t_sectors_miss_l1tex_atomic                                , //     13
            lts__t_sectors_ilwal                                            , //     14
            lts__t_sectors_miss_crd_i_condrd                                , //     15
            lts__mccif_write_sectors_excluding_16                           , //     16
            sm__cycles_active_vs                                            , //     17
            lts__t_sectors_miss_gpc_concat_cas                              , //     18
            gcc__tsl2_requests                                              , //     19
            smsp__inst_exelwted_pipe_su_pred_off_all                        , //     20
            lts__t_sectors_hit_raster_wr                                    , //     21
            lts__t_sectors_miss_hub_membar                                  , //     22
            swdx__cycles_active                                             , //     23
            lts__t_sectors_hit_ltc                                          , //     24
            vpc__input_prims_triangle                                       , //     25
            lts__t_sectors_hit_atomic                                       , //     26
            sm__mios_shmem_accesses_pe_read_isbe                            , //     27
            gpmsd__input_samples_2d                                         , //     28
            gpmsd__input_active_3d                                          , //     29
            l1tex__t_set_accesses_hit_tex_format_3d                         , //     30
            lts__t_requests_ltc_cas                                         , //     31
            sm__ctas_launched                                               , //     32
            mpc__isbe_allocations_alpha                                     , //     33
            l1tex__m_read_sectors_tex_format_lwbemap                        , //     34
            smsp__inst_exelwted_pipe_lsu                                    , //     35
            lts__t_requests_wr                                              , //     36
            lts__t_requests_crop_prefetch                                   , //     37
            swdx__tc_replayer_bin_flushes_reason_pagepool_full              , //     38
            mmu__hubtlb_requests_hit_under_miss                             , //     39
            l1tex__samp_samp2mipb_active                                    , //     40
            lts__t_requests_iso_wr                                          , //     41
            smsp__warps_cant_issue_no_instructions_0                        , //     42
            sm__ps_quads_launched                                           , //     43
            pda__input_prims_patch                                          , //     44
            lts__t_requests_host_cpu_membar                                 , //     45
            lts__t_requests_host_noncpu_rd                                  , //     46
            sm__ps_warps_killed                                             , //     47
            vaf__beta_acache_requests_acache_hit                            , //     48
            lts__t_sectors_miss_zrop_condrd                                 , //     49
            lts__t_requests_gcc_rd                                          , //     50
            lts__t_sectors_hit_zrop_prefetch                                , //     51
            lts__t_sectors_zrd_ni                                           , //     52
            cwd__output_ctas                                                , //     53
            gcc__l15_requests_hit                                           , //     54
            lts__t_sectors_miss_membar                                      , //     55
            crop__input_requests_all_color_channels                         , //     56
            l1tex__t_set_conflicts_surface_ld                               , //     57
            swdx__tc_replayer_bin_flushes_reason_drain_timeout              , //     58
            lts__t_requests_iso                                             , //     59
            smsp__inst_exelwted_tex_wb_pending                              , //     60
            smsp__warps_cant_issue_allocation_stall_0                       , //     61
            vaf__beta_cycles_active                                         , //     62
            zrop__read_requests                                             , //     63
            mme__input_method_dwords                                        , //     64
            vpc__beta_tasks_active                                          , //     65
            pda__input_prims_triflat                                        , //     66
            lts__t_sectors_ltc_clear                                        , //     67
            mmu__hubtlb_stalled_on_tag_allocation                           , //     68
            zrop__read_subpackets_stencil                                   , //     69
            lts__t_sectors_ltc_ilwal                                        , //     70
            prop__earlyz_killed_quads                                       , //     71
            l1tex__t_set_accesses_hit_lg_local_ld                           , //     72
            sys__fb_read_dwords                                             , //     73
            lts__mccif_read_request_latency_576                             , //     74
            gpu__dispatch_count                                             , //     75
            lts__t_sectors_miss_zrop_ilwal                                  , //     76
            sm__mios_shmem_accesses_lsu_write                               , //     77
            gpmsd__input_active_compute                                     , //     78
            lts__t_sectors_hit_host_cpu_membar                              , //     79
            sm__threads_launched_gs                                         , //     80
            mpc__warp_launch_stalled_gs_fast_beta                           , //     81
            lts__t_sectors_miss_pe                                          , //     82
            raster__zlwll_lwlled_occluders_near_far_clipped                 , //     83
            lts__t_sectors_hit_ltc_atomic                                   , //     84
            lts__t_sectors_hit_host_noncpu_clear                            , //     85
            gcc__l15_requests_hit_constant                                  , //     86
            lts__mccif_read_request_latency_256                             , //     87
            sm__miop_pq_read_active_smp0                                    , //     88
            gpmsd__input_samples                                            , //     89
            lts__t_requests_hub_condrd                                      , //     90
            lts__t_sectors_hit_hub_rd                                       , //     91
            lts__t_sectors_niso_rd                                          , //     92
            lts__t_sectors_crd_ni_condrd                                    , //     93
            prop__latez_rstrz_input_quads                                   , //     94
            sm__cycles_active_cs                                            , //     95
            smsp__inst_exelwted_pipe_fma64plus_pred_on                      , //     96
            lts__t_sectors_miss_ltc_atomic                                  , //     97
            gpc__rg_utlb_stalled_on_tag_allocation                          , //     98
            sked__dispatch_active_scg0                                      , //     99
            vpc__input_isbes                                                , //    100
            pdb__output_stalled                                             , //    101
            lts__t_requests_host_noncpu_ilwal                               , //    102
            l1tex__t_set_conflicts_surface_red                              , //    103
            l1tex__t_set_conflicts_lg_global_atom                           , //    104
            vaf__beta_cycles_elapsed                                        , //    105
            smsp__inst_exelwted_pipe_tex                                    , //    106
            lts__t_sectors_zrop_prefetch                                    , //    107
            lts__t_requests_crd_i                                           , //    108
            smsp__inst_exelwted_vs                                          , //    109
            raster__setup_lwlled_prims                                      , //    110
            tga__input_patches                                              , //    111
            smsp__warps_eligible                                            , //    112
            mpc__input_batches                                              , //    113
            lts__t_sectors_miss_zrop_rd                                     , //    114
            swdx__tc_replayer_bin_flushes_reason_explicit                   , //    115
            raster__crstr_lwlled_prims_no_tile_coverage                     , //    116
            l1tex__samp_samp2mipb_busy                                      , //    117
            l1tex__samp_samp2mipb_backpressured                             , //    118
            l1tex__t_set_accesses_lg_local_cctl                             , //    119
            l1tex__w_w2d_busy                                               , //    120
            sm__subtiles_active                                             , //    121
            vaf__beta_tasks_active                                          , //    122
            raster__zlwll_lwlled_occluders_depth_bounds                     , //    123
            raster__frstr_output_subtiles_4_cycle                           , //    124
            fe__i2m_write_stalled_request_fifo_full                         , //    125
            lts__t_requests_gpc_condrd                                      , //    126
            sm__warps_launched_vsb                                          , //    127
            lts__t_sectors_gpc_atomic                                       , //    128
            l1tex__m_read_request_stalled                                   , //    129
            l1tex__samp_input_quads_tex_format_3d                           , //    130
            lts__t_sectors_hit_zrop_condrd                                  , //    131
            lts__t_sectors_zrd_i_condrd                                     , //    132
            lts__t_sectors_miss_crop_ilwal                                  , //    133
            sm__warps_launched_cs                                           , //    134
            gpmsd__input_samples_3d                                         , //    135
            smsp__thread_inst_exelwted                                      , //    136
            lts__d_atomic_resseq_cycles_bank_conflict                       , //    137
            smsp__inst_exelwted_pipe_bru_pred_on                            , //    138
            vpc__clip_input_prims_clipped_multi_plane                       , //    139
            prop__cdp_alpha_blendopt_killed_pixels                          , //    140
            sm__cycles_active_ps                                            , //    141
            lts__t_sectors_raster_wr                                        , //    142
            scc__load_constants_miss                                        , //    143
            l1tex__t_set_accesses_surface_atom                              , //    144
            l1tex__samp_input_quads_filter_trilinear                        , //    145
            lts__t_requests_zrd_i_prefetch                                  , //    146
            lts__t_requests_host_noncpu_atomic                              , //    147
            lts__t_sectors_hit_valid                                        , //    148
            lts__t_sectors_miss_crd_ni                                      , //    149
            cbmgr__beta_cbe_allocations                                     , //    150
            lts__t_requests_host_noncpu_condrd                              , //    151
            vpc__lwll_lwlled_prims_reason_backfacing                        , //    152
            lts__t_sectors_miss_mmu_rd                                      , //    153
            l1tex__texin_requests_surface_atom_cas                          , //    154
            gpc__tpc0_utlb_requests_miss                                    , //    155
            vpc__lwll_lwlled_prims_line                                     , //    156
            smsp__inst_exelwted_pipe_tex_pred_off_all                       , //    157
            gr__cycles_idle                                                 , //    158
            gpc__prop_utlb_requests_hit_under_miss                          , //    159
            l1tex__samp_input_quads_tex_format_2d_nomipmap                  , //    160
            lts__t_sectors_hit_zrop_rd                                      , //    161
            smsp__inst_exelwted_vsa                                         , //    162
            lts__t_sectors_hit_gpc_clear                                    , //    163
            l1tex__texin_requests_null_lg                                   , //    164
            lts__t_sectors_fill                                             , //    165
            lts__t_sectors_cas                                              , //    166
            lts__t_requests_l1tex                                           , //    167
            lts__t_sectors_host_cpu_atomic                                  , //    168
            smsp__inst_exelwted_pipe_fma64plus_pred_off_all                 , //    169
            mmu__hubtlb_stalled_request_fifo_full                           , //    170
            lts__t_requests_crop_wr                                         , //    171
            sm__warps_launched_ps                                           , //    172
            raster__tc_output_subtiles                                      , //    173
            zrop__input_stalled                                             , //    174
            smsp__warps_cant_issue_misc_0                                   , //    175
            lts__mccif_read_cycles_64                                       , //    176
            prop__earlyz_output_samples                                     , //    177
            sm__warps_active_vsa                                            , //    178
            l1tex__samp_samp2mipb_stalled                                   , //    179
            prop__cycles_active                                             , //    180
            vpc__clip_input_prims_clipped                                   , //    181
            lts__t_sectors_rd                                               , //    182
            l1tex__t_atomic_address_conflicts_surface_red                   , //    183
            lts__t_sectors_hit_iso_rd                                       , //    184
            smsp__tex_requests                                              , //    185
            lts__t_sectors_niso                                             , //    186
            crop__processed_subpackets                                      , //    187
            lts__t_sectors_hit_host_cpu_atomic                              , //    188
            prop__latez_output_samples                                      , //    189
            vpc__input_prims_point                                          , //    190
            gpu__time_active                                                , //    191
            lts__t_sectors_host_cpu                                         , //    192
            l1tex__t_set_conflicts_lg_local_ld                              , //    193
            pdb__input_batches                                              , //    194
            lts__t_sectors_hit_zrd_ni_prefetch                              , //    195
            lts__t_sectors_ltc_membar                                       , //    196
            lts__t_sectors_hit_hub                                          , //    197
            swdx__tc_replayer_bin_flushes_reason_level_1_threshold          , //    198
            lts__t_sectors_hub_wr                                           , //    199
            lts__mccif_write_request_latency_64                             , //    200
            pel__in_active                                                  , //    201
            gpmpd__cycles_active                                            , //    202
            prop__csb_output_stalled                                        , //    203
            mpc__tram_allocation_stalled                                    , //    204
            l1tex__m_read_sectors_tex_format_1d2d_tex                       , //    205
            lts__t_sectors_miss_zrop_prefetch                               , //    206
            l1tex__m_read_request_active                                    , //    207
            l1tex__t_set_conflicts_surface_atom                             , //    208
            smsp__inst_exelwted_pipe_bar_pred_on                            , //    209
            mmu__hubtlb_requests_miss                                       , //    210
            vaf__alpha_fetched_attr_scalar_post_cbf                         , //    211
            lts__mccif_read_request_latency_128                             , //    212
            smsp__lsu_write_back_active                                     , //    213
            lts__t_sectors_raster                                           , //    214
            lts__t_sectors_miss_condrd                                      , //    215
            vpc__lwll_lwlled_prims_triangle                                 , //    216
            lts__t_sectors_hit_prefetch                                     , //    217
            pda__output_verts                                               , //    218
            gpc__tpc0_utlb_requests_sparse                                  , //    219
            smsp__inst_exelwted_pipe_bru                                    , //    220
            smsp__inst_exelwted_pipe_lsu_pred_on                            , //    221
            lts__t_sectors_miss_l1tex                                       , //    222
            lts__t_sectors_host_noncpu_ilwal                                , //    223
            prop__earlyz_killed_pixels                                      , //    224
            lts__t_sectors_condrd                                           , //    225
            l1tex__texin2m_fifo_output_busy                                 , //    226
            gpc__rg_utlb_stalled_request_fifo_full                          , //    227
            prop__gnic_port1_stalled                                        , //    228
            pda__input_prims_line                                           , //    229
            lts__t_requests_rd                                              , //    230
            lts__t_sectors_miss_crop_prefetch                               , //    231
            l1tex__x_x2w_stalled                                            , //    232
            zrop__processed_requests_type_shdz_biquad                       , //    233
            vaf__alpha_fetched_attr_scalar_indexed                          , //    234
            lts__t_sectors_hit_host_cpu_clear                               , //    235
            lts__t_sectors_hit_niso                                         , //    236
            lts__t_sectors_miss_raster                                      , //    237
            pel__out_l2_requests_read                                       , //    238
            sm__icc_requests_miss_covered                                   , //    239
            lts__t_sectors_hit_ltc_membar                                   , //    240
            stri__to_acache_stalled_on_read                                 , //    241
            gpmsd__input_pixels                                             , //    242
            sm__icc_requests_miss_no_tags                                   , //    243
            prop__input_stalled                                             , //    244
            sm__miop_pq_read_active_tex_smp0                                , //    245
            smsp__miop_pq_write_active                                      , //    246
            l1tex__x_x2w_backpressured                                      , //    247
            l1tex__samp_samp2x_backpressured                                , //    248
            l1tex__x_x2t_busy                                               , //    249
            lts__t_sectors_miss_gpc_membar                                  , //    250
            gpc__tpc0_utlb_requests_hit_under_miss                          , //    251
            vpc__output_verts                                               , //    252
            lts__t_requests_host_noncpu                                     , //    253
            lts__t_sectors_hit_host_noncpu_prefetch                         , //    254
            lts__t_sectors_miss_crop_wr                                     , //    255
            lts__t_sectors_miss_hub_cas                                     , //    256
            swdx__input_messages                                            , //    257
            prop__pixel_shader_barriers                                     , //    258
            raster__frstr_output_subtiles_2_cycle                           , //    259
            sm__warps_launched_gs_fast_alpha                                , //    260
            lts__t_requests_gpc_rd                                          , //    261
            lts__mccif_write_request_latency_48                             , //    262
            l1tex__t_set_conflicts_lg_global_cctl                           , //    263
            sm__cycles_active_3d_ps                                         , //    264
            lts__t_tags_writeback_tier2_issued                              , //    265
            l1tex__w_w2d_active                                             , //    266
            prop__latez_shdz_input_samples                                  , //    267
            mmu__pte_requests_miss                                          , //    268
            prop__earlyz_output_pixels                                      , //    269
            smsp__inst_exelwted_pipe_ldc                                    , //    270
            raster__setup_output_prims                                      , //    271
            vaf__alpha_acache_requests_local_hit                            , //    272
            sys__fb_write_requests                                          , //    273
            lts__cbc_requests_miss_clear_zbc                                , //    274
            sm__warps_retiring_ps_stalled_not_selected                      , //    275
            gcc__l15_requests_miss_instruction                              , //    276
            raster__setup_output_prims_line                                 , //    277
            smsp__inst_exelwted_pipe_xu_pred_on                             , //    278
            lts__t_sectors_host_cpu_membar                                  , //    279
            sm__miop_pq_read_active_pixout_smp1                             , //    280
            raster__crstr_search_stalling_input                             , //    281
            lts__t_sectors_hit_gpc_wr                                       , //    282
            swdx__output_active                                             , //    283
            mpc__input_verts_alpha                                          , //    284
            lts__t_sectors_hit_host_cpu_cas                                 , //    285
            zrop__write_subpackets_stencil                                  , //    286
            lts__t_sectors_hit_zrd_i_rd                                     , //    287
            lts__t_sectors_ltc_cas                                          , //    288
            lts__t_sectors_concat_cas                                       , //    289
            lts__t_sectors_ltc_prefetch                                     , //    290
            pdb__output_stalled_beta                                        , //    291
            sm__mios_shmem_accesses_pe_write_isbe                           , //    292
            lts__t_sectors_zrop_condrd                                      , //    293
            crop__input_stalled                                             , //    294
            l1tex__t_set_accesses                                           , //    295
            lts__t_requests_condrd                                          , //    296
            pda__input_stalled_index_fetch                                  , //    297
            fe__i2m_write_requests                                          , //    298
            smsp__warps_cant_issue_not_selected                             , //    299
            lts__t_sectors_crd_i_prefetch                                   , //    300
            lts__t_sectors_miss_host_noncpu_prefetch                        , //    301
            raster__zlwll_occluders_zfail                                   , //    302
            prop__csb_input_samples_3d                                      , //    303
            l1tex__t_set_conflicts_tex_bilinear                             , //    304
            sys__pushbuffer_dwords                                          , //    305
            sm__miop_pq_read_active_lsu_smp0                                , //    306
            lts__t_requests_crd_ni_prefetch                                 , //    307
            lts__d_decomp_input_stalled                                     , //    308
            prop__csb_input_pixels_3d                                       , //    309
            lts__cycles_active                                              , //    310
            vaf__alpha_acache_requests_acache_hit                           , //    311
            lts__t_sectors_ltc_atomic                                       , //    312
            wwdx__input_tasks                                               , //    313
            smsp__inst_exelwted_pipe_adu                                    , //    314
            pdb__cycles_active                                              , //    315
            tgb__output_verts_complex_interior                              , //    316
            lts__t_requests_ltc_rd                                          , //    317
            lts__t_sectors_miss_crd_ni_prefetch                             , //    318
            lts__ltcx_read_requests                                         , //    319
            pes__stream_output_prims                                        , //    320
            vpc__lwll_lwlled_prims_reason_bounding_box                      , //    321
            l1tex__t_set_accesses_lg_global_atom                            , //    322
            smsp__inst_exelwted_tes                                         , //    323
            lts__d_atomic_reqseq_stalled_source_not_ready                   , //    324
            swdx__input_pixel_shader_barriers                               , //    325
            lts__t_sectors_miss_gpc_prefetch                                , //    326
            lts__t_sectors_miss_pe_ilwal                                    , //    327
            lts__t_sectors_pe_rd                                            , //    328
            lts__t_sectors_miss_hub_ilwal                                   , //    329
            l1tex__t_set_accesses_miss_tag_miss_tex_format_no_mipmap        , //    330
            lts__t_sectors_hit_gpc                                          , //    331
            scc__load_constants_stalled_max_versions                        , //    332
            sm__idc_requests_miss_to_gcc                                    , //    333
            lts__t_requests_zrop                                            , //    334
            vaf__alpha_acache_requests                                      , //    335
            gcc__l15_requests_instruction                                   , //    336
            lts__r_input_fifo_stalled                                       , //    337
            prop__csb_output_crop_requests                                  , //    338
            sked__dispatch_count                                            , //    339
            zrop__input_requests_type_bundles                               , //    340
            lts__d_atomic_reqseq_cycles_bank_conflict                       , //    341
            prop__gnic_port0_stalled                                        , //    342
            l1tex__samp_input_quads_sz_96b_nearest                          , //    343
            l1tex__t_set_conflicts_cctlt                                    , //    344
            lts__t_sectors_miss_host_cpu_clear                              , //    345
            scc__load_constants_page_first_update                           , //    346
            lts__t_sectors_cbc                                              , //    347
            lts__t_sectors_host_noncpu_atomic                               , //    348
            mpc__isbe_allocation_stalled_alpha                              , //    349
            prop__cdp_alpha_to_coverage_output_pixels                       , //    350
            lts__t_sectors_zrop                                             , //    351
            l1tex__m_stalled_on_tag_allocation                              , //    352
            l1tex__t_set_accesses_miss_surface_ld                           , //    353
            lts__t_requests_pe_ilwal                                        , //    354
            l1tex__w_w2d_stalled                                            , //    355
            lts__t_requests_l1tex_rd                                        , //    356
            vaf__alpha_input_batches_post_cbf                               , //    357
            lts__t_sectors_miss_rd                                          , //    358
            lts__t_requests_prefetch                                        , //    359
            lts__t_sectors_miss_host_cpu_condrd                             , //    360
            lts__t_sectors_hit_zrd_i_prefetch                               , //    361
            vaf__alpha_to_acache_stalled_on_tag_allocation                  , //    362
            l1tex__m_write_sectors_surface_atom_cas                         , //    363
            lts__t_sectors_host_noncpu_membar                               , //    364
            lts__t_sectors_hit_concat_cas                                   , //    365
            raster__frstr_killed_subtiles_input                             , //    366
            sm__miop_pq_read_active_pixout_smp0                             , //    367
            sm__cycles_active_tes                                           , //    368
            lts__t_sectors_miss_zrd_ni_condrd                               , //    369
            lts__t_tags_writeback_tier1_killed                              , //    370
            lts__t_sectors_raster_rd                                        , //    371
            vpc__output_prims_stippled_line                                 , //    372
            swdx__tc_replayer_bin_flushes_replay_tiled                      , //    373
            gcc__tsl2_requests_hit                                          , //    374
            smsp__warps_cant_issue_math_pipe_throttle                       , //    375
            pdb__output_pkts                                                , //    376
            lts__cbc_requests_hit_clear_zbc_crop                            , //    377
            zrop__input_requests_type_plane_eq                              , //    378
            vaf__alpha_cycles_elapsed                                       , //    379
            smsp__imc_requests_miss_covered                                 , //    380
            smsp__inst_exelwted_pipe_fma64plus                              , //    381
            lts__t_sectors_gpc_condrd                                       , //    382
            lts__t_requests_host_noncpu_clear                               , //    383
            zrop__input_requests                                            , //    384
            mpc__tram_fill_fifo_stalled                                     , //    385
            lts__t_sectors_hit_crd_ni                                       , //    386
            l1tex__t_set_accesses_hit_lg_global_ld                          , //    387
            vaf__alpha_fetched_attr_scalar_indexed_constant                 , //    388
            l1tex__texin_requests_lg_local_cctl                             , //    389
            rdm__cycles_active                                              , //    390
            lts__t_requests_clear                                           , //    391
            prop__input_active                                              , //    392
            sm__warps_draining_ps                                           , //    393
            zrop__cycles_active                                             , //    394
            tga__output_prims                                               , //    395
            rdm__crop_output_stalled                                        , //    396
            zrop__input_requests_type_shdz_biquad                           , //    397
            lts__t_sectors_miss_ltc_wr                                      , //    398
            lts__t_sectors_pe                                               , //    399
            lts__t_sectors_hit_zrd_ni_condrd                                , //    400
            l1tex__t_atomic_address_conflicts_surface_atom                  , //    401
            lts__t_requests_ltc_condrd                                      , //    402
            lts__t_sectors_hit_host_cpu_wr                                  , //    403
            gcc__l15_requests_constant                                      , //    404
            lts__t_sectors_hit_host_noncpu_wr                               , //    405
            lts__t_sectors_hit_pe_wr                                        , //    406
            pda__input_prims_lineadj                                        , //    407
            lts__t_sectors_gpc_ilwal                                        , //    408
            lts__t_requests_raster                                          , //    409
            smsp__warps_cant_issue_tile_allocation_stall                    , //    410
            lts__t_sectors_hub_prefetch                                     , //    411
            vpc__input_isbes_prim                                           , //    412
            swdx__input_active                                              , //    413
            lts__t_requests_host_noncpu_cas                                 , //    414
            tga__output_tasks_complex_boundary                              , //    415
            l1tex__texin2m_fifo_output_active                               , //    416
            l1tex__m_read_sectors_tex_format_3d                             , //    417
            gpmsd__input_stalled                                            , //    418
            l1tex__t_set_accesses_hit_tex_format_no_mipmap                  , //    419
            prop__input_pixels_3d                                           , //    420
            lts__t_sectors_ltc_condrd                                       , //    421
            lts__t_requests_host_noncpu_prefetch                            , //    422
            lts__t_requests_ltc_prefetch                                    , //    423
            gpc__tpc1_utlb_stalled_on_tag_allocation                        , //    424
            lts__t_sectors_host_noncpu_wr                                   , //    425
            lts__t_tags_writeback_tier1_issued                              , //    426
            mpc__warp_launch_stalled_vsb                                    , //    427
            lts__t_sectors_mmu_rd                                           , //    428
            crop__input_requests_clear                                      , //    429
            gpc__prop_utlb_stalled_request_fifo_full                        , //    430
            lts__t_sectors_hit_zrop_ilwal                                   , //    431
            lts__t_sectors_hit_host_noncpu                                  , //    432
            mmu__pte_requests_small_page_1                                  , //    433
            l1tex__texin_sm2tex_stalled                                     , //    434
            crop__input_stalled_upstream_fifo_full                          , //    435
            l1tex__t_set_accesses_miss_tag_hit_tex_format_1d_buffer         , //    436
            l1tex__m_read_sector_lwmulative_miss_latency                    , //    437
            l1tex__texin_stalled_on_tsl2_miss                               , //    438
            gpc__cycles_elapsed                                             , //    439
            sm__warps_active                                                , //    440
            lts__t_sectors_crd_i_condrd                                     , //    441
            lts__t_sectors_niso_wr                                          , //    442
            prop__input_quads                                               , //    443
            fe__i2m_write_stalled_data_buffer_full                          , //    444
            pda__input_prims_tri                                            , //    445
            crop__write_requests_compressed_4to1                            , //    446
            l1tex__samp_input_quads_sz_32b_nearest                          , //    447
            smsp__inst_exelwted                                             , //    448
            lts__t_sectors_hit_gpc_membar                                   , //    449
            lts__t_sectors_ltc_wr                                           , //    450
            lts__t_sectors_miss_gpc_clear                                   , //    451
            prop__csb_output_active                                         , //    452
            lts__t_sectors_miss_gpc                                         , //    453
            sm__cycles_active_vsa                                           , //    454
            stri__cycles_stalled                                            , //    455
            lts__t_sectors_zrop_rd                                          , //    456
            lts__t_sectors_zrop_ilwal                                       , //    457
            lts__t_sectors_miss_host_cpu                                    , //    458
            lts__t_sectors_miss_mmu_wr                                      , //    459
            l1tex__t_set_accesses_miss_tag_miss_tex_format_lwbemap          , //    460
            crop__input_requests_2d                                         , //    461
            l1tex__m_write_sectors_lg_global_atom                           , //    462
            lts__t_sectors_gpc_wr                                           , //    463
            wwdx__output_stalled                                            , //    464
            gcc__tsl2_requests_miss                                         , //    465
            lts__t_requests_zrop_rd                                         , //    466
            cwd__output_stalled_no_free_slots                               , //    467
            lts__t_requests_raster_rd                                       , //    468
            sm__warps_launched_tes                                          , //    469
            sm__warps_launched_gs                                           , //    470
            lts__t_sectors_hit_l1tex                                        , //    471
            l1tex__t_sectors_promoted                                       , //    472
            lts__t_sectors_mmu                                              , //    473
            vaf__cycles_active                                              , //    474
            prop__cdp_alpha_blendopt_read_avoid                             , //    475
            l1tex__texin_requests_lg_global_st                              , //    476
            mmu__pte_requests_big_page_3                                    , //    477
            l1tex__t_set_accesses_surface_st                                , //    478
            l1tex__m_write_sectors_surface_st                               , //    479
            swdx__tc_replayer_bin_flushes_reason_constant_table_full        , //    480
            lts__t_sectors_miss_host_cpu_cas                                , //    481
            lts__t_sectors_miss_cbc_wr                                      , //    482
            lts__t_requests_mmu                                             , //    483
            smsp__warps_active                                              , //    484
            sm__threads_launched_vsb                                        , //    485
            tgb__output_prims_triangle                                      , //    486
            l1tex__texin2m_fifo_output_backpressured                        , //    487
            sm__mios_shmem_accesses_pe_write_tram                           , //    488
            lts__t_sectors_hit_l1tex_wr                                     , //    489
            crop__processed_requests_compressed                             , //    490
            smsp__inst_exelwted_ps                                          , //    491
            lts__t_requests_gpc_membar                                      , //    492
            l1tex__lod_output_wavefronts                                    , //    493
            vpc__cycles_active                                              , //    494
            lts__t_requests_niso_wr                                         , //    495
            cbmgr__beta_cbe_allocation_stalled_max_tasks                    , //    496
            stri__to_acache_stalled_on_tag_allocation                       , //    497
            l1tex__t_set_accesses_lg_global_atom_cas                        , //    498
            lts__t_sectors_miss_host_cpu_membar                             , //    499
            lts__t_sectors_hit_iso_wr                                       , //    500
            l1tex__t_set_accesses_tex_format_1d2d_tex                       , //    501
            sm__inst_exelwted_pipe_adu_divergent_smp0                       , //    502
            zrop__processed_subpackets_stencil                              , //    503
            lts__t_sectors_hit_membar                                       , //    504
            lts__t_sectors_zrd_ni_prefetch                                  , //    505
            gpmpd__output_packets                                           , //    506
            vaf__alpha_to_acache_stalled_on_read                            , //    507
            zrop__processed_requests                                        , //    508
            tpc__cycles_elapsed                                             , //    509
            l1tex__texin_requests_lg_global_cctl                            , //    510
            l1tex__t_set_accesses_miss_tag_hit_tex_format_no_mipmap         , //    511
            lts__t_requests_host_cpu_ilwal                                  , //    512
            zrop__read_returns                                              , //    513
            lts__t_requests_host_cpu_prefetch                               , //    514
            fe__output_ops                                                  , //    515
            sm__warps_active_tcs                                            , //    516
            pel__out_active                                                 , //    517
            l1tex__t_set_accesses_tex_format_3d                             , //    518
            lts__mccif_write_request_latency_high                           , //    519
            vpc__output_prims_line                                          , //    520
            gpc__tpc1_utlb_stalled_write_buffer_full                        , //    521
            lts__cbc_requests_comptag_miss                                  , //    522
            lts__t_sectors_zrd_i_rd                                         , //    523
            lts__t_sectors_miss_zrd_i_rd                                    , //    524
            lts__t_requests_zrop_wr                                         , //    525
            lts__mccif_read_request_latency_512                             , //    526
            l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_tex         , //    527
            stri__acache_requests_stri_local_hit                            , //    528
            cwd__cycles_active                                              , //    529
            lts__t_sectors_hit_gpc_ilwal                                    , //    530
            pel__out_l2_requests_write_512b                                 , //    531
            lts__t_requests_concat_cas                                      , //    532
            sm__idc_requests_miss_no_tags                                   , //    533
            gpc__tpc0_utlb_requests_hit                                     , //    534
            lts__t_requests_host_cpu_clear                                  , //    535
            lts__t_sectors_gpc_concat_cas                                   , //    536
            prop__cdp_alpha_test_killed_pixels                              , //    537
            sm__warps_active_vsb                                            , //    538
            lts__t_requests_crd_ni                                          , //    539
            lts__t_requests_ltc_concat_cas                                  , //    540
            sm__miop_adu_replays_smp1                                       , //    541
            pda__input_verts                                                , //    542
            lts__t_sectors_pe_wr                                            , //    543
            mmu__pte_requests_small_page_0                                  , //    544
            lts__t_requests_gpc_ilwal                                       , //    545
            lts__t_sectors_hub_ilwal                                        , //    546
            l1tex__samp_input_quads_tex_format_1d_buffer                    , //    547
            lts__t_sectors_hit_mmu_rd                                       , //    548
            mpc__warp_launch_stalled_gs_fast_alpha                          , //    549
            gpc__rg_utlb_requests_hit                                       , //    550
            mpc__output_batches                                             , //    551
            lts__t_sectors_ltc_concat_cas                                   , //    552
            lts__ltcx_write_stalled_fifo_full                               , //    553
            lts__t_sectors_hit_pe_rd                                        , //    554
            lts__t_requests_raster_wr                                       , //    555
            gpu__time_duration                                              , //    556
            raster__crstr_output_tiles                                      , //    557
            smsp__imc_requests_miss_to_gcc                                  , //    558
            l1tex__t_set_conflicts_lg_global_atom_cas                       , //    559
            lts__t_sectors_miss_iso_rd                                      , //    560
            crop__write_requests                                            , //    561
            lts__t_sectors_hit_crd_ni_prefetch                              , //    562
            crop__input_requests_aamode_8xaa                                , //    563
            lts__d_atomic_block_cycles_serialized                           , //    564
            lts__t_sectors_pe_ilwal                                         , //    565
            prop__gnic_port1_active                                         , //    566
            swdx__tc_replayer_bin_flushes_replay_not_tiled                  , //    567
            lts__t_sectors_hit_ltc_concat_cas                               , //    568
            lts__t_sectors_hit_hub_cas                                      , //    569
            wwdx__cycles_active                                             , //    570
            lts__t_sectors_hit_gcc                                          , //    571
            l1tex__samp_input_quads_tex_format_lwbemap                      , //    572
            sm__ps_quads_killed                                             , //    573
            l1tex__t_t2d_busy                                               , //    574
            lts__t_requests_hub_rd                                          , //    575
            l1tex__d_sectors_fill                                           , //    576
            l1tex__x_x2w_active                                             , //    577
            lts__t_sectors_hit_gcc_rd                                       , //    578
            smsp__ps_threads_killed                                         , //    579
            lts__t_sectors_hit_ltc_prefetch                                 , //    580
            vpc__output_prims_point                                         , //    581
            gcc__l15_requests_hit_instruction                               , //    582
            lts__t_sectors_hit_rd                                           , //    583
            lts__t_sectors_hub_atomic                                       , //    584
            l1tex__d_output_wavefronts                                      , //    585
            lts__t_sectors_hit_gpc_rd                                       , //    586
            gpmpd__output_stalled_batch                                     , //    587
            lts__t_sectors_miss_zrop                                        , //    588
            l1tex__f_output_tex2sm_stalled                                  , //    589
            prop__gnic_port0_active                                         , //    590
            sm__miop_pq_read_active_smp1                                    , //    591
            lts__t_sectors_miss_cbc_rd                                      , //    592
            mmu__hubtlb_requests_hit                                        , //    593
            lts__t_sectors_hit_host_cpu_prefetch                            , //    594
            lts__t_sectors_host_cpu_clear                                   , //    595
            l1tex__t_atomic_address_conflicts_lg_global_atom_cas            , //    596
            lts__t_requests_gpc_prefetch                                    , //    597
            lts__t_sectors_hub                                              , //    598
            l1tex__m_read_sectors                                           , //    599
            lts__t_sectors_hit_crd_ni_rd                                    , //    600
            pes__cycles_active                                              , //    601
            lts__t_sectors_miss_ilwal                                       , //    602
            cbmgr__alpha_cbe_allocations                                    , //    603
            gpc__rg_utlb_requests_miss                                      , //    604
            lts__t_sectors_hit_ltc_clear                                    , //    605
            pel__out_read_stalled_vaf_alpha                                 , //    606
            mpc__isbe_allocations                                           , //    607
            rdm__input_data0_active                                         , //    608
            crop__input_requests_blend_enabled                              , //    609
            l1tex__texin2m_fifo_output_stalled                              , //    610
            smsp__inst_exelwted_lsu_wb_pending                              , //    611
            lts__t_requests_host_cpu                                        , //    612
            raster__setup_output_prims_triangle                             , //    613
            lts__t_sectors_miss_iso                                         , //    614
            l1tex__x_x2t_backpressured                                      , //    615
            prop__latez_shdz_input_quads                                    , //    616
            pel__in_write_requests_stalled                                  , //    617
            smsp__inst_exelwted_lsu_wb                                      , //    618
            lts__t_requests_ltc_clear                                       , //    619
            l1tex__m_read_sectors_surface_ld_d                              , //    620
            sys__fb_read_requests                                           , //    621
            lts__t_tag_requests_hit                                         , //    622
            lts__t_requests_niso_rd                                         , //    623
            gpmsd__cycles_active                                            , //    624
            sm__warps_active_gs                                             , //    625
            l1tex__d_d2f_stalled                                            , //    626
            smsp__inst_exelwted_pipe_su                                     , //    627
            lts__t_sectors_miss_host_noncpu_membar                          , //    628
            gpmsd__input_pixels_2d                                          , //    629
            lts__t_sectors_hit_crd_ni_condrd                                , //    630
            lts__t_requests_hub_prefetch                                    , //    631
            lts__t_sectors_miss_ltc_clear                                   , //    632
            sm__cycles_active                                               , //    633
            l1tex__w_output_wavefronts                                      , //    634
            sm__threads_launched_tcs                                        , //    635
            l1tex__texin_requests_surface_red                               , //    636
            lts__t_sectors_crd_ni_rd                                        , //    637
            l1tex__samp_samp2x_stalled                                      , //    638
            lts__t_sectors_mmu_wr                                           , //    639
            swdx__tc_binner_binned_op_tiled_cache_barriers                  , //    640
            smsp__warps_cant_issue_short_scoreboard_1                       , //    641
            lts__t_sectors_host_cpu_concat_cas                              , //    642
            swdx__output_stalled                                            , //    643
            lts__t_sectors_ltc_rd                                           , //    644
            l1tex__t_sectors_miss                                           , //    645
            l1tex__samp_input_quads_tex_format_2d_mipmap                    , //    646
            gpmsd__input_quads_2d                                           , //    647
            raster__crstr_discover_stalling_setup                           , //    648
            lts__t_requests_crop_ilwal                                      , //    649
            mpc__warp_launch_stalled_tcs                                    , //    650
            lts__t_sectors_hub_condrd                                       , //    651
            vaf__beta_input_tasks                                           , //    652
            prop__cdp_alpha_blendopt_pixels_fill_override                   , //    653
            l1tex__t_set_accesses_tex_format_no_mipmap                      , //    654
            l1tex__t_set_accesses_tex_format_lwbemap                        , //    655
            raster__frstr_output_subtiles_3_cycle                           , //    656
            raster__crstr_input_prims                                       , //    657
            scc__load_constants_hit                                         , //    658
            lts__t_sectors_miss_gpc_ilwal                                   , //    659
            l1tex__m_read_sectors_surface_ld_p                              , //    660
            pel__in_write_requests_stalled_vaf_alpha                        , //    661
            lts__t_sectors_hit_ltc_condrd                                   , //    662
            lts__t_requests_pe_rd                                           , //    663
            lts__d_atomic_reqseq_stalled_pending_store_same_address         , //    664
            lts__t_sectors_miss_hub                                         , //    665
            l1tex__m_read_sectors_tex_format_1d2d_array                     , //    666
            l1tex__texin_requests_surface_st                                , //    667
            l1tex__texin_sm2tex_busy                                        , //    668
            l1tex__m_write_sectors_surface_atom                             , //    669
            tgb__output_prims_point                                         , //    670
            lts__t_sectors_hit_hub_wr                                       , //    671
            lts__cbc_requests_miss_clear_zbc_zrop                           , //    672
            lts__t_sectors_hit_zrd_i                                        , //    673
            cwd__output_stalled_state_ack                                   , //    674
            l1tex__t_set_accesses_miss_lg_global_ld                         , //    675
            lts__t_sectors_miss_hub_atomic                                  , //    676
            gpc__gpcl1tlb_stalled_request_fifo_full                         , //    677
            lts__t_requests_cas                                             , //    678
            lts__t_sectors_miss_crd_i                                       , //    679
            l1tex__texin_tsl1_requests_hit_texhdr                           , //    680
            mpc__warp_launch_stalled_rf_free                                , //    681
            sm__cycles_active_3d                                            , //    682
            sm__icc_requests_miss_to_gcc                                    , //    683
            lts__t_requests_gpc_concat_cas                                  , //    684
            gpmpd__input_tasks                                              , //    685
            lts__t_sectors_miss_host_cpu_wr                                 , //    686
            smsp__warps_cant_issue_tex_throttle                             , //    687
            lts__t_sectors_miss_l1tex_rd                                    , //    688
            smsp__inst_exelwted_pipe_ldc_pred_on                            , //    689
            lts__t_requests_l1tex_wr                                        , //    690
            tgb__cycles_active                                              , //    691
            lts__t_sectors_host_cpu_prefetch                                , //    692
            fe__cycles_wfi_host_scg0                                        , //    693
            lts__d_atomic_block_stalled_pending_miss                        , //    694
            smsp__warps_cant_issue_allocation_stall_1                       , //    695
            vaf__gpm_stalled_by_state_processing                            , //    696
            lts__t_sectors_hit_hub_ilwal                                    , //    697
            gpc__prop_utlb_requests_sparse                                  , //    698
            lts__t_requests_host_cpu_concat_cas                             , //    699
            l1tex__samp_samp2x_active                                       , //    700
            prop__input_stalled_waiting_for_pixel_shader_barrier_release    , //    701
            lts__t_sectors_host_noncpu_concat_cas                           , //    702
            lts__t_sectors_host_noncpu_condrd                               , //    703
            lts__t_sectors_host_noncpu                                      , //    704
            l1tex__t_set_accesses_hit_tex_format_1d2d_array                 , //    705
            l1tex__t_set_conflicts_lg_global_ld                             , //    706
            l1tex__samp_input_quads_sz_32b                                  , //    707
            lts__r_input_fifo_active                                        , //    708
            lts__t_sectors_miss_gcc                                         , //    709
            lts__t_sectors_hit_cas                                          , //    710
            swdx__tc_replayer_bin_flushes_reason_clear                      , //    711
            smsp__imc_requests_miss_no_tags                                 , //    712
            smsp__inst_exelwted_pipe_fp16                                   , //    713
            smsp__inst_exelwted_tex_wb                                      , //    714
            vpc__clip_output_attrs                                          , //    715
            lts__t_sectors_hit_crd_i_prefetch                               , //    716
            lts__t_sectors_hit_hub_concat_cas                               , //    717
            lts__t_sectors_iso                                              , //    718
            lts__t_sectors_hit_ltc_cas                                      , //    719
            vaf__alpha_fetched_attr_vector_post_cbf                         , //    720
            lts__t_sectors_ltc                                              , //    721
            smsp__inst_exelwted_pipe_fmai                                   , //    722
            vaf__alpha_acache_requests_acache_miss                          , //    723
            l1tex__x_x2t_stalled                                            , //    724
            lts__t_sectors_hit_l1tex_rd                                     , //    725
            lts__t_sectors_hit_cbc_rd                                       , //    726
            lts__t_sectors_hit_gpc_condrd                                   , //    727
            raster__frstr_output_cycles                                     , //    728
            smsp__inst_exelwted_pipe_fma64plusplus                          , //    729
            l1tex__t_set_conflicts_surface_st                               , //    730
            l1tex__t_t2d_active                                             , //    731
            lts__t_sectors_hit_cbc_wr                                       , //    732
            l1tex__texin_requests_null_surface                              , //    733
            gpu__time_start                                                 , //    734
            lts__t_sectors_crd_i                                            , //    735
            lts__cbc_requests_miss_clear_zbc_crop                           , //    736
            scc__load_constants_stalled_update_scoreboard_full              , //    737
            lts__t_sectors_hit_zrop                                         , //    738
            sm__miop_pq_read_active_tex_smp1                                , //    739
            lts__t_sectors_hit_host_noncpu_rd                               , //    740
            rdm__input_data1_active                                         , //    741
            prop__zrop_output_quads                                         , //    742
            fe__input_methods                                               , //    743
            lts__t_sectors_hit_host_noncpu_condrd                           , //    744
            raster__crstr_output_tiles_empty                                , //    745
            lts__t_sectors_hit_cbc                                          , //    746
            l1tex__texin_requests_lg_global_red                             , //    747
            cbmgr__cycles_active                                            , //    748
            lts__t_sectors_miss_hub_clear                                   , //    749
            rdm__crop_output_active                                         , //    750
            lts__t_sectors_hit_ltc_rd                                       , //    751
            lts__mccif_read_request_latency_high                            , //    752
            smsp__inst_exelwted_pipe_xu_pred_off_all                        , //    753
            gpc__prop_utlb_stalled_on_tag_allocation                        , //    754
            prop__latez_output_pixels                                       , //    755
            sm__miop_adu_replays_smp0                                       , //    756
            vpc__lwll_lwlled_prims_reason_scissor                           , //    757
            lts__cbc_requests_comptag_fill                                  , //    758
            mpc__warp_launch_stalled_gs                                     , //    759
            crop__read_returns                                              , //    760
            l1tex__t_set_accesses_tex_format_1d2d_array                     , //    761
            lts__t_sectors_host_noncpu_rd                                   , //    762
            crop__read_stalled                                              , //    763
            vpc__output_cbes                                                , //    764
            lts__t_sectors_iso_wr                                           , //    765
            mpc__input_subtiles                                             , //    766
            gcc__cycles_active                                              , //    767
            lts__t_sectors_gpc_rd                                           , //    768
            crop__input_requests_aamode_2xaa                                , //    769
            l1tex__t_set_accesses_miss_tag_hit_tex_format_3d                , //    770
            l1tex__t_set_accesses_lg_local_st                               , //    771
            lts__t_sectors_gcc                                              , //    772
            zrop__write_requests                                            , //    773
            sm__threads_launched_cs                                         , //    774
            lts__t_sectors_hit_zrd_ni                                       , //    775
            lts__t_requests_hub_cas                                         , //    776
            lts__t_sectors_crd_ni_prefetch                                  , //    777
            lts__t_sectors_miss_zrd_ni                                      , //    778
            swdx__tc_replayer_bin_flushes_reason_cbe_slots_threshold        , //    779
            mpc__warp_launch_stalled_ps                                     , //    780
            lts__t_sectors_prefetch                                         , //    781
            smsp__issue_active                                              , //    782
            l1tex__texin_tsl1_requests_hit_samphdr                          , //    783
            lts__t_sectors_hit_host_noncpu_ilwal                            , //    784
            lts__d_atomic_block_stalled_same_address                        , //    785
            lts__t_sectors_miss_hub_wr                                      , //    786
            lts__t_sectors_miss_host_noncpu_rd                              , //    787
            pel__out_input_stalled                                          , //    788
            l1tex__texin_tsl1_requests_miss_texhdr                          , //    789
            l1tex__t_set_accesses_surface_ld                                , //    790
            lts__ltcx_read_sectors                                          , //    791
            zrop__processed_subpackets                                      , //    792
            smsp__inst_exelwted_gs                                          , //    793
            l1tex__samp_input_quads_tex_format_1d_tex                       , //    794
            gpc__tpc1_utlb_requests_hit                                     , //    795
            lts__t_sectors_miss_host_cpu_rd                                 , //    796
            lts__t_sectors_miss_ltc_concat_cas                              , //    797
            lts__t_sectors_hub_clear                                        , //    798
            vpc__output_prims                                               , //    799
            sm__warps_launched_tcs                                          , //    800
            swdx__tc_replayer_bin_flushes_reason_timeout                    , //    801
            l1tex__d_d2f_backpressured                                      , //    802
            mme__cycles_stalled                                             , //    803
            gpc__tpc0_utlb_stalled_request_fifo_full                        , //    804
            lts__t_sectors_miss_zrd_i                                       , //    805
            vpc__read_isbes                                                 , //    806
            lts__t_sectors                                                  , //    807
            lts__t_requests_mmu_wr                                          , //    808
            lts__t_sectors_crop_wr                                          , //    809
            zrop__processed_requests_type_bundle                            , //    810
            l1tex__samp_input_quads_sz_128b_nearest                         , //    811
            mpc__isbe_allocation_stalled_alpha_on_vsc                       , //    812
            lts__t_requests_cbc_wr                                          , //    813
            lts__t_sectors_gpc_prefetch                                     , //    814
            swdx__input_prims_single_binned                                 , //    815
            prop__input_pixels                                              , //    816
            lts__t_sectors_miss_zrop_wr                                     , //    817
            l1tex__m_read_sectors_tex_format_1d_buffer                      , //    818
            l1tex__t_sectors_miss_lg                                        , //    819
            smsp__inst_exelwted_pipe_xu                                     , //    820
            vpc__alpha_cycles_active                                        , //    821
            lts__t_requests_host_cpu_atomic                                 , //    822
            lts__t_sectors_miss_host_cpu_atomic                             , //    823
            pes__stream_output_attrs                                        , //    824
            lts__cbc_requests_hit_clear_zbc_zrop                            , //    825
            prop__latez_shdz_input_pixels                                   , //    826
            sm__subtiles_launched_smp0                                      , //    827
            lts__t_sectors_hit_host_noncpu_membar                           , //    828
            lts__t_sectors_miss_hub_condrd                                  , //    829
            lts__t_sectors_miss_iso_wr                                      , //    830
            swdx__tc_replayer_bin_flushes_reason_cbe_memory_threshold       , //    831
            gpmsd__input_pixels_fully_covered                               , //    832
            lts__t_requests_gpc_atomic                                      , //    833
            lts__t_sectors_miss_clear                                       , //    834
            smsp__warps_cant_issue_imc_miss                                 , //    835
            lts__cbc_requests_comptag_hit                                   , //    836
            mpc__alpha_beta_mode_switches                                   , //    837
            l1tex__t_set_accesses_surface_red                               , //    838
            sys__cycles_elapsed                                             , //    839
            smsp__inst_exelwted_pipe_lsu_pred_off_all                       , //    840
            sm__idc_requests_hit                                            , //    841
            smsp__inst_exelwted_pipe_fma64plusplus_pred_off_all             , //    842
            raster__zlwll_lwlled_occluders                                  , //    843
            lts__mccif_read_request_latency_640                             , //    844
            l1tex__texin_requests_lg_local_st                               , //    845
            lts__t_requests_hub_clear                                       , //    846
            lts__t_requests_membar                                          , //    847
            lts__t_sectors_host_noncpu_clear                                , //    848
            fbp__cycles_elapsed                                             , //    849
            l1tex__texin_requests_null_tex                                  , //    850
            raster__frstr_output_subtiles                                   , //    851
            stri__acache_requests_stri                                      , //    852
            cbmgr__beta_cbe_allocation_stalled                              , //    853
            l1tex__t_set_accesses_miss_tag_miss                             , //    854
            sm__warps_completed_ooo_3d_vtg                                  , //    855
            gpc__gpcl1tlb_requests_miss                                     , //    856
            sys__gpfifo_dwords                                              , //    857
            sm__mios_shmem_accesses_lsu_read                                , //    858
            mpc__warp_launch_stalled_vsa                                    , //    859
            raster__zlwll_output_occluders_trivial_accept                   , //    860
            vaf__gpm_stalled_by_beta_input_fifo                             , //    861
            sked__cycles_active                                             , //    862
            lts__t_sectors_hit_crop_wr                                      , //    863
            lts__t_sectors_hit_host_noncpu_cas                              , //    864
            lts__mccif_read_request_latency_448                             , //    865
            fe__i2m_write_bytes                                             , //    866
            stri__acache_requests_stri_miss                                 , //    867
            cbmgr__beta_cbe_allocation_stalled_no_space                     , //    868
            cbmgr__alpha_cbe_allocation_stalled_no_space                    , //    869
            lts__t_requests_ilwal                                           , //    870
            lts__t_sectors_hit_host_cpu_condrd                              , //    871
            fe__output_ops_vert                                             , //    872
            smsp__inst_issued                                               , //    873
            l1tex__t_texels                                                 , //    874
            sm__mios_shmem_accesses_su_read_tram_bank_conflict              , //    875
            lts__t_sectors_hit_raster_rd                                    , //    876
            lts__t_sectors_host_cpu_ilwal                                   , //    877
            vpc__input_prims_patch                                          , //    878
            mmu__pte_requests_small_page_3                                  , //    879
            swdx__tc_replayer_bin_flushes_reason_non_binnable_line          , //    880
            sm__ps_quads_sent_to_pixout                                     , //    881
            l1tex__t_set_accesses_miss_tag_hit_tex_format_lwbemap           , //    882
            sm__warps_active_tes                                            , //    883
            l1tex__t_set_conflicts_lg_global_red                            , //    884
            cwd__feedback_mpc_messages                                      , //    885
            lts__cbc_requests_hit_clear_zbc                                 , //    886
            lts__t_sectors_hit_crop_prefetch                                , //    887
            lts__t_sectors_miss_gpc_wr                                      , //    888
            lts__t_requests_gcc                                             , //    889
            raster__frstr_output_subtiles_2d                                , //    890
            lts__t_sectors_miss_host_noncpu_ilwal                           , //    891
            l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_array       , //    892
            lts__t_sectors_crop_ilwal                                       , //    893
            zrop__input_samples_part2                                       , //    894
            prop__latez_rstrz_input_samples                                 , //    895
            scc__load_constants                                             , //    896
            vpc__clip_output_prims                                          , //    897
            smsp__warps_cant_issue_dispatch_stall                           , //    898
            lts__t_sectors_cbc_rd                                           , //    899
            lts__t_sectors_miss_zrd_i_condrd                                , //    900
            lts__t_requests_hub_concat_cas                                  , //    901
            prop__zrop_output_active                                        , //    902
            prop__csb_output_quads                                          , //    903
            vaf__gpm_stalled_by_state_fifo                                  , //    904
            scc__cycles_active                                              , //    905
            prop__cdp_alpha_to_coverage_killed_pixels                       , //    906
            lts__t_requests_host_noncpu_wr                                  , //    907
            lts__t_sectors_hit_crd_i                                        , //    908
            l1tex__m_write_sectors_lg_global_atom_cas                       , //    909
            crop__input_requests                                            , //    910
            vpc__output_prims_triangle                                      , //    911
            l1tex__t_set_accesses_lg_global_st                              , //    912
            lts__t_sectors_hit_hub_clear                                    , //    913
            l1tex__texin_tsl1_requests_miss_samphdr                         , //    914
            l1tex__d_d2f_active                                             , //    915
            fe__output_ops_bundle_scg0_go_idle                              , //    916
            prop__csb_output_quads_3d                                       , //    917
            raster__tc_output_tiles                                         , //    918
            vpc__output_attrs_scalar                                        , //    919
            lts__t_sectors_miss_atomic                                      , //    920
            l1tex__samp_input_quads_sz_64b_nearest                          , //    921
            mme__cycles_active                                              , //    922
            scc__load_constants_page_same                                   , //    923
            prop__input_quads_2d                                            , //    924
            l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_tex          , //    925
            lts__t_requests_hub_atomic                                      , //    926
            lts__t_sectors_miss_crd_i_prefetch                              , //    927
            lts__t_sectors_crd_ni                                           , //    928
            mme__output_method_dwords                                       , //    929
            sm__threads_launched_tes                                        , //    930
            smsp__inst_exelwted_pipe_tex_pred_on                            , //    931
            stri__cycles_active                                             , //    932
            lts__t_sectors_hit_mmu_wr                                       , //    933
            l1tex__texin_requests_cctlt                                     , //    934
            crop__read_requests                                             , //    935
            mmu__pte_requests_big_page_2                                    , //    936
            pdb__input_stalled_alpha                                        , //    937
            swdx__tc_replayer_bin_flushes_reason_level_0_threshold          , //    938
            lts__t_requests_crd_i_prefetch                                  , //    939
            sm__miop_pq_read_active_lsu_smp1                                , //    940
            lts__t_sectors_hit_ltc_ilwal                                    , //    941
            l1tex__t_t2d_stalled                                            , //    942
            lts__t_sectors_miss_ltc_membar                                  , //    943
            fe__output_ops_bundle_scg1_go_idle                              , //    944
            lts__t_sectors_miss_concat_cas                                  , //    945
            sm__cycles_active_tcs                                           , //    946
            raster__tc_flushes                                              , //    947
            fe__output_ops_bundle                                           , //    948
            l1tex__m_read_sectors_tex_format_no_mipmap                      , //    949
            lts__t_sectors_miss_cbc                                         , //    950
            lts__d_atomic_reqseq_input_stalled_fifo_full                    , //    951
            lts__d_cycles_bank_conflict                                     , //    952
            crop__input_requests_aamode_4xaa                                , //    953
            gcc__l15_requests_miss                                          , //    954
            mmu__cycles_active                                              , //    955
            lts__t_requests_zrop_condrd                                     , //    956
            l1tex__t_set_conflicts_surface_atom_cas                         , //    957
            pel__out_read_stalled_vaf_beta                                  , //    958
            raster__frstr_output_subtiles_1_cycle                           , //    959
            lts__t_sectors_hit_hub_condrd                                   , //    960
            l1tex__t_set_accesses_miss_lg_local_ld                          , //    961
            mpc__input_prims_alpha_patches                                  , //    962
            crop__write_requests_compressed_2to1                            , //    963
            swdx__input_prims_dual_binned                                   , //    964
            sm__warps_launched_vsa                                          , //    965
            mmu__pte_requests_big_page_0                                    , //    966
            raster__crstr_discover_working_no_tile_stalling_setup           , //    967
            l1tex__samp_samp2x_busy                                         , //    968
            prop__csb_input_quads_3d                                        , //    969
            crop__write_subpackets                                          , //    970
            lts__t_sectors_hit_zrd_i_condrd                                 , //    971
            l1tex__texin_requests_surface_atom                              , //    972
            vaf__beta_to_acache_stalled_on_read                             , //    973
            lts__xbar_output_active                                         , //    974
            cbmgr__alpha_cbe_allocation_stalled_max_tasks                   , //    975
            sm__warps_retiring_ps_stalled_out_of_order                      , //    976
            gpmsd__input_pixels_3d                                          , //    977
            lts__t_sectors_hub_rd                                           , //    978
            mpc__input_prims_beta                                           , //    979
            pel__in_write_requests_stalled_vaf_beta                         , //    980
            mmu__pte_requests_big_page_1                                    , //    981
            vaf__beta_acache_requests_local_hit                             , //    982
            l1tex__texin_requests_surface_ld                                , //    983
            lts__t_sectors_hit_gpc_prefetch                                 , //    984
            l1tex__t_set_accesses_lg_local_ld                               , //    985
            fe__input_method_dwords_i2m                                     , //    986
            tgb__output_prims_line                                          , //    987
            lts__ltcx_read_stalled_fifo_full                                , //    988
            lts__t_sectors_miss_zrd_i_prefetch                              , //    989
            l1tex__t_set_accesses_hit_in_warp_surface_ld                    , //    990
            smsp__warps_launched                                            , //    991
            l1tex__x_x2w_busy                                               , //    992
            lts__t_requests_ltc                                             , //    993
            l1tex__t_set_accesses_tex_format_1d_buffer                      , //    994
            l1tex__w_w2d_backpressured                                      , //    995
            gpu__time_end                                                   , //    996
            lts__t_sectors_zrd_ni_condrd                                    , //    997
            crop__processed_samples_part2                                   , //    998
            lts__t_sectors_miss_gpc_condrd                                  , //    999
            lts__t_requests_hub_ilwal                                       , //   1000
            gpc__tpc1_utlb_requests_hit_under_miss                          , //   1001
            fe__output_ops_ld_const                                         , //   1002
            l1tex__m_read_sectors_lg_global_ld                              , //   1003
            vaf__alpha_cycles_active                                        , //   1004
            raster__setup_input_prims                                       , //   1005
            zrop__processed_requests_type_plane_eq                          , //   1006
            lts__t_sectors_miss_host_cpu_prefetch                           , //   1007
            sm__mios_shmem_accesses_lsu_read_bank_conflict                  , //   1008
            smsp__warps_cant_issue_mio_throttle                             , //   1009
            pdb__output_stalled_alpha                                       , //   1010
            vaf__beta_input_task_fifo_full                                  , //   1011
            gcc__l15_requests_miss_constant                                 , //   1012
            zrop__write_subpackets_depth                                    , //   1013
            lts__t_sectors_miss_pe_rd                                       , //   1014
            fe__cycles_wfi_ctxsw                                            , //   1015
            lts__t_sectors_miss_crop_condrd                                 , //   1016
            sm__threads_launched_ps_not_killed                              , //   1017
            lts__t_sectors_hit_ilwal                                        , //   1018
            fe__output_ops_bundle_scg1_wfi_host                             , //   1019
            crop__input_requests_blend_blendopt_fill_over                   , //   1020
            gpc__gpcl1tlb_stalled_on_tag_allocation                         , //   1021
            mpc__warp_launch_stalled_tes                                    , //   1022
            lts__t_sectors_miss_raster_rd                                   , //   1023
            vpc__lwll_cycles_active                                         , //   1024
            smsp__inst_exelwted_pipe_bru_pred_off_all                       , //   1025
            gpmsd__sm2gpmsd_pixout_cdp0_active_color                        , //   1026
            lts__t_sectors_hub_membar                                       , //   1027
            crop__processed_samples_part1                                   , //   1028
            lts__t_requests_zrd_ni                                          , //   1029
            smsp__inst_exelwted_pipe_adu_pred_on                            , //   1030
            lts__t_sectors_miss_zrd_ni_rd                                   , //   1031
            tgb__output_verts_simple                                        , //   1032
            lts__t_requests_host_cpu_condrd                                 , //   1033
            vpc__lwll_lwlled_prims_reason_zero_area                         , //   1034
            l1tex__samp_input_quads_sz_96b                                  , //   1035
            lts__t_requests_ltc_atomic                                      , //   1036
            l1tex__t_output_wavefronts                                      , //   1037
            lts__t_requests_iso_rd                                          , //   1038
            smsp__warps_cant_issue_long_scoreboard_0                        , //   1039
            lts__t_sectors_miss_wr                                          , //   1040
            lts__t_sectors_crop_prefetch                                    , //   1041
            smsp__warps_cant_issue_short_scoreboard_0                       , //   1042
            lts__t_sectors_miss_cas                                         , //   1043
            sm__warps_launched_gs_fast_beta                                 , //   1044
            l1tex__m_write_sectors_lg_global_st                             , //   1045
            pda__input_prims_point                                          , //   1046
            prop__latez_output_quads                                        , //   1047
            lts__t_requests_hub_wr                                          , //   1048
            smsp__inst_exelwted_pipe_bar_pred_off_all                       , //   1049
            raster__setup_output_prims_point                                , //   1050
            raster__zlwll_input_tiles                                       , //   1051
            mpc__input_prims_alpha                                          , //   1052
            lts__d_atomic_resseq_stalled_backpressure                       , //   1053
            smsp__warps_cant_issue_long_scoreboard_1                        , //   1054
            sm__threads_launched                                            , //   1055
            vaf__alpha_input_prims                                          , //   1056
            smsp__inst_exelwted_pipe_fxu                                    , //   1057
            prop__input_samples                                             , //   1058
            smsp__warps_cant_issue_drain                                    , //   1059
            lts__mccif_write_stalled                                        , //   1060
            vpc__alpha_batches_active                                       , //   1061
            fe__cycles_active                                               , //   1062
            lts__t_sectors_miss_raster_wr                                   , //   1063
            vaf__alpha_input_batches_pre_cbf                                , //   1064
            tga__output_tasks_primary                                       , //   1065
            l1tex__t_set_accesses_hit                                       , //   1066
            raster__zlwll_output_occluders_zfail_no_lwll                    , //   1067
            lts__mccif_write_cycles_16                                      , //   1068
            lts__t_sectors_hit_crd_i_rd                                     , //   1069
            smsp__warps_cant_issue_wait                                     , //   1070
            gpc__rg_utlb_stalled_write_buffer_full                          , //   1071
            lts__t_sectors_crop_condrd                                      , //   1072
            lts__t_sectors_hit_zrop_wr                                      , //   1073
            l1tex__samp_input_quads_filter_aniso                            , //   1074
            lts__t_sectors_gpc_membar                                       , //   1075
            lts__mccif_read_request_latency_384                             , //   1076
            lts__t_sectors_miss                                             , //   1077
            lts__t_sectors_host_cpu_rd                                      , //   1078
            smsp__inst_exelwted_vsb                                         , //   1079
            smsp__warps_cant_issue_misc_1                                   , //   1080
            gpc__prop_utlb_requests_hit                                     , //   1081
            gpc__prop_utlb_requests_miss                                    , //   1082
            smsp__inst_exelwted_cs                                          , //   1083
            vaf__alpha_batches_active                                       , //   1084
            l1tex__t_set_accesses_cctlt                                     , //   1085
            l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_array        , //   1086
            mmu__pde_requests_miss                                          , //   1087
            lts__mccif_read_cycles_32                                       , //   1088
            lts__t_requests_gpc_cas                                         , //   1089
            lts__t_requests_host_noncpu_concat_cas                          , //   1090
            lts__t_sectors_miss_gcc_rd                                      , //   1091
            lts__t_requests_mmu_rd                                          , //   1092
            wwdx__input_cbes                                                , //   1093
            pda__input_prims                                                , //   1094
            stri__acache_requests_stri_hit                                  , //   1095
            lts__t_sectors_miss_host_cpu_ilwal                              , //   1096
            vpc__clip_output_verts                                          , //   1097
            mpc__tram_startxy_fifo_stalled                                  , //   1098
            l1tex__m_write_sectors_lg_global_red                            , //   1099
            lts__t_sectors_miss_prefetch                                    , //   1100
            pda__input_prims_triadj                                         , //   1101
            crop__processed_requests_uncompressed                           , //   1102
            lts__t_sectors_atomic                                           , //   1103
            lts__t_sectors_clear                                            , //   1104
            vpc__lwll_lwlled_prims_reason_frustum                           , //   1105
            pel__in_write_requests                                          , //   1106
            lts__t_requests_zrop_prefetch                                   , //   1107
            lts__t_sectors_wr                                               , //   1108
            crop__input_samples_part2                                       , //   1109
            l1tex__m_read_sectors_surface_atom_cas                          , //   1110
            vpc__lwll_lwlled_prims_reason_rotated_grid                      , //   1111
            l1tex__t_set_accesses_hit_in_warp_lg_local_st                   , //   1112
            lts__t_requests_host_noncpu_membar                              , //   1113
            sm__mios_datapath_active                                        , //   1114
            lts__t_requests_l1tex_atomic                                    , //   1115
            mme__input_methods_shadow_filtered                              , //   1116
            lts__t_requests_zrd_ni_prefetch                                 , //   1117
            lts__t_sectors_zrd_ni_rd                                        , //   1118
            zrop__zlwll_cycles_active                                       , //   1119
            l1tex__d_cycles_bank_conflict_2x2                               , //   1120
            l1tex__t_set_accesses_hit_tex_format_1d_buffer                  , //   1121
            lts__t_sectors_miss_host_noncpu_wr                              , //   1122
            lts__t_sectors_miss_niso_wr                                     , //   1123
            mpc__cycles_elapsed_beta                                        , //   1124
            l1tex__m_write_sectors_lg_local_st                              , //   1125
            vpc__lwll_lwlled_prims_point                                    , //   1126
            vaf__alpha_fetched_attr_vector_pre_cbf                          , //   1127
            gpc__gpcl1tlb_requests_hit_under_miss                           , //   1128
            gr__cycles_active                                               , //   1129
            lts__t_requests_ltc_membar                                      , //   1130
            pda__output_batches                                             , //   1131
            zrop__write_subpackets_coalesced                                , //   1132
            lts__t_sectors_hit_host_cpu_rd                                  , //   1133
            smsp__inst_exelwted_tcs                                         , //   1134
            l1tex__t_set_accesses_hit_in_warp_lg_global_st                  , //   1135
            lts__t_sectors_miss_pe_wr                                       , //   1136
            lts__t_sectors_host_cpu_condrd                                  , //   1137
            sm__warps_active_ps                                             , //   1138
            lts__t_sectors_zrop_wr                                          , //   1139
            fe__output_ops_bundle_scg0_wfi_host                             , //   1140
            stri__cycles_busy                                               , //   1141
            tga__output_tasks_complex_interior                              , //   1142
            sm__inst_exelwted_pipe_adu_divergent_smp1                       , //   1143
            lts__t_sectors_miss_ltc                                         , //   1144
            smsp__tex_write_back_active                                     , //   1145
            rdm__zrop_output_active                                         , //   1146
            lts__cbc_requests_comptag_writeback                             , //   1147
            smsp__cycles_active                                             , //   1148
            tga__input_isbes                                                , //   1149
            lts__t_requests_cbc_rd                                          , //   1150
            lts__t_requests                                                 , //   1151
            lts__t_sectors_zrd_i_prefetch                                   , //   1152
            vpc__clip_input_prims_clipped_single_plane                      , //   1153
            vaf__beta_cycles_stalled_on_mpc                                 , //   1154
            swdx__tc_replayer_bin_flushes_reason_non_binnable_state         , //   1155
            lts__t_sectors_miss_host_noncpu_condrd                          , //   1156
            lts__t_sectors_hit_host_noncpu_atomic                           , //   1157
            l1tex__m_read_sectors_lg_global_atom                            , //   1158
            lts__t_sectors_miss_zrd_ni_prefetch                             , //   1159
            lts__t_tag_requests_miss                                        , //   1160
            lts__t_sectors_hit_crop                                         , //   1161
            lts__t_requests_ltc_ilwal                                       , //   1162
            lts__t_sectors_gpc_cas                                          , //   1163
            lts__t_sectors_miss_host_noncpu_atomic                          , //   1164
            lts__t_sectors_miss_host_noncpu                                 , //   1165
            l1tex__t_set_conflicts_tex_trilinear                            , //   1166
            sm__cycles_active_gs                                            , //   1167
            crop__cycles_active                                             , //   1168
            l1tex__t_set_accesses_miss_tag_miss_tex_format_3d               , //   1169
            prop__input_samples_3d                                          , //   1170
            vpc__clip_cycles_stalled                                        , //   1171
            lts__t_sectors_hit_l1tex_atomic                                 , //   1172
            scc__load_constants_stalled_no_gcc_credits                      , //   1173
            lts__t_requests_crop                                            , //   1174
            lts__t_requests_hub                                             , //   1175
            gpc__tpc0_utlb_stalled_write_buffer_full                        , //   1176
            raster__frstr_killed_subtiles_output                            , //   1177
            lts__t_requests_zrd_i_rd                                        , //   1178
            lts__t_sectors_miss_host_noncpu_cas                             , //   1179
            smsp__warps_cant_issue_selected                                 , //   1180
            lts__t_sectors_hit_hub_atomic                                   , //   1181
            lts__t_sectors_hit_crd_i_condrd                                 , //   1182
            lts__t_requests_hub_membar                                      , //   1183
            gpmpd__output_stalled_task                                      , //   1184
            mpc__isbe_allocation_stalled_beta_on_vsc                        , //   1185
            lts__t_sectors_miss_host_noncpu_clear                           , //   1186
            fe__cycles_wfi_subch_switch_scg0                                , //   1187
            lts__t_sectors_miss_ltc_cas                                     , //   1188
            l1tex__samp_pre_nop_wavefronts                                  , //   1189
            prop__csb_output_samples_3d                                     , //   1190
            l1tex__m_read_sectors_surface_atom                              , //   1191
            l1tex__t_atomic_address_conflicts_surface_atom_cas              , //   1192
            l1tex__texin_requests_lg_local_ld                               , //   1193
            pdb__input_tasks                                                , //   1194
            sm__subtiles_launched_smp1                                      , //   1195
            l1tex__d_d2f_busy                                               , //   1196
            sm__icc_prefetches                                              , //   1197
            lts__d_atomic_block_stalled_priority                            , //   1198
            l1tex__texin_sm2tex_active                                      , //   1199
            swdx__tc_replayer_bin_flushes                                   , //   1200
            l1tex__t_set_conflicts_lg_global_st                             , //   1201
            lts__t_sectors_miss_hub_rd                                      , //   1202
            raster__frstr_output_subtiles_fully_covered                     , //   1203
            sm__warps_active_cs                                             , //   1204
            vaf__beta_input_patches                                         , //   1205
            lts__t_sectors_host_noncpu_prefetch                             , //   1206
            raster__zlwll_lwlled_occluders_stencil                          , //   1207
            smsp__inst_exelwted_pipe_fe                                     , //   1208
            l1tex__texin_requests_lg_global_atom                            , //   1209
            lts__t_sectors_gpc                                              , //   1210
            l1tex__t_set_conflicts_lg_local_st                              , //   1211
            swdx__tc_replayer_bin_flushes_reason_z_mode_transition          , //   1212
            smsp__thread_inst_exelwted_pred_on                              , //   1213
            gpmsd__sm2gpmsd_pixout_cdp1_active_shdz                         , //   1214
            lts__t_sectors_miss_crd_i_rd                                    , //   1215
            gpc__gpcl1tlb_requests_hit                                      , //   1216
            raster__tc_input_transactions                                   , //   1217
            lts__t_sectors_miss_hub_concat_cas                              , //   1218
            lts__t_sectors_miss_ltc_prefetch                                , //   1219
            lts__t_requests_pe_wr                                           , //   1220
            vaf__beta_to_acache_stalled_on_tag_allocation                   , //   1221
            host__chsw_switches                                             , //   1222
            pdb__input_stalled_beta                                         , //   1223
            gpc__tpc0_utlb_stalled_on_tag_allocation                        , //   1224
            prop__csb_output_crop_requests_color                            , //   1225
            lts__t_sectors_miss_crd_ni_rd                                   , //   1226
            smsp__warps_cant_issue_no_instructions_1                        , //   1227
            lts__t_sectors_l1tex                                            , //   1228
            lts__t_requests_crop_rd                                         , //   1229
            lts__t_sectors_hit                                              , //   1230
            mpc__cycles_active_beta                                         , //   1231
            pel__out_l2_requests                                            , //   1232
            sm__miop_ldc_replays_smp1                                       , //   1233
            vaf__gpm_stalled_by_alpha_input_fifo                            , //   1234
            crop__input_requests_blend_blendopt_read_avoid                  , //   1235
            lts__t_requests_pe                                              , //   1236
            smsp__inst_exelwted_pipe_adu_pred_off_all                       , //   1237
            vaf__alpha_input_verts                                          , //   1238
            lts__t_requests_zrd_ni_rd                                       , //   1239
            tga__isbes_active                                               , //   1240
            vpc__lwll_lwlled_prims_reason_diamond_exit_rule                 , //   1241
            raster__zlwll_input_occluders                                   , //   1242
            l1tex__x_x2t_active                                             , //   1243
            lts__t_sectors_miss_host_noncpu_concat_cas                      , //   1244
            lts__t_requests_crd_ni_rd                                       , //   1245
            l1tex__texin_requests_tex                                       , //   1246
            smsp__inst_exelwted_pipe_ldc_pred_off_all                       , //   1247
            zrop__write_subpackets                                          , //   1248
            lts__t_requests_host_cpu_wr                                     , //   1249
            smsp__inst_exelwted_pipe_su_pred_on                             , //   1250
            prop__input_pixels_2d                                           , //   1251
            lts__mccif_write_request_latency_24                             , //   1252
            lts__t_sectors_cbc_wr                                           , //   1253
            l1tex__texin_requests_membar                                    , //   1254
            lts__d_decomp_processed_sectors                                 , //   1255
            lts__t_sectors_miss_crop                                        , //   1256
            lts__t_sectors_miss_gpc_cas                                     , //   1257
            crop__write_requests_compressed_8to1_or_fastclear               , //   1258
            prop__input_samples_2d                                          , //   1259
            l1tex__texin_sm2tex_backpressured                               , //   1260
            gpmpd__input_batches                                            , //   1261
            gpc__rg_utlb_requests_hit_under_miss                            , //   1262
            lts__t_sectors_miss_niso                                        , //   1263
            gpc__tpc1_utlb_requests_miss                                    , //   1264
            lts__t_requests_zrd_i                                           , //   1265
            mmu__pde_requests_hit                                           , //   1266
            vpc__input_prims                                                , //   1267
            pel__in_write_requests_stalled_tg                               , //   1268
            mpc__cycles_active_alpha                                        , //   1269
            mmu__pte_requests_hit                                           , //   1270
            sm__ctas_active                                                 , //   1271
            lts__d_atomic_resseq_stalled_output_arbitration                 , //   1272
            lts__t_sectors_hit_pe                                           , //   1273
            lts__t_sectors_miss_host_cpu_concat_cas                         , //   1274
            gpmsd__output_quads                                             , //   1275
            l1tex__texin2m_fifo_input_stalled                               , //   1276
            l1tex__x_output_wavefronts                                      , //   1277
            sm__idc_requests_miss_covered                                   , //   1278
            lts__t_sectors_miss_l1tex_wr                                    , //   1279
            lts__t_sectors_hit_host_noncpu_concat_cas                       , //   1280
            lts__t_sectors_hub_cas                                          , //   1281
            lts__t_sectors_miss_crd_ni_condrd                               , //   1282
            fe__input_methods_stalled                                       , //   1283
            pda__cycles_active                                              , //   1284
            sm__mios_shmem_accesses_lsu_write_bank_conflict                 , //   1285
            lts__mccif_read_requests                                        , //   1286
            smsp__imc_requests_hit                                          , //   1287
            gpc__tpc1_utlb_stalled_request_fifo_full                        , //   1288
            l1tex__t_set_accesses_hit_tex_format_1d2d_tex                   , //   1289
            l1tex__m_read_sectors_lg_local_ld                               , //   1290
            lts__t_sectors_hit_host_cpu_ilwal                               , //   1291
            swdx__output_barriers                                           , //   1292
            lts__t_sectors_l1tex_wr                                         , //   1293
            vaf__beta_acache_requests_acache_miss                           , //   1294
            lts__t_sectors_miss_gpc_rd                                      , //   1295
            l1tex__samp_input_quads_filter_bilinear                         , //   1296
            vpc__lwll_cycles_stalled                                        , //   1297
            lts__t_requests_ltc_wr                                          , //   1298
            swdx__tc_replayer_bin_flushes_reason_state_full                 , //   1299
            raster__crstr_discover_working_no_tile_output                   , //   1300
            smsp__inst_exelwted_pipe_fma64plusplus_pred_on                  , //   1301
            cbmgr__alpha_cbe_allocation_stalled                             , //   1302
            gpmsd__sm2gpmsd_pixout_cdp1_active_color                        , //   1303
            lts__t_sectors_hit_gpc_cas                                      , //   1304
            pel__cycles_active                                              , //   1305
            lts__t_sectors_crd_i_rd                                         , //   1306
            lts__t_requests_zrd_i_condrd                                    , //   1307
            wwdx__input_prims                                               , //   1308
            lts__t_sectors_miss_ltc_ilwal                                   , //   1309
            prop__csb_output_pixels_3d                                      , //   1310
            lts__t_sectors_hit_ltc_wr                                       , //   1311
            l1tex__samp_output_wavefronts                                   , //   1312
            lts__t_sectors_miss_niso_rd                                     , //   1313
            l1tex__t_sectors                                                , //   1314
            lts__t_sectors_l1tex_rd                                         , //   1315
            pel__out_l2_requests_write_256b                                 , //   1316
            zrop__read_subpackets                                           , //   1317
            lts__t_sectors_hit_crop_ilwal                                   , //   1318
            lts__t_requests_crop_condrd                                     , //   1319
            mpc__warp_launch_stalled_vtg                                    , //   1320
            tga__batches_active                                             , //   1321
            zrop__input_requests_containing_stencil                         , //   1322
            raster__zlwll_lwlled_occluders_depth                            , //   1323
            gpu__draw_count                                                 , //   1324
            lts__t_sectors_hit_wr                                           , //   1325
            mpc__input_verts_beta                                           , //   1326
            l1tex__t_set_accesses_hit_in_warp_surface_st                    , //   1327
            prop__latez_rstrz_input_pixels                                  , //   1328
            smsp__warps_cant_issue_barrier                                  , //   1329
            scc__load_constants_page_pool_allocations                       , //   1330
            l1tex__t_set_accesses_surface_atom_cas                          , //   1331
            l1tex__d_cycles_bank_conflict_bilerp                            , //   1332
            l1tex__t_atomic_address_conflicts_lg_global_atom                , //   1333
            vaf__alpha_fetched_attr_scalar_pre_cbf                          , //   1334
            lts__t_sectors_hit_gpc_concat_cas                               , //   1335
            sys__fb_write_dwords                                            , //   1336
            vpc__lwll_lwlled_prims_reason_zero_length                       , //   1337
            vpc__write_sectors                                              , //   1338
            crop__input_requests_aamode_1xaa                                , //   1339
            mmu__pte_requests_small_page_2                                  , //   1340
            lts__t_sectors_hit_condrd                                       , //   1341
            scc__input_state_changes                                        , //   1342
            crop__processed_requests                                        , //   1343
            lts__t_sectors_miss_ltc_condrd                                  , //   1344
            lts__t_sectors_miss_crop_rd                                     , //   1345
            prop__zrop_output_stalled                                       , //   1346
            sm__warps_retiring_ps_stalled_backpressure                      , //   1347
            crop__input_requests_3d                                         , //   1348
            sm__icc_requests_hit                                            , //   1349
            l1tex__t_set_conflicts_lg_local_cctl                            , //   1350
            l1tex__t_set_accesses_miss_tag_hit                              , //   1351
            lts__t_sectors_crop_rd                                          , //   1352
            vpc__lwll_lwlled_prims                                          , //   1353
            vpc__input_prims_line                                           , //   1354
            tga__output_tasks                                               , //   1355
            lts__t_requests_crd_ni_condrd                                   , //   1356
            sm__miop_ldc_replays_smp0                                       , //   1357
            smsp__warps_cant_issue_membar                                   , //   1358
            pes__stream_output_verts                                        , //   1359
            gpmsd__sm2gpmsd_pixout_cdp0_active_shdz                         , //   1360
            lts__t_requests_host_cpu_rd                                     , //   1361
            l1tex__m_read_sectors_lg_global_atom_cas                        , //   1362
            lts__t_sectors_host_cpu_wr                                      , //   1363
            mpc__input_tasks                                                , //   1364
            swdx__output_prims                                              , //   1365
            prop__input_quads_3d                                            , //   1366
            mpc__cycles_elapsed_alpha                                       , //   1367
            lts__t_requests_crd_i_rd                                        , //   1368
            l1tex__texin_requests_lg_global_atom_cas                        , //   1369
            lts__t_sectors_iso_rd                                           , //   1370
            lts__t_sectors_host_noncpu_cas                                  , //   1371
            swdx__binner_active                                             , //   1372
            sm__warps_active_vtg                                            , //   1373
            zrop__input_samples_part1                                       , //   1374
            swdx__tc_replayer_bin_flushes_reason_bin_memory_threshold       , //   1375
            lts__t_sectors_miss_gpc_atomic                                  , //   1376
            raster__frstr_processed_3_edges                                 , //   1377
            l1tex__t_set_accesses_lg_global_cctl                            , //   1378
            pel__out_read_stalled_stri                                      , //   1379
            l1tex__samp_input_quads_sz_128b                                 , //   1380
            lts__t_requests_gpc_wr                                          , //   1381
            mme__call_count                                                 , //   1382
            lts__t_sectors_hit_iso                                          , //   1383
            tga__input_batches                                              , //   1384
            crop__write_stalled                                             , //   1385
            lts__t_sectors_miss_mmu                                         , //   1386
            smsp__inst_exelwted_pipe_bar                                    , //   1387
            lts__t_requests_host_cpu_cas                                    , //   1388
            lts__t_sectors_hit_raster                                       , //   1389
            l1tex__samp_input_quads_sz_64b                                  , //   1390
            lts__t_sectors_gcc_rd                                           , //   1391
            lts__t_sectors_crop                                             , //   1392
            lts__t_requests_gpc                                             , //   1393
            lts__t_sectors_hit_host_cpu_concat_cas                          , //   1394
            zrop__input_requests_expanded_to_samples                        , //   1395
            lts__t_sectors_hit_zrd_ni_rd                                    , //   1396
            sm__mios_shmem_accesses_su_read_tram                            , //   1397
            gpc__rg_utlb_requests_sparse                                    , //   1398
            vaf__alpha_cycles_stalled_on_mpc                                , //   1399
            lts__t_sectors_zrd_i                                            , //   1400
            l1tex__texin_requests_lg_global_ld                              , //   1401
            mpc__isbe_allocation_stalled                                    , //   1402
            prop__zrop_output_samples                                       , //   1403
            prop__earlyz_output_quads                                       , //   1404
            lts__t_sectors_hit_crop_rd                                      , //   1405
            gpc__tpc1_utlb_requests_sparse                                  , //   1406
            lts__t_requests_niso                                            , //   1407
            crop__write_requests_compressed_1to1                            , //   1408
            l1tex__f_output_tex2sm_backpressured                            , //   1409
            lts__t_sectors_hit_niso_rd                                      , //   1410
            crop__input_requests_pitch_linear                               , //   1411
            lts__t_sectors_hit_gpc_atomic                                   , //   1412
            lts__t_sectors_hit_host_cpu                                     , //   1413
            l1tex__f_output_tex2sm_busy                                     , //   1414
            lts__t_sectors_miss_hub_prefetch                                , //   1415
            lts__t_sectors_hit_clear                                        , //   1416
            lts__t_requests_atomic                                          , //   1417
            lts__mccif_write_requests                                       , //   1418
            pel__out_l2_requests_ilwalidate_256b                            , //   1419
            lts__mccif_read_stalled                                         , //   1420
            lts__t_requests_crd_i_condrd                                    , //   1421
            lts__t_requests_zrop_ilwal                                      , //   1422
            swdx__input_stalled                                             , //   1423
            l1tex__t_set_accesses_hit_in_warp_lg_local_ld                   , //   1424
            lts__t_sectors_miss_ltc_rd                                      , //   1425
            lts__t_sectors_gpc_clear                                        , //   1426
            lts__d_atomic_block_stalled_backpressure                        , //   1427
            l1tex__t_set_accesses_hit_in_warp_lg_global_ld                  , //   1428
            lts__t_requests_cbc                                             , //   1429
            lts__t_sectors_host_cpu_cas                                     , //   1430
            pel__in_write_requests_active                                   , //   1431
            zrop__input_requests_write_compression_ok                       , //   1432
            lts__t_sectors_hit_mmu                                          , //   1433
            l1tex__t_set_accesses_lg_global_red                             , //   1434
            mpc__isbe_allocation_stalled_beta                               , //   1435
            l1tex__f_output_tex2sm_active                                   , //   1436
            pda__input_restart_indices                                      , //   1437
            lts__t_requests_gpc_clear                                       , //   1438
            lts__t_sectors_hub_concat_cas                                   , //   1439
            lts__d_decomp_input_sectors                                     , //   1440
            sm__threads_launched_vsa                                        , //   1441
            l1tex__t_t2d_backpressured                                      , //   1442
            l1tex__t_set_accesses_lg_global_ld                              , //   1443
            lts__t_sectors_hit_crop_condrd                                  , //   1444
            zrop__processed_requests_type_plane_eq_fast                     , //   1445
            tgb__tasks_active                                               , //   1446
            lts__t_sectors_hit_hub_prefetch                                 , //   1447
            sm__cycles_active_vsb                                           , //   1448
            prop__csb_output_pixels                                         , //   1449
            sm__warps_completed_ooo_3d_ps                                   , //   1450
            prop__csb_killed_quadquads                                      , //   1451
            lts__t_sectors_membar                                           , //   1452
            vpc__beta_cycles_active                                         , //   1453
            lts__t_sectors_l1tex_atomic                                     , //   1454
            vaf__cycles_stalled                                             , //   1455
            lts__t_sectors_hit_pe_ilwal                                     , //   1456
            lts__t_sectors_hit_hub_membar                                   , //   1457
            crop__input_requests_blend_blendopt_killed                      , //   1458
            stri__attrs                                                     , //   1459
            l1tex__t_set_accesses_hit_tex_format_lwbemap                    , //   1460
            tga__cycles_active                                              , //   1461
            l1tex__t_atomic_address_conflicts_lg_global_red                 , //   1462
            COUNT // = 1463
        };
    }

    struct RawMetricsStorage : public RawMetricsContext
    {
        double values[RawMetricIdx::COUNT];
        uint16_t counts[RawMetricIdx::COUNT];
        double devAttrValues[1];
        enum { NumDeviceAttributes = 0 };

        MetricValue GetValue(RawMetricIdx::Enum rawMetricIdx, double sustainedRate, double cycles_elapsed)
        {
            MetricValue metricValue; // single exit-point to encourage RVO
            if (configuring)
            {
                counts[rawMetricIdx] = 1;
                // fill dummy data to avoid compiler or linter warnings about uninitialized data
                metricValue = {};
            }
            else
            {
                const double unitCount = counts[rawMetricIdx];
                metricValue.sum = values[rawMetricIdx];
                metricValue.avg = values[rawMetricIdx] / unitCount;
                metricValue.peak_sustained = sustainedRate;
                metricValue.cycles_elapsed = cycles_elapsed;
            }
            return metricValue;
        }

        // helpers for templating
        static const size_t NumRawMetricIds = RawMetricIdx::COUNT;
        static const uint64_t* GetRawMetricIds()
        {
            static const uint64_t s_rawMetricIds[] = {
                0x005655beacc28eb6, // gpmsd__input_quads_3d
                0x0081e75a4672299c, // prop__earlyz_killed_samples
                0x00d667ee178a3f79, // tgb__output_verts_complex_boundary
                0x00e356a6041e4dd8, // lts__mccif_read_request_latency_320
                0x0103198ad47d9408, // gpmsd__input_quads
                0x010eb23db39704c4, // lts__t_requests_zrd_ni_condrd
                0x01148deaeaec8923, // crop__read_subpackets
                0x012c887f4367ac62, // vpc__clip_cycles_active
                0x015107064e16da28, // sm__cycles_active_3d_vtg
                0x015bd62c4e335801, // mpc__isbe_allocations_beta
                0x01d8564d70f66975, // crop__input_samples_part1
                0x01dbc2377c4f18f0, // lts__t_sectors_hit_niso_wr
                0x025a8259cd601a0b, // l1tex__m_write_sectors_surface_red
                0x0268b61166de8461, // lts__t_sectors_miss_l1tex_atomic
                0x02b0791797c45907, // lts__t_sectors_ilwal
                0x02d0c00e51dc0d15, // lts__t_sectors_miss_crd_i_condrd
                0x0300ff820bddded9, // lts__mccif_write_sectors_excluding_16
                0x033274c4aa2f4013, // sm__cycles_active_vs
                0x03d0c329d1b5f9e9, // lts__t_sectors_miss_gpc_concat_cas
                0x0444fd364f99ad45, // gcc__tsl2_requests
                0x04450160aa004cb8, // smsp__inst_exelwted_pipe_su_pred_off_all
                0x049a3104a6ce00c1, // lts__t_sectors_hit_raster_wr
                0x04d839d6f4074af5, // lts__t_sectors_miss_hub_membar
                0x04e0ca769bde521d, // swdx__cycles_active
                0x051b08900cb1cb17, // lts__t_sectors_hit_ltc
                0x0530a4846837c249, // vpc__input_prims_triangle
                0x055a3c719599e902, // lts__t_sectors_hit_atomic
                0x057ac0f9c9f4b532, // sm__mios_shmem_accesses_pe_read_isbe
                0x05f7ae016ca73a1e, // gpmsd__input_samples_2d
                0x06041a0f6c25b291, // gpmsd__input_active_3d
                0x060a194120aec4a2, // l1tex__t_set_accesses_hit_tex_format_3d
                0x0679d67c9a616e39, // lts__t_requests_ltc_cas
                0x068e23859205f429, // sm__ctas_launched
                0x0694d5a13327aa6e, // mpc__isbe_allocations_alpha
                0x06f618d0694a4bde, // l1tex__m_read_sectors_tex_format_lwbemap
                0x071aaf7217f2c0b6, // smsp__inst_exelwted_pipe_lsu
                0x07425156c039cd52, // lts__t_requests_wr
                0x0746b5c83924bff7, // lts__t_requests_crop_prefetch
                0x074eb08cfdf772dc, // swdx__tc_replayer_bin_flushes_reason_pagepool_full
                0x0752f24d502b94cd, // mmu__hubtlb_requests_hit_under_miss
                0x0762032bca7fef4a, // l1tex__samp_samp2mipb_active
                0x076b2d5187063b14, // lts__t_requests_iso_wr
                0x07cae68117042559, // smsp__warps_cant_issue_no_instructions_0
                0x0817a2fca93d7346, // sm__ps_quads_launched
                0x08368c3d357f6dc4, // pda__input_prims_patch
                0x0843e884eb416e01, // lts__t_requests_host_cpu_membar
                0x0845037cc88295f9, // lts__t_requests_host_noncpu_rd
                0x08776a3112e3ca7c, // sm__ps_warps_killed
                0x0893a5bc39ce0cc8, // vaf__beta_acache_requests_acache_hit
                0x0895e6397c65abb1, // lts__t_sectors_miss_zrop_condrd
                0x08bde7b644b79bf5, // lts__t_requests_gcc_rd
                0x08df2b071f653a29, // lts__t_sectors_hit_zrop_prefetch
                0x08e1a731568a7dd2, // lts__t_sectors_zrd_ni
                0x08e99fee07619830, // cwd__output_ctas
                0x09006af5b6d9e466, // gcc__l15_requests_hit
                0x097b7a8414578f30, // lts__t_sectors_miss_membar
                0x09a4089e17582144, // crop__input_requests_all_color_channels
                0x09e3ef550e0ea184, // l1tex__t_set_conflicts_surface_ld
                0x09ff11307f622b32, // swdx__tc_replayer_bin_flushes_reason_drain_timeout
                0x0a8bc75d5f749c88, // lts__t_requests_iso
                0x0a9682a84b386651, // smsp__inst_exelwted_tex_wb_pending
                0x0afa99b554a10acc, // smsp__warps_cant_issue_allocation_stall_0
                0x0b0b2869764dfe45, // vaf__beta_cycles_active
                0x0b47a6a5884a95eb, // zrop__read_requests
                0x0b6d28f3ffd107e8, // mme__input_method_dwords
                0x0ba0ba4d7cda3c0e, // vpc__beta_tasks_active
                0x0bab73e7dcf41525, // pda__input_prims_triflat
                0x0be29d4b26bbb9db, // lts__t_sectors_ltc_clear
                0x0bfdcd0d251740df, // mmu__hubtlb_stalled_on_tag_allocation
                0x0c0677d236d60d65, // zrop__read_subpackets_stencil
                0x0c0d220b08222c25, // lts__t_sectors_ltc_ilwal
                0x0ce0e9f41e3870a6, // prop__earlyz_killed_quads
                0x0d03791579c186d5, // l1tex__t_set_accesses_hit_lg_local_ld
                0x0d2c34d9f560616d, // sys__fb_read_dwords
                0x0d46ffe2c3d6ef7f, // lts__mccif_read_request_latency_576
                0x0d656f6dd95f083c, // gpu__dispatch_count
                0x0d6f9a70b78f2bbc, // lts__t_sectors_miss_zrop_ilwal
                0x0dbec5eb24933026, // sm__mios_shmem_accesses_lsu_write
                0x0e010bff1d011a13, // gpmsd__input_active_compute
                0x0e147720691b010d, // lts__t_sectors_hit_host_cpu_membar
                0x0e32fa07cf4704ad, // sm__threads_launched_gs
                0x0e3ecef17a363215, // mpc__warp_launch_stalled_gs_fast_beta
                0x0e7a5bd99d48dee3, // lts__t_sectors_miss_pe
                0x0e841997540d8305, // raster__zlwll_lwlled_occluders_near_far_clipped
                0x0e935d7b084e2cc8, // lts__t_sectors_hit_ltc_atomic
                0x0ebd7bd0b08cc749, // lts__t_sectors_hit_host_noncpu_clear
                0x0ee14294c2ca8786, // gcc__l15_requests_hit_constant
                0x0f1651baa1041c58, // lts__mccif_read_request_latency_256
                0x0f8c1bc7fb7df31a, // sm__miop_pq_read_active_smp0
                0x0fa764421427929a, // gpmsd__input_samples
                0x0fa824f02a4c6c01, // lts__t_requests_hub_condrd
                0x0fdf39cffb0da1d8, // lts__t_sectors_hit_hub_rd
                0x1043cab20a0fd9b3, // lts__t_sectors_niso_rd
                0x108406a8dee0e38e, // lts__t_sectors_crd_ni_condrd
                0x109bd5c16520f58b, // prop__latez_rstrz_input_quads
                0x10b1a7a08e071fba, // sm__cycles_active_cs
                0x10eb6ff80c2ba784, // smsp__inst_exelwted_pipe_fma64plus_pred_on
                0x11077b4965706ead, // lts__t_sectors_miss_ltc_atomic
                0x1116c12e988f70ce, // gpc__rg_utlb_stalled_on_tag_allocation
                0x1123f8e6e275fb3b, // sked__dispatch_active_scg0
                0x114c6eb963f28310, // vpc__input_isbes
                0x115956a6de5ed958, // pdb__output_stalled
                0x118b9e5799336b9a, // lts__t_requests_host_noncpu_ilwal
                0x119dc8e495bb91d8, // l1tex__t_set_conflicts_surface_red
                0x11a5f226287b1b65, // l1tex__t_set_conflicts_lg_global_atom
                0x11bfbead6e960dea, // vaf__beta_cycles_elapsed
                0x11c181ede1a3af6e, // smsp__inst_exelwted_pipe_tex
                0x11d59ac77ecd2669, // lts__t_sectors_zrop_prefetch
                0x1204ef2699a79011, // lts__t_requests_crd_i
                0x125bbc5640417859, // smsp__inst_exelwted_vs
                0x125ff5c25b1465cd, // raster__setup_lwlled_prims
                0x1264db54b22e3dad, // tga__input_patches
                0x12bfce3c5ea7eb8c, // smsp__warps_eligible
                0x130b5b8db17bf13c, // mpc__input_batches
                0x132c73af3fe58be7, // lts__t_sectors_miss_zrop_rd
                0x135a2824134ccb55, // swdx__tc_replayer_bin_flushes_reason_explicit
                0x135ed7852811091c, // raster__crstr_lwlled_prims_no_tile_coverage
                0x13b01de7d58ddc9b, // l1tex__samp_samp2mipb_busy
                0x13d1c2811c611c7f, // l1tex__samp_samp2mipb_backpressured
                0x13ebe7b4a947076d, // l1tex__t_set_accesses_lg_local_cctl
                0x13f612d4f9bfd9b8, // l1tex__w_w2d_busy
                0x148fc02c5c4a085f, // sm__subtiles_active
                0x14ce7ba33da83dd3, // vaf__beta_tasks_active
                0x14dff203b80154cb, // raster__zlwll_lwlled_occluders_depth_bounds
                0x150bf00893dd3866, // raster__frstr_output_subtiles_4_cycle
                0x157bb296913cf9db, // fe__i2m_write_stalled_request_fifo_full
                0x16daff633139b72a, // lts__t_requests_gpc_condrd
                0x171ed4b2e4945772, // sm__warps_launched_vsb
                0x17479bb16cd1f5d9, // lts__t_sectors_gpc_atomic
                0x176d458c10184647, // l1tex__m_read_request_stalled
                0x17951e04fae39ef5, // l1tex__samp_input_quads_tex_format_3d
                0x181e07f5bb3c86ee, // lts__t_sectors_hit_zrop_condrd
                0x182f10248573a148, // lts__t_sectors_zrd_i_condrd
                0x18deaadbd6e36c8b, // lts__t_sectors_miss_crop_ilwal
                0x193f4f27083c1ca3, // sm__warps_launched_cs
                0x196cb42bdd502350, // gpmsd__input_samples_3d
                0x198d6f4f463e46d5, // smsp__thread_inst_exelwted
                0x19982fc5e4e8077c, // lts__d_atomic_resseq_cycles_bank_conflict
                0x199d24c327ae2141, // smsp__inst_exelwted_pipe_bru_pred_on
                0x1a0e8b916115c3f9, // vpc__clip_input_prims_clipped_multi_plane
                0x1a2d250a9ed08955, // prop__cdp_alpha_blendopt_killed_pixels
                0x1a5905b5625843c2, // sm__cycles_active_ps
                0x1a82cbefc757b55b, // lts__t_sectors_raster_wr
                0x1ab761f5f1b9e6ab, // scc__load_constants_miss
                0x1ad8471db9982e34, // l1tex__t_set_accesses_surface_atom
                0x1afbb4e8ba57df7e, // l1tex__samp_input_quads_filter_trilinear
                0x1b944d1843086ca6, // lts__t_requests_zrd_i_prefetch
                0x1bc989d49343b02b, // lts__t_requests_host_noncpu_atomic
                0x1c227b8a5a51abb8, // lts__t_sectors_hit_valid
                0x1c2a73a24abc49f8, // lts__t_sectors_miss_crd_ni
                0x1c4862e3792b4cae, // cbmgr__beta_cbe_allocations
                0x1c9476a9aec01196, // lts__t_requests_host_noncpu_condrd
                0x1cb8e6fb32801058, // vpc__lwll_lwlled_prims_reason_backfacing
                0x1cf06be6b8fc2994, // lts__t_sectors_miss_mmu_rd
                0x1cf835c9873cab19, // l1tex__texin_requests_surface_atom_cas
                0x1cfc77896b505d30, // gpc__tpc0_utlb_requests_miss
                0x1d0360a0ebe63033, // vpc__lwll_lwlled_prims_line
                0x1d3045c504a0738d, // smsp__inst_exelwted_pipe_tex_pred_off_all
                0x1d37a88bcb89bd29, // gr__cycles_idle
                0x1d96d5bed2351fbb, // gpc__prop_utlb_requests_hit_under_miss
                0x1dbfe97441064e0f, // l1tex__samp_input_quads_tex_format_2d_nomipmap
                0x1dc96a2df05a0eed, // lts__t_sectors_hit_zrop_rd
                0x1dd0a273d5f62f90, // smsp__inst_exelwted_vsa
                0x1dde193ba021699e, // lts__t_sectors_hit_gpc_clear
                0x1e1df4ba11bc094c, // l1tex__texin_requests_null_lg
                0x1e5ba36d1067063f, // lts__t_sectors_fill
                0x1e8990f0191f56e9, // lts__t_sectors_cas
                0x1eae4448c2246ecb, // lts__t_requests_l1tex
                0x1eba63057343d636, // lts__t_sectors_host_cpu_atomic
                0x1ec12de61746247e, // smsp__inst_exelwted_pipe_fma64plus_pred_off_all
                0x1ecfcaf894994112, // mmu__hubtlb_stalled_request_fifo_full
                0x1ed70fc2d81182dc, // lts__t_requests_crop_wr
                0x1edfe970a034098e, // sm__warps_launched_ps
                0x1ee0f0bae00d3188, // raster__tc_output_subtiles
                0x1ef207a983a492b1, // zrop__input_stalled
                0x1f0a7998df441a86, // smsp__warps_cant_issue_misc_0
                0x1f0bb77efbbafc8a, // lts__mccif_read_cycles_64
                0x1f184664c44efbf9, // prop__earlyz_output_samples
                0x1f457effbd4e26f6, // sm__warps_active_vsa
                0x1f4582266bb65655, // l1tex__samp_samp2mipb_stalled
                0x1f48d244f3c9b333, // prop__cycles_active
                0x1f6d4fe87c5c675d, // vpc__clip_input_prims_clipped
                0x1f7bf69cdf135e2e, // lts__t_sectors_rd
                0x1f80787c4615893f, // l1tex__t_atomic_address_conflicts_surface_red
                0x1f9b13999850c452, // lts__t_sectors_hit_iso_rd
                0x1fb823281048b429, // smsp__tex_requests
                0x1fbba9fafc03da85, // lts__t_sectors_niso
                0x20281e1ce3e329c9, // crop__processed_subpackets
                0x2030d93da77e9359, // lts__t_sectors_hit_host_cpu_atomic
                0x205af9f35af9eb97, // prop__latez_output_samples
                0x208500aa12cae928, // vpc__input_prims_point
                0x20a525aff0090259, // gpu__time_active
                0x20f41842d3b6ee0e, // lts__t_sectors_host_cpu
                0x20f5f62d1a340238, // l1tex__t_set_conflicts_lg_local_ld
                0x210413ff3a04c945, // pdb__input_batches
                0x21246106aa74cce7, // lts__t_sectors_hit_zrd_ni_prefetch
                0x2167f25f6f53abbf, // lts__t_sectors_ltc_membar
                0x21703d9eb184eeac, // lts__t_sectors_hit_hub
                0x21a7f71402d66540, // swdx__tc_replayer_bin_flushes_reason_level_1_threshold
                0x21b4dfe7b66d027e, // lts__t_sectors_hub_wr
                0x21db6924319afe57, // lts__mccif_write_request_latency_64
                0x220bfdc7362f8e63, // pel__in_active
                0x2233b707cd1d5a13, // gpmpd__cycles_active
                0x225bd8c19386c2b4, // prop__csb_output_stalled
                0x22666c2ec8aa6640, // mpc__tram_allocation_stalled
                0x22741b172014bc4e, // l1tex__m_read_sectors_tex_format_1d2d_tex
                0x22ab3daed523a9f2, // lts__t_sectors_miss_zrop_prefetch
                0x22ccd4e10b9a5d35, // l1tex__m_read_request_active
                0x22dff09b69d96b80, // l1tex__t_set_conflicts_surface_atom
                0x23169f2a01c42d35, // smsp__inst_exelwted_pipe_bar_pred_on
                0x236b167767c58fa5, // mmu__hubtlb_requests_miss
                0x23a895214a027c25, // vaf__alpha_fetched_attr_scalar_post_cbf
                0x23cc2d495952795e, // lts__mccif_read_request_latency_128
                0x23d388a1d6355af6, // smsp__lsu_write_back_active
                0x23f4142eb5c08346, // lts__t_sectors_raster
                0x240dd2107f5f4c0e, // lts__t_sectors_miss_condrd
                0x247e2178997a166b, // vpc__lwll_lwlled_prims_triangle
                0x24df78366b50edaa, // lts__t_sectors_hit_prefetch
                0x25398abe48789c25, // pda__output_verts
                0x25471d0eba5e9877, // gpc__tpc0_utlb_requests_sparse
                0x254bf5815672c0fc, // smsp__inst_exelwted_pipe_bru
                0x255f347910f1b522, // smsp__inst_exelwted_pipe_lsu_pred_on
                0x255f85aed2c8d648, // lts__t_sectors_miss_l1tex
                0x25a1ddaf70c760df, // lts__t_sectors_host_noncpu_ilwal
                0x25f58d24f486364a, // prop__earlyz_killed_pixels
                0x262427c871695d2d, // lts__t_sectors_condrd
                0x269def64bd248147, // l1tex__texin2m_fifo_output_busy
                0x26e29434d000e984, // gpc__rg_utlb_stalled_request_fifo_full
                0x2702a1227f134f69, // prop__gnic_port1_stalled
                0x275c0452e38be6d5, // pda__input_prims_line
                0x2769c24acc4b306b, // lts__t_requests_rd
                0x277072462811f118, // lts__t_sectors_miss_crop_prefetch
                0x280caa6ae792cc48, // l1tex__x_x2w_stalled
                0x28730c7af7a9d249, // zrop__processed_requests_type_shdz_biquad
                0x28e42086d0634cbb, // vaf__alpha_fetched_attr_scalar_indexed
                0x292e482bd5ceb357, // lts__t_sectors_hit_host_cpu_clear
                0x29ab0d044e6931b1, // lts__t_sectors_hit_niso
                0x29d918f5f5893131, // lts__t_sectors_miss_raster
                0x29efabb6e0aab685, // pel__out_l2_requests_read
                0x2a5127b91144ca0c, // sm__icc_requests_miss_covered
                0x2a8d40e6d6a40790, // lts__t_sectors_hit_ltc_membar
                0x2ae407d1c2f3a299, // stri__to_acache_stalled_on_read
                0x2b12701ee61eac5a, // gpmsd__input_pixels
                0x2b1c9d89d524e967, // sm__icc_requests_miss_no_tags
                0x2b4d87a5c156f76e, // prop__input_stalled
                0x2b7750b02f7bcb5b, // sm__miop_pq_read_active_tex_smp0
                0x2b99fae0531919cb, // smsp__miop_pq_write_active
                0x2b9ac231595241a0, // l1tex__x_x2w_backpressured
                0x2bb4d075eee1f0f3, // l1tex__samp_samp2x_backpressured
                0x2bd5f7ff9aa5cd6f, // l1tex__x_x2t_busy
                0x2c0ccf32d3bac6c8, // lts__t_sectors_miss_gpc_membar
                0x2cc9a252552d0407, // gpc__tpc0_utlb_requests_hit_under_miss
                0x2cec3a80892a7fce, // vpc__output_verts
                0x2d3a1d6d2068c8fe, // lts__t_requests_host_noncpu
                0x2d44a0fd7c725bcd, // lts__t_sectors_hit_host_noncpu_prefetch
                0x2d4e29255fb82562, // lts__t_sectors_miss_crop_wr
                0x2d72021d1dca97a0, // lts__t_sectors_miss_hub_cas
                0x2d767069b410d8ad, // swdx__input_messages
                0x2da1045ffe3234fc, // prop__pixel_shader_barriers
                0x2dbe3b155592ccf7, // raster__frstr_output_subtiles_2_cycle
                0x2dc009333590c60f, // sm__warps_launched_gs_fast_alpha
                0x2dc7dd0ce9d7e3fb, // lts__t_requests_gpc_rd
                0x2de750379191df20, // lts__mccif_write_request_latency_48
                0x2e0af35b5dad9dad, // l1tex__t_set_conflicts_lg_global_cctl
                0x2e57316f867b46a8, // sm__cycles_active_3d_ps
                0x2e8770f43b969d22, // lts__t_tags_writeback_tier2_issued
                0x2ec2baa64794e6de, // l1tex__w_w2d_active
                0x2f1612412c391c1e, // prop__latez_shdz_input_samples
                0x2f1b95695347f235, // mmu__pte_requests_miss
                0x2f2603657fec192d, // prop__earlyz_output_pixels
                0x2f53ec6e26062762, // smsp__inst_exelwted_pipe_ldc
                0x2f6c83f1f1547f36, // raster__setup_output_prims
                0x2fa1c695a280856a, // vaf__alpha_acache_requests_local_hit
                0x2fcc576f769fe47c, // sys__fb_write_requests
                0x301daeacad635445, // lts__cbc_requests_miss_clear_zbc
                0x30376d862207eab7, // sm__warps_retiring_ps_stalled_not_selected
                0x304dd9a73c42cce0, // gcc__l15_requests_miss_instruction
                0x30b38252ee920335, // raster__setup_output_prims_line
                0x30c1fd758090698a, // smsp__inst_exelwted_pipe_xu_pred_on
                0x30cc27d21e0c88ab, // lts__t_sectors_host_cpu_membar
                0x30d32d5b193ee525, // sm__miop_pq_read_active_pixout_smp1
                0x3125eb61508b2b5f, // raster__crstr_search_stalling_input
                0x3158d1ec8e9b7860, // lts__t_sectors_hit_gpc_wr
                0x31a74642781e5199, // swdx__output_active
                0x31b2a48dd1cffc03, // mpc__input_verts_alpha
                0x31e9608b25104126, // lts__t_sectors_hit_host_cpu_cas
                0x323b0f9a1490b8ca, // zrop__write_subpackets_stencil
                0x324dac55f047b7e1, // lts__t_sectors_hit_zrd_i_rd
                0x32541918a0ff5a36, // lts__t_sectors_ltc_cas
                0x32b24e764f1b6ed3, // lts__t_sectors_concat_cas
                0x32c19db9804d6c94, // lts__t_sectors_ltc_prefetch
                0x32e5d08b60c22f19, // pdb__output_stalled_beta
                0x32eb2db5130a5ab8, // sm__mios_shmem_accesses_pe_write_isbe
                0x3367cf37de98db82, // lts__t_sectors_zrop_condrd
                0x33780cc0c42fed14, // crop__input_stalled
                0x3396c11370bc032d, // l1tex__t_set_accesses
                0x33973f907195d834, // lts__t_requests_condrd
                0x33b0c4cd0710f3e5, // pda__input_stalled_index_fetch
                0x33df86b93b469519, // fe__i2m_write_requests
                0x34179deedb7a9b47, // smsp__warps_cant_issue_not_selected
                0x344720060c5c155c, // lts__t_sectors_crd_i_prefetch
                0x348232ec84c8e8f1, // lts__t_sectors_miss_host_noncpu_prefetch
                0x34e387ed4e539b60, // raster__zlwll_occluders_zfail
                0x34edbb56e5581aa9, // prop__csb_input_samples_3d
                0x350b732060f5df9f, // l1tex__t_set_conflicts_tex_bilinear
                0x351e65b46ab40dbd, // sys__pushbuffer_dwords
                0x35cf9a3d0c3fcf8f, // sm__miop_pq_read_active_lsu_smp0
                0x35dcacd398e19006, // lts__t_requests_crd_ni_prefetch
                0x361d59f4055a2901, // lts__d_decomp_input_stalled
                0x36c6844e76be8529, // prop__csb_input_pixels_3d
                0x37425272f5ba5f37, // lts__cycles_active
                0x3751ece9a12b1689, // vaf__alpha_acache_requests_acache_hit
                0x375c80872348955a, // lts__t_sectors_ltc_atomic
                0x3767064c13ec807b, // wwdx__input_tasks
                0x376e9d544683aec3, // smsp__inst_exelwted_pipe_adu
                0x37e40a760c4cd198, // pdb__cycles_active
                0x383b391a9e1e67a6, // tgb__output_verts_complex_interior
                0x38920d2ea0e0f224, // lts__t_requests_ltc_rd
                0x38b40fd77fe706ec, // lts__t_sectors_miss_crd_ni_prefetch
                0x391880580ad27f2c, // lts__ltcx_read_requests
                0x391e80519a5833d3, // pes__stream_output_prims
                0x3985a6d0032d0865, // vpc__lwll_lwlled_prims_reason_bounding_box
                0x39afc26445bf41b7, // l1tex__t_set_accesses_lg_global_atom
                0x39d2bff03f49faf3, // smsp__inst_exelwted_tes
                0x3a2798a870e3714d, // lts__d_atomic_reqseq_stalled_source_not_ready
                0x3a52c2445c8b813d, // swdx__input_pixel_shader_barriers
                0x3ab607e1c87dec7f, // lts__t_sectors_miss_gpc_prefetch
                0x3ab9b6c81e13f2a1, // lts__t_sectors_miss_pe_ilwal
                0x3abfc2a4095e0266, // lts__t_sectors_pe_rd
                0x3b0635ff83409920, // lts__t_sectors_miss_hub_ilwal
                0x3b692a84e19e6fb2, // l1tex__t_set_accesses_miss_tag_miss_tex_format_no_mipmap
                0x3bcb58d2a832a3cb, // lts__t_sectors_hit_gpc
                0x3bdc0dfc63d13d76, // scc__load_constants_stalled_max_versions
                0x3c086382824a3148, // sm__idc_requests_miss_to_gcc
                0x3c287250bcdd03ed, // lts__t_requests_zrop
                0x3c35e25bf0b1dedd, // vaf__alpha_acache_requests
                0x3c3fdb607d9e391c, // gcc__l15_requests_instruction
                0x3c766de62a123ae4, // lts__r_input_fifo_stalled
                0x3c7977a202a7f6f3, // prop__csb_output_crop_requests
                0x3c8d4a71189bd0dd, // sked__dispatch_count
                0x3c9fc5687be19e55, // zrop__input_requests_type_bundles
                0x3cd4e5cdeae591ed, // lts__d_atomic_reqseq_cycles_bank_conflict
                0x3cfa77a41504b63f, // prop__gnic_port0_stalled
                0x3d2feb260d9467fa, // l1tex__samp_input_quads_sz_96b_nearest
                0x3e769b26beffc081, // l1tex__t_set_conflicts_cctlt
                0x3e9820c565b24389, // lts__t_sectors_miss_host_cpu_clear
                0x3f0aaf062f2cec09, // scc__load_constants_page_first_update
                0x3f188560b327a9f2, // lts__t_sectors_cbc
                0x3f4f7ee846f65693, // lts__t_sectors_host_noncpu_atomic
                0x3f5fed38213aac64, // mpc__isbe_allocation_stalled_alpha
                0x3f7a1d4ab64aeed7, // prop__cdp_alpha_to_coverage_output_pixels
                0x3f85b31d14665d22, // lts__t_sectors_zrop
                0x3f8c8c9c01c6f449, // l1tex__m_stalled_on_tag_allocation
                0x3f9221781944733d, // l1tex__t_set_accesses_miss_surface_ld
                0x40118f17d692cd3c, // lts__t_requests_pe_ilwal
                0x402743ff6f93bd13, // l1tex__w_w2d_stalled
                0x4036391f3843c88b, // lts__t_requests_l1tex_rd
                0x409b00baf51995a8, // vaf__alpha_input_batches_post_cbf
                0x40c8d659ac4d37a0, // lts__t_sectors_miss_rd
                0x410fb18aba2cab1d, // lts__t_requests_prefetch
                0x41275cb20614c05c, // lts__t_sectors_miss_host_cpu_condrd
                0x41674e9dc6b3373a, // lts__t_sectors_hit_zrd_i_prefetch
                0x419f2dd1de90e2a8, // vaf__alpha_to_acache_stalled_on_tag_allocation
                0x41a7b5470c048a97, // l1tex__m_write_sectors_surface_atom_cas
                0x4201a3e331769a9d, // lts__t_sectors_host_noncpu_membar
                0x426b4b2f7f658544, // lts__t_sectors_hit_concat_cas
                0x42cb4c2c53a7544c, // raster__frstr_killed_subtiles_input
                0x42f9833a6f0ece72, // sm__miop_pq_read_active_pixout_smp0
                0x433d36bc597920d7, // sm__cycles_active_tes
                0x436862370dc5341e, // lts__t_sectors_miss_zrd_ni_condrd
                0x439c4858bc9341fa, // lts__t_tags_writeback_tier1_killed
                0x441827fe078ae949, // lts__t_sectors_raster_rd
                0x4439aef01ff03887, // vpc__output_prims_stippled_line
                0x4446773ffc91da0e, // swdx__tc_replayer_bin_flushes_replay_tiled
                0x447089b5b354d82c, // gcc__tsl2_requests_hit
                0x44a2184127c49409, // smsp__warps_cant_issue_math_pipe_throttle
                0x44ac9584d24de32c, // pdb__output_pkts
                0x44da0f6807185944, // lts__cbc_requests_hit_clear_zbc_crop
                0x44e83dd97ce91172, // zrop__input_requests_type_plane_eq
                0x450940571b7da9a7, // vaf__alpha_cycles_elapsed
                0x457b947874ed9063, // smsp__imc_requests_miss_covered
                0x45a63a0b8cd4ddd3, // smsp__inst_exelwted_pipe_fma64plus
                0x45bbef99d5209bdd, // lts__t_sectors_gpc_condrd
                0x45d22bc7b815fd8b, // lts__t_requests_host_noncpu_clear
                0x45ef341669bcf22b, // zrop__input_requests
                0x466175c74c33bd3e, // mpc__tram_fill_fifo_stalled
                0x46665335bd139d11, // lts__t_sectors_hit_crd_ni
                0x46b956370b550dca, // l1tex__t_set_accesses_hit_lg_global_ld
                0x46d69f285cb8d42e, // vaf__alpha_fetched_attr_scalar_indexed_constant
                0x46f306edaf72bca7, // l1tex__texin_requests_lg_local_cctl
                0x472849a601b09c22, // rdm__cycles_active
                0x47636983720ff6e6, // lts__t_requests_clear
                0x478c94ab6e4e5f86, // prop__input_active
                0x47937796fa4374ba, // sm__warps_draining_ps
                0x47b0bcb296e86f28, // zrop__cycles_active
                0x47d7cee5fe68978c, // tga__output_prims
                0x482b9c6d4b045168, // rdm__crop_output_stalled
                0x488d5c52d28e776b, // zrop__input_requests_type_shdz_biquad
                0x488f302803956d63, // lts__t_sectors_miss_ltc_wr
                0x48aee0e448e09315, // lts__t_sectors_pe
                0x48c24f3e182cecc7, // lts__t_sectors_hit_zrd_ni_condrd
                0x490130bc625f4704, // l1tex__t_atomic_address_conflicts_surface_atom
                0x49046257640de113, // lts__t_requests_ltc_condrd
                0x49188c0fc3c4fa79, // lts__t_sectors_hit_host_cpu_wr
                0x491e326f3dc3f2fb, // gcc__l15_requests_constant
                0x49292bb1eff2e782, // lts__t_sectors_hit_host_noncpu_wr
                0x4940c15104ecfd4f, // lts__t_sectors_hit_pe_wr
                0x4957bdb1c873e0c7, // pda__input_prims_lineadj
                0x499edc507624295b, // lts__t_sectors_gpc_ilwal
                0x49e3cabb5afd6ce1, // lts__t_requests_raster
                0x4a02b4e190a24f84, // smsp__warps_cant_issue_tile_allocation_stall
                0x4a0bc7786a8c8458, // lts__t_sectors_hub_prefetch
                0x4a461c7d74b2bd93, // vpc__input_isbes_prim
                0x4a6f59b8f17fcd26, // swdx__input_active
                0x4a958fac45b19fee, // lts__t_requests_host_noncpu_cas
                0x4a9c64c0f4b127cd, // tga__output_tasks_complex_boundary
                0x4aa1686bee1a995c, // l1tex__texin2m_fifo_output_active
                0x4ac950e7cef3e559, // l1tex__m_read_sectors_tex_format_3d
                0x4b0e4dc133fbdf20, // gpmsd__input_stalled
                0x4b1ffc8ae872708c, // l1tex__t_set_accesses_hit_tex_format_no_mipmap
                0x4bc2ff23c0c9e139, // prop__input_pixels_3d
                0x4bc69131042dc7e3, // lts__t_sectors_ltc_condrd
                0x4bed824ea7a2c60a, // lts__t_requests_host_noncpu_prefetch
                0x4c272b1076284bcd, // lts__t_requests_ltc_prefetch
                0x4c38d6464fec377f, // gpc__tpc1_utlb_stalled_on_tag_allocation
                0x4c6372571850ad71, // lts__t_sectors_host_noncpu_wr
                0x4c925266e21e2e03, // lts__t_tags_writeback_tier1_issued
                0x4cab66078824d541, // mpc__warp_launch_stalled_vsb
                0x4cb79e44819169e1, // lts__t_sectors_mmu_rd
                0x4cdf36f0edfe06ec, // crop__input_requests_clear
                0x4d0d8408e94b45d4, // gpc__prop_utlb_stalled_request_fifo_full
                0x4dfbfc86d70f385d, // lts__t_sectors_hit_zrop_ilwal
                0x4e4675bfb83dd341, // lts__t_sectors_hit_host_noncpu
                0x4e4e012e6c3e1a9e, // mmu__pte_requests_small_page_1
                0x4e74e9317fb1a7ca, // l1tex__texin_sm2tex_stalled
                0x4e756420a47670a4, // crop__input_stalled_upstream_fifo_full
                0x4f0142dbdeee97cc, // l1tex__t_set_accesses_miss_tag_hit_tex_format_1d_buffer
                0x4f723217c56b73fe, // l1tex__m_read_sector_lwmulative_miss_latency
                0x4f81f27cbd9afd34, // l1tex__texin_stalled_on_tsl2_miss
                0x4f8f98832f71e3c1, // gpc__cycles_elapsed
                0x4fc35fed50b6a866, // sm__warps_active
                0x503b81725ee34a3b, // lts__t_sectors_crd_i_condrd
                0x503d33d71def5ea7, // lts__t_sectors_niso_wr
                0x5058510c75061962, // prop__input_quads
                0x50c1ab3171946cbd, // fe__i2m_write_stalled_data_buffer_full
                0x50c5fd2e7bc9c13a, // pda__input_prims_tri
                0x511a965dc8d66718, // crop__write_requests_compressed_4to1
                0x51704d986666d85d, // l1tex__samp_input_quads_sz_32b_nearest
                0x51cc28456f06151a, // smsp__inst_exelwted
                0x523ae980deb4d101, // lts__t_sectors_hit_gpc_membar
                0x5251071e51e52b5a, // lts__t_sectors_ltc_wr
                0x52b767369c4f1a6a, // lts__t_sectors_miss_gpc_clear
                0x52bd6fbfb6a6e6bf, // prop__csb_output_active
                0x52e7ceedccaa49ed, // lts__t_sectors_miss_gpc
                0x530bc121e4184cd5, // sm__cycles_active_vsa
                0x534c9cac9a6f3771, // stri__cycles_stalled
                0x537a15cf68f7df2b, // lts__t_sectors_zrop_rd
                0x53a47dee6100c37c, // lts__t_sectors_zrop_ilwal
                0x53d2a13572ac84f1, // lts__t_sectors_miss_host_cpu
                0x53e942b5baa44a96, // lts__t_sectors_miss_mmu_wr
                0x540dbcb3b133e83b, // l1tex__t_set_accesses_miss_tag_miss_tex_format_lwbemap
                0x54168b48583139d5, // crop__input_requests_2d
                0x548923436243ec73, // l1tex__m_write_sectors_lg_global_atom
                0x54c00750807ca401, // lts__t_sectors_gpc_wr
                0x54ec88843e0e4f61, // wwdx__output_stalled
                0x5537050c080c395c, // gcc__tsl2_requests_miss
                0x5538f2be5921befc, // lts__t_requests_zrop_rd
                0x5544f18b839ce55d, // cwd__output_stalled_no_free_slots
                0x554d9097cfd310f8, // lts__t_requests_raster_rd
                0x55754ad0e957dc38, // sm__warps_launched_tes
                0x55795f89a0bf6e38, // sm__warps_launched_gs
                0x55e166fdfac2aa3f, // lts__t_sectors_hit_l1tex
                0x564e3323e3833687, // l1tex__t_sectors_promoted
                0x568bfde7cf37186d, // lts__t_sectors_mmu
                0x57209ba5b83afa4d, // vaf__cycles_active
                0x5771a5cc9b8d3adb, // prop__cdp_alpha_blendopt_read_avoid
                0x5775f9e14f20d851, // l1tex__texin_requests_lg_global_st
                0x58115185bf3022e2, // mmu__pte_requests_big_page_3
                0x588f857644500d95, // l1tex__t_set_accesses_surface_st
                0x58c01518ca285be7, // l1tex__m_write_sectors_surface_st
                0x58ceac4faaad5db2, // swdx__tc_replayer_bin_flushes_reason_constant_table_full
                0x58e7c8247fbf7920, // lts__t_sectors_miss_host_cpu_cas
                0x5904c5a2bc708efa, // lts__t_sectors_miss_cbc_wr
                0x5912be72bb85b770, // lts__t_requests_mmu
                0x592df9080455e392, // smsp__warps_active
                0x5956231907408670, // sm__threads_launched_vsb
                0x59a328e66ff5dbf3, // tgb__output_prims_triangle
                0x5a198eb9870d9005, // l1tex__texin2m_fifo_output_backpressured
                0x5a3d00cb9ba775ec, // sm__mios_shmem_accesses_pe_write_tram
                0x5a52313ebcb116d8, // lts__t_sectors_hit_l1tex_wr
                0x5a8d6174544bf29b, // crop__processed_requests_compressed
                0x5ad9effc6d00dfb1, // smsp__inst_exelwted_ps
                0x5ae2e872fc9dcec5, // lts__t_requests_gpc_membar
                0x5aea6f7ff31f853b, // l1tex__lod_output_wavefronts
                0x5af74edaba62d136, // vpc__cycles_active
                0x5af79a92e4fb3a14, // lts__t_requests_niso_wr
                0x5b532665b3271976, // cbmgr__beta_cbe_allocation_stalled_max_tasks
                0x5b55b366baedcd72, // stri__to_acache_stalled_on_tag_allocation
                0x5b562b3336aec771, // l1tex__t_set_accesses_lg_global_atom_cas
                0x5b9ca7388de61447, // lts__t_sectors_miss_host_cpu_membar
                0x5be60db73d132b05, // lts__t_sectors_hit_iso_wr
                0x5beb7b8ff535250e, // l1tex__t_set_accesses_tex_format_1d2d_tex
                0x5c1e19e9e6151b60, // sm__inst_exelwted_pipe_adu_divergent_smp0
                0x5c69ddb849f06d9d, // zrop__processed_subpackets_stencil
                0x5cc876caee50c57b, // lts__t_sectors_hit_membar
                0x5cd9454b81a08968, // lts__t_sectors_zrd_ni_prefetch
                0x5ced0cca1f3fff16, // gpmpd__output_packets
                0x5d01b81443b8e12e, // vaf__alpha_to_acache_stalled_on_read
                0x5d75970a90ac3e11, // zrop__processed_requests
                0x5d8a1887d485c0c8, // tpc__cycles_elapsed
                0x5dd4f23434aa4427, // l1tex__texin_requests_lg_global_cctl
                0x5de3822bf27cd3e4, // l1tex__t_set_accesses_miss_tag_hit_tex_format_no_mipmap
                0x5e45ac59b840e363, // lts__t_requests_host_cpu_ilwal
                0x5e77399dc9f03125, // zrop__read_returns
                0x5e8eac0affa40188, // lts__t_requests_host_cpu_prefetch
                0x5eb849324678dd87, // fe__output_ops
                0x5ed5167a214fc36e, // sm__warps_active_tcs
                0x5ede8a335ea5a8b1, // pel__out_active
                0x5f00911b8393896b, // l1tex__t_set_accesses_tex_format_3d
                0x5f2fd1a9d591474b, // lts__mccif_write_request_latency_high
                0x600b7ce3d7151586, // vpc__output_prims_line
                0x600cb0d5840cd74c, // gpc__tpc1_utlb_stalled_write_buffer_full
                0x6090110db22a99d4, // lts__cbc_requests_comptag_miss
                0x60dbfe8063240697, // lts__t_sectors_zrd_i_rd
                0x61160a87ea7e4032, // lts__t_sectors_miss_zrd_i_rd
                0x61185f3a5a4e808a, // lts__t_requests_zrop_wr
                0x611e90a95c3297f9, // lts__mccif_read_request_latency_512
                0x61420ec1a8658339, // l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_tex
                0x6194e6c555bc4958, // stri__acache_requests_stri_local_hit
                0x619826068501904b, // cwd__cycles_active
                0x61a3c4dd71b016d7, // lts__t_sectors_hit_gpc_ilwal
                0x61c4c2a515f01711, // pel__out_l2_requests_write_512b
                0x6230645cf09d20d8, // lts__t_requests_concat_cas
                0x62312343a692ad25, // sm__idc_requests_miss_no_tags
                0x62966a6d6118b862, // gpc__tpc0_utlb_requests_hit
                0x62f14f4b29105e96, // lts__t_requests_host_cpu_clear
                0x630208d7a7a0aea1, // lts__t_sectors_gpc_concat_cas
                0x6303fa851c4d3617, // prop__cdp_alpha_test_killed_pixels
                0x637e4f0ff3aa4d18, // sm__warps_active_vsb
                0x638c31440680cebd, // lts__t_requests_crd_ni
                0x63a3b76bdfb1b937, // lts__t_requests_ltc_concat_cas
                0x63a5754b73be0502, // sm__miop_adu_replays_smp1
                0x6417737f0c7fb8a2, // pda__input_verts
                0x6462e87365f91961, // lts__t_sectors_pe_wr
                0x64f472413c6b7c22, // mmu__pte_requests_small_page_0
                0x6506cdee9ad29c6f, // lts__t_requests_gpc_ilwal
                0x658a71a93ae05fe7, // lts__t_sectors_hub_ilwal
                0x65968f7ee6f51995, // l1tex__samp_input_quads_tex_format_1d_buffer
                0x659f4ff0f763b2f5, // lts__t_sectors_hit_mmu_rd
                0x65c0d6c64e82a668, // mpc__warp_launch_stalled_gs_fast_alpha
                0x65d6c9dc770c2a47, // gpc__rg_utlb_requests_hit
                0x65de6dade2c4cace, // mpc__output_batches
                0x660935ffbdfd78d3, // lts__t_sectors_ltc_concat_cas
                0x662081f39f09f545, // lts__ltcx_write_stalled_fifo_full
                0x6639d39aa4122694, // lts__t_sectors_hit_pe_rd
                0x6648bce081e73305, // lts__t_requests_raster_wr
                0x6652b6c819f64f72, // gpu__time_duration
                0x666175661f2abb6a, // raster__crstr_output_tiles
                0x6668e672a86ca6d6, // smsp__imc_requests_miss_to_gcc
                0x667409558ec595b8, // l1tex__t_set_conflicts_lg_global_atom_cas
                0x6674910fc40f7fe9, // lts__t_sectors_miss_iso_rd
                0x667d192393a9668f, // crop__write_requests
                0x6690f79223aebdf9, // lts__t_sectors_hit_crd_ni_prefetch
                0x66910a15128e3a58, // crop__input_requests_aamode_8xaa
                0x66bf0c5b635b6755, // lts__d_atomic_block_cycles_serialized
                0x66e4e630afb0d3e6, // lts__t_sectors_pe_ilwal
                0x66ebac642936a839, // prop__gnic_port1_active
                0x66f73b1b47d0e2c3, // swdx__tc_replayer_bin_flushes_replay_not_tiled
                0x672509d333ecb55e, // lts__t_sectors_hit_ltc_concat_cas
                0x6742ae0c1e4f8be4, // lts__t_sectors_hit_hub_cas
                0x67490da756c27f28, // wwdx__cycles_active
                0x678cd126fe178fe6, // lts__t_sectors_hit_gcc
                0x67950cc8d0c74158, // l1tex__samp_input_quads_tex_format_lwbemap
                0x67b0c79fa5498093, // sm__ps_quads_killed
                0x681e60c2b98865be, // l1tex__t_t2d_busy
                0x6868508a3d581b76, // lts__t_requests_hub_rd
                0x68c03666b2d1d0fb, // l1tex__d_sectors_fill
                0x68d5fb65beca3e22, // l1tex__x_x2w_active
                0x691a980b8ebc2395, // lts__t_sectors_hit_gcc_rd
                0x6926b743ba79aedf, // smsp__ps_threads_killed
                0x69b7d59055f44e3e, // lts__t_sectors_hit_ltc_prefetch
                0x69c2f861f269a179, // vpc__output_prims_point
                0x6a102505277be773, // gcc__l15_requests_hit_instruction
                0x6a61accd2ad14a1c, // lts__t_sectors_hit_rd
                0x6a949bf1da630d18, // lts__t_sectors_hub_atomic
                0x6a9ee7c79e5ab52b, // l1tex__d_output_wavefronts
                0x6aba5e8340548e19, // lts__t_sectors_hit_gpc_rd
                0x6acf2e7869a85269, // gpmpd__output_stalled_batch
                0x6ae39b3b9f4d9e2e, // lts__t_sectors_miss_zrop
                0x6aef363927927510, // l1tex__f_output_tex2sm_stalled
                0x6aefdc470926957e, // prop__gnic_port0_active
                0x6aff3851f16ca120, // sm__miop_pq_read_active_smp1
                0x6b018173d94adefd, // lts__t_sectors_miss_cbc_rd
                0x6b18ad13019fefa7, // mmu__hubtlb_requests_hit
                0x6b9e6de1bdde37c2, // lts__t_sectors_hit_host_cpu_prefetch
                0x6c0d2981c676681d, // lts__t_sectors_host_cpu_clear
                0x6c579bbfbc66b723, // l1tex__t_atomic_address_conflicts_lg_global_atom_cas
                0x6c67b97faa2ab00b, // lts__t_requests_gpc_prefetch
                0x6ce268b025f2f992, // lts__t_sectors_hub
                0x6cef78272a4c30b6, // l1tex__m_read_sectors
                0x6d34124ad11d2751, // lts__t_sectors_hit_crd_ni_rd
                0x6d4cc2fec5f6b890, // pes__cycles_active
                0x6d973ebf285b2706, // lts__t_sectors_miss_ilwal
                0x6daa7cc043bc8321, // cbmgr__alpha_cbe_allocations
                0x6e1013dfa6ee1605, // gpc__rg_utlb_requests_miss
                0x6e2603d3d736e93a, // lts__t_sectors_hit_ltc_clear
                0x6e27b3e3f7af1aa1, // pel__out_read_stalled_vaf_alpha
                0x6e2888e625aded43, // mpc__isbe_allocations
                0x6e5b74cda072159e, // rdm__input_data0_active
                0x6e65b310d296e830, // crop__input_requests_blend_enabled
                0x6ef5d4a739a223d4, // l1tex__texin2m_fifo_output_stalled
                0x6f2b169193fea548, // smsp__inst_exelwted_lsu_wb_pending
                0x6f594d3e427f22d6, // lts__t_requests_host_cpu
                0x6f5985fba1f5132b, // raster__setup_output_prims_triangle
                0x6f5b3383ce59e0df, // lts__t_sectors_miss_iso
                0x6f78c43182de7996, // l1tex__x_x2t_backpressured
                0x6f88de7e56bcf97e, // prop__latez_shdz_input_quads
                0x6fc7f82dde174259, // pel__in_write_requests_stalled
                0x6fd9138a9b2c69cf, // smsp__inst_exelwted_lsu_wb
                0x704aa67fbf31ee2c, // lts__t_requests_ltc_clear
                0x70db62a1ed1786e3, // l1tex__m_read_sectors_surface_ld_d
                0x71013ed0ec04da45, // sys__fb_read_requests
                0x710f053fb3cd00f3, // lts__t_tag_requests_hit
                0x712581f91e1d9173, // lts__t_requests_niso_rd
                0x713e71e0ffd25227, // gpmsd__cycles_active
                0x7166f1f66ee14253, // sm__warps_active_gs
                0x7169a84119970331, // l1tex__d_d2f_stalled
                0x71be3a6e5ed821e1, // smsp__inst_exelwted_pipe_su
                0x71e3482308c488d9, // lts__t_sectors_miss_host_noncpu_membar
                0x71f2ad773dafa2ef, // gpmsd__input_pixels_2d
                0x72470418e22ca45b, // lts__t_sectors_hit_crd_ni_condrd
                0x725cfb45c679ba48, // lts__t_requests_hub_prefetch
                0x726b89d9bf107406, // lts__t_sectors_miss_ltc_clear
                0x729172b0ec7860a7, // sm__cycles_active
                0x72bd13c86366be3e, // l1tex__w_output_wavefronts
                0x72ef698ae53c5da8, // sm__threads_launched_tcs
                0x73325cf5f94987fb, // l1tex__texin_requests_surface_red
                0x738b70b1e5fec918, // lts__t_sectors_crd_ni_rd
                0x73a32fcfa7432233, // l1tex__samp_samp2x_stalled
                0x73c6229d06f365f0, // lts__t_sectors_mmu_wr
                0x73c8ae9bb7f5f023, // swdx__tc_binner_binned_op_tiled_cache_barriers
                0x73cd0c027f239f69, // smsp__warps_cant_issue_short_scoreboard_1
                0x73d4317db7c61e57, // lts__t_sectors_host_cpu_concat_cas
                0x73d76e71258e43c7, // swdx__output_stalled
                0x741284787ab667b3, // lts__t_sectors_ltc_rd
                0x742177e139008d5f, // l1tex__t_sectors_miss
                0x742727b228b1c0f9, // l1tex__samp_input_quads_tex_format_2d_mipmap
                0x7483e8cd9a60a0d8, // gpmsd__input_quads_2d
                0x748b75fb36c7c8d8, // raster__crstr_discover_stalling_setup
                0x74bbee56bd20a2b2, // lts__t_requests_crop_ilwal
                0x74d53a78feb18bf5, // mpc__warp_launch_stalled_tcs
                0x74df13d4a1d16f46, // lts__t_sectors_hub_condrd
                0x74fe6a60eb5694be, // vaf__beta_input_tasks
                0x7542ed93d392e12e, // prop__cdp_alpha_blendopt_pixels_fill_override
                0x75e4cdf6c8995d79, // l1tex__t_set_accesses_tex_format_no_mipmap
                0x760bf5c3618b7056, // l1tex__t_set_accesses_tex_format_lwbemap
                0x761020deffa97ac2, // raster__frstr_output_subtiles_3_cycle
                0x761117116d2b78bc, // raster__crstr_input_prims
                0x76876cd5e51861ae, // scc__load_constants_hit
                0x76b58cafe2c5489f, // lts__t_sectors_miss_gpc_ilwal
                0x77c463aa8e12abd3, // l1tex__m_read_sectors_surface_ld_p
                0x77f7cd959ff61828, // pel__in_write_requests_stalled_vaf_alpha
                0x7850155d7f0ffe25, // lts__t_sectors_hit_ltc_condrd
                0x78d30dc94ca92aa6, // lts__t_requests_pe_rd
                0x78fe7ac784396216, // lts__d_atomic_reqseq_stalled_pending_store_same_address
                0x790d86d8e5bfc7c1, // lts__t_sectors_miss_hub
                0x79187a9172784e84, // l1tex__m_read_sectors_tex_format_1d2d_array
                0x79449060b66e0947, // l1tex__texin_requests_surface_st
                0x797b1a1767f92f04, // l1tex__texin_sm2tex_busy
                0x7986156b5d207824, // l1tex__m_write_sectors_surface_atom
                0x79e10bd36b9709dc, // tgb__output_prims_point
                0x7a22a469a9e5be1c, // lts__t_sectors_hit_hub_wr
                0x7a7f9c832213400f, // lts__cbc_requests_miss_clear_zbc_zrop
                0x7abb8ef1045fc625, // lts__t_sectors_hit_zrd_i
                0x7b071fb7b44b682f, // cwd__output_stalled_state_ack
                0x7b0bfbf67f1325e3, // l1tex__t_set_accesses_miss_lg_global_ld
                0x7b0d47b085a75089, // lts__t_sectors_miss_hub_atomic
                0x7b43f85b96c44c54, // gpc__gpcl1tlb_stalled_request_fifo_full
                0x7b735a23ac980ac2, // lts__t_requests_cas
                0x7b771ecd214566e8, // lts__t_sectors_miss_crd_i
                0x7bad702609fde330, // l1tex__texin_tsl1_requests_hit_texhdr
                0x7bdaa4cdc00fb355, // mpc__warp_launch_stalled_rf_free
                0x7bdcd2fd8e08ebf6, // sm__cycles_active_3d
                0x7c3df99ac306f9f0, // sm__icc_requests_miss_to_gcc
                0x7c59366e80358abb, // lts__t_requests_gpc_concat_cas
                0x7c9e5d90d9865a94, // gpmpd__input_tasks
                0x7c9fa78ce08f28de, // lts__t_sectors_miss_host_cpu_wr
                0x7cdae65b31262492, // smsp__warps_cant_issue_tex_throttle
                0x7d45e530aefcf4d3, // lts__t_sectors_miss_l1tex_rd
                0x7d4a74fba2ad099d, // smsp__inst_exelwted_pipe_ldc_pred_on
                0x7d550d2083f2249c, // lts__t_requests_l1tex_wr
                0x7db59d10b2c29861, // tgb__cycles_active
                0x7db91450dc5ae27a, // lts__t_sectors_host_cpu_prefetch
                0x7dc7e95f8622f5b9, // fe__cycles_wfi_host_scg0
                0x7defdc59788b37e1, // lts__d_atomic_block_stalled_pending_miss
                0x7e0fdd230208b477, // smsp__warps_cant_issue_allocation_stall_1
                0x7e1285551f912949, // vaf__gpm_stalled_by_state_processing
                0x7e13e4d8c22abe52, // lts__t_sectors_hit_hub_ilwal
                0x7e76329e8c232e94, // gpc__prop_utlb_requests_sparse
                0x7e7c33e6bb96879f, // lts__t_requests_host_cpu_concat_cas
                0x7e9f98f83aca27a1, // l1tex__samp_samp2x_active
                0x7ea335f224db55c3, // prop__input_stalled_waiting_for_pixel_shader_barrier_release
                0x7eadf0e2e87c6e43, // lts__t_sectors_host_noncpu_concat_cas
                0x7ebf906bea36d92c, // lts__t_sectors_host_noncpu_condrd
                0x7ef98d99ae4eaeeb, // lts__t_sectors_host_noncpu
                0x7f5fb032b904db10, // l1tex__t_set_accesses_hit_tex_format_1d2d_array
                0x7f8b3033c97c99a9, // l1tex__t_set_conflicts_lg_global_ld
                0x7fcd3b86f57b150f, // l1tex__samp_input_quads_sz_32b
                0x7fe4af9551ea91cf, // lts__r_input_fifo_active
                0x7fe85467792e6cf8, // lts__t_sectors_miss_gcc
                0x80063c976d4d7952, // lts__t_sectors_hit_cas
                0x8041cac430bfe98c, // swdx__tc_replayer_bin_flushes_reason_clear
                0x8068a9adb85dd73c, // smsp__imc_requests_miss_no_tags
                0x809686a8db7be021, // smsp__inst_exelwted_pipe_fp16
                0x80e2aa8997c520cf, // smsp__inst_exelwted_tex_wb
                0x8132d39c7456ed20, // vpc__clip_output_attrs
                0x816d4701e1a27144, // lts__t_sectors_hit_crd_i_prefetch
                0x818611e484b85c1e, // lts__t_sectors_hit_hub_concat_cas
                0x819d2e6963b3be1b, // lts__t_sectors_iso
                0x81a666959c4f9154, // lts__t_sectors_hit_ltc_cas
                0x81e5c1a2be57a263, // vaf__alpha_fetched_attr_vector_post_cbf
                0x8214e17abeed1f9f, // lts__t_sectors_ltc
                0x826a829246b651a3, // smsp__inst_exelwted_pipe_fmai
                0x82a1b1f512dd2737, // vaf__alpha_acache_requests_acache_miss
                0x82bb72c27710cc02, // l1tex__x_x2t_stalled
                0x831e3b32a1e90d8a, // lts__t_sectors_hit_l1tex_rd
                0x8331b747bce571d7, // lts__t_sectors_hit_cbc_rd
                0x8394dde8f5d00bea, // lts__t_sectors_hit_gpc_condrd
                0x83ae943630ca9799, // raster__frstr_output_cycles
                0x83e86d249bf4f529, // smsp__inst_exelwted_pipe_fma64plusplus
                0x83f7aacdcde28827, // l1tex__t_set_conflicts_surface_st
                0x83f85d970e0e7116, // l1tex__t_t2d_active
                0x84350f16e3c39094, // lts__t_sectors_hit_cbc_wr
                0x8439e79f6b50a470, // l1tex__texin_requests_null_surface
                0x844535ecfe490359, // gpu__time_start
                0x846882949a65bab2, // lts__t_sectors_crd_i
                0x859b47fc25441b1b, // lts__cbc_requests_miss_clear_zbc_crop
                0x85daa596f9cae297, // scc__load_constants_stalled_update_scoreboard_full
                0x85e98bbd705b39c8, // lts__t_sectors_hit_zrop
                0x865841595a5f3103, // sm__miop_pq_read_active_tex_smp1
                0x86639e177eb4a35e, // lts__t_sectors_hit_host_noncpu_rd
                0x8670c43898e6b4bd, // rdm__input_data1_active
                0x8678e0409ac7f7cf, // prop__zrop_output_quads
                0x86ba9d8efb6dffc4, // fe__input_methods
                0x86e9e36dcc5bfa5e, // lts__t_sectors_hit_host_noncpu_condrd
                0x8732afcc3453ec03, // raster__crstr_output_tiles_empty
                0x882416dd00ab7db4, // lts__t_sectors_hit_cbc
                0x88944230aaa9a214, // l1tex__texin_requests_lg_global_red
                0x889823853a19316c, // cbmgr__cycles_active
                0x88d918389780b365, // lts__t_sectors_miss_hub_clear
                0x88f20e175502388a, // rdm__crop_output_active
                0x8918ef7438996c67, // lts__t_sectors_hit_ltc_rd
                0x89763832ab664daa, // lts__mccif_read_request_latency_high
                0x89b23b5ce7b878ab, // smsp__inst_exelwted_pipe_xu_pred_off_all
                0x89b2823e3b3fdf1e, // gpc__prop_utlb_stalled_on_tag_allocation
                0x89d24c8bae582878, // prop__latez_output_pixels
                0x8a01c6456c84e1bf, // sm__miop_adu_replays_smp0
                0x8a1c295e17b81e63, // vpc__lwll_lwlled_prims_reason_scissor
                0x8a29003b2eb0e8ae, // lts__cbc_requests_comptag_fill
                0x8a2b187990077013, // mpc__warp_launch_stalled_gs
                0x8a5d23962b2784fc, // crop__read_returns
                0x8a707d682ae81765, // l1tex__t_set_accesses_tex_format_1d2d_array
                0x8ada1a88f44e3b5e, // lts__t_sectors_host_noncpu_rd
                0x8ae4d82a45cea026, // crop__read_stalled
                0x8af15385a73622d0, // vpc__output_cbes
                0x8af5ed6063c96f87, // lts__t_sectors_iso_wr
                0x8b650a49eaef165f, // mpc__input_subtiles
                0x8b69100d2d1e8f67, // gcc__cycles_active
                0x8b79c8411cc64542, // lts__t_sectors_gpc_rd
                0x8bb5c6ad4425a671, // crop__input_requests_aamode_2xaa
                0x8bdf39983f3e2f3a, // l1tex__t_set_accesses_miss_tag_hit_tex_format_3d
                0x8bf5dd4cd1c48e53, // l1tex__t_set_accesses_lg_local_st
                0x8c01ec8e1005c97b, // lts__t_sectors_gcc
                0x8c241742e335b454, // zrop__write_requests
                0x8ca6bebae68b0997, // sm__threads_launched_cs
                0x8ca8aedc7015db30, // lts__t_sectors_hit_zrd_ni
                0x8cb1bba817ff6be4, // lts__t_requests_hub_cas
                0x8cfbd4e0ea2e50c4, // lts__t_sectors_crd_ni_prefetch
                0x8d1c49444a2fdc05, // lts__t_sectors_miss_zrd_ni
                0x8d87387f2939e3ed, // swdx__tc_replayer_bin_flushes_reason_cbe_slots_threshold
                0x8e04d8f2c23164a0, // mpc__warp_launch_stalled_ps
                0x8e5f8961e20cbcc0, // lts__t_sectors_prefetch
                0x8e6c6f67afb202bb, // smsp__issue_active
                0x8e6c9114370a98f4, // l1tex__texin_tsl1_requests_hit_samphdr
                0x8ea71db1aa611450, // lts__t_sectors_hit_host_noncpu_ilwal
                0x8f68bf9576380cff, // lts__d_atomic_block_stalled_same_address
                0x8f6d1fa2a0345da7, // lts__t_sectors_miss_hub_wr
                0x8f99ad7745d0f564, // lts__t_sectors_miss_host_noncpu_rd
                0x8fd47dc551ac1714, // pel__out_input_stalled
                0x8fd5e38be151b3dc, // l1tex__texin_tsl1_requests_miss_texhdr
                0x900ae17afb98dff8, // l1tex__t_set_accesses_surface_ld
                0x901961897d6546f2, // lts__ltcx_read_sectors
                0x902f50f71ee9ba3e, // zrop__processed_subpackets
                0x908a5a9023031be9, // smsp__inst_exelwted_gs
                0x9097f7cc83f1f1a2, // l1tex__samp_input_quads_tex_format_1d_tex
                0x90b5be4e2db4d097, // gpc__tpc1_utlb_requests_hit
                0x90b7c5d6e18dfe45, // lts__t_sectors_miss_host_cpu_rd
                0x90c5a6b58955475c, // lts__t_sectors_miss_ltc_concat_cas
                0x90e364762bfc0e63, // lts__t_sectors_hub_clear
                0x910aab9901911284, // vpc__output_prims
                0x911d060edcc6588d, // sm__warps_launched_tcs
                0x912e1f570883a28b, // swdx__tc_replayer_bin_flushes_reason_timeout
                0x9168e42bc61bb1a3, // l1tex__d_d2f_backpressured
                0x9175507832de89e0, // mme__cycles_stalled
                0x918d9b53e0ab32e0, // gpc__tpc0_utlb_stalled_request_fifo_full
                0x91952aa6c7bad10c, // lts__t_sectors_miss_zrd_i
                0x91b920caa82ceda5, // vpc__read_isbes
                0x91e1ce9a7fa7e74a, // lts__t_sectors
                0x91f1c8f2bcaf3eae, // lts__t_requests_mmu_wr
                0x928edc8ac4e59793, // lts__t_sectors_crop_wr
                0x92ae3d11ceb72ade, // zrop__processed_requests_type_bundle
                0x92b019350fa70e28, // l1tex__samp_input_quads_sz_128b_nearest
                0x930d82874265e258, // mpc__isbe_allocation_stalled_alpha_on_vsc
                0x931ce6683b382075, // lts__t_requests_cbc_wr
                0x9357ce58f3160a54, // lts__t_sectors_gpc_prefetch
                0x93662b675bd2ffe8, // swdx__input_prims_single_binned
                0x936cae2c239414dc, // prop__input_pixels
                0x939576cafe8fe56d, // lts__t_sectors_miss_zrop_wr
                0x93ba86467b604222, // l1tex__m_read_sectors_tex_format_1d_buffer
                0x93eb0e713e006308, // l1tex__t_sectors_miss_lg
                0x93fdfc93f794b931, // smsp__inst_exelwted_pipe_xu
                0x940b58293aca7ea4, // vpc__alpha_cycles_active
                0x940c43d0f64a2219, // lts__t_requests_host_cpu_atomic
                0x941b77475267d172, // lts__t_sectors_miss_host_cpu_atomic
                0x9420876652c763b3, // pes__stream_output_attrs
                0x9485873394af1e5f, // lts__cbc_requests_hit_clear_zbc_zrop
                0x94b8538479deb671, // prop__latez_shdz_input_pixels
                0x94eda0c2fabbd739, // sm__subtiles_launched_smp0
                0x94fdf3130a5ff7b8, // lts__t_sectors_hit_host_noncpu_membar
                0x951036752dc89573, // lts__t_sectors_miss_hub_condrd
                0x953f323fd0d95679, // lts__t_sectors_miss_iso_wr
                0x9551d44059e9feae, // swdx__tc_replayer_bin_flushes_reason_cbe_memory_threshold
                0x95615a4d6d934ad2, // gpmsd__input_pixels_fully_covered
                0x956a68a4c21cd940, // lts__t_requests_gpc_atomic
                0x956d19e4805b8d48, // lts__t_sectors_miss_clear
                0x95d5f5e4bf8ac4a7, // smsp__warps_cant_issue_imc_miss
                0x95f45230df636dab, // lts__cbc_requests_comptag_hit
                0x9631823713f833a9, // mpc__alpha_beta_mode_switches
                0x969a8981de5318dd, // l1tex__t_set_accesses_surface_red
                0x971c2144063dc931, // sys__cycles_elapsed
                0x975bcfdae3490bce, // smsp__inst_exelwted_pipe_lsu_pred_off_all
                0x978df963e704a77e, // sm__idc_requests_hit
                0x9790f201c6eb6076, // smsp__inst_exelwted_pipe_fma64plusplus_pred_off_all
                0x97a5941f575afda1, // raster__zlwll_lwlled_occluders
                0x97b164710f0d7bbf, // lts__mccif_read_request_latency_640
                0x97b1f95bd0bb8283, // l1tex__texin_requests_lg_local_st
                0x97da66ba063daed2, // lts__t_requests_hub_clear
                0x97f0fb0e42548629, // lts__t_requests_membar
                0x980778bbf890c2b6, // lts__t_sectors_host_noncpu_clear
                0x9848b3171eaffc88, // fbp__cycles_elapsed
                0x9874835db8d0bf26, // l1tex__texin_requests_null_tex
                0x9876a25e402a3dfc, // raster__frstr_output_subtiles
                0x990f9985b0ada3d0, // stri__acache_requests_stri
                0x9917f6d72a481afb, // cbmgr__beta_cbe_allocation_stalled
                0x99282399ffa78720, // l1tex__t_set_accesses_miss_tag_miss
                0x99311c374f3397b3, // sm__warps_completed_ooo_3d_vtg
                0x994ff06d8f37dda2, // gpc__gpcl1tlb_requests_miss
                0x995310b188eb1ea5, // sys__gpfifo_dwords
                0x9a28bcbb2b46510a, // sm__mios_shmem_accesses_lsu_read
                0x9a4a73f6492f46f6, // mpc__warp_launch_stalled_vsa
                0x9a633904a994a1e9, // raster__zlwll_output_occluders_trivial_accept
                0x9a69a16ed9795e76, // vaf__gpm_stalled_by_beta_input_fifo
                0x9a745c916c4dac90, // sked__cycles_active
                0x9a776c95d17fed11, // lts__t_sectors_hit_crop_wr
                0x9b37fefcbf655808, // lts__t_sectors_hit_host_noncpu_cas
                0x9b3e7753d70ea4c3, // lts__mccif_read_request_latency_448
                0x9b4193bf1c204380, // fe__i2m_write_bytes
                0x9b494cdb20e8b179, // stri__acache_requests_stri_miss
                0x9b5e21cc14d04e9d, // cbmgr__beta_cbe_allocation_stalled_no_space
                0x9b8be79e5d50e23f, // cbmgr__alpha_cbe_allocation_stalled_no_space
                0x9b8f759637376cfd, // lts__t_requests_ilwal
                0x9bbb9c9bac707504, // lts__t_sectors_hit_host_cpu_condrd
                0x9bbc093e933af3a4, // fe__output_ops_vert
                0x9bfd37ac408c2f36, // smsp__inst_issued
                0x9c05a46d423e0c1a, // l1tex__t_texels
                0x9c1a6e286a04ab8a, // sm__mios_shmem_accesses_su_read_tram_bank_conflict
                0x9c300379d0467d36, // lts__t_sectors_hit_raster_rd
                0x9c39f66b5099e54a, // lts__t_sectors_host_cpu_ilwal
                0x9c6c6951a24b04c9, // vpc__input_prims_patch
                0x9c77cb5392c7fad7, // mmu__pte_requests_small_page_3
                0x9d0263b8a149844a, // swdx__tc_replayer_bin_flushes_reason_non_binnable_line
                0x9d0a8ba3b87b6dd3, // sm__ps_quads_sent_to_pixout
                0x9d9e1c861aea2ab9, // l1tex__t_set_accesses_miss_tag_hit_tex_format_lwbemap
                0x9dc5e82116687afa, // sm__warps_active_tes
                0x9e964c7603e3e26f, // l1tex__t_set_conflicts_lg_global_red
                0x9ebb83fe43da5848, // cwd__feedback_mpc_messages
                0x9eebbb8de8ce6e42, // lts__cbc_requests_hit_clear_zbc
                0x9f30f8c1bb651773, // lts__t_sectors_hit_crop_prefetch
                0x9f6b88f2ea323ff8, // lts__t_sectors_miss_gpc_wr
                0x9f9b280f41321027, // lts__t_requests_gcc
                0x9fa52c8324f5ac59, // raster__frstr_output_subtiles_2d
                0x9fcdf60643ef31af, // lts__t_sectors_miss_host_noncpu_ilwal
                0x9fe2486047565627, // l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_array
                0xa0013c35fc7e3af2, // lts__t_sectors_crop_ilwal
                0xa008903e0e011bf0, // zrop__input_samples_part2
                0xa01719ce28b6f5f6, // prop__latez_rstrz_input_samples
                0xa01e13725126a692, // scc__load_constants
                0xa0255ce56c3ccae2, // vpc__clip_output_prims
                0xa02b84de00ffde9d, // smsp__warps_cant_issue_dispatch_stall
                0xa0554e4d089a0a41, // lts__t_sectors_cbc_rd
                0xa07158d28d185056, // lts__t_sectors_miss_zrd_i_condrd
                0xa09e504da1317125, // lts__t_requests_hub_concat_cas
                0xa0d496308394903a, // prop__zrop_output_active
                0xa1067416c7d5e46b, // prop__csb_output_quads
                0xa1485bafa273ae4c, // vaf__gpm_stalled_by_state_fifo
                0xa1a2a51d534bcebd, // scc__cycles_active
                0xa1b2e3c47cb06e47, // prop__cdp_alpha_to_coverage_killed_pixels
                0xa201dd2e244cbdd7, // lts__t_requests_host_noncpu_wr
                0xa2181d3fcea742ae, // lts__t_sectors_hit_crd_i
                0xa2377eaaeb4449d2, // l1tex__m_write_sectors_lg_global_atom_cas
                0xa25d2acb503fb13e, // crop__input_requests
                0xa2af300229f065b7, // vpc__output_prims_triangle
                0xa2f63edea093d9cb, // l1tex__t_set_accesses_lg_global_st
                0xa3090b314510f9ec, // lts__t_sectors_hit_hub_clear
                0xa31a75b0db8e3f5d, // l1tex__texin_tsl1_requests_miss_samphdr
                0xa37fea8cb26b3781, // l1tex__d_d2f_active
                0xa3ad1e3384c6d437, // fe__output_ops_bundle_scg0_go_idle
                0xa3c8c6a2e4b84358, // prop__csb_output_quads_3d
                0xa3d4855f24bae0ab, // raster__tc_output_tiles
                0xa3db65313f80abd9, // vpc__output_attrs_scalar
                0xa435ee06202fc699, // lts__t_sectors_miss_atomic
                0xa43dc2c1c4526827, // l1tex__samp_input_quads_sz_64b_nearest
                0xa472753ffaf64017, // mme__cycles_active
                0xa4780961e3d675b3, // scc__load_constants_page_same
                0xa48bee423bc735f5, // prop__input_quads_2d
                0xa4d193d685761d22, // l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_tex
                0xa4e1a0b1c427b889, // lts__t_requests_hub_atomic
                0xa4e7e575f7ad5a59, // lts__t_sectors_miss_crd_i_prefetch
                0xa519e64ba32437f1, // lts__t_sectors_crd_ni
                0xa652073ceef8c45f, // mme__output_method_dwords
                0xa68088869011d32d, // sm__threads_launched_tes
                0xa686033354b09ca6, // smsp__inst_exelwted_pipe_tex_pred_on
                0xa695f817d74b5a0e, // stri__cycles_active
                0xa6a90c9805045eb3, // lts__t_sectors_hit_mmu_wr
                0xa6b22de0c069dedf, // l1tex__texin_requests_cctlt
                0xa6ba7dc18984a458, // crop__read_requests
                0xa6d3b3e269ab8145, // mmu__pte_requests_big_page_2
                0xa6e3dc7c64e90966, // pdb__input_stalled_alpha
                0xa6f922c40fe12134, // swdx__tc_replayer_bin_flushes_reason_level_0_threshold
                0xa776cb63dadf4928, // lts__t_requests_crd_i_prefetch
                0xa7bf760293f1641b, // sm__miop_pq_read_active_lsu_smp1
                0xa7df7a39b563583d, // lts__t_sectors_hit_ltc_ilwal
                0xa800727f059c25ab, // l1tex__t_t2d_stalled
                0xa811005f7a6900c9, // lts__t_sectors_miss_ltc_membar
                0xa83c0259eb35a48e, // fe__output_ops_bundle_scg1_go_idle
                0xa851daeab55d1f0c, // lts__t_sectors_miss_concat_cas
                0xa851ffe7af4e89eb, // sm__cycles_active_tcs
                0xa86cbd6926347e26, // raster__tc_flushes
                0xa8a6e44757c73f8b, // fe__output_ops_bundle
                0xa8c20e4ba85734b6, // l1tex__m_read_sectors_tex_format_no_mipmap
                0xa914182ecb97cba0, // lts__t_sectors_miss_cbc
                0xa97b362998d5fda8, // lts__d_atomic_reqseq_input_stalled_fifo_full
                0xa9980c5bfa812985, // lts__d_cycles_bank_conflict
                0xa9f20288b10d7707, // crop__input_requests_aamode_4xaa
                0xa9fc96785ed5c658, // gcc__l15_requests_miss
                0xaa5bf40d369c74fd, // mmu__cycles_active
                0xaace511cf7a0beff, // lts__t_requests_zrop_condrd
                0xab0af8fe4b14e5fa, // l1tex__t_set_conflicts_surface_atom_cas
                0xab34cde7af2f4c1a, // pel__out_read_stalled_vaf_beta
                0xab51daf5401301a4, // raster__frstr_output_subtiles_1_cycle
                0xab86491aea8d6ff7, // lts__t_sectors_hit_hub_condrd
                0xab87f4afebce5b7a, // l1tex__t_set_accesses_miss_lg_local_ld
                0xab91fca54ccdae16, // mpc__input_prims_alpha_patches
                0xaba75991e1891d32, // crop__write_requests_compressed_2to1
                0xabaf7b35e298ade4, // swdx__input_prims_dual_binned
                0xabc2b8d7400a0074, // sm__warps_launched_vsa
                0xabf7aebb829f6edb, // mmu__pte_requests_big_page_0
                0xabf99fd2227678c4, // raster__crstr_discover_working_no_tile_stalling_setup
                0xac10176d181a51d8, // l1tex__samp_samp2x_busy
                0xac3704609b2a70e2, // prop__csb_input_quads_3d
                0xac5913e18d41780a, // crop__write_subpackets
                0xac6622ba952482df, // lts__t_sectors_hit_zrd_i_condrd
                0xac729e8f35860819, // l1tex__texin_requests_surface_atom
                0xacff7c8139f21631, // vaf__beta_to_acache_stalled_on_read
                0xad0931b910cc10ab, // lts__xbar_output_active
                0xad0e730b720f80a8, // cbmgr__alpha_cbe_allocation_stalled_max_tasks
                0xad2fba5b48dd8681, // sm__warps_retiring_ps_stalled_out_of_order
                0xad60bade184696d7, // gpmsd__input_pixels_3d
                0xad67cb63a3dd7048, // lts__t_sectors_hub_rd
                0xadd30cf82deb0673, // mpc__input_prims_beta
                0xadfd769a93367e1e, // pel__in_write_requests_stalled_vaf_beta
                0xae2cf6ca6b5b4582, // mmu__pte_requests_big_page_1
                0xae36612d24e80741, // vaf__beta_acache_requests_local_hit
                0xae384b2571c0c4dc, // l1tex__texin_requests_surface_ld
                0xae47219c08cce0f7, // lts__t_sectors_hit_gpc_prefetch
                0xae646f46f04fef2f, // l1tex__t_set_accesses_lg_local_ld
                0xaef03692a2d4ecb8, // fe__input_method_dwords_i2m
                0xaf0979801e256bb4, // tgb__output_prims_line
                0xaf5bd69489a57532, // lts__ltcx_read_stalled_fifo_full
                0xaf5de5f5e3b324f7, // lts__t_sectors_miss_zrd_i_prefetch
                0xaf62a8b264fdac84, // l1tex__t_set_accesses_hit_in_warp_surface_ld
                0xaf77cd38536af97c, // smsp__warps_launched
                0xb046826d727d2742, // l1tex__x_x2w_busy
                0xb04bb3ed8df2ac57, // lts__t_requests_ltc
                0xb06f1f91a0b5de4e, // l1tex__t_set_accesses_tex_format_1d_buffer
                0xb08dd15cfe4b9be9, // l1tex__w_w2d_backpressured
                0xb0a4f906a0a59d7f, // gpu__time_end
                0xb0e4772c8df05ad9, // lts__t_sectors_zrd_ni_condrd
                0xb10f01930a262f73, // crop__processed_samples_part2
                0xb1448e6f0b9e0ff8, // lts__t_sectors_miss_gpc_condrd
                0xb1507d1739e39398, // lts__t_requests_hub_ilwal
                0xb18cc28e894c4cce, // gpc__tpc1_utlb_requests_hit_under_miss
                0xb1919e31437cf745, // fe__output_ops_ld_const
                0xb1e2ab9f05cbfc99, // l1tex__m_read_sectors_lg_global_ld
                0xb1edf650f105644f, // vaf__alpha_cycles_active
                0xb1f57ce6815882ad, // raster__setup_input_prims
                0xb25293643c0ea950, // zrop__processed_requests_type_plane_eq
                0xb269d7a968346de3, // lts__t_sectors_miss_host_cpu_prefetch
                0xb28448f6c7718d02, // sm__mios_shmem_accesses_lsu_read_bank_conflict
                0xb2d83043cfe3c8a9, // smsp__warps_cant_issue_mio_throttle
                0xb2e65aa013a88058, // pdb__output_stalled_alpha
                0xb2f7613a01f7206c, // vaf__beta_input_task_fifo_full
                0xb2fa3f0aaa1d7496, // gcc__l15_requests_miss_constant
                0xb2fc71315ca7bb16, // zrop__write_subpackets_depth
                0xb31dae4674dbd8f9, // lts__t_sectors_miss_pe_rd
                0xb3619ee0b59b05c2, // fe__cycles_wfi_ctxsw
                0xb38c397412351b98, // lts__t_sectors_miss_crop_condrd
                0xb3939bad3ca294aa, // sm__threads_launched_ps_not_killed
                0xb3c456ed308167f3, // lts__t_sectors_hit_ilwal
                0xb3d558bf584e3879, // fe__output_ops_bundle_scg1_wfi_host
                0xb421b5d0c79129f9, // crop__input_requests_blend_blendopt_fill_over
                0xb4336808d89a3553, // gpc__gpcl1tlb_stalled_on_tag_allocation
                0xb439217ff003e6a1, // mpc__warp_launch_stalled_tes
                0xb441674702f268bf, // lts__t_sectors_miss_raster_rd
                0xb4471591c0aa199c, // vpc__lwll_cycles_active
                0xb4544b8e17175d0e, // smsp__inst_exelwted_pipe_bru_pred_off_all
                0xb495a6c8956eb03c, // gpmsd__sm2gpmsd_pixout_cdp0_active_color
                0xb4fd12bf870d7ece, // lts__t_sectors_hub_membar
                0xb512160e489c05ff, // crop__processed_samples_part1
                0xb53125cdf42271b0, // lts__t_requests_zrd_ni
                0xb57ad87dab7a4d6a, // smsp__inst_exelwted_pipe_adu_pred_on
                0xb58d07d94b29c99a, // lts__t_sectors_miss_zrd_ni_rd
                0xb5bd7359a99acafd, // tgb__output_verts_simple
                0xb664a7cfbfa0adef, // lts__t_requests_host_cpu_condrd
                0xb682757e1168ccc6, // vpc__lwll_lwlled_prims_reason_zero_area
                0xb6b549c30f0978ea, // l1tex__samp_input_quads_sz_96b
                0xb6d76964b364ddd7, // lts__t_requests_ltc_atomic
                0xb760e71c8922b4b5, // l1tex__t_output_wavefronts
                0xb7b3a44b79df562a, // lts__t_requests_iso_rd
                0xb7b53cb0a4b615df, // smsp__warps_cant_issue_long_scoreboard_0
                0xb7b908281f29c7b3, // lts__t_sectors_miss_wr
                0xb7dc70b3af86d49a, // lts__t_sectors_crop_prefetch
                0xb7f819583f082d50, // smsp__warps_cant_issue_short_scoreboard_0
                0xb807df058c2eb2d4, // lts__t_sectors_miss_cas
                0xb85f9dbb7fb96e97, // sm__warps_launched_gs_fast_beta
                0xb864d8489c252f5d, // l1tex__m_write_sectors_lg_global_st
                0xb8962fc997bc7544, // pda__input_prims_point
                0xb896ee912c8b0ce0, // prop__latez_output_quads
                0xb90b0983d2e03c33, // lts__t_requests_hub_wr
                0xb92737d020e0eb15, // smsp__inst_exelwted_pipe_bar_pred_off_all
                0xb92fcefb02b8d1b4, // raster__setup_output_prims_point
                0xb9ab5a71b4e5df03, // raster__zlwll_input_tiles
                0xb9d96f590b330ee5, // mpc__input_prims_alpha
                0xb9e4de705da84625, // lts__d_atomic_resseq_stalled_backpressure
                0xb9e5d6c152001c13, // smsp__warps_cant_issue_long_scoreboard_1
                0xba1d6516feeb41dd, // sm__threads_launched
                0xba9a992a57f8baad, // vaf__alpha_input_prims
                0xbaa7e0967dc43c31, // smsp__inst_exelwted_pipe_fxu
                0xbac3765c513a8844, // prop__input_samples
                0xbae0031440580af9, // smsp__warps_cant_issue_drain
                0xbb31aeeb4929be34, // lts__mccif_write_stalled
                0xbb5791b2f4966c5f, // vpc__alpha_batches_active
                0xbb6a6b77e6fd1687, // fe__cycles_active
                0xbb83ab344e81eb99, // lts__t_sectors_miss_raster_wr
                0xbbaba6b2a9d91085, // vaf__alpha_input_batches_pre_cbf
                0xbbbe1f6be4efbcbc, // tga__output_tasks_primary
                0xbbc2f52f36382e05, // l1tex__t_set_accesses_hit
                0xbbfb62302ab11ea1, // raster__zlwll_output_occluders_zfail_no_lwll
                0xbc05af209e10df4d, // lts__mccif_write_cycles_16
                0xbc24960efbb53f5b, // lts__t_sectors_hit_crd_i_rd
                0xbc3cbd9c5f5cfcb3, // smsp__warps_cant_issue_wait
                0xbc40fd93ae58cd4a, // gpc__rg_utlb_stalled_write_buffer_full
                0xbc417f25d9255b55, // lts__t_sectors_crop_condrd
                0xbc661c0d2f60c129, // lts__t_sectors_hit_zrop_wr
                0xbc9a4a734d5bd5e6, // l1tex__samp_input_quads_filter_aniso
                0xbcaeca9daacf9b4d, // lts__t_sectors_gpc_membar
                0xbcf68625e179b7d4, // lts__mccif_read_request_latency_384
                0xbd2f0064c2ded5f7, // lts__t_sectors_miss
                0xbd986c53ba5f7f50, // lts__t_sectors_host_cpu_rd
                0xbda86d7febfc438b, // smsp__inst_exelwted_vsb
                0xbdc982ea3aa070ea, // smsp__warps_cant_issue_misc_1
                0xbdca4e99ab2f6899, // gpc__prop_utlb_requests_hit
                0xbde626b9ba8410bb, // gpc__prop_utlb_requests_miss
                0xbe1f5b4404f9f1a1, // smsp__inst_exelwted_cs
                0xbecdea42c2b591df, // vaf__alpha_batches_active
                0xbef9e2b8fd6f3c7d, // l1tex__t_set_accesses_cctlt
                0xbf2aa0d728a90840, // l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_array
                0xbf3350f1d997af51, // mmu__pde_requests_miss
                0xbf7c8478c221ac2a, // lts__mccif_read_cycles_32
                0xbf8000616662e481, // lts__t_requests_gpc_cas
                0xbfa05d5a54aae2dd, // lts__t_requests_host_noncpu_concat_cas
                0xbfa54b39565f9eb5, // lts__t_sectors_miss_gcc_rd
                0xbfbc998fc7b584eb, // lts__t_requests_mmu_rd
                0xbfc750850c0114dc, // wwdx__input_cbes
                0xbfc7cdf2725bcd3f, // pda__input_prims
                0xbff8c169585a3447, // stri__acache_requests_stri_hit
                0xc01df2c8fdb922b8, // lts__t_sectors_miss_host_cpu_ilwal
                0xc02b37e92d603ed9, // vpc__clip_output_verts
                0xc04ebb5ea6a601ec, // mpc__tram_startxy_fifo_stalled
                0xc0a1086622701f4c, // l1tex__m_write_sectors_lg_global_red
                0xc0ed933f9a8fb830, // lts__t_sectors_miss_prefetch
                0xc1018b1cc909fccc, // pda__input_prims_triadj
                0xc1115d3a4a5ba283, // crop__processed_requests_uncompressed
                0xc1831233f05d1ee9, // lts__t_sectors_atomic
                0xc190ae9185457490, // lts__t_sectors_clear
                0xc1a64cfa27e3a935, // vpc__lwll_lwlled_prims_reason_frustum
                0xc1e672f4c446b97a, // pel__in_write_requests
                0xc20e9d414ff845a6, // lts__t_requests_zrop_prefetch
                0xc227626dd899e179, // lts__t_sectors_wr
                0xc2581a68907788a4, // crop__input_samples_part2
                0xc2a05687d091fa3e, // l1tex__m_read_sectors_surface_atom_cas
                0xc2bc507dfbcc0d9a, // vpc__lwll_lwlled_prims_reason_rotated_grid
                0xc2ca931395b51610, // l1tex__t_set_accesses_hit_in_warp_lg_local_st
                0xc2dd4d1ee3c6a6a6, // lts__t_requests_host_noncpu_membar
                0xc356004d92dbb0e3, // sm__mios_datapath_active
                0xc36e48b6e3f51623, // lts__t_requests_l1tex_atomic
                0xc383777e4bcc64d0, // mme__input_methods_shadow_filtered
                0xc38fddeeaeeb4f11, // lts__t_requests_zrd_ni_prefetch
                0xc3978df8aa6df8b5, // lts__t_sectors_zrd_ni_rd
                0xc3dbd0775c1ee14e, // zrop__zlwll_cycles_active
                0xc3e27ae6fc88e1c5, // l1tex__d_cycles_bank_conflict_2x2
                0xc41c3fbc62b3fbd3, // l1tex__t_set_accesses_hit_tex_format_1d_buffer
                0xc4345bb767ecc3fd, // lts__t_sectors_miss_host_noncpu_wr
                0xc47511bb42dfd87e, // lts__t_sectors_miss_niso_wr
                0xc489f4df1cee62d2, // mpc__cycles_elapsed_beta
                0xc4ac2a600b8a7ae1, // l1tex__m_write_sectors_lg_local_st
                0xc4af129284839418, // vpc__lwll_lwlled_prims_point
                0xc4c008eba731764b, // vaf__alpha_fetched_attr_vector_pre_cbf
                0xc4cac6755efe8fc4, // gpc__gpcl1tlb_requests_hit_under_miss
                0xc4cb01d9bb4b4663, // gr__cycles_active
                0xc4cf6ecb21466829, // lts__t_requests_ltc_membar
                0xc540c967d3684d13, // pda__output_batches
                0xc5ccee2cb5a28048, // zrop__write_subpackets_coalesced
                0xc5dbf68130d5ec75, // lts__t_sectors_hit_host_cpu_rd
                0xc5f7b04dd166c243, // smsp__inst_exelwted_tcs
                0xc618da31d3dcc719, // l1tex__t_set_accesses_hit_in_warp_lg_global_st
                0xc67aa264235c4ae0, // lts__t_sectors_miss_pe_wr
                0xc685f523bc20b362, // lts__t_sectors_host_cpu_condrd
                0xc6dbfb73c845838e, // sm__warps_active_ps
                0xc709d78a44aaec73, // lts__t_sectors_zrop_wr
                0xc7213d30af89ca88, // fe__output_ops_bundle_scg0_wfi_host
                0xc7304eedad835776, // stri__cycles_busy
                0xc777d57708b338bb, // tga__output_tasks_complex_interior
                0xc78fc13d9c71ac16, // sm__inst_exelwted_pipe_adu_divergent_smp1
                0xc7f98d3e46708aaf, // lts__t_sectors_miss_ltc
                0xc7fdcde8d627563a, // smsp__tex_write_back_active
                0xc80afbbcc95dacf3, // rdm__zrop_output_active
                0xc813a443bd74fb32, // lts__cbc_requests_comptag_writeback
                0xc83c3fa8e952fd31, // smsp__cycles_active
                0xc84d64dad9c597b6, // tga__input_isbes
                0xc87df455043ae3c6, // lts__t_requests_cbc_rd
                0xc885d5a129ffeeb8, // lts__t_requests
                0xc8c2ba28a833f22e, // lts__t_sectors_zrd_i_prefetch
                0xc8e3c3e00c5bb580, // vpc__clip_input_prims_clipped_single_plane
                0xc8ee407239d191d9, // vaf__beta_cycles_stalled_on_mpc
                0xc8f5132459391741, // swdx__tc_replayer_bin_flushes_reason_non_binnable_state
                0xc9071291fc9159a6, // lts__t_sectors_miss_host_noncpu_condrd
                0xc99dc9675c17957e, // lts__t_sectors_hit_host_noncpu_atomic
                0xc9e5791793ce02ad, // l1tex__m_read_sectors_lg_global_atom
                0xca09bb1b61b3ba87, // lts__t_sectors_miss_zrd_ni_prefetch
                0xca2a684e9bdf2b08, // lts__t_tag_requests_miss
                0xcaa81c971b40596c, // lts__t_sectors_hit_crop
                0xcab81f4de987ea7c, // lts__t_requests_ltc_ilwal
                0xcac287a5bb0804e2, // lts__t_sectors_gpc_cas
                0xcae54faa9838e95c, // lts__t_sectors_miss_host_noncpu_atomic
                0xcae880ceb516871f, // lts__t_sectors_miss_host_noncpu
                0xcaf9b0725d103e9d, // l1tex__t_set_conflicts_tex_trilinear
                0xcb0e3e7f6d4e4558, // sm__cycles_active_gs
                0xcb1aa07c9db2e30d, // crop__cycles_active
                0xcb32f8bad2cdbab9, // l1tex__t_set_accesses_miss_tag_miss_tex_format_3d
                0xcb4c33e661d03761, // prop__input_samples_3d
                0xcbf60d9c418dcf3d, // vpc__clip_cycles_stalled
                0xcc30b26e33fa659b, // lts__t_sectors_hit_l1tex_atomic
                0xcc42b5a2665baab2, // scc__load_constants_stalled_no_gcc_credits
                0xcc89902c6e450416, // lts__t_requests_crop
                0xcccf579072b5a706, // lts__t_requests_hub
                0xccd8b50882bed32e, // gpc__tpc0_utlb_stalled_write_buffer_full
                0xccf53c2e80cd8c07, // raster__frstr_killed_subtiles_output
                0xcd0d49182c9b3a2a, // lts__t_requests_zrd_i_rd
                0xcd69a5b855f4f541, // lts__t_sectors_miss_host_noncpu_cas
                0xcd6e052297de14f9, // smsp__warps_cant_issue_selected
                0xcd813d7fbb42a991, // lts__t_sectors_hit_hub_atomic
                0xcd847e667f652cd5, // lts__t_sectors_hit_crd_i_condrd
                0xcd972c0409190b48, // lts__t_requests_hub_membar
                0xcdb5e127468fd6ad, // gpmpd__output_stalled_task
                0xce711f90199f3946, // mpc__isbe_allocation_stalled_beta_on_vsc
                0xceb3993cda892343, // lts__t_sectors_miss_host_noncpu_clear
                0xceb8f82cc7d6028e, // fe__cycles_wfi_subch_switch_scg0
                0xcebc92b8f9757729, // lts__t_sectors_miss_ltc_cas
                0xcec6693871d0ad40, // l1tex__samp_pre_nop_wavefronts
                0xcf7d4f426cbc6ca4, // prop__csb_output_samples_3d
                0xcf93f0f432faa3de, // l1tex__m_read_sectors_surface_atom
                0xcfdb6f1abdd0d8c5, // l1tex__t_atomic_address_conflicts_surface_atom_cas
                0xcfeefd2e7e67bbd2, // l1tex__texin_requests_lg_local_ld
                0xd0202d23444bea50, // pdb__input_tasks
                0xd021ad743db1dc67, // sm__subtiles_launched_smp1
                0xd0d3cf6bbad1f3b6, // l1tex__d_d2f_busy
                0xd0d5fe5d08b7c8a6, // sm__icc_prefetches
                0xd0ec5f70d3240ac3, // lts__d_atomic_block_stalled_priority
                0xd0f3223f49266c2d, // l1tex__texin_sm2tex_active
                0xd108899736a28d7a, // swdx__tc_replayer_bin_flushes
                0xd141071985eff47f, // l1tex__t_set_conflicts_lg_global_st
                0xd15f26b634c986c8, // lts__t_sectors_miss_hub_rd
                0xd174a71f9e971492, // raster__frstr_output_subtiles_fully_covered
                0xd19334cc68d13432, // sm__warps_active_cs
                0xd1a75bb30e3308f2, // vaf__beta_input_patches
                0xd1ac25323a1b678b, // lts__t_sectors_host_noncpu_prefetch
                0xd1d084442eb276c5, // raster__zlwll_lwlled_occluders_stencil
                0xd21afa5475c5c81a, // smsp__inst_exelwted_pipe_fe
                0xd23adc6f7715ab79, // l1tex__texin_requests_lg_global_atom
                0xd30717731f097b9e, // lts__t_sectors_gpc
                0xd32a72312b2c0b51, // l1tex__t_set_conflicts_lg_local_st
                0xd35af832cac67f84, // swdx__tc_replayer_bin_flushes_reason_z_mode_transition
                0xd3710e387404738e, // smsp__thread_inst_exelwted_pred_on
                0xd3757d8c2f57843c, // gpmsd__sm2gpmsd_pixout_cdp1_active_shdz
                0xd38b3a65f1bc047f, // lts__t_sectors_miss_crd_i_rd
                0xd391a628fb8064ea, // gpc__gpcl1tlb_requests_hit
                0xd397075225627d7c, // raster__tc_input_transactions
                0xd3f4bb80510347f4, // lts__t_sectors_miss_hub_concat_cas
                0xd3fa102bb9e6b5ac, // lts__t_sectors_miss_ltc_prefetch
                0xd4062fd6644af8fa, // lts__t_requests_pe_wr
                0xd43accbdfc963dc4, // vaf__beta_to_acache_stalled_on_tag_allocation
                0xd47128715a0a0fac, // host__chsw_switches
                0xd489c8a2edd1c4bc, // pdb__input_stalled_beta
                0xd48ef93bc8688804, // gpc__tpc0_utlb_stalled_on_tag_allocation
                0xd4f03ac8a518cafb, // prop__csb_output_crop_requests_color
                0xd51199f7e78e7259, // lts__t_sectors_miss_crd_ni_rd
                0xd51f7aefc94c8c80, // smsp__warps_cant_issue_no_instructions_1
                0xd533bf1a902d1e1b, // lts__t_sectors_l1tex
                0xd59f18e6c8b52426, // lts__t_requests_crop_rd
                0xd5c2f38eff794215, // lts__t_sectors_hit
                0xd5df607ad009a72d, // mpc__cycles_active_beta
                0xd64571f75525a473, // pel__out_l2_requests
                0xd650e2aa026d99c6, // sm__miop_ldc_replays_smp1
                0xd65963aa63a41876, // vaf__gpm_stalled_by_alpha_input_fifo
                0xd659db620fea6fe1, // crop__input_requests_blend_blendopt_read_avoid
                0xd6721435869baf4f, // lts__t_requests_pe
                0xd67f3780c537a58a, // smsp__inst_exelwted_pipe_adu_pred_off_all
                0xd6f46c5384c285ae, // vaf__alpha_input_verts
                0xd7016700c986728f, // lts__t_requests_zrd_ni_rd
                0xd70e2d762ce589cf, // tga__isbes_active
                0xd72aafb1a78526fb, // vpc__lwll_lwlled_prims_reason_diamond_exit_rule
                0xd74e55805b0470c5, // raster__zlwll_input_occluders
                0xd7f6fa62a82fd6ae, // l1tex__x_x2t_active
                0xd834d09b3603b584, // lts__t_sectors_miss_host_noncpu_concat_cas
                0xd83a471d223f6d5e, // lts__t_requests_crd_ni_rd
                0xd85194f87ebab807, // l1tex__texin_requests_tex
                0xd872db5886561920, // smsp__inst_exelwted_pipe_ldc_pred_off_all
                0xd8b6982aa6f9f242, // zrop__write_subpackets
                0xd8ea53404d2836ad, // lts__t_requests_host_cpu_wr
                0xd9551d6473b6ac1c, // smsp__inst_exelwted_pipe_su_pred_on
                0xd9596e1d53103a9d, // prop__input_pixels_2d
                0xd9d0a9da5f6f0ce9, // lts__mccif_write_request_latency_24
                0xd9d8167e32742214, // lts__t_sectors_cbc_wr
                0xda1a05f3d6f835d1, // l1tex__texin_requests_membar
                0xda40794b46389f71, // lts__d_decomp_processed_sectors
                0xda4f9e7792f92201, // lts__t_sectors_miss_crop
                0xda56ca7e0e1a2e5d, // lts__t_sectors_miss_gpc_cas
                0xda5bed00d39e8364, // crop__write_requests_compressed_8to1_or_fastclear
                0xda74ad9c51aa8d53, // prop__input_samples_2d
                0xdaa39b32eeefefc4, // l1tex__texin_sm2tex_backpressured
                0xdb056c0a554057ba, // gpmpd__input_batches
                0xdb63a5a6410fce2b, // gpc__rg_utlb_requests_hit_under_miss
                0xdba2a89ee823b108, // lts__t_sectors_miss_niso
                0xdbb896c5b2452894, // gpc__tpc1_utlb_requests_miss
                0xdbd30e147c730f11, // lts__t_requests_zrd_i
                0xdbd8b40aca621095, // mmu__pde_requests_hit
                0xdbea9684b75c975b, // vpc__input_prims
                0xdc20b72e37d6d86f, // pel__in_write_requests_stalled_tg
                0xdc23f1759e28697f, // mpc__cycles_active_alpha
                0xdc3a790b206b37bb, // mmu__pte_requests_hit
                0xdc4bab27f3c27a78, // sm__ctas_active
                0xdc6f71f3ea4d8468, // lts__d_atomic_resseq_stalled_output_arbitration
                0xdcd927e7b2a1f6bd, // lts__t_sectors_hit_pe
                0xdcfb99171ce5a904, // lts__t_sectors_miss_host_cpu_concat_cas
                0xdd424a448db1c72f, // gpmsd__output_quads
                0xdd81c1f86d5e0a0e, // l1tex__texin2m_fifo_input_stalled
                0xdda12a3d369e9a76, // l1tex__x_output_wavefronts
                0xde01eb4729c1c81b, // sm__idc_requests_miss_covered
                0xde457b353c34b507, // lts__t_sectors_miss_l1tex_wr
                0xde560bd49270adca, // lts__t_sectors_hit_host_noncpu_concat_cas
                0xde67489fdfda5023, // lts__t_sectors_hub_cas
                0xde6807ef260ac54a, // lts__t_sectors_miss_crd_ni_condrd
                0xde7232cfb43237c7, // fe__input_methods_stalled
                0xdeb4176e4b27b419, // pda__cycles_active
                0xdec92a5c0c7f9d34, // sm__mios_shmem_accesses_lsu_write_bank_conflict
                0xdecdc240cd3ce616, // lts__mccif_read_requests
                0xded1353399ed2edc, // smsp__imc_requests_hit
                0xdf90a5b7d5b780d1, // gpc__tpc1_utlb_stalled_request_fifo_full
                0xdf989e6da6087f2e, // l1tex__t_set_accesses_hit_tex_format_1d2d_tex
                0xdfadc6fec51d3b02, // l1tex__m_read_sectors_lg_local_ld
                0xdfd74cb9ffdede26, // lts__t_sectors_hit_host_cpu_ilwal
                0xe06729a769f78805, // swdx__output_barriers
                0xe06f71f21ab3d8a6, // lts__t_sectors_l1tex_wr
                0xe07e4b3063c24bbd, // vaf__beta_acache_requests_acache_miss
                0xe088eb5a8bad2795, // lts__t_sectors_miss_gpc_rd
                0xe08e1ea5df38cbd0, // l1tex__samp_input_quads_filter_bilinear
                0xe08f89e6603353b0, // vpc__lwll_cycles_stalled
                0xe0e7b4ebad0ab985, // lts__t_requests_ltc_wr
                0xe10decc2feeb207b, // swdx__tc_replayer_bin_flushes_reason_state_full
                0xe11e0d6e419fd1ad, // raster__crstr_discover_working_no_tile_output
                0xe1aaa32b570fcb6a, // smsp__inst_exelwted_pipe_fma64plusplus_pred_on
                0xe2083da958fd65b9, // cbmgr__alpha_cbe_allocation_stalled
                0xe2621f39a91d412b, // gpmsd__sm2gpmsd_pixout_cdp1_active_color
                0xe2806d146baa30a5, // lts__t_sectors_hit_gpc_cas
                0xe28706b39cf04f58, // pel__cycles_active
                0xe2a00ef5f8c352c6, // lts__t_sectors_crd_i_rd
                0xe2a7e472575820ae, // lts__t_requests_zrd_i_condrd
                0xe2e0555210f43286, // wwdx__input_prims
                0xe2f17f8780c22e61, // lts__t_sectors_miss_ltc_ilwal
                0xe2f9f1626bdd6755, // prop__csb_output_pixels_3d
                0xe357b438918e71ab, // lts__t_sectors_hit_ltc_wr
                0xe3a5e1c4c84520e8, // l1tex__samp_output_wavefronts
                0xe3c90f63bec5e243, // lts__t_sectors_miss_niso_rd
                0xe3f0b76746b978c2, // l1tex__t_sectors
                0xe41d225f1f30601a, // lts__t_sectors_l1tex_rd
                0xe442e3247c05ba56, // pel__out_l2_requests_write_256b
                0xe46e451e0ac3632d, // zrop__read_subpackets
                0xe4b08cf13393f6f4, // lts__t_sectors_hit_crop_ilwal
                0xe4bcdc9dda138612, // lts__t_requests_crop_condrd
                0xe4ccbb342dff4dae, // mpc__warp_launch_stalled_vtg
                0xe4e77a69aaaf5969, // tga__batches_active
                0xe51c4adf2287ab69, // zrop__input_requests_containing_stencil
                0xe55b6fd68b49358d, // raster__zlwll_lwlled_occluders_depth
                0xe587eb4ab2509dab, // gpu__draw_count
                0xe588514b91d0829d, // lts__t_sectors_hit_wr
                0xe62d1e98cd19f830, // mpc__input_verts_beta
                0xe6645e16be16b844, // l1tex__t_set_accesses_hit_in_warp_surface_st
                0xe6e8255c4eaff0cc, // prop__latez_rstrz_input_pixels
                0xe6f1852be334d4d5, // smsp__warps_cant_issue_barrier
                0xe71c8bf3c44e887e, // scc__load_constants_page_pool_allocations
                0xe72a1bacfdedae58, // l1tex__t_set_accesses_surface_atom_cas
                0xe73926cd958e5d13, // l1tex__d_cycles_bank_conflict_bilerp
                0xe74d929c972be17d, // l1tex__t_atomic_address_conflicts_lg_global_atom
                0xe752e39423d64a01, // vaf__alpha_fetched_attr_scalar_pre_cbf
                0xe771f7de70d0e375, // lts__t_sectors_hit_gpc_concat_cas
                0xe7d42dfaa6195b69, // sys__fb_write_dwords
                0xe7f652f5e6d226b6, // vpc__lwll_lwlled_prims_reason_zero_length
                0xe803a16e97efe6bc, // vpc__write_sectors
                0xe8178cb790ba3c71, // crop__input_requests_aamode_1xaa
                0xe8c0100a80b1d2ea, // mmu__pte_requests_small_page_2
                0xe8ed6cea2cd11424, // lts__t_sectors_hit_condrd
                0xe91929ce43255ee5, // scc__input_state_changes
                0xe9330ef7f39efd3f, // crop__processed_requests
                0xe9a491d200ec0c4e, // lts__t_sectors_miss_ltc_condrd
                0xe9b8f6e3ba18a8f2, // lts__t_sectors_miss_crop_rd
                0xe9d0c23e87bb1b83, // prop__zrop_output_stalled
                0xe9df2a5d0d249a49, // sm__warps_retiring_ps_stalled_backpressure
                0xea3ecd3d4a834a58, // crop__input_requests_3d
                0xea6f3fbe270d1aa8, // sm__icc_requests_hit
                0xea75e5898b025f9d, // l1tex__t_set_conflicts_lg_local_cctl
                0xea9510f1f31c1a93, // l1tex__t_set_accesses_miss_tag_hit
                0xead2e717711e73cd, // lts__t_sectors_crop_rd
                0xeb157bcd9495415a, // vpc__lwll_lwlled_prims
                0xeb4981e5db4d7ef4, // vpc__input_prims_line
                0xeb8af93cf3dda2cc, // tga__output_tasks
                0xec218e00e2419906, // lts__t_requests_crd_ni_condrd
                0xec636b60c24def6b, // sm__miop_ldc_replays_smp0
                0xec643447095d5ab1, // smsp__warps_cant_issue_membar
                0xec83faa97fbccda5, // pes__stream_output_verts
                0xec89aa350a0e1f4c, // gpmsd__sm2gpmsd_pixout_cdp0_active_shdz
                0xecae2fb31e54df64, // lts__t_requests_host_cpu_rd
                0xed03e05210a199a9, // l1tex__m_read_sectors_lg_global_atom_cas
                0xed0dcf3796e46e12, // lts__t_sectors_host_cpu_wr
                0xed46cfeae5515471, // mpc__input_tasks
                0xed90306e7d3e7e4b, // swdx__output_prims
                0xeddf30c82a10664a, // prop__input_quads_3d
                0xee2c1c012c9d57fa, // mpc__cycles_elapsed_alpha
                0xee32020ead73eb4b, // lts__t_requests_crd_i_rd
                0xee4143c8eb31b151, // l1tex__texin_requests_lg_global_atom_cas
                0xee5c029591bad687, // lts__t_sectors_iso_rd
                0xee7f3309e3e157d9, // lts__t_sectors_host_noncpu_cas
                0xee8c5f6fc0fb8645, // swdx__binner_active
                0xeee6c2143f335074, // sm__warps_active_vtg
                0xeef22f533015de24, // zrop__input_samples_part1
                0xef00250e3e2568a9, // swdx__tc_replayer_bin_flushes_reason_bin_memory_threshold
                0xef7a25094411419b, // lts__t_sectors_miss_gpc_atomic
                0xef80b9a7a3c1ec62, // raster__frstr_processed_3_edges
                0xf0383d7154090996, // l1tex__t_set_accesses_lg_global_cctl
                0xf0a02737ea14afc1, // pel__out_read_stalled_stri
                0xf0b14eedc4020660, // l1tex__samp_input_quads_sz_128b
                0xf12517b659646648, // lts__t_requests_gpc_wr
                0xf1376d2bcbf65771, // mme__call_count
                0xf161e50d15b6d74d, // lts__t_sectors_hit_iso
                0xf1672876e1100c47, // tga__input_batches
                0xf1c922d29601b582, // crop__write_stalled
                0xf1e7d87ee6185992, // lts__t_sectors_miss_mmu
                0xf229adbd047e343b, // smsp__inst_exelwted_pipe_bar
                0xf23c44cb7879955b, // lts__t_requests_host_cpu_cas
                0xf27a1ef3954e2733, // lts__t_sectors_hit_raster
                0xf2d5074352ae91e3, // l1tex__samp_input_quads_sz_64b
                0xf2f2e68f592b5007, // lts__t_sectors_gcc_rd
                0xf31e512d663932f4, // lts__t_sectors_crop
                0xf3365abcc09a40da, // lts__t_requests_gpc
                0xf35addef8236a682, // lts__t_sectors_hit_host_cpu_concat_cas
                0xf3603af578007eb4, // zrop__input_requests_expanded_to_samples
                0xf36a4eeb64b1afab, // lts__t_sectors_hit_zrd_ni_rd
                0xf37edf6af03c45f2, // sm__mios_shmem_accesses_su_read_tram
                0xf3890b531098ea64, // gpc__rg_utlb_requests_sparse
                0xf3896e507a12df65, // vaf__alpha_cycles_stalled_on_mpc
                0xf39311b1bbe87658, // lts__t_sectors_zrd_i
                0xf3bbff58aeaaa834, // l1tex__texin_requests_lg_global_ld
                0xf4516c3a0986613d, // mpc__isbe_allocation_stalled
                0xf45544dc5a1e6415, // prop__zrop_output_samples
                0xf475b7b3db425c42, // prop__earlyz_output_quads
                0xf4790b330eb98183, // lts__t_sectors_hit_crop_rd
                0xf47c608220f58f08, // gpc__tpc1_utlb_requests_sparse
                0xf53a2e6da1d3c0c7, // lts__t_requests_niso
                0xf54489465a12fbf7, // crop__write_requests_compressed_1to1
                0xf55aacc2501016e1, // l1tex__f_output_tex2sm_backpressured
                0xf57e10dbeea070de, // lts__t_sectors_hit_niso_rd
                0xf5887cba6695c953, // crop__input_requests_pitch_linear
                0xf6077c6404fd8ee8, // lts__t_sectors_hit_gpc_atomic
                0xf64bc3184577c089, // lts__t_sectors_hit_host_cpu
                0xf650ee53fdec33cd, // l1tex__f_output_tex2sm_busy
                0xf661bec0d8508c25, // lts__t_sectors_miss_hub_prefetch
                0xf674b23b690a3001, // lts__t_sectors_hit_clear
                0xf6b1438196b42349, // lts__t_requests_atomic
                0xf6cf19b80ac9ab46, // lts__mccif_write_requests
                0xf6de119ed816ba79, // pel__out_l2_requests_ilwalidate_256b
                0xf7c917ff6b30c98d, // lts__mccif_read_stalled
                0xf83d1a4c595b3d88, // lts__t_requests_crd_i_condrd
                0xf8b74806d9a6f036, // lts__t_requests_zrop_ilwal
                0xf8bd2799b6777bb5, // swdx__input_stalled
                0xf8d4bdc21c61252d, // l1tex__t_set_accesses_hit_in_warp_lg_local_ld
                0xf90917cef71b5622, // lts__t_sectors_miss_ltc_rd
                0xf91858732d7eeabe, // lts__t_sectors_gpc_clear
                0xf95ef7b041238f9a, // lts__d_atomic_block_stalled_backpressure
                0xf9f6ae5c5bf79a58, // l1tex__t_set_accesses_hit_in_warp_lg_global_ld
                0xfa15508c76c84966, // lts__t_requests_cbc
                0xfa73feacf6c4ba9f, // lts__t_sectors_host_cpu_cas
                0xfaa60e0a6ee2fffd, // pel__in_write_requests_active
                0xfac490dd4366ac57, // zrop__input_requests_write_compression_ok
                0xfaf6e584a1ed6124, // lts__t_sectors_hit_mmu
                0xfb08be11d68c7f71, // l1tex__t_set_accesses_lg_global_red
                0xfb186c27dce65f25, // mpc__isbe_allocation_stalled_beta
                0xfb20d07776b4c7ba, // l1tex__f_output_tex2sm_active
                0xfb669b6ede852742, // pda__input_restart_indices
                0xfb836e74765afa0d, // lts__t_requests_gpc_clear
                0xfb8616e58f505ca7, // lts__t_sectors_hub_concat_cas
                0xfb89ea10ab417822, // lts__d_decomp_input_sectors
                0xfbe9e897fe808741, // sm__threads_launched_vsa
                0xfbf021181535f6c5, // l1tex__t_t2d_backpressured
                0xfc06ceafc63a95bc, // l1tex__t_set_accesses_lg_global_ld
                0xfc7e958326f16afb, // lts__t_sectors_hit_crop_condrd
                0xfce0d8f7a6937f7b, // zrop__processed_requests_type_plane_eq_fast
                0xfcf91bd531865131, // tgb__tasks_active
                0xfd47092654281ec2, // lts__t_sectors_hit_hub_prefetch
                0xfd57232c50596865, // sm__cycles_active_vsb
                0xfd604195af9c8506, // prop__csb_output_pixels
                0xfd6189be511227ab, // sm__warps_completed_ooo_3d_ps
                0xfdc5edccfe957879, // prop__csb_killed_quadquads
                0xfdf6b10fc570ae48, // lts__t_sectors_membar
                0xfdf945569fc825fc, // vpc__beta_cycles_active
                0xfe4102e42df4fa43, // lts__t_sectors_l1tex_atomic
                0xfe6ddce29af7bf6a, // vaf__cycles_stalled
                0xfe8a85530a412507, // lts__t_sectors_hit_pe_ilwal
                0xfeacbdb2817151d4, // lts__t_sectors_hit_hub_membar
                0xfebc0ec229a69a4d, // crop__input_requests_blend_blendopt_killed
                0xfee1144ae96f0469, // stri__attrs
                0xff1abaf51bb645e5, // l1tex__t_set_accesses_hit_tex_format_lwbemap
                0xff40a3ad70ac21d1, // tga__cycles_active
                0xffc36d4d02d85412, // l1tex__t_atomic_address_conflicts_lg_global_red
                0
            };
            return s_rawMetricIds;
        }
        static const char* const* GetRawMetricNames()
        {
            static const char* const s_rawMetricNames[] = {
                "gpmsd__input_quads_3d",
                "prop__earlyz_killed_samples",
                "tgb__output_verts_complex_boundary",
                "lts__mccif_read_request_latency_320",
                "gpmsd__input_quads",
                "lts__t_requests_zrd_ni_condrd",
                "crop__read_subpackets",
                "vpc__clip_cycles_active",
                "sm__cycles_active_3d_vtg",
                "mpc__isbe_allocations_beta",
                "crop__input_samples_part1",
                "lts__t_sectors_hit_niso_wr",
                "l1tex__m_write_sectors_surface_red",
                "lts__t_sectors_miss_l1tex_atomic",
                "lts__t_sectors_ilwal",
                "lts__t_sectors_miss_crd_i_condrd",
                "lts__mccif_write_sectors_excluding_16",
                "sm__cycles_active_vs",
                "lts__t_sectors_miss_gpc_concat_cas",
                "gcc__tsl2_requests",
                "smsp__inst_exelwted_pipe_su_pred_off_all",
                "lts__t_sectors_hit_raster_wr",
                "lts__t_sectors_miss_hub_membar",
                "swdx__cycles_active",
                "lts__t_sectors_hit_ltc",
                "vpc__input_prims_triangle",
                "lts__t_sectors_hit_atomic",
                "sm__mios_shmem_accesses_pe_read_isbe",
                "gpmsd__input_samples_2d",
                "gpmsd__input_active_3d",
                "l1tex__t_set_accesses_hit_tex_format_3d",
                "lts__t_requests_ltc_cas",
                "sm__ctas_launched",
                "mpc__isbe_allocations_alpha",
                "l1tex__m_read_sectors_tex_format_lwbemap",
                "smsp__inst_exelwted_pipe_lsu",
                "lts__t_requests_wr",
                "lts__t_requests_crop_prefetch",
                "swdx__tc_replayer_bin_flushes_reason_pagepool_full",
                "mmu__hubtlb_requests_hit_under_miss",
                "l1tex__samp_samp2mipb_active",
                "lts__t_requests_iso_wr",
                "smsp__warps_cant_issue_no_instructions_0",
                "sm__ps_quads_launched",
                "pda__input_prims_patch",
                "lts__t_requests_host_cpu_membar",
                "lts__t_requests_host_noncpu_rd",
                "sm__ps_warps_killed",
                "vaf__beta_acache_requests_acache_hit",
                "lts__t_sectors_miss_zrop_condrd",
                "lts__t_requests_gcc_rd",
                "lts__t_sectors_hit_zrop_prefetch",
                "lts__t_sectors_zrd_ni",
                "cwd__output_ctas",
                "gcc__l15_requests_hit",
                "lts__t_sectors_miss_membar",
                "crop__input_requests_all_color_channels",
                "l1tex__t_set_conflicts_surface_ld",
                "swdx__tc_replayer_bin_flushes_reason_drain_timeout",
                "lts__t_requests_iso",
                "smsp__inst_exelwted_tex_wb_pending",
                "smsp__warps_cant_issue_allocation_stall_0",
                "vaf__beta_cycles_active",
                "zrop__read_requests",
                "mme__input_method_dwords",
                "vpc__beta_tasks_active",
                "pda__input_prims_triflat",
                "lts__t_sectors_ltc_clear",
                "mmu__hubtlb_stalled_on_tag_allocation",
                "zrop__read_subpackets_stencil",
                "lts__t_sectors_ltc_ilwal",
                "prop__earlyz_killed_quads",
                "l1tex__t_set_accesses_hit_lg_local_ld",
                "sys__fb_read_dwords",
                "lts__mccif_read_request_latency_576",
                "gpu__dispatch_count",
                "lts__t_sectors_miss_zrop_ilwal",
                "sm__mios_shmem_accesses_lsu_write",
                "gpmsd__input_active_compute",
                "lts__t_sectors_hit_host_cpu_membar",
                "sm__threads_launched_gs",
                "mpc__warp_launch_stalled_gs_fast_beta",
                "lts__t_sectors_miss_pe",
                "raster__zlwll_lwlled_occluders_near_far_clipped",
                "lts__t_sectors_hit_ltc_atomic",
                "lts__t_sectors_hit_host_noncpu_clear",
                "gcc__l15_requests_hit_constant",
                "lts__mccif_read_request_latency_256",
                "sm__miop_pq_read_active_smp0",
                "gpmsd__input_samples",
                "lts__t_requests_hub_condrd",
                "lts__t_sectors_hit_hub_rd",
                "lts__t_sectors_niso_rd",
                "lts__t_sectors_crd_ni_condrd",
                "prop__latez_rstrz_input_quads",
                "sm__cycles_active_cs",
                "smsp__inst_exelwted_pipe_fma64plus_pred_on",
                "lts__t_sectors_miss_ltc_atomic",
                "gpc__rg_utlb_stalled_on_tag_allocation",
                "sked__dispatch_active_scg0",
                "vpc__input_isbes",
                "pdb__output_stalled",
                "lts__t_requests_host_noncpu_ilwal",
                "l1tex__t_set_conflicts_surface_red",
                "l1tex__t_set_conflicts_lg_global_atom",
                "vaf__beta_cycles_elapsed",
                "smsp__inst_exelwted_pipe_tex",
                "lts__t_sectors_zrop_prefetch",
                "lts__t_requests_crd_i",
                "smsp__inst_exelwted_vs",
                "raster__setup_lwlled_prims",
                "tga__input_patches",
                "smsp__warps_eligible",
                "mpc__input_batches",
                "lts__t_sectors_miss_zrop_rd",
                "swdx__tc_replayer_bin_flushes_reason_explicit",
                "raster__crstr_lwlled_prims_no_tile_coverage",
                "l1tex__samp_samp2mipb_busy",
                "l1tex__samp_samp2mipb_backpressured",
                "l1tex__t_set_accesses_lg_local_cctl",
                "l1tex__w_w2d_busy",
                "sm__subtiles_active",
                "vaf__beta_tasks_active",
                "raster__zlwll_lwlled_occluders_depth_bounds",
                "raster__frstr_output_subtiles_4_cycle",
                "fe__i2m_write_stalled_request_fifo_full",
                "lts__t_requests_gpc_condrd",
                "sm__warps_launched_vsb",
                "lts__t_sectors_gpc_atomic",
                "l1tex__m_read_request_stalled",
                "l1tex__samp_input_quads_tex_format_3d",
                "lts__t_sectors_hit_zrop_condrd",
                "lts__t_sectors_zrd_i_condrd",
                "lts__t_sectors_miss_crop_ilwal",
                "sm__warps_launched_cs",
                "gpmsd__input_samples_3d",
                "smsp__thread_inst_exelwted",
                "lts__d_atomic_resseq_cycles_bank_conflict",
                "smsp__inst_exelwted_pipe_bru_pred_on",
                "vpc__clip_input_prims_clipped_multi_plane",
                "prop__cdp_alpha_blendopt_killed_pixels",
                "sm__cycles_active_ps",
                "lts__t_sectors_raster_wr",
                "scc__load_constants_miss",
                "l1tex__t_set_accesses_surface_atom",
                "l1tex__samp_input_quads_filter_trilinear",
                "lts__t_requests_zrd_i_prefetch",
                "lts__t_requests_host_noncpu_atomic",
                "lts__t_sectors_hit_valid",
                "lts__t_sectors_miss_crd_ni",
                "cbmgr__beta_cbe_allocations",
                "lts__t_requests_host_noncpu_condrd",
                "vpc__lwll_lwlled_prims_reason_backfacing",
                "lts__t_sectors_miss_mmu_rd",
                "l1tex__texin_requests_surface_atom_cas",
                "gpc__tpc0_utlb_requests_miss",
                "vpc__lwll_lwlled_prims_line",
                "smsp__inst_exelwted_pipe_tex_pred_off_all",
                "gr__cycles_idle",
                "gpc__prop_utlb_requests_hit_under_miss",
                "l1tex__samp_input_quads_tex_format_2d_nomipmap",
                "lts__t_sectors_hit_zrop_rd",
                "smsp__inst_exelwted_vsa",
                "lts__t_sectors_hit_gpc_clear",
                "l1tex__texin_requests_null_lg",
                "lts__t_sectors_fill",
                "lts__t_sectors_cas",
                "lts__t_requests_l1tex",
                "lts__t_sectors_host_cpu_atomic",
                "smsp__inst_exelwted_pipe_fma64plus_pred_off_all",
                "mmu__hubtlb_stalled_request_fifo_full",
                "lts__t_requests_crop_wr",
                "sm__warps_launched_ps",
                "raster__tc_output_subtiles",
                "zrop__input_stalled",
                "smsp__warps_cant_issue_misc_0",
                "lts__mccif_read_cycles_64",
                "prop__earlyz_output_samples",
                "sm__warps_active_vsa",
                "l1tex__samp_samp2mipb_stalled",
                "prop__cycles_active",
                "vpc__clip_input_prims_clipped",
                "lts__t_sectors_rd",
                "l1tex__t_atomic_address_conflicts_surface_red",
                "lts__t_sectors_hit_iso_rd",
                "smsp__tex_requests",
                "lts__t_sectors_niso",
                "crop__processed_subpackets",
                "lts__t_sectors_hit_host_cpu_atomic",
                "prop__latez_output_samples",
                "vpc__input_prims_point",
                "gpu__time_active",
                "lts__t_sectors_host_cpu",
                "l1tex__t_set_conflicts_lg_local_ld",
                "pdb__input_batches",
                "lts__t_sectors_hit_zrd_ni_prefetch",
                "lts__t_sectors_ltc_membar",
                "lts__t_sectors_hit_hub",
                "swdx__tc_replayer_bin_flushes_reason_level_1_threshold",
                "lts__t_sectors_hub_wr",
                "lts__mccif_write_request_latency_64",
                "pel__in_active",
                "gpmpd__cycles_active",
                "prop__csb_output_stalled",
                "mpc__tram_allocation_stalled",
                "l1tex__m_read_sectors_tex_format_1d2d_tex",
                "lts__t_sectors_miss_zrop_prefetch",
                "l1tex__m_read_request_active",
                "l1tex__t_set_conflicts_surface_atom",
                "smsp__inst_exelwted_pipe_bar_pred_on",
                "mmu__hubtlb_requests_miss",
                "vaf__alpha_fetched_attr_scalar_post_cbf",
                "lts__mccif_read_request_latency_128",
                "smsp__lsu_write_back_active",
                "lts__t_sectors_raster",
                "lts__t_sectors_miss_condrd",
                "vpc__lwll_lwlled_prims_triangle",
                "lts__t_sectors_hit_prefetch",
                "pda__output_verts",
                "gpc__tpc0_utlb_requests_sparse",
                "smsp__inst_exelwted_pipe_bru",
                "smsp__inst_exelwted_pipe_lsu_pred_on",
                "lts__t_sectors_miss_l1tex",
                "lts__t_sectors_host_noncpu_ilwal",
                "prop__earlyz_killed_pixels",
                "lts__t_sectors_condrd",
                "l1tex__texin2m_fifo_output_busy",
                "gpc__rg_utlb_stalled_request_fifo_full",
                "prop__gnic_port1_stalled",
                "pda__input_prims_line",
                "lts__t_requests_rd",
                "lts__t_sectors_miss_crop_prefetch",
                "l1tex__x_x2w_stalled",
                "zrop__processed_requests_type_shdz_biquad",
                "vaf__alpha_fetched_attr_scalar_indexed",
                "lts__t_sectors_hit_host_cpu_clear",
                "lts__t_sectors_hit_niso",
                "lts__t_sectors_miss_raster",
                "pel__out_l2_requests_read",
                "sm__icc_requests_miss_covered",
                "lts__t_sectors_hit_ltc_membar",
                "stri__to_acache_stalled_on_read",
                "gpmsd__input_pixels",
                "sm__icc_requests_miss_no_tags",
                "prop__input_stalled",
                "sm__miop_pq_read_active_tex_smp0",
                "smsp__miop_pq_write_active",
                "l1tex__x_x2w_backpressured",
                "l1tex__samp_samp2x_backpressured",
                "l1tex__x_x2t_busy",
                "lts__t_sectors_miss_gpc_membar",
                "gpc__tpc0_utlb_requests_hit_under_miss",
                "vpc__output_verts",
                "lts__t_requests_host_noncpu",
                "lts__t_sectors_hit_host_noncpu_prefetch",
                "lts__t_sectors_miss_crop_wr",
                "lts__t_sectors_miss_hub_cas",
                "swdx__input_messages",
                "prop__pixel_shader_barriers",
                "raster__frstr_output_subtiles_2_cycle",
                "sm__warps_launched_gs_fast_alpha",
                "lts__t_requests_gpc_rd",
                "lts__mccif_write_request_latency_48",
                "l1tex__t_set_conflicts_lg_global_cctl",
                "sm__cycles_active_3d_ps",
                "lts__t_tags_writeback_tier2_issued",
                "l1tex__w_w2d_active",
                "prop__latez_shdz_input_samples",
                "mmu__pte_requests_miss",
                "prop__earlyz_output_pixels",
                "smsp__inst_exelwted_pipe_ldc",
                "raster__setup_output_prims",
                "vaf__alpha_acache_requests_local_hit",
                "sys__fb_write_requests",
                "lts__cbc_requests_miss_clear_zbc",
                "sm__warps_retiring_ps_stalled_not_selected",
                "gcc__l15_requests_miss_instruction",
                "raster__setup_output_prims_line",
                "smsp__inst_exelwted_pipe_xu_pred_on",
                "lts__t_sectors_host_cpu_membar",
                "sm__miop_pq_read_active_pixout_smp1",
                "raster__crstr_search_stalling_input",
                "lts__t_sectors_hit_gpc_wr",
                "swdx__output_active",
                "mpc__input_verts_alpha",
                "lts__t_sectors_hit_host_cpu_cas",
                "zrop__write_subpackets_stencil",
                "lts__t_sectors_hit_zrd_i_rd",
                "lts__t_sectors_ltc_cas",
                "lts__t_sectors_concat_cas",
                "lts__t_sectors_ltc_prefetch",
                "pdb__output_stalled_beta",
                "sm__mios_shmem_accesses_pe_write_isbe",
                "lts__t_sectors_zrop_condrd",
                "crop__input_stalled",
                "l1tex__t_set_accesses",
                "lts__t_requests_condrd",
                "pda__input_stalled_index_fetch",
                "fe__i2m_write_requests",
                "smsp__warps_cant_issue_not_selected",
                "lts__t_sectors_crd_i_prefetch",
                "lts__t_sectors_miss_host_noncpu_prefetch",
                "raster__zlwll_occluders_zfail",
                "prop__csb_input_samples_3d",
                "l1tex__t_set_conflicts_tex_bilinear",
                "sys__pushbuffer_dwords",
                "sm__miop_pq_read_active_lsu_smp0",
                "lts__t_requests_crd_ni_prefetch",
                "lts__d_decomp_input_stalled",
                "prop__csb_input_pixels_3d",
                "lts__cycles_active",
                "vaf__alpha_acache_requests_acache_hit",
                "lts__t_sectors_ltc_atomic",
                "wwdx__input_tasks",
                "smsp__inst_exelwted_pipe_adu",
                "pdb__cycles_active",
                "tgb__output_verts_complex_interior",
                "lts__t_requests_ltc_rd",
                "lts__t_sectors_miss_crd_ni_prefetch",
                "lts__ltcx_read_requests",
                "pes__stream_output_prims",
                "vpc__lwll_lwlled_prims_reason_bounding_box",
                "l1tex__t_set_accesses_lg_global_atom",
                "smsp__inst_exelwted_tes",
                "lts__d_atomic_reqseq_stalled_source_not_ready",
                "swdx__input_pixel_shader_barriers",
                "lts__t_sectors_miss_gpc_prefetch",
                "lts__t_sectors_miss_pe_ilwal",
                "lts__t_sectors_pe_rd",
                "lts__t_sectors_miss_hub_ilwal",
                "l1tex__t_set_accesses_miss_tag_miss_tex_format_no_mipmap",
                "lts__t_sectors_hit_gpc",
                "scc__load_constants_stalled_max_versions",
                "sm__idc_requests_miss_to_gcc",
                "lts__t_requests_zrop",
                "vaf__alpha_acache_requests",
                "gcc__l15_requests_instruction",
                "lts__r_input_fifo_stalled",
                "prop__csb_output_crop_requests",
                "sked__dispatch_count",
                "zrop__input_requests_type_bundles",
                "lts__d_atomic_reqseq_cycles_bank_conflict",
                "prop__gnic_port0_stalled",
                "l1tex__samp_input_quads_sz_96b_nearest",
                "l1tex__t_set_conflicts_cctlt",
                "lts__t_sectors_miss_host_cpu_clear",
                "scc__load_constants_page_first_update",
                "lts__t_sectors_cbc",
                "lts__t_sectors_host_noncpu_atomic",
                "mpc__isbe_allocation_stalled_alpha",
                "prop__cdp_alpha_to_coverage_output_pixels",
                "lts__t_sectors_zrop",
                "l1tex__m_stalled_on_tag_allocation",
                "l1tex__t_set_accesses_miss_surface_ld",
                "lts__t_requests_pe_ilwal",
                "l1tex__w_w2d_stalled",
                "lts__t_requests_l1tex_rd",
                "vaf__alpha_input_batches_post_cbf",
                "lts__t_sectors_miss_rd",
                "lts__t_requests_prefetch",
                "lts__t_sectors_miss_host_cpu_condrd",
                "lts__t_sectors_hit_zrd_i_prefetch",
                "vaf__alpha_to_acache_stalled_on_tag_allocation",
                "l1tex__m_write_sectors_surface_atom_cas",
                "lts__t_sectors_host_noncpu_membar",
                "lts__t_sectors_hit_concat_cas",
                "raster__frstr_killed_subtiles_input",
                "sm__miop_pq_read_active_pixout_smp0",
                "sm__cycles_active_tes",
                "lts__t_sectors_miss_zrd_ni_condrd",
                "lts__t_tags_writeback_tier1_killed",
                "lts__t_sectors_raster_rd",
                "vpc__output_prims_stippled_line",
                "swdx__tc_replayer_bin_flushes_replay_tiled",
                "gcc__tsl2_requests_hit",
                "smsp__warps_cant_issue_math_pipe_throttle",
                "pdb__output_pkts",
                "lts__cbc_requests_hit_clear_zbc_crop",
                "zrop__input_requests_type_plane_eq",
                "vaf__alpha_cycles_elapsed",
                "smsp__imc_requests_miss_covered",
                "smsp__inst_exelwted_pipe_fma64plus",
                "lts__t_sectors_gpc_condrd",
                "lts__t_requests_host_noncpu_clear",
                "zrop__input_requests",
                "mpc__tram_fill_fifo_stalled",
                "lts__t_sectors_hit_crd_ni",
                "l1tex__t_set_accesses_hit_lg_global_ld",
                "vaf__alpha_fetched_attr_scalar_indexed_constant",
                "l1tex__texin_requests_lg_local_cctl",
                "rdm__cycles_active",
                "lts__t_requests_clear",
                "prop__input_active",
                "sm__warps_draining_ps",
                "zrop__cycles_active",
                "tga__output_prims",
                "rdm__crop_output_stalled",
                "zrop__input_requests_type_shdz_biquad",
                "lts__t_sectors_miss_ltc_wr",
                "lts__t_sectors_pe",
                "lts__t_sectors_hit_zrd_ni_condrd",
                "l1tex__t_atomic_address_conflicts_surface_atom",
                "lts__t_requests_ltc_condrd",
                "lts__t_sectors_hit_host_cpu_wr",
                "gcc__l15_requests_constant",
                "lts__t_sectors_hit_host_noncpu_wr",
                "lts__t_sectors_hit_pe_wr",
                "pda__input_prims_lineadj",
                "lts__t_sectors_gpc_ilwal",
                "lts__t_requests_raster",
                "smsp__warps_cant_issue_tile_allocation_stall",
                "lts__t_sectors_hub_prefetch",
                "vpc__input_isbes_prim",
                "swdx__input_active",
                "lts__t_requests_host_noncpu_cas",
                "tga__output_tasks_complex_boundary",
                "l1tex__texin2m_fifo_output_active",
                "l1tex__m_read_sectors_tex_format_3d",
                "gpmsd__input_stalled",
                "l1tex__t_set_accesses_hit_tex_format_no_mipmap",
                "prop__input_pixels_3d",
                "lts__t_sectors_ltc_condrd",
                "lts__t_requests_host_noncpu_prefetch",
                "lts__t_requests_ltc_prefetch",
                "gpc__tpc1_utlb_stalled_on_tag_allocation",
                "lts__t_sectors_host_noncpu_wr",
                "lts__t_tags_writeback_tier1_issued",
                "mpc__warp_launch_stalled_vsb",
                "lts__t_sectors_mmu_rd",
                "crop__input_requests_clear",
                "gpc__prop_utlb_stalled_request_fifo_full",
                "lts__t_sectors_hit_zrop_ilwal",
                "lts__t_sectors_hit_host_noncpu",
                "mmu__pte_requests_small_page_1",
                "l1tex__texin_sm2tex_stalled",
                "crop__input_stalled_upstream_fifo_full",
                "l1tex__t_set_accesses_miss_tag_hit_tex_format_1d_buffer",
                "l1tex__m_read_sector_lwmulative_miss_latency",
                "l1tex__texin_stalled_on_tsl2_miss",
                "gpc__cycles_elapsed",
                "sm__warps_active",
                "lts__t_sectors_crd_i_condrd",
                "lts__t_sectors_niso_wr",
                "prop__input_quads",
                "fe__i2m_write_stalled_data_buffer_full",
                "pda__input_prims_tri",
                "crop__write_requests_compressed_4to1",
                "l1tex__samp_input_quads_sz_32b_nearest",
                "smsp__inst_exelwted",
                "lts__t_sectors_hit_gpc_membar",
                "lts__t_sectors_ltc_wr",
                "lts__t_sectors_miss_gpc_clear",
                "prop__csb_output_active",
                "lts__t_sectors_miss_gpc",
                "sm__cycles_active_vsa",
                "stri__cycles_stalled",
                "lts__t_sectors_zrop_rd",
                "lts__t_sectors_zrop_ilwal",
                "lts__t_sectors_miss_host_cpu",
                "lts__t_sectors_miss_mmu_wr",
                "l1tex__t_set_accesses_miss_tag_miss_tex_format_lwbemap",
                "crop__input_requests_2d",
                "l1tex__m_write_sectors_lg_global_atom",
                "lts__t_sectors_gpc_wr",
                "wwdx__output_stalled",
                "gcc__tsl2_requests_miss",
                "lts__t_requests_zrop_rd",
                "cwd__output_stalled_no_free_slots",
                "lts__t_requests_raster_rd",
                "sm__warps_launched_tes",
                "sm__warps_launched_gs",
                "lts__t_sectors_hit_l1tex",
                "l1tex__t_sectors_promoted",
                "lts__t_sectors_mmu",
                "vaf__cycles_active",
                "prop__cdp_alpha_blendopt_read_avoid",
                "l1tex__texin_requests_lg_global_st",
                "mmu__pte_requests_big_page_3",
                "l1tex__t_set_accesses_surface_st",
                "l1tex__m_write_sectors_surface_st",
                "swdx__tc_replayer_bin_flushes_reason_constant_table_full",
                "lts__t_sectors_miss_host_cpu_cas",
                "lts__t_sectors_miss_cbc_wr",
                "lts__t_requests_mmu",
                "smsp__warps_active",
                "sm__threads_launched_vsb",
                "tgb__output_prims_triangle",
                "l1tex__texin2m_fifo_output_backpressured",
                "sm__mios_shmem_accesses_pe_write_tram",
                "lts__t_sectors_hit_l1tex_wr",
                "crop__processed_requests_compressed",
                "smsp__inst_exelwted_ps",
                "lts__t_requests_gpc_membar",
                "l1tex__lod_output_wavefronts",
                "vpc__cycles_active",
                "lts__t_requests_niso_wr",
                "cbmgr__beta_cbe_allocation_stalled_max_tasks",
                "stri__to_acache_stalled_on_tag_allocation",
                "l1tex__t_set_accesses_lg_global_atom_cas",
                "lts__t_sectors_miss_host_cpu_membar",
                "lts__t_sectors_hit_iso_wr",
                "l1tex__t_set_accesses_tex_format_1d2d_tex",
                "sm__inst_exelwted_pipe_adu_divergent_smp0",
                "zrop__processed_subpackets_stencil",
                "lts__t_sectors_hit_membar",
                "lts__t_sectors_zrd_ni_prefetch",
                "gpmpd__output_packets",
                "vaf__alpha_to_acache_stalled_on_read",
                "zrop__processed_requests",
                "tpc__cycles_elapsed",
                "l1tex__texin_requests_lg_global_cctl",
                "l1tex__t_set_accesses_miss_tag_hit_tex_format_no_mipmap",
                "lts__t_requests_host_cpu_ilwal",
                "zrop__read_returns",
                "lts__t_requests_host_cpu_prefetch",
                "fe__output_ops",
                "sm__warps_active_tcs",
                "pel__out_active",
                "l1tex__t_set_accesses_tex_format_3d",
                "lts__mccif_write_request_latency_high",
                "vpc__output_prims_line",
                "gpc__tpc1_utlb_stalled_write_buffer_full",
                "lts__cbc_requests_comptag_miss",
                "lts__t_sectors_zrd_i_rd",
                "lts__t_sectors_miss_zrd_i_rd",
                "lts__t_requests_zrop_wr",
                "lts__mccif_read_request_latency_512",
                "l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_tex",
                "stri__acache_requests_stri_local_hit",
                "cwd__cycles_active",
                "lts__t_sectors_hit_gpc_ilwal",
                "pel__out_l2_requests_write_512b",
                "lts__t_requests_concat_cas",
                "sm__idc_requests_miss_no_tags",
                "gpc__tpc0_utlb_requests_hit",
                "lts__t_requests_host_cpu_clear",
                "lts__t_sectors_gpc_concat_cas",
                "prop__cdp_alpha_test_killed_pixels",
                "sm__warps_active_vsb",
                "lts__t_requests_crd_ni",
                "lts__t_requests_ltc_concat_cas",
                "sm__miop_adu_replays_smp1",
                "pda__input_verts",
                "lts__t_sectors_pe_wr",
                "mmu__pte_requests_small_page_0",
                "lts__t_requests_gpc_ilwal",
                "lts__t_sectors_hub_ilwal",
                "l1tex__samp_input_quads_tex_format_1d_buffer",
                "lts__t_sectors_hit_mmu_rd",
                "mpc__warp_launch_stalled_gs_fast_alpha",
                "gpc__rg_utlb_requests_hit",
                "mpc__output_batches",
                "lts__t_sectors_ltc_concat_cas",
                "lts__ltcx_write_stalled_fifo_full",
                "lts__t_sectors_hit_pe_rd",
                "lts__t_requests_raster_wr",
                "gpu__time_duration",
                "raster__crstr_output_tiles",
                "smsp__imc_requests_miss_to_gcc",
                "l1tex__t_set_conflicts_lg_global_atom_cas",
                "lts__t_sectors_miss_iso_rd",
                "crop__write_requests",
                "lts__t_sectors_hit_crd_ni_prefetch",
                "crop__input_requests_aamode_8xaa",
                "lts__d_atomic_block_cycles_serialized",
                "lts__t_sectors_pe_ilwal",
                "prop__gnic_port1_active",
                "swdx__tc_replayer_bin_flushes_replay_not_tiled",
                "lts__t_sectors_hit_ltc_concat_cas",
                "lts__t_sectors_hit_hub_cas",
                "wwdx__cycles_active",
                "lts__t_sectors_hit_gcc",
                "l1tex__samp_input_quads_tex_format_lwbemap",
                "sm__ps_quads_killed",
                "l1tex__t_t2d_busy",
                "lts__t_requests_hub_rd",
                "l1tex__d_sectors_fill",
                "l1tex__x_x2w_active",
                "lts__t_sectors_hit_gcc_rd",
                "smsp__ps_threads_killed",
                "lts__t_sectors_hit_ltc_prefetch",
                "vpc__output_prims_point",
                "gcc__l15_requests_hit_instruction",
                "lts__t_sectors_hit_rd",
                "lts__t_sectors_hub_atomic",
                "l1tex__d_output_wavefronts",
                "lts__t_sectors_hit_gpc_rd",
                "gpmpd__output_stalled_batch",
                "lts__t_sectors_miss_zrop",
                "l1tex__f_output_tex2sm_stalled",
                "prop__gnic_port0_active",
                "sm__miop_pq_read_active_smp1",
                "lts__t_sectors_miss_cbc_rd",
                "mmu__hubtlb_requests_hit",
                "lts__t_sectors_hit_host_cpu_prefetch",
                "lts__t_sectors_host_cpu_clear",
                "l1tex__t_atomic_address_conflicts_lg_global_atom_cas",
                "lts__t_requests_gpc_prefetch",
                "lts__t_sectors_hub",
                "l1tex__m_read_sectors",
                "lts__t_sectors_hit_crd_ni_rd",
                "pes__cycles_active",
                "lts__t_sectors_miss_ilwal",
                "cbmgr__alpha_cbe_allocations",
                "gpc__rg_utlb_requests_miss",
                "lts__t_sectors_hit_ltc_clear",
                "pel__out_read_stalled_vaf_alpha",
                "mpc__isbe_allocations",
                "rdm__input_data0_active",
                "crop__input_requests_blend_enabled",
                "l1tex__texin2m_fifo_output_stalled",
                "smsp__inst_exelwted_lsu_wb_pending",
                "lts__t_requests_host_cpu",
                "raster__setup_output_prims_triangle",
                "lts__t_sectors_miss_iso",
                "l1tex__x_x2t_backpressured",
                "prop__latez_shdz_input_quads",
                "pel__in_write_requests_stalled",
                "smsp__inst_exelwted_lsu_wb",
                "lts__t_requests_ltc_clear",
                "l1tex__m_read_sectors_surface_ld_d",
                "sys__fb_read_requests",
                "lts__t_tag_requests_hit",
                "lts__t_requests_niso_rd",
                "gpmsd__cycles_active",
                "sm__warps_active_gs",
                "l1tex__d_d2f_stalled",
                "smsp__inst_exelwted_pipe_su",
                "lts__t_sectors_miss_host_noncpu_membar",
                "gpmsd__input_pixels_2d",
                "lts__t_sectors_hit_crd_ni_condrd",
                "lts__t_requests_hub_prefetch",
                "lts__t_sectors_miss_ltc_clear",
                "sm__cycles_active",
                "l1tex__w_output_wavefronts",
                "sm__threads_launched_tcs",
                "l1tex__texin_requests_surface_red",
                "lts__t_sectors_crd_ni_rd",
                "l1tex__samp_samp2x_stalled",
                "lts__t_sectors_mmu_wr",
                "swdx__tc_binner_binned_op_tiled_cache_barriers",
                "smsp__warps_cant_issue_short_scoreboard_1",
                "lts__t_sectors_host_cpu_concat_cas",
                "swdx__output_stalled",
                "lts__t_sectors_ltc_rd",
                "l1tex__t_sectors_miss",
                "l1tex__samp_input_quads_tex_format_2d_mipmap",
                "gpmsd__input_quads_2d",
                "raster__crstr_discover_stalling_setup",
                "lts__t_requests_crop_ilwal",
                "mpc__warp_launch_stalled_tcs",
                "lts__t_sectors_hub_condrd",
                "vaf__beta_input_tasks",
                "prop__cdp_alpha_blendopt_pixels_fill_override",
                "l1tex__t_set_accesses_tex_format_no_mipmap",
                "l1tex__t_set_accesses_tex_format_lwbemap",
                "raster__frstr_output_subtiles_3_cycle",
                "raster__crstr_input_prims",
                "scc__load_constants_hit",
                "lts__t_sectors_miss_gpc_ilwal",
                "l1tex__m_read_sectors_surface_ld_p",
                "pel__in_write_requests_stalled_vaf_alpha",
                "lts__t_sectors_hit_ltc_condrd",
                "lts__t_requests_pe_rd",
                "lts__d_atomic_reqseq_stalled_pending_store_same_address",
                "lts__t_sectors_miss_hub",
                "l1tex__m_read_sectors_tex_format_1d2d_array",
                "l1tex__texin_requests_surface_st",
                "l1tex__texin_sm2tex_busy",
                "l1tex__m_write_sectors_surface_atom",
                "tgb__output_prims_point",
                "lts__t_sectors_hit_hub_wr",
                "lts__cbc_requests_miss_clear_zbc_zrop",
                "lts__t_sectors_hit_zrd_i",
                "cwd__output_stalled_state_ack",
                "l1tex__t_set_accesses_miss_lg_global_ld",
                "lts__t_sectors_miss_hub_atomic",
                "gpc__gpcl1tlb_stalled_request_fifo_full",
                "lts__t_requests_cas",
                "lts__t_sectors_miss_crd_i",
                "l1tex__texin_tsl1_requests_hit_texhdr",
                "mpc__warp_launch_stalled_rf_free",
                "sm__cycles_active_3d",
                "sm__icc_requests_miss_to_gcc",
                "lts__t_requests_gpc_concat_cas",
                "gpmpd__input_tasks",
                "lts__t_sectors_miss_host_cpu_wr",
                "smsp__warps_cant_issue_tex_throttle",
                "lts__t_sectors_miss_l1tex_rd",
                "smsp__inst_exelwted_pipe_ldc_pred_on",
                "lts__t_requests_l1tex_wr",
                "tgb__cycles_active",
                "lts__t_sectors_host_cpu_prefetch",
                "fe__cycles_wfi_host_scg0",
                "lts__d_atomic_block_stalled_pending_miss",
                "smsp__warps_cant_issue_allocation_stall_1",
                "vaf__gpm_stalled_by_state_processing",
                "lts__t_sectors_hit_hub_ilwal",
                "gpc__prop_utlb_requests_sparse",
                "lts__t_requests_host_cpu_concat_cas",
                "l1tex__samp_samp2x_active",
                "prop__input_stalled_waiting_for_pixel_shader_barrier_release",
                "lts__t_sectors_host_noncpu_concat_cas",
                "lts__t_sectors_host_noncpu_condrd",
                "lts__t_sectors_host_noncpu",
                "l1tex__t_set_accesses_hit_tex_format_1d2d_array",
                "l1tex__t_set_conflicts_lg_global_ld",
                "l1tex__samp_input_quads_sz_32b",
                "lts__r_input_fifo_active",
                "lts__t_sectors_miss_gcc",
                "lts__t_sectors_hit_cas",
                "swdx__tc_replayer_bin_flushes_reason_clear",
                "smsp__imc_requests_miss_no_tags",
                "smsp__inst_exelwted_pipe_fp16",
                "smsp__inst_exelwted_tex_wb",
                "vpc__clip_output_attrs",
                "lts__t_sectors_hit_crd_i_prefetch",
                "lts__t_sectors_hit_hub_concat_cas",
                "lts__t_sectors_iso",
                "lts__t_sectors_hit_ltc_cas",
                "vaf__alpha_fetched_attr_vector_post_cbf",
                "lts__t_sectors_ltc",
                "smsp__inst_exelwted_pipe_fmai",
                "vaf__alpha_acache_requests_acache_miss",
                "l1tex__x_x2t_stalled",
                "lts__t_sectors_hit_l1tex_rd",
                "lts__t_sectors_hit_cbc_rd",
                "lts__t_sectors_hit_gpc_condrd",
                "raster__frstr_output_cycles",
                "smsp__inst_exelwted_pipe_fma64plusplus",
                "l1tex__t_set_conflicts_surface_st",
                "l1tex__t_t2d_active",
                "lts__t_sectors_hit_cbc_wr",
                "l1tex__texin_requests_null_surface",
                "gpu__time_start",
                "lts__t_sectors_crd_i",
                "lts__cbc_requests_miss_clear_zbc_crop",
                "scc__load_constants_stalled_update_scoreboard_full",
                "lts__t_sectors_hit_zrop",
                "sm__miop_pq_read_active_tex_smp1",
                "lts__t_sectors_hit_host_noncpu_rd",
                "rdm__input_data1_active",
                "prop__zrop_output_quads",
                "fe__input_methods",
                "lts__t_sectors_hit_host_noncpu_condrd",
                "raster__crstr_output_tiles_empty",
                "lts__t_sectors_hit_cbc",
                "l1tex__texin_requests_lg_global_red",
                "cbmgr__cycles_active",
                "lts__t_sectors_miss_hub_clear",
                "rdm__crop_output_active",
                "lts__t_sectors_hit_ltc_rd",
                "lts__mccif_read_request_latency_high",
                "smsp__inst_exelwted_pipe_xu_pred_off_all",
                "gpc__prop_utlb_stalled_on_tag_allocation",
                "prop__latez_output_pixels",
                "sm__miop_adu_replays_smp0",
                "vpc__lwll_lwlled_prims_reason_scissor",
                "lts__cbc_requests_comptag_fill",
                "mpc__warp_launch_stalled_gs",
                "crop__read_returns",
                "l1tex__t_set_accesses_tex_format_1d2d_array",
                "lts__t_sectors_host_noncpu_rd",
                "crop__read_stalled",
                "vpc__output_cbes",
                "lts__t_sectors_iso_wr",
                "mpc__input_subtiles",
                "gcc__cycles_active",
                "lts__t_sectors_gpc_rd",
                "crop__input_requests_aamode_2xaa",
                "l1tex__t_set_accesses_miss_tag_hit_tex_format_3d",
                "l1tex__t_set_accesses_lg_local_st",
                "lts__t_sectors_gcc",
                "zrop__write_requests",
                "sm__threads_launched_cs",
                "lts__t_sectors_hit_zrd_ni",
                "lts__t_requests_hub_cas",
                "lts__t_sectors_crd_ni_prefetch",
                "lts__t_sectors_miss_zrd_ni",
                "swdx__tc_replayer_bin_flushes_reason_cbe_slots_threshold",
                "mpc__warp_launch_stalled_ps",
                "lts__t_sectors_prefetch",
                "smsp__issue_active",
                "l1tex__texin_tsl1_requests_hit_samphdr",
                "lts__t_sectors_hit_host_noncpu_ilwal",
                "lts__d_atomic_block_stalled_same_address",
                "lts__t_sectors_miss_hub_wr",
                "lts__t_sectors_miss_host_noncpu_rd",
                "pel__out_input_stalled",
                "l1tex__texin_tsl1_requests_miss_texhdr",
                "l1tex__t_set_accesses_surface_ld",
                "lts__ltcx_read_sectors",
                "zrop__processed_subpackets",
                "smsp__inst_exelwted_gs",
                "l1tex__samp_input_quads_tex_format_1d_tex",
                "gpc__tpc1_utlb_requests_hit",
                "lts__t_sectors_miss_host_cpu_rd",
                "lts__t_sectors_miss_ltc_concat_cas",
                "lts__t_sectors_hub_clear",
                "vpc__output_prims",
                "sm__warps_launched_tcs",
                "swdx__tc_replayer_bin_flushes_reason_timeout",
                "l1tex__d_d2f_backpressured",
                "mme__cycles_stalled",
                "gpc__tpc0_utlb_stalled_request_fifo_full",
                "lts__t_sectors_miss_zrd_i",
                "vpc__read_isbes",
                "lts__t_sectors",
                "lts__t_requests_mmu_wr",
                "lts__t_sectors_crop_wr",
                "zrop__processed_requests_type_bundle",
                "l1tex__samp_input_quads_sz_128b_nearest",
                "mpc__isbe_allocation_stalled_alpha_on_vsc",
                "lts__t_requests_cbc_wr",
                "lts__t_sectors_gpc_prefetch",
                "swdx__input_prims_single_binned",
                "prop__input_pixels",
                "lts__t_sectors_miss_zrop_wr",
                "l1tex__m_read_sectors_tex_format_1d_buffer",
                "l1tex__t_sectors_miss_lg",
                "smsp__inst_exelwted_pipe_xu",
                "vpc__alpha_cycles_active",
                "lts__t_requests_host_cpu_atomic",
                "lts__t_sectors_miss_host_cpu_atomic",
                "pes__stream_output_attrs",
                "lts__cbc_requests_hit_clear_zbc_zrop",
                "prop__latez_shdz_input_pixels",
                "sm__subtiles_launched_smp0",
                "lts__t_sectors_hit_host_noncpu_membar",
                "lts__t_sectors_miss_hub_condrd",
                "lts__t_sectors_miss_iso_wr",
                "swdx__tc_replayer_bin_flushes_reason_cbe_memory_threshold",
                "gpmsd__input_pixels_fully_covered",
                "lts__t_requests_gpc_atomic",
                "lts__t_sectors_miss_clear",
                "smsp__warps_cant_issue_imc_miss",
                "lts__cbc_requests_comptag_hit",
                "mpc__alpha_beta_mode_switches",
                "l1tex__t_set_accesses_surface_red",
                "sys__cycles_elapsed",
                "smsp__inst_exelwted_pipe_lsu_pred_off_all",
                "sm__idc_requests_hit",
                "smsp__inst_exelwted_pipe_fma64plusplus_pred_off_all",
                "raster__zlwll_lwlled_occluders",
                "lts__mccif_read_request_latency_640",
                "l1tex__texin_requests_lg_local_st",
                "lts__t_requests_hub_clear",
                "lts__t_requests_membar",
                "lts__t_sectors_host_noncpu_clear",
                "fbp__cycles_elapsed",
                "l1tex__texin_requests_null_tex",
                "raster__frstr_output_subtiles",
                "stri__acache_requests_stri",
                "cbmgr__beta_cbe_allocation_stalled",
                "l1tex__t_set_accesses_miss_tag_miss",
                "sm__warps_completed_ooo_3d_vtg",
                "gpc__gpcl1tlb_requests_miss",
                "sys__gpfifo_dwords",
                "sm__mios_shmem_accesses_lsu_read",
                "mpc__warp_launch_stalled_vsa",
                "raster__zlwll_output_occluders_trivial_accept",
                "vaf__gpm_stalled_by_beta_input_fifo",
                "sked__cycles_active",
                "lts__t_sectors_hit_crop_wr",
                "lts__t_sectors_hit_host_noncpu_cas",
                "lts__mccif_read_request_latency_448",
                "fe__i2m_write_bytes",
                "stri__acache_requests_stri_miss",
                "cbmgr__beta_cbe_allocation_stalled_no_space",
                "cbmgr__alpha_cbe_allocation_stalled_no_space",
                "lts__t_requests_ilwal",
                "lts__t_sectors_hit_host_cpu_condrd",
                "fe__output_ops_vert",
                "smsp__inst_issued",
                "l1tex__t_texels",
                "sm__mios_shmem_accesses_su_read_tram_bank_conflict",
                "lts__t_sectors_hit_raster_rd",
                "lts__t_sectors_host_cpu_ilwal",
                "vpc__input_prims_patch",
                "mmu__pte_requests_small_page_3",
                "swdx__tc_replayer_bin_flushes_reason_non_binnable_line",
                "sm__ps_quads_sent_to_pixout",
                "l1tex__t_set_accesses_miss_tag_hit_tex_format_lwbemap",
                "sm__warps_active_tes",
                "l1tex__t_set_conflicts_lg_global_red",
                "cwd__feedback_mpc_messages",
                "lts__cbc_requests_hit_clear_zbc",
                "lts__t_sectors_hit_crop_prefetch",
                "lts__t_sectors_miss_gpc_wr",
                "lts__t_requests_gcc",
                "raster__frstr_output_subtiles_2d",
                "lts__t_sectors_miss_host_noncpu_ilwal",
                "l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_array",
                "lts__t_sectors_crop_ilwal",
                "zrop__input_samples_part2",
                "prop__latez_rstrz_input_samples",
                "scc__load_constants",
                "vpc__clip_output_prims",
                "smsp__warps_cant_issue_dispatch_stall",
                "lts__t_sectors_cbc_rd",
                "lts__t_sectors_miss_zrd_i_condrd",
                "lts__t_requests_hub_concat_cas",
                "prop__zrop_output_active",
                "prop__csb_output_quads",
                "vaf__gpm_stalled_by_state_fifo",
                "scc__cycles_active",
                "prop__cdp_alpha_to_coverage_killed_pixels",
                "lts__t_requests_host_noncpu_wr",
                "lts__t_sectors_hit_crd_i",
                "l1tex__m_write_sectors_lg_global_atom_cas",
                "crop__input_requests",
                "vpc__output_prims_triangle",
                "l1tex__t_set_accesses_lg_global_st",
                "lts__t_sectors_hit_hub_clear",
                "l1tex__texin_tsl1_requests_miss_samphdr",
                "l1tex__d_d2f_active",
                "fe__output_ops_bundle_scg0_go_idle",
                "prop__csb_output_quads_3d",
                "raster__tc_output_tiles",
                "vpc__output_attrs_scalar",
                "lts__t_sectors_miss_atomic",
                "l1tex__samp_input_quads_sz_64b_nearest",
                "mme__cycles_active",
                "scc__load_constants_page_same",
                "prop__input_quads_2d",
                "l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_tex",
                "lts__t_requests_hub_atomic",
                "lts__t_sectors_miss_crd_i_prefetch",
                "lts__t_sectors_crd_ni",
                "mme__output_method_dwords",
                "sm__threads_launched_tes",
                "smsp__inst_exelwted_pipe_tex_pred_on",
                "stri__cycles_active",
                "lts__t_sectors_hit_mmu_wr",
                "l1tex__texin_requests_cctlt",
                "crop__read_requests",
                "mmu__pte_requests_big_page_2",
                "pdb__input_stalled_alpha",
                "swdx__tc_replayer_bin_flushes_reason_level_0_threshold",
                "lts__t_requests_crd_i_prefetch",
                "sm__miop_pq_read_active_lsu_smp1",
                "lts__t_sectors_hit_ltc_ilwal",
                "l1tex__t_t2d_stalled",
                "lts__t_sectors_miss_ltc_membar",
                "fe__output_ops_bundle_scg1_go_idle",
                "lts__t_sectors_miss_concat_cas",
                "sm__cycles_active_tcs",
                "raster__tc_flushes",
                "fe__output_ops_bundle",
                "l1tex__m_read_sectors_tex_format_no_mipmap",
                "lts__t_sectors_miss_cbc",
                "lts__d_atomic_reqseq_input_stalled_fifo_full",
                "lts__d_cycles_bank_conflict",
                "crop__input_requests_aamode_4xaa",
                "gcc__l15_requests_miss",
                "mmu__cycles_active",
                "lts__t_requests_zrop_condrd",
                "l1tex__t_set_conflicts_surface_atom_cas",
                "pel__out_read_stalled_vaf_beta",
                "raster__frstr_output_subtiles_1_cycle",
                "lts__t_sectors_hit_hub_condrd",
                "l1tex__t_set_accesses_miss_lg_local_ld",
                "mpc__input_prims_alpha_patches",
                "crop__write_requests_compressed_2to1",
                "swdx__input_prims_dual_binned",
                "sm__warps_launched_vsa",
                "mmu__pte_requests_big_page_0",
                "raster__crstr_discover_working_no_tile_stalling_setup",
                "l1tex__samp_samp2x_busy",
                "prop__csb_input_quads_3d",
                "crop__write_subpackets",
                "lts__t_sectors_hit_zrd_i_condrd",
                "l1tex__texin_requests_surface_atom",
                "vaf__beta_to_acache_stalled_on_read",
                "lts__xbar_output_active",
                "cbmgr__alpha_cbe_allocation_stalled_max_tasks",
                "sm__warps_retiring_ps_stalled_out_of_order",
                "gpmsd__input_pixels_3d",
                "lts__t_sectors_hub_rd",
                "mpc__input_prims_beta",
                "pel__in_write_requests_stalled_vaf_beta",
                "mmu__pte_requests_big_page_1",
                "vaf__beta_acache_requests_local_hit",
                "l1tex__texin_requests_surface_ld",
                "lts__t_sectors_hit_gpc_prefetch",
                "l1tex__t_set_accesses_lg_local_ld",
                "fe__input_method_dwords_i2m",
                "tgb__output_prims_line",
                "lts__ltcx_read_stalled_fifo_full",
                "lts__t_sectors_miss_zrd_i_prefetch",
                "l1tex__t_set_accesses_hit_in_warp_surface_ld",
                "smsp__warps_launched",
                "l1tex__x_x2w_busy",
                "lts__t_requests_ltc",
                "l1tex__t_set_accesses_tex_format_1d_buffer",
                "l1tex__w_w2d_backpressured",
                "gpu__time_end",
                "lts__t_sectors_zrd_ni_condrd",
                "crop__processed_samples_part2",
                "lts__t_sectors_miss_gpc_condrd",
                "lts__t_requests_hub_ilwal",
                "gpc__tpc1_utlb_requests_hit_under_miss",
                "fe__output_ops_ld_const",
                "l1tex__m_read_sectors_lg_global_ld",
                "vaf__alpha_cycles_active",
                "raster__setup_input_prims",
                "zrop__processed_requests_type_plane_eq",
                "lts__t_sectors_miss_host_cpu_prefetch",
                "sm__mios_shmem_accesses_lsu_read_bank_conflict",
                "smsp__warps_cant_issue_mio_throttle",
                "pdb__output_stalled_alpha",
                "vaf__beta_input_task_fifo_full",
                "gcc__l15_requests_miss_constant",
                "zrop__write_subpackets_depth",
                "lts__t_sectors_miss_pe_rd",
                "fe__cycles_wfi_ctxsw",
                "lts__t_sectors_miss_crop_condrd",
                "sm__threads_launched_ps_not_killed",
                "lts__t_sectors_hit_ilwal",
                "fe__output_ops_bundle_scg1_wfi_host",
                "crop__input_requests_blend_blendopt_fill_over",
                "gpc__gpcl1tlb_stalled_on_tag_allocation",
                "mpc__warp_launch_stalled_tes",
                "lts__t_sectors_miss_raster_rd",
                "vpc__lwll_cycles_active",
                "smsp__inst_exelwted_pipe_bru_pred_off_all",
                "gpmsd__sm2gpmsd_pixout_cdp0_active_color",
                "lts__t_sectors_hub_membar",
                "crop__processed_samples_part1",
                "lts__t_requests_zrd_ni",
                "smsp__inst_exelwted_pipe_adu_pred_on",
                "lts__t_sectors_miss_zrd_ni_rd",
                "tgb__output_verts_simple",
                "lts__t_requests_host_cpu_condrd",
                "vpc__lwll_lwlled_prims_reason_zero_area",
                "l1tex__samp_input_quads_sz_96b",
                "lts__t_requests_ltc_atomic",
                "l1tex__t_output_wavefronts",
                "lts__t_requests_iso_rd",
                "smsp__warps_cant_issue_long_scoreboard_0",
                "lts__t_sectors_miss_wr",
                "lts__t_sectors_crop_prefetch",
                "smsp__warps_cant_issue_short_scoreboard_0",
                "lts__t_sectors_miss_cas",
                "sm__warps_launched_gs_fast_beta",
                "l1tex__m_write_sectors_lg_global_st",
                "pda__input_prims_point",
                "prop__latez_output_quads",
                "lts__t_requests_hub_wr",
                "smsp__inst_exelwted_pipe_bar_pred_off_all",
                "raster__setup_output_prims_point",
                "raster__zlwll_input_tiles",
                "mpc__input_prims_alpha",
                "lts__d_atomic_resseq_stalled_backpressure",
                "smsp__warps_cant_issue_long_scoreboard_1",
                "sm__threads_launched",
                "vaf__alpha_input_prims",
                "smsp__inst_exelwted_pipe_fxu",
                "prop__input_samples",
                "smsp__warps_cant_issue_drain",
                "lts__mccif_write_stalled",
                "vpc__alpha_batches_active",
                "fe__cycles_active",
                "lts__t_sectors_miss_raster_wr",
                "vaf__alpha_input_batches_pre_cbf",
                "tga__output_tasks_primary",
                "l1tex__t_set_accesses_hit",
                "raster__zlwll_output_occluders_zfail_no_lwll",
                "lts__mccif_write_cycles_16",
                "lts__t_sectors_hit_crd_i_rd",
                "smsp__warps_cant_issue_wait",
                "gpc__rg_utlb_stalled_write_buffer_full",
                "lts__t_sectors_crop_condrd",
                "lts__t_sectors_hit_zrop_wr",
                "l1tex__samp_input_quads_filter_aniso",
                "lts__t_sectors_gpc_membar",
                "lts__mccif_read_request_latency_384",
                "lts__t_sectors_miss",
                "lts__t_sectors_host_cpu_rd",
                "smsp__inst_exelwted_vsb",
                "smsp__warps_cant_issue_misc_1",
                "gpc__prop_utlb_requests_hit",
                "gpc__prop_utlb_requests_miss",
                "smsp__inst_exelwted_cs",
                "vaf__alpha_batches_active",
                "l1tex__t_set_accesses_cctlt",
                "l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_array",
                "mmu__pde_requests_miss",
                "lts__mccif_read_cycles_32",
                "lts__t_requests_gpc_cas",
                "lts__t_requests_host_noncpu_concat_cas",
                "lts__t_sectors_miss_gcc_rd",
                "lts__t_requests_mmu_rd",
                "wwdx__input_cbes",
                "pda__input_prims",
                "stri__acache_requests_stri_hit",
                "lts__t_sectors_miss_host_cpu_ilwal",
                "vpc__clip_output_verts",
                "mpc__tram_startxy_fifo_stalled",
                "l1tex__m_write_sectors_lg_global_red",
                "lts__t_sectors_miss_prefetch",
                "pda__input_prims_triadj",
                "crop__processed_requests_uncompressed",
                "lts__t_sectors_atomic",
                "lts__t_sectors_clear",
                "vpc__lwll_lwlled_prims_reason_frustum",
                "pel__in_write_requests",
                "lts__t_requests_zrop_prefetch",
                "lts__t_sectors_wr",
                "crop__input_samples_part2",
                "l1tex__m_read_sectors_surface_atom_cas",
                "vpc__lwll_lwlled_prims_reason_rotated_grid",
                "l1tex__t_set_accesses_hit_in_warp_lg_local_st",
                "lts__t_requests_host_noncpu_membar",
                "sm__mios_datapath_active",
                "lts__t_requests_l1tex_atomic",
                "mme__input_methods_shadow_filtered",
                "lts__t_requests_zrd_ni_prefetch",
                "lts__t_sectors_zrd_ni_rd",
                "zrop__zlwll_cycles_active",
                "l1tex__d_cycles_bank_conflict_2x2",
                "l1tex__t_set_accesses_hit_tex_format_1d_buffer",
                "lts__t_sectors_miss_host_noncpu_wr",
                "lts__t_sectors_miss_niso_wr",
                "mpc__cycles_elapsed_beta",
                "l1tex__m_write_sectors_lg_local_st",
                "vpc__lwll_lwlled_prims_point",
                "vaf__alpha_fetched_attr_vector_pre_cbf",
                "gpc__gpcl1tlb_requests_hit_under_miss",
                "gr__cycles_active",
                "lts__t_requests_ltc_membar",
                "pda__output_batches",
                "zrop__write_subpackets_coalesced",
                "lts__t_sectors_hit_host_cpu_rd",
                "smsp__inst_exelwted_tcs",
                "l1tex__t_set_accesses_hit_in_warp_lg_global_st",
                "lts__t_sectors_miss_pe_wr",
                "lts__t_sectors_host_cpu_condrd",
                "sm__warps_active_ps",
                "lts__t_sectors_zrop_wr",
                "fe__output_ops_bundle_scg0_wfi_host",
                "stri__cycles_busy",
                "tga__output_tasks_complex_interior",
                "sm__inst_exelwted_pipe_adu_divergent_smp1",
                "lts__t_sectors_miss_ltc",
                "smsp__tex_write_back_active",
                "rdm__zrop_output_active",
                "lts__cbc_requests_comptag_writeback",
                "smsp__cycles_active",
                "tga__input_isbes",
                "lts__t_requests_cbc_rd",
                "lts__t_requests",
                "lts__t_sectors_zrd_i_prefetch",
                "vpc__clip_input_prims_clipped_single_plane",
                "vaf__beta_cycles_stalled_on_mpc",
                "swdx__tc_replayer_bin_flushes_reason_non_binnable_state",
                "lts__t_sectors_miss_host_noncpu_condrd",
                "lts__t_sectors_hit_host_noncpu_atomic",
                "l1tex__m_read_sectors_lg_global_atom",
                "lts__t_sectors_miss_zrd_ni_prefetch",
                "lts__t_tag_requests_miss",
                "lts__t_sectors_hit_crop",
                "lts__t_requests_ltc_ilwal",
                "lts__t_sectors_gpc_cas",
                "lts__t_sectors_miss_host_noncpu_atomic",
                "lts__t_sectors_miss_host_noncpu",
                "l1tex__t_set_conflicts_tex_trilinear",
                "sm__cycles_active_gs",
                "crop__cycles_active",
                "l1tex__t_set_accesses_miss_tag_miss_tex_format_3d",
                "prop__input_samples_3d",
                "vpc__clip_cycles_stalled",
                "lts__t_sectors_hit_l1tex_atomic",
                "scc__load_constants_stalled_no_gcc_credits",
                "lts__t_requests_crop",
                "lts__t_requests_hub",
                "gpc__tpc0_utlb_stalled_write_buffer_full",
                "raster__frstr_killed_subtiles_output",
                "lts__t_requests_zrd_i_rd",
                "lts__t_sectors_miss_host_noncpu_cas",
                "smsp__warps_cant_issue_selected",
                "lts__t_sectors_hit_hub_atomic",
                "lts__t_sectors_hit_crd_i_condrd",
                "lts__t_requests_hub_membar",
                "gpmpd__output_stalled_task",
                "mpc__isbe_allocation_stalled_beta_on_vsc",
                "lts__t_sectors_miss_host_noncpu_clear",
                "fe__cycles_wfi_subch_switch_scg0",
                "lts__t_sectors_miss_ltc_cas",
                "l1tex__samp_pre_nop_wavefronts",
                "prop__csb_output_samples_3d",
                "l1tex__m_read_sectors_surface_atom",
                "l1tex__t_atomic_address_conflicts_surface_atom_cas",
                "l1tex__texin_requests_lg_local_ld",
                "pdb__input_tasks",
                "sm__subtiles_launched_smp1",
                "l1tex__d_d2f_busy",
                "sm__icc_prefetches",
                "lts__d_atomic_block_stalled_priority",
                "l1tex__texin_sm2tex_active",
                "swdx__tc_replayer_bin_flushes",
                "l1tex__t_set_conflicts_lg_global_st",
                "lts__t_sectors_miss_hub_rd",
                "raster__frstr_output_subtiles_fully_covered",
                "sm__warps_active_cs",
                "vaf__beta_input_patches",
                "lts__t_sectors_host_noncpu_prefetch",
                "raster__zlwll_lwlled_occluders_stencil",
                "smsp__inst_exelwted_pipe_fe",
                "l1tex__texin_requests_lg_global_atom",
                "lts__t_sectors_gpc",
                "l1tex__t_set_conflicts_lg_local_st",
                "swdx__tc_replayer_bin_flushes_reason_z_mode_transition",
                "smsp__thread_inst_exelwted_pred_on",
                "gpmsd__sm2gpmsd_pixout_cdp1_active_shdz",
                "lts__t_sectors_miss_crd_i_rd",
                "gpc__gpcl1tlb_requests_hit",
                "raster__tc_input_transactions",
                "lts__t_sectors_miss_hub_concat_cas",
                "lts__t_sectors_miss_ltc_prefetch",
                "lts__t_requests_pe_wr",
                "vaf__beta_to_acache_stalled_on_tag_allocation",
                "host__chsw_switches",
                "pdb__input_stalled_beta",
                "gpc__tpc0_utlb_stalled_on_tag_allocation",
                "prop__csb_output_crop_requests_color",
                "lts__t_sectors_miss_crd_ni_rd",
                "smsp__warps_cant_issue_no_instructions_1",
                "lts__t_sectors_l1tex",
                "lts__t_requests_crop_rd",
                "lts__t_sectors_hit",
                "mpc__cycles_active_beta",
                "pel__out_l2_requests",
                "sm__miop_ldc_replays_smp1",
                "vaf__gpm_stalled_by_alpha_input_fifo",
                "crop__input_requests_blend_blendopt_read_avoid",
                "lts__t_requests_pe",
                "smsp__inst_exelwted_pipe_adu_pred_off_all",
                "vaf__alpha_input_verts",
                "lts__t_requests_zrd_ni_rd",
                "tga__isbes_active",
                "vpc__lwll_lwlled_prims_reason_diamond_exit_rule",
                "raster__zlwll_input_occluders",
                "l1tex__x_x2t_active",
                "lts__t_sectors_miss_host_noncpu_concat_cas",
                "lts__t_requests_crd_ni_rd",
                "l1tex__texin_requests_tex",
                "smsp__inst_exelwted_pipe_ldc_pred_off_all",
                "zrop__write_subpackets",
                "lts__t_requests_host_cpu_wr",
                "smsp__inst_exelwted_pipe_su_pred_on",
                "prop__input_pixels_2d",
                "lts__mccif_write_request_latency_24",
                "lts__t_sectors_cbc_wr",
                "l1tex__texin_requests_membar",
                "lts__d_decomp_processed_sectors",
                "lts__t_sectors_miss_crop",
                "lts__t_sectors_miss_gpc_cas",
                "crop__write_requests_compressed_8to1_or_fastclear",
                "prop__input_samples_2d",
                "l1tex__texin_sm2tex_backpressured",
                "gpmpd__input_batches",
                "gpc__rg_utlb_requests_hit_under_miss",
                "lts__t_sectors_miss_niso",
                "gpc__tpc1_utlb_requests_miss",
                "lts__t_requests_zrd_i",
                "mmu__pde_requests_hit",
                "vpc__input_prims",
                "pel__in_write_requests_stalled_tg",
                "mpc__cycles_active_alpha",
                "mmu__pte_requests_hit",
                "sm__ctas_active",
                "lts__d_atomic_resseq_stalled_output_arbitration",
                "lts__t_sectors_hit_pe",
                "lts__t_sectors_miss_host_cpu_concat_cas",
                "gpmsd__output_quads",
                "l1tex__texin2m_fifo_input_stalled",
                "l1tex__x_output_wavefronts",
                "sm__idc_requests_miss_covered",
                "lts__t_sectors_miss_l1tex_wr",
                "lts__t_sectors_hit_host_noncpu_concat_cas",
                "lts__t_sectors_hub_cas",
                "lts__t_sectors_miss_crd_ni_condrd",
                "fe__input_methods_stalled",
                "pda__cycles_active",
                "sm__mios_shmem_accesses_lsu_write_bank_conflict",
                "lts__mccif_read_requests",
                "smsp__imc_requests_hit",
                "gpc__tpc1_utlb_stalled_request_fifo_full",
                "l1tex__t_set_accesses_hit_tex_format_1d2d_tex",
                "l1tex__m_read_sectors_lg_local_ld",
                "lts__t_sectors_hit_host_cpu_ilwal",
                "swdx__output_barriers",
                "lts__t_sectors_l1tex_wr",
                "vaf__beta_acache_requests_acache_miss",
                "lts__t_sectors_miss_gpc_rd",
                "l1tex__samp_input_quads_filter_bilinear",
                "vpc__lwll_cycles_stalled",
                "lts__t_requests_ltc_wr",
                "swdx__tc_replayer_bin_flushes_reason_state_full",
                "raster__crstr_discover_working_no_tile_output",
                "smsp__inst_exelwted_pipe_fma64plusplus_pred_on",
                "cbmgr__alpha_cbe_allocation_stalled",
                "gpmsd__sm2gpmsd_pixout_cdp1_active_color",
                "lts__t_sectors_hit_gpc_cas",
                "pel__cycles_active",
                "lts__t_sectors_crd_i_rd",
                "lts__t_requests_zrd_i_condrd",
                "wwdx__input_prims",
                "lts__t_sectors_miss_ltc_ilwal",
                "prop__csb_output_pixels_3d",
                "lts__t_sectors_hit_ltc_wr",
                "l1tex__samp_output_wavefronts",
                "lts__t_sectors_miss_niso_rd",
                "l1tex__t_sectors",
                "lts__t_sectors_l1tex_rd",
                "pel__out_l2_requests_write_256b",
                "zrop__read_subpackets",
                "lts__t_sectors_hit_crop_ilwal",
                "lts__t_requests_crop_condrd",
                "mpc__warp_launch_stalled_vtg",
                "tga__batches_active",
                "zrop__input_requests_containing_stencil",
                "raster__zlwll_lwlled_occluders_depth",
                "gpu__draw_count",
                "lts__t_sectors_hit_wr",
                "mpc__input_verts_beta",
                "l1tex__t_set_accesses_hit_in_warp_surface_st",
                "prop__latez_rstrz_input_pixels",
                "smsp__warps_cant_issue_barrier",
                "scc__load_constants_page_pool_allocations",
                "l1tex__t_set_accesses_surface_atom_cas",
                "l1tex__d_cycles_bank_conflict_bilerp",
                "l1tex__t_atomic_address_conflicts_lg_global_atom",
                "vaf__alpha_fetched_attr_scalar_pre_cbf",
                "lts__t_sectors_hit_gpc_concat_cas",
                "sys__fb_write_dwords",
                "vpc__lwll_lwlled_prims_reason_zero_length",
                "vpc__write_sectors",
                "crop__input_requests_aamode_1xaa",
                "mmu__pte_requests_small_page_2",
                "lts__t_sectors_hit_condrd",
                "scc__input_state_changes",
                "crop__processed_requests",
                "lts__t_sectors_miss_ltc_condrd",
                "lts__t_sectors_miss_crop_rd",
                "prop__zrop_output_stalled",
                "sm__warps_retiring_ps_stalled_backpressure",
                "crop__input_requests_3d",
                "sm__icc_requests_hit",
                "l1tex__t_set_conflicts_lg_local_cctl",
                "l1tex__t_set_accesses_miss_tag_hit",
                "lts__t_sectors_crop_rd",
                "vpc__lwll_lwlled_prims",
                "vpc__input_prims_line",
                "tga__output_tasks",
                "lts__t_requests_crd_ni_condrd",
                "sm__miop_ldc_replays_smp0",
                "smsp__warps_cant_issue_membar",
                "pes__stream_output_verts",
                "gpmsd__sm2gpmsd_pixout_cdp0_active_shdz",
                "lts__t_requests_host_cpu_rd",
                "l1tex__m_read_sectors_lg_global_atom_cas",
                "lts__t_sectors_host_cpu_wr",
                "mpc__input_tasks",
                "swdx__output_prims",
                "prop__input_quads_3d",
                "mpc__cycles_elapsed_alpha",
                "lts__t_requests_crd_i_rd",
                "l1tex__texin_requests_lg_global_atom_cas",
                "lts__t_sectors_iso_rd",
                "lts__t_sectors_host_noncpu_cas",
                "swdx__binner_active",
                "sm__warps_active_vtg",
                "zrop__input_samples_part1",
                "swdx__tc_replayer_bin_flushes_reason_bin_memory_threshold",
                "lts__t_sectors_miss_gpc_atomic",
                "raster__frstr_processed_3_edges",
                "l1tex__t_set_accesses_lg_global_cctl",
                "pel__out_read_stalled_stri",
                "l1tex__samp_input_quads_sz_128b",
                "lts__t_requests_gpc_wr",
                "mme__call_count",
                "lts__t_sectors_hit_iso",
                "tga__input_batches",
                "crop__write_stalled",
                "lts__t_sectors_miss_mmu",
                "smsp__inst_exelwted_pipe_bar",
                "lts__t_requests_host_cpu_cas",
                "lts__t_sectors_hit_raster",
                "l1tex__samp_input_quads_sz_64b",
                "lts__t_sectors_gcc_rd",
                "lts__t_sectors_crop",
                "lts__t_requests_gpc",
                "lts__t_sectors_hit_host_cpu_concat_cas",
                "zrop__input_requests_expanded_to_samples",
                "lts__t_sectors_hit_zrd_ni_rd",
                "sm__mios_shmem_accesses_su_read_tram",
                "gpc__rg_utlb_requests_sparse",
                "vaf__alpha_cycles_stalled_on_mpc",
                "lts__t_sectors_zrd_i",
                "l1tex__texin_requests_lg_global_ld",
                "mpc__isbe_allocation_stalled",
                "prop__zrop_output_samples",
                "prop__earlyz_output_quads",
                "lts__t_sectors_hit_crop_rd",
                "gpc__tpc1_utlb_requests_sparse",
                "lts__t_requests_niso",
                "crop__write_requests_compressed_1to1",
                "l1tex__f_output_tex2sm_backpressured",
                "lts__t_sectors_hit_niso_rd",
                "crop__input_requests_pitch_linear",
                "lts__t_sectors_hit_gpc_atomic",
                "lts__t_sectors_hit_host_cpu",
                "l1tex__f_output_tex2sm_busy",
                "lts__t_sectors_miss_hub_prefetch",
                "lts__t_sectors_hit_clear",
                "lts__t_requests_atomic",
                "lts__mccif_write_requests",
                "pel__out_l2_requests_ilwalidate_256b",
                "lts__mccif_read_stalled",
                "lts__t_requests_crd_i_condrd",
                "lts__t_requests_zrop_ilwal",
                "swdx__input_stalled",
                "l1tex__t_set_accesses_hit_in_warp_lg_local_ld",
                "lts__t_sectors_miss_ltc_rd",
                "lts__t_sectors_gpc_clear",
                "lts__d_atomic_block_stalled_backpressure",
                "l1tex__t_set_accesses_hit_in_warp_lg_global_ld",
                "lts__t_requests_cbc",
                "lts__t_sectors_host_cpu_cas",
                "pel__in_write_requests_active",
                "zrop__input_requests_write_compression_ok",
                "lts__t_sectors_hit_mmu",
                "l1tex__t_set_accesses_lg_global_red",
                "mpc__isbe_allocation_stalled_beta",
                "l1tex__f_output_tex2sm_active",
                "pda__input_restart_indices",
                "lts__t_requests_gpc_clear",
                "lts__t_sectors_hub_concat_cas",
                "lts__d_decomp_input_sectors",
                "sm__threads_launched_vsa",
                "l1tex__t_t2d_backpressured",
                "l1tex__t_set_accesses_lg_global_ld",
                "lts__t_sectors_hit_crop_condrd",
                "zrop__processed_requests_type_plane_eq_fast",
                "tgb__tasks_active",
                "lts__t_sectors_hit_hub_prefetch",
                "sm__cycles_active_vsb",
                "prop__csb_output_pixels",
                "sm__warps_completed_ooo_3d_ps",
                "prop__csb_killed_quadquads",
                "lts__t_sectors_membar",
                "vpc__beta_cycles_active",
                "lts__t_sectors_l1tex_atomic",
                "vaf__cycles_stalled",
                "lts__t_sectors_hit_pe_ilwal",
                "lts__t_sectors_hit_hub_membar",
                "crop__input_requests_blend_blendopt_killed",
                "stri__attrs",
                "l1tex__t_set_accesses_hit_tex_format_lwbemap",
                "tga__cycles_active",
                "l1tex__t_atomic_address_conflicts_lg_global_red",
                0
            };
            return s_rawMetricNames;
        }

        RawMetricsStorage() {}
    };

    struct RawMetrics : public RawMetricsStorage
    {
        MetricValue gpmsd__input_quads_3d()
        {
            return GetValue(RawMetricIdx::gpmsd__input_quads_3d, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__earlyz_killed_samples()
        {
            return GetValue(RawMetricIdx::prop__earlyz_killed_samples, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue tgb__output_verts_complex_boundary()
        {
            return GetValue(RawMetricIdx::tgb__output_verts_complex_boundary, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_320()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_320, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_quads()
        {
            return GetValue(RawMetricIdx::gpmsd__input_quads, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrd_ni_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrd_ni_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__read_subpackets()
        {
            return GetValue(RawMetricIdx::crop__read_subpackets, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__clip_cycles_active()
        {
            return GetValue(RawMetricIdx::vpc__clip_cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_3d_vtg()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_3d_vtg, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__isbe_allocations_beta()
        {
            return GetValue(RawMetricIdx::mpc__isbe_allocations_beta, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__input_samples_part1()
        {
            return GetValue(RawMetricIdx::crop__input_samples_part1, 63.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_niso_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_niso_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_surface_red()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_surface_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_l1tex_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_l1tex_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crd_i_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crd_i_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_write_sectors_excluding_16()
        {
            return GetValue(RawMetricIdx::lts__mccif_write_sectors_excluding_16, 2.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_vs()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_vs, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gcc__tsl2_requests()
        {
            return GetValue(RawMetricIdx::gcc__tsl2_requests, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_su_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_su_pred_off_all, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_raster_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_raster_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__cycles_active()
        {
            return GetValue(RawMetricIdx::swdx__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__input_prims_triangle()
        {
            return GetValue(RawMetricIdx::vpc__input_prims_triangle, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_pe_read_isbe()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_pe_read_isbe, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_samples_2d()
        {
            return GetValue(RawMetricIdx::gpmsd__input_samples_2d, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_active_3d()
        {
            return GetValue(RawMetricIdx::gpmsd__input_active_3d, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_tex_format_3d()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_tex_format_3d, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__ctas_launched()
        {
            return GetValue(RawMetricIdx::sm__ctas_launched, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__isbe_allocations_alpha()
        {
            return GetValue(RawMetricIdx::mpc__isbe_allocations_alpha, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_tex_format_lwbemap()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_tex_format_lwbemap, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_lsu()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_lsu, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crop_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crop_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_pagepool_full()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_pagepool_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mmu__hubtlb_requests_hit_under_miss()
        {
            return GetValue(RawMetricIdx::mmu__hubtlb_requests_hit_under_miss, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_samp2mipb_active()
        {
            return GetValue(RawMetricIdx::l1tex__samp_samp2mipb_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_iso_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_iso_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_no_instructions_0()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_no_instructions_0, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__ps_quads_launched()
        {
            return GetValue(RawMetricIdx::sm__ps_quads_launched, 2.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pda__input_prims_patch()
        {
            return GetValue(RawMetricIdx::pda__input_prims_patch, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_membar()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_membar, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__ps_warps_killed()
        {
            return GetValue(RawMetricIdx::sm__ps_warps_killed, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_acache_requests_acache_hit()
        {
            return GetValue(RawMetricIdx::vaf__beta_acache_requests_acache_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrop_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrop_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gcc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gcc_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrop_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrop_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrd_ni()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrd_ni, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue cwd__output_ctas()
        {
            return GetValue(RawMetricIdx::cwd__output_ctas, 0.5, sys__cycles_elapsed().avg);
        }

        MetricValue gcc__l15_requests_hit()
        {
            return GetValue(RawMetricIdx::gcc__l15_requests_hit, 0.5, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_all_color_channels()
        {
            return GetValue(RawMetricIdx::crop__input_requests_all_color_channels, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_surface_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_surface_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_drain_timeout()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_drain_timeout, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_iso()
        {
            return GetValue(RawMetricIdx::lts__t_requests_iso, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_tex_wb_pending()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_tex_wb_pending, 80.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_allocation_stall_0()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_allocation_stall_0, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_cycles_active()
        {
            return GetValue(RawMetricIdx::vaf__beta_cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__read_requests()
        {
            return GetValue(RawMetricIdx::zrop__read_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mme__input_method_dwords()
        {
            return GetValue(RawMetricIdx::mme__input_method_dwords, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue vpc__beta_tasks_active()
        {
            return GetValue(RawMetricIdx::vpc__beta_tasks_active, 63.0, gpc__cycles_elapsed().avg);
        }

        MetricValue pda__input_prims_triflat()
        {
            return GetValue(RawMetricIdx::pda__input_prims_triflat, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mmu__hubtlb_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::mmu__hubtlb_stalled_on_tag_allocation, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue zrop__read_subpackets_stencil()
        {
            return GetValue(RawMetricIdx::zrop__read_subpackets_stencil, 8.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__earlyz_killed_quads()
        {
            return GetValue(RawMetricIdx::prop__earlyz_killed_quads, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_lg_local_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_lg_local_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sys__fb_read_dwords()
        {
            return GetValue(RawMetricIdx::sys__fb_read_dwords, 4.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_576()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_576, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpu__dispatch_count()
        {
            return GetValue(RawMetricIdx::gpu__dispatch_count, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrop_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrop_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_lsu_write()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_lsu_write, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_active_compute()
        {
            return GetValue(RawMetricIdx::gpmsd__input_active_compute, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__threads_launched_gs()
        {
            return GetValue(RawMetricIdx::sm__threads_launched_gs, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_gs_fast_beta()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_gs_fast_beta, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_pe()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_pe, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_lwlled_occluders_near_far_clipped()
        {
            return GetValue(RawMetricIdx::raster__zlwll_lwlled_occluders_near_far_clipped, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gcc__l15_requests_hit_constant()
        {
            return GetValue(RawMetricIdx::gcc__l15_requests_hit_constant, 0.5, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_256()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_256, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__miop_pq_read_active_smp0()
        {
            return GetValue(RawMetricIdx::sm__miop_pq_read_active_smp0, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_samples()
        {
            return GetValue(RawMetricIdx::gpmsd__input_samples, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_niso_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_niso_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crd_ni_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crd_ni_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__latez_rstrz_input_quads()
        {
            return GetValue(RawMetricIdx::prop__latez_rstrz_input_quads, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_cs()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_cs, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fma64plus_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fma64plus_pred_on, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__rg_utlb_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::gpc__rg_utlb_stalled_on_tag_allocation, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sked__dispatch_active_scg0()
        {
            return GetValue(RawMetricIdx::sked__dispatch_active_scg0, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue vpc__input_isbes()
        {
            return GetValue(RawMetricIdx::vpc__input_isbes, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue pdb__output_stalled()
        {
            return GetValue(RawMetricIdx::pdb__output_stalled, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_surface_red()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_surface_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_global_atom()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_global_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_cycles_elapsed()
        {
            return GetValue(RawMetricIdx::vaf__beta_cycles_elapsed, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_tex()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_tex, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrop_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrop_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crd_i()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crd_i, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_vs()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_vs, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__setup_lwlled_prims()
        {
            return GetValue(RawMetricIdx::raster__setup_lwlled_prims, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue tga__input_patches()
        {
            return GetValue(RawMetricIdx::tga__input_patches, 32.0, sys__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_eligible()
        {
            return GetValue(RawMetricIdx::smsp__warps_eligible, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__input_batches()
        {
            return GetValue(RawMetricIdx::mpc__input_batches, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrop_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrop_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_explicit()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_explicit, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__crstr_lwlled_prims_no_tile_coverage()
        {
            return GetValue(RawMetricIdx::raster__crstr_lwlled_prims_no_tile_coverage, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_samp2mipb_busy()
        {
            return GetValue(RawMetricIdx::l1tex__samp_samp2mipb_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_samp2mipb_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__samp_samp2mipb_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_local_cctl()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_local_cctl, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__w_w2d_busy()
        {
            return GetValue(RawMetricIdx::l1tex__w_w2d_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__subtiles_active()
        {
            return GetValue(RawMetricIdx::sm__subtiles_active, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_tasks_active()
        {
            return GetValue(RawMetricIdx::vaf__beta_tasks_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_lwlled_occluders_depth_bounds()
        {
            return GetValue(RawMetricIdx::raster__zlwll_lwlled_occluders_depth_bounds, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_output_subtiles_4_cycle()
        {
            return GetValue(RawMetricIdx::raster__frstr_output_subtiles_4_cycle, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue fe__i2m_write_stalled_request_fifo_full()
        {
            return GetValue(RawMetricIdx::fe__i2m_write_stalled_request_fifo_full, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_vsb()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_vsb, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_request_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_request_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_tex_format_3d()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_tex_format_3d, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrop_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrop_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrd_i_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrd_i_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crop_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crop_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_cs()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_cs, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_samples_3d()
        {
            return GetValue(RawMetricIdx::gpmsd__input_samples_3d, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__thread_inst_exelwted()
        {
            return GetValue(RawMetricIdx::smsp__thread_inst_exelwted, 48.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_resseq_cycles_bank_conflict()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_resseq_cycles_bank_conflict, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_bru_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_bru_pred_on, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue vpc__clip_input_prims_clipped_multi_plane()
        {
            return GetValue(RawMetricIdx::vpc__clip_input_prims_clipped_multi_plane, 0.007692307692307693, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__cdp_alpha_blendopt_killed_pixels()
        {
            return GetValue(RawMetricIdx::prop__cdp_alpha_blendopt_killed_pixels, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_ps()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_ps, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_raster_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_raster_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants_miss()
        {
            return GetValue(RawMetricIdx::scc__load_constants_miss, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_surface_atom()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_surface_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_filter_trilinear()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_filter_trilinear, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrd_i_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrd_i_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_atomic, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_valid()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_valid, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crd_ni()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crd_ni, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue cbmgr__beta_cbe_allocations()
        {
            return GetValue(RawMetricIdx::cbmgr__beta_cbe_allocations, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_reason_backfacing()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_reason_backfacing, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_mmu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_mmu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_surface_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_surface_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc0_utlb_requests_miss()
        {
            return GetValue(RawMetricIdx::gpc__tpc0_utlb_requests_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_line()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_line, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_tex_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_tex_pred_off_all, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue gr__cycles_idle()
        {
            return GetValue(RawMetricIdx::gr__cycles_idle, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue gpc__prop_utlb_requests_hit_under_miss()
        {
            return GetValue(RawMetricIdx::gpc__prop_utlb_requests_hit_under_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_tex_format_2d_nomipmap()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_tex_format_2d_nomipmap, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrop_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrop_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_vsa()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_vsa, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_null_lg()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_null_lg, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_fill()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_fill, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_l1tex()
        {
            return GetValue(RawMetricIdx::lts__t_requests_l1tex, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fma64plus_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fma64plus_pred_off_all, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue mmu__hubtlb_stalled_request_fifo_full()
        {
            return GetValue(RawMetricIdx::mmu__hubtlb_stalled_request_fifo_full, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crop_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crop_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_ps()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_ps, 0.25, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__tc_output_subtiles()
        {
            return GetValue(RawMetricIdx::raster__tc_output_subtiles, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue zrop__input_stalled()
        {
            return GetValue(RawMetricIdx::zrop__input_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_misc_0()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_misc_0, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_cycles_64()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_cycles_64, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__earlyz_output_samples()
        {
            return GetValue(RawMetricIdx::prop__earlyz_output_samples, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_active_vsa()
        {
            return GetValue(RawMetricIdx::sm__warps_active_vsa, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_samp2mipb_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__samp_samp2mipb_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__cycles_active()
        {
            return GetValue(RawMetricIdx::prop__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__clip_input_prims_clipped()
        {
            return GetValue(RawMetricIdx::vpc__clip_input_prims_clipped, 0.014492753623188406, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_atomic_address_conflicts_surface_red()
        {
            return GetValue(RawMetricIdx::l1tex__t_atomic_address_conflicts_surface_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_iso_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_iso_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__tex_requests()
        {
            return GetValue(RawMetricIdx::smsp__tex_requests, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_niso()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_niso, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__processed_subpackets()
        {
            return GetValue(RawMetricIdx::crop__processed_subpackets, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__latez_output_samples()
        {
            return GetValue(RawMetricIdx::prop__latez_output_samples, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__input_prims_point()
        {
            return GetValue(RawMetricIdx::vpc__input_prims_point, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue gpu__time_active()
        {
            return GetValue(RawMetricIdx::gpu__time_active, 10.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_local_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_local_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pdb__input_batches()
        {
            return GetValue(RawMetricIdx::pdb__input_batches, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrd_ni_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrd_ni_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_level_1_threshold()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_level_1_threshold, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_write_request_latency_64()
        {
            return GetValue(RawMetricIdx::lts__mccif_write_request_latency_64, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__in_active()
        {
            return GetValue(RawMetricIdx::pel__in_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmpd__cycles_active()
        {
            return GetValue(RawMetricIdx::gpmpd__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_stalled()
        {
            return GetValue(RawMetricIdx::prop__csb_output_stalled, 128.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__tram_allocation_stalled()
        {
            return GetValue(RawMetricIdx::mpc__tram_allocation_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_tex_format_1d2d_tex()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_tex_format_1d2d_tex, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrop_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrop_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_request_active()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_request_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_surface_atom()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_surface_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_bar_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_bar_pred_on, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue mmu__hubtlb_requests_miss()
        {
            return GetValue(RawMetricIdx::mmu__hubtlb_requests_miss, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_fetched_attr_scalar_post_cbf()
        {
            return GetValue(RawMetricIdx::vaf__alpha_fetched_attr_scalar_post_cbf, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_128()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_128, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__lsu_write_back_active()
        {
            return GetValue(RawMetricIdx::smsp__lsu_write_back_active, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_raster()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_raster, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_triangle()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_triangle, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pda__output_verts()
        {
            return GetValue(RawMetricIdx::pda__output_verts, 12.0, sys__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc0_utlb_requests_sparse()
        {
            return GetValue(RawMetricIdx::gpc__tpc0_utlb_requests_sparse, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_bru()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_bru, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_lsu_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_lsu_pred_on, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_l1tex()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_l1tex, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__earlyz_killed_pixels()
        {
            return GetValue(RawMetricIdx::prop__earlyz_killed_pixels, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin2m_fifo_output_busy()
        {
            return GetValue(RawMetricIdx::l1tex__texin2m_fifo_output_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__rg_utlb_stalled_request_fifo_full()
        {
            return GetValue(RawMetricIdx::gpc__rg_utlb_stalled_request_fifo_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__gnic_port1_stalled()
        {
            return GetValue(RawMetricIdx::prop__gnic_port1_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue pda__input_prims_line()
        {
            return GetValue(RawMetricIdx::pda__input_prims_line, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crop_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crop_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_x2w_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__x_x2w_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__processed_requests_type_shdz_biquad()
        {
            return GetValue(RawMetricIdx::zrop__processed_requests_type_shdz_biquad, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_fetched_attr_scalar_indexed()
        {
            return GetValue(RawMetricIdx::vaf__alpha_fetched_attr_scalar_indexed, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_niso()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_niso, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_raster()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_raster, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__out_l2_requests_read()
        {
            return GetValue(RawMetricIdx::pel__out_l2_requests_read, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__icc_requests_miss_covered()
        {
            return GetValue(RawMetricIdx::sm__icc_requests_miss_covered, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue stri__to_acache_stalled_on_read()
        {
            return GetValue(RawMetricIdx::stri__to_acache_stalled_on_read, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_pixels()
        {
            return GetValue(RawMetricIdx::gpmsd__input_pixels, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__icc_requests_miss_no_tags()
        {
            return GetValue(RawMetricIdx::sm__icc_requests_miss_no_tags, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__input_stalled()
        {
            return GetValue(RawMetricIdx::prop__input_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__miop_pq_read_active_tex_smp0()
        {
            return GetValue(RawMetricIdx::sm__miop_pq_read_active_tex_smp0, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__miop_pq_write_active()
        {
            return GetValue(RawMetricIdx::smsp__miop_pq_write_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_x2w_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__x_x2w_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_samp2x_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__samp_samp2x_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_x2t_busy()
        {
            return GetValue(RawMetricIdx::l1tex__x_x2t_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc0_utlb_requests_hit_under_miss()
        {
            return GetValue(RawMetricIdx::gpc__tpc0_utlb_requests_hit_under_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__output_verts()
        {
            return GetValue(RawMetricIdx::vpc__output_verts, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crop_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crop_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__input_messages()
        {
            return GetValue(RawMetricIdx::swdx__input_messages, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__pixel_shader_barriers()
        {
            return GetValue(RawMetricIdx::prop__pixel_shader_barriers, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_output_subtiles_2_cycle()
        {
            return GetValue(RawMetricIdx::raster__frstr_output_subtiles_2_cycle, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_gs_fast_alpha()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_gs_fast_alpha, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_write_request_latency_48()
        {
            return GetValue(RawMetricIdx::lts__mccif_write_request_latency_48, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_global_cctl()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_global_cctl, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_3d_ps()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_3d_ps, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_tags_writeback_tier2_issued()
        {
            return GetValue(RawMetricIdx::lts__t_tags_writeback_tier2_issued, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__w_w2d_active()
        {
            return GetValue(RawMetricIdx::l1tex__w_w2d_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__latez_shdz_input_samples()
        {
            return GetValue(RawMetricIdx::prop__latez_shdz_input_samples, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_miss()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_miss, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue prop__earlyz_output_pixels()
        {
            return GetValue(RawMetricIdx::prop__earlyz_output_pixels, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_ldc()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_ldc, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__setup_output_prims()
        {
            return GetValue(RawMetricIdx::raster__setup_output_prims, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_acache_requests_local_hit()
        {
            return GetValue(RawMetricIdx::vaf__alpha_acache_requests_local_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sys__fb_write_requests()
        {
            return GetValue(RawMetricIdx::sys__fb_write_requests, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_miss_clear_zbc()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_miss_clear_zbc, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__warps_retiring_ps_stalled_not_selected()
        {
            return GetValue(RawMetricIdx::sm__warps_retiring_ps_stalled_not_selected, 128.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gcc__l15_requests_miss_instruction()
        {
            return GetValue(RawMetricIdx::gcc__l15_requests_miss_instruction, 0.5, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__setup_output_prims_line()
        {
            return GetValue(RawMetricIdx::raster__setup_output_prims_line, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_xu_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_xu_pred_on, 0.25, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__miop_pq_read_active_pixout_smp1()
        {
            return GetValue(RawMetricIdx::sm__miop_pq_read_active_pixout_smp1, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__crstr_search_stalling_input()
        {
            return GetValue(RawMetricIdx::raster__crstr_search_stalling_input, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__output_active()
        {
            return GetValue(RawMetricIdx::swdx__output_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__input_verts_alpha()
        {
            return GetValue(RawMetricIdx::mpc__input_verts_alpha, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__write_subpackets_stencil()
        {
            return GetValue(RawMetricIdx::zrop__write_subpackets_stencil, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrd_i_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrd_i_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pdb__output_stalled_beta()
        {
            return GetValue(RawMetricIdx::pdb__output_stalled_beta, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_pe_write_isbe()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_pe_write_isbe, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrop_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrop_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_stalled()
        {
            return GetValue(RawMetricIdx::crop__input_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pda__input_stalled_index_fetch()
        {
            return GetValue(RawMetricIdx::pda__input_stalled_index_fetch, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue fe__i2m_write_requests()
        {
            return GetValue(RawMetricIdx::fe__i2m_write_requests, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_not_selected()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_not_selected, 5.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crd_i_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crd_i_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_occluders_zfail()
        {
            return GetValue(RawMetricIdx::raster__zlwll_occluders_zfail, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__csb_input_samples_3d()
        {
            return GetValue(RawMetricIdx::prop__csb_input_samples_3d, 128.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_tex_bilinear()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_tex_bilinear, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sys__pushbuffer_dwords()
        {
            return GetValue(RawMetricIdx::sys__pushbuffer_dwords, 16.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__miop_pq_read_active_lsu_smp0()
        {
            return GetValue(RawMetricIdx::sm__miop_pq_read_active_lsu_smp0, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crd_ni_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crd_ni_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_decomp_input_stalled()
        {
            return GetValue(RawMetricIdx::lts__d_decomp_input_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__csb_input_pixels_3d()
        {
            return GetValue(RawMetricIdx::prop__csb_input_pixels_3d, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__cycles_active()
        {
            return GetValue(RawMetricIdx::lts__cycles_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_acache_requests_acache_hit()
        {
            return GetValue(RawMetricIdx::vaf__alpha_acache_requests_acache_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue wwdx__input_tasks()
        {
            return GetValue(RawMetricIdx::wwdx__input_tasks, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_adu()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_adu, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue pdb__cycles_active()
        {
            return GetValue(RawMetricIdx::pdb__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue tgb__output_verts_complex_interior()
        {
            return GetValue(RawMetricIdx::tgb__output_verts_complex_interior, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crd_ni_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crd_ni_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__ltcx_read_requests()
        {
            return GetValue(RawMetricIdx::lts__ltcx_read_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pes__stream_output_prims()
        {
            return GetValue(RawMetricIdx::pes__stream_output_prims, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_reason_bounding_box()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_reason_bounding_box, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_global_atom()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_global_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_tes()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_tes, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_reqseq_stalled_source_not_ready()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_reqseq_stalled_source_not_ready, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__input_pixel_shader_barriers()
        {
            return GetValue(RawMetricIdx::swdx__input_pixel_shader_barriers, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_pe_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_pe_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_pe_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_pe_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_miss_tex_format_no_mipmap()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_miss_tex_format_no_mipmap, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants_stalled_max_versions()
        {
            return GetValue(RawMetricIdx::scc__load_constants_stalled_max_versions, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__idc_requests_miss_to_gcc()
        {
            return GetValue(RawMetricIdx::sm__idc_requests_miss_to_gcc, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrop()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrop, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_acache_requests()
        {
            return GetValue(RawMetricIdx::vaf__alpha_acache_requests, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gcc__l15_requests_instruction()
        {
            return GetValue(RawMetricIdx::gcc__l15_requests_instruction, 0.5, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__r_input_fifo_stalled()
        {
            return GetValue(RawMetricIdx::lts__r_input_fifo_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_crop_requests()
        {
            return GetValue(RawMetricIdx::prop__csb_output_crop_requests, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sked__dispatch_count()
        {
            return GetValue(RawMetricIdx::sked__dispatch_count, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue zrop__input_requests_type_bundles()
        {
            return GetValue(RawMetricIdx::zrop__input_requests_type_bundles, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_reqseq_cycles_bank_conflict()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_reqseq_cycles_bank_conflict, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__gnic_port0_stalled()
        {
            return GetValue(RawMetricIdx::prop__gnic_port0_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_sz_96b_nearest()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_sz_96b_nearest, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_cctlt()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_cctlt, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants_page_first_update()
        {
            return GetValue(RawMetricIdx::scc__load_constants_page_first_update, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_cbc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_cbc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__isbe_allocation_stalled_alpha()
        {
            return GetValue(RawMetricIdx::mpc__isbe_allocation_stalled_alpha, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__cdp_alpha_to_coverage_output_pixels()
        {
            return GetValue(RawMetricIdx::prop__cdp_alpha_to_coverage_output_pixels, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrop()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrop, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::l1tex__m_stalled_on_tag_allocation, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_surface_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_surface_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_pe_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_pe_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__w_w2d_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__w_w2d_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_l1tex_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_l1tex_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_input_batches_post_cbf()
        {
            return GetValue(RawMetricIdx::vaf__alpha_input_batches_post_cbf, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrd_i_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrd_i_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_to_acache_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::vaf__alpha_to_acache_stalled_on_tag_allocation, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_surface_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_surface_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_killed_subtiles_input()
        {
            return GetValue(RawMetricIdx::raster__frstr_killed_subtiles_input, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__miop_pq_read_active_pixout_smp0()
        {
            return GetValue(RawMetricIdx::sm__miop_pq_read_active_pixout_smp0, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_tes()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_tes, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrd_ni_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrd_ni_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_tags_writeback_tier1_killed()
        {
            return GetValue(RawMetricIdx::lts__t_tags_writeback_tier1_killed, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_raster_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_raster_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__output_prims_stippled_line()
        {
            return GetValue(RawMetricIdx::vpc__output_prims_stippled_line, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_replay_tiled()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_replay_tiled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gcc__tsl2_requests_hit()
        {
            return GetValue(RawMetricIdx::gcc__tsl2_requests_hit, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_math_pipe_throttle()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_math_pipe_throttle, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pdb__output_pkts()
        {
            return GetValue(RawMetricIdx::pdb__output_pkts, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_hit_clear_zbc_crop()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_hit_clear_zbc_crop, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__input_requests_type_plane_eq()
        {
            return GetValue(RawMetricIdx::zrop__input_requests_type_plane_eq, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_cycles_elapsed()
        {
            return GetValue(RawMetricIdx::vaf__alpha_cycles_elapsed, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__imc_requests_miss_covered()
        {
            return GetValue(RawMetricIdx::smsp__imc_requests_miss_covered, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fma64plus()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fma64plus, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_clear()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_clear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__input_requests()
        {
            return GetValue(RawMetricIdx::zrop__input_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__tram_fill_fifo_stalled()
        {
            return GetValue(RawMetricIdx::mpc__tram_fill_fifo_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crd_ni()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crd_ni, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_lg_global_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_lg_global_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_fetched_attr_scalar_indexed_constant()
        {
            return GetValue(RawMetricIdx::vaf__alpha_fetched_attr_scalar_indexed_constant, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_lg_local_cctl()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_local_cctl, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue rdm__cycles_active()
        {
            return GetValue(RawMetricIdx::rdm__cycles_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_clear()
        {
            return GetValue(RawMetricIdx::lts__t_requests_clear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__input_active()
        {
            return GetValue(RawMetricIdx::prop__input_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_draining_ps()
        {
            return GetValue(RawMetricIdx::sm__warps_draining_ps, 128.0, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__cycles_active()
        {
            return GetValue(RawMetricIdx::zrop__cycles_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue tga__output_prims()
        {
            return GetValue(RawMetricIdx::tga__output_prims, 32.0, sys__cycles_elapsed().avg);
        }

        MetricValue rdm__crop_output_stalled()
        {
            return GetValue(RawMetricIdx::rdm__crop_output_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__input_requests_type_shdz_biquad()
        {
            return GetValue(RawMetricIdx::zrop__input_requests_type_shdz_biquad, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_pe()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_pe, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrd_ni_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrd_ni_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_atomic_address_conflicts_surface_atom()
        {
            return GetValue(RawMetricIdx::l1tex__t_atomic_address_conflicts_surface_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gcc__l15_requests_constant()
        {
            return GetValue(RawMetricIdx::gcc__l15_requests_constant, 0.5, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_pe_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_pe_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pda__input_prims_lineadj()
        {
            return GetValue(RawMetricIdx::pda__input_prims_lineadj, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_raster()
        {
            return GetValue(RawMetricIdx::lts__t_requests_raster, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_tile_allocation_stall()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_tile_allocation_stall, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__input_isbes_prim()
        {
            return GetValue(RawMetricIdx::vpc__input_isbes_prim, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue swdx__input_active()
        {
            return GetValue(RawMetricIdx::swdx__input_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue tga__output_tasks_complex_boundary()
        {
            return GetValue(RawMetricIdx::tga__output_tasks_complex_boundary, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin2m_fifo_output_active()
        {
            return GetValue(RawMetricIdx::l1tex__texin2m_fifo_output_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_tex_format_3d()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_tex_format_3d, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_stalled()
        {
            return GetValue(RawMetricIdx::gpmsd__input_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_tex_format_no_mipmap()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_tex_format_no_mipmap, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__input_pixels_3d()
        {
            return GetValue(RawMetricIdx::prop__input_pixels_3d, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc1_utlb_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::gpc__tpc1_utlb_stalled_on_tag_allocation, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_tags_writeback_tier1_issued()
        {
            return GetValue(RawMetricIdx::lts__t_tags_writeback_tier1_issued, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_vsb()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_vsb, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_mmu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_mmu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_clear()
        {
            return GetValue(RawMetricIdx::crop__input_requests_clear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__prop_utlb_stalled_request_fifo_full()
        {
            return GetValue(RawMetricIdx::gpc__prop_utlb_stalled_request_fifo_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrop_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrop_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_small_page_1()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_small_page_1, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_sm2tex_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__texin_sm2tex_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__input_stalled_upstream_fifo_full()
        {
            return GetValue(RawMetricIdx::crop__input_stalled_upstream_fifo_full, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_hit_tex_format_1d_buffer()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_hit_tex_format_1d_buffer, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sector_lwmulative_miss_latency()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sector_lwmulative_miss_latency, 1023.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_stalled_on_tsl2_miss()
        {
            return GetValue(RawMetricIdx::l1tex__texin_stalled_on_tsl2_miss, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__cycles_elapsed()
        {
            MetricValue metricValue;
            if (configuring)
            {
                counts[RawMetricIdx::gpc__cycles_elapsed] = 1;
                metricValue = {};
            }
            else
            {
                const double unitCount = counts[RawMetricIdx::gpc__cycles_elapsed];
                metricValue.sum = values[RawMetricIdx::gpc__cycles_elapsed];
                metricValue.avg = metricValue.sum / unitCount;
                metricValue.peak_sustained = 1.0;
                metricValue.cycles_elapsed = metricValue.avg;
            }
            return metricValue;
        }

        MetricValue sm__warps_active()
        {
            return GetValue(RawMetricIdx::sm__warps_active, 128.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crd_i_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crd_i_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_niso_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_niso_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__input_quads()
        {
            return GetValue(RawMetricIdx::prop__input_quads, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue fe__i2m_write_stalled_data_buffer_full()
        {
            return GetValue(RawMetricIdx::fe__i2m_write_stalled_data_buffer_full, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue pda__input_prims_tri()
        {
            return GetValue(RawMetricIdx::pda__input_prims_tri, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue crop__write_requests_compressed_4to1()
        {
            return GetValue(RawMetricIdx::crop__write_requests_compressed_4to1, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_sz_32b_nearest()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_sz_32b_nearest, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_active()
        {
            return GetValue(RawMetricIdx::prop__csb_output_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_vsa()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_vsa, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue stri__cycles_stalled()
        {
            return GetValue(RawMetricIdx::stri__cycles_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrop_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrop_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrop_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrop_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_mmu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_mmu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_miss_tex_format_lwbemap()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_miss_tex_format_lwbemap, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_2d()
        {
            return GetValue(RawMetricIdx::crop__input_requests_2d, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_lg_global_atom()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_lg_global_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue wwdx__output_stalled()
        {
            return GetValue(RawMetricIdx::wwdx__output_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gcc__tsl2_requests_miss()
        {
            return GetValue(RawMetricIdx::gcc__tsl2_requests_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrop_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrop_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue cwd__output_stalled_no_free_slots()
        {
            return GetValue(RawMetricIdx::cwd__output_stalled_no_free_slots, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_raster_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_raster_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_tes()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_tes, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_gs()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_gs, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_l1tex()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_l1tex, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_sectors_promoted()
        {
            return GetValue(RawMetricIdx::l1tex__t_sectors_promoted, 12.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_mmu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_mmu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__cycles_active()
        {
            return GetValue(RawMetricIdx::vaf__cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__cdp_alpha_blendopt_read_avoid()
        {
            return GetValue(RawMetricIdx::prop__cdp_alpha_blendopt_read_avoid, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_lg_global_st()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_global_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_big_page_3()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_big_page_3, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_surface_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_surface_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_surface_st()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_surface_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_constant_table_full()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_constant_table_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_cbc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_cbc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_mmu()
        {
            return GetValue(RawMetricIdx::lts__t_requests_mmu, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_active()
        {
            return GetValue(RawMetricIdx::smsp__warps_active, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__threads_launched_vsb()
        {
            return GetValue(RawMetricIdx::sm__threads_launched_vsb, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue tgb__output_prims_triangle()
        {
            return GetValue(RawMetricIdx::tgb__output_prims_triangle, 4.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin2m_fifo_output_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__texin2m_fifo_output_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_pe_write_tram()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_pe_write_tram, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_l1tex_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_l1tex_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__processed_requests_compressed()
        {
            return GetValue(RawMetricIdx::crop__processed_requests_compressed, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_ps()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_ps, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_membar()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_membar, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__lod_output_wavefronts()
        {
            return GetValue(RawMetricIdx::l1tex__lod_output_wavefronts, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vpc__cycles_active()
        {
            return GetValue(RawMetricIdx::vpc__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_niso_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_niso_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue cbmgr__beta_cbe_allocation_stalled_max_tasks()
        {
            return GetValue(RawMetricIdx::cbmgr__beta_cbe_allocation_stalled_max_tasks, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue stri__to_acache_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::stri__to_acache_stalled_on_tag_allocation, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_global_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_global_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_iso_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_iso_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_tex_format_1d2d_tex()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_tex_format_1d2d_tex, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__inst_exelwted_pipe_adu_divergent_smp0()
        {
            return GetValue(RawMetricIdx::sm__inst_exelwted_pipe_adu_divergent_smp0, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__processed_subpackets_stencil()
        {
            return GetValue(RawMetricIdx::zrop__processed_subpackets_stencil, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrd_ni_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrd_ni_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpmpd__output_packets()
        {
            return GetValue(RawMetricIdx::gpmpd__output_packets, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_to_acache_stalled_on_read()
        {
            return GetValue(RawMetricIdx::vaf__alpha_to_acache_stalled_on_read, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__processed_requests()
        {
            return GetValue(RawMetricIdx::zrop__processed_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue tpc__cycles_elapsed()
        {
            MetricValue metricValue;
            if (configuring)
            {
                counts[RawMetricIdx::tpc__cycles_elapsed] = 1;
                metricValue = {};
            }
            else
            {
                const double unitCount = counts[RawMetricIdx::tpc__cycles_elapsed];
                metricValue.sum = values[RawMetricIdx::tpc__cycles_elapsed];
                metricValue.avg = metricValue.sum / unitCount;
                metricValue.peak_sustained = 1.0;
                metricValue.cycles_elapsed = metricValue.avg;
            }
            return metricValue;
        }

        MetricValue l1tex__texin_requests_lg_global_cctl()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_global_cctl, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_hit_tex_format_no_mipmap()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_hit_tex_format_no_mipmap, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__read_returns()
        {
            return GetValue(RawMetricIdx::zrop__read_returns, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__output_ops()
        {
            return GetValue(RawMetricIdx::fe__output_ops, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__warps_active_tcs()
        {
            return GetValue(RawMetricIdx::sm__warps_active_tcs, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pel__out_active()
        {
            return GetValue(RawMetricIdx::pel__out_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_tex_format_3d()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_tex_format_3d, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_write_request_latency_high()
        {
            return GetValue(RawMetricIdx::lts__mccif_write_request_latency_high, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__output_prims_line()
        {
            return GetValue(RawMetricIdx::vpc__output_prims_line, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc1_utlb_stalled_write_buffer_full()
        {
            return GetValue(RawMetricIdx::gpc__tpc1_utlb_stalled_write_buffer_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_comptag_miss()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_comptag_miss, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrd_i_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrd_i_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrd_i_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrd_i_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrop_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrop_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_512()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_512, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_tex()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_tex, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue stri__acache_requests_stri_local_hit()
        {
            return GetValue(RawMetricIdx::stri__acache_requests_stri_local_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue cwd__cycles_active()
        {
            return GetValue(RawMetricIdx::cwd__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__out_l2_requests_write_512b()
        {
            return GetValue(RawMetricIdx::pel__out_l2_requests_write_512b, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_concat_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__idc_requests_miss_no_tags()
        {
            return GetValue(RawMetricIdx::sm__idc_requests_miss_no_tags, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc0_utlb_requests_hit()
        {
            return GetValue(RawMetricIdx::gpc__tpc0_utlb_requests_hit, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_clear()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_clear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__cdp_alpha_test_killed_pixels()
        {
            return GetValue(RawMetricIdx::prop__cdp_alpha_test_killed_pixels, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_active_vsb()
        {
            return GetValue(RawMetricIdx::sm__warps_active_vsb, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crd_ni()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crd_ni, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_concat_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__miop_adu_replays_smp1()
        {
            return GetValue(RawMetricIdx::sm__miop_adu_replays_smp1, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pda__input_verts()
        {
            return GetValue(RawMetricIdx::pda__input_verts, 32.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_pe_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_pe_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_small_page_0()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_small_page_0, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_tex_format_1d_buffer()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_tex_format_1d_buffer, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_mmu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_mmu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_gs_fast_alpha()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_gs_fast_alpha, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__rg_utlb_requests_hit()
        {
            return GetValue(RawMetricIdx::gpc__rg_utlb_requests_hit, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__output_batches()
        {
            return GetValue(RawMetricIdx::mpc__output_batches, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__ltcx_write_stalled_fifo_full()
        {
            return GetValue(RawMetricIdx::lts__ltcx_write_stalled_fifo_full, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_pe_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_pe_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_raster_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_raster_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpu__time_duration()
        {
            return GetValue(RawMetricIdx::gpu__time_duration, 10.0, sys__cycles_elapsed().avg);
        }

        MetricValue raster__crstr_output_tiles()
        {
            return GetValue(RawMetricIdx::raster__crstr_output_tiles, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__imc_requests_miss_to_gcc()
        {
            return GetValue(RawMetricIdx::smsp__imc_requests_miss_to_gcc, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_global_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_global_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_iso_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_iso_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__write_requests()
        {
            return GetValue(RawMetricIdx::crop__write_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crd_ni_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crd_ni_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_aamode_8xaa()
        {
            return GetValue(RawMetricIdx::crop__input_requests_aamode_8xaa, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_block_cycles_serialized()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_block_cycles_serialized, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_pe_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_pe_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__gnic_port1_active()
        {
            return GetValue(RawMetricIdx::prop__gnic_port1_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_replay_not_tiled()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_replay_not_tiled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue wwdx__cycles_active()
        {
            return GetValue(RawMetricIdx::wwdx__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gcc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gcc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_tex_format_lwbemap()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_tex_format_lwbemap, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__ps_quads_killed()
        {
            return GetValue(RawMetricIdx::sm__ps_quads_killed, 8.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_t2d_busy()
        {
            return GetValue(RawMetricIdx::l1tex__t_t2d_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__d_sectors_fill()
        {
            return GetValue(RawMetricIdx::l1tex__d_sectors_fill, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_x2w_active()
        {
            return GetValue(RawMetricIdx::l1tex__x_x2w_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gcc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gcc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__ps_threads_killed()
        {
            return GetValue(RawMetricIdx::smsp__ps_threads_killed, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__output_prims_point()
        {
            return GetValue(RawMetricIdx::vpc__output_prims_point, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue gcc__l15_requests_hit_instruction()
        {
            return GetValue(RawMetricIdx::gcc__l15_requests_hit_instruction, 0.5, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__d_output_wavefronts()
        {
            return GetValue(RawMetricIdx::l1tex__d_output_wavefronts, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpmpd__output_stalled_batch()
        {
            return GetValue(RawMetricIdx::gpmpd__output_stalled_batch, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrop()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrop, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__f_output_tex2sm_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__f_output_tex2sm_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__gnic_port0_active()
        {
            return GetValue(RawMetricIdx::prop__gnic_port0_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__miop_pq_read_active_smp1()
        {
            return GetValue(RawMetricIdx::sm__miop_pq_read_active_smp1, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_cbc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_cbc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mmu__hubtlb_requests_hit()
        {
            return GetValue(RawMetricIdx::mmu__hubtlb_requests_hit, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_atomic_address_conflicts_lg_global_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__t_atomic_address_conflicts_lg_global_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crd_ni_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crd_ni_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pes__cycles_active()
        {
            return GetValue(RawMetricIdx::pes__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue cbmgr__alpha_cbe_allocations()
        {
            return GetValue(RawMetricIdx::cbmgr__alpha_cbe_allocations, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gpc__rg_utlb_requests_miss()
        {
            return GetValue(RawMetricIdx::gpc__rg_utlb_requests_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__out_read_stalled_vaf_alpha()
        {
            return GetValue(RawMetricIdx::pel__out_read_stalled_vaf_alpha, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__isbe_allocations()
        {
            return GetValue(RawMetricIdx::mpc__isbe_allocations, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue rdm__input_data0_active()
        {
            return GetValue(RawMetricIdx::rdm__input_data0_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_blend_enabled()
        {
            return GetValue(RawMetricIdx::crop__input_requests_blend_enabled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin2m_fifo_output_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__texin2m_fifo_output_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_lsu_wb_pending()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_lsu_wb_pending, 128.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__setup_output_prims_triangle()
        {
            return GetValue(RawMetricIdx::raster__setup_output_prims_triangle, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_iso()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_iso, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_x2t_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__x_x2t_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__latez_shdz_input_quads()
        {
            return GetValue(RawMetricIdx::prop__latez_shdz_input_quads, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue pel__in_write_requests_stalled()
        {
            return GetValue(RawMetricIdx::pel__in_write_requests_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_lsu_wb()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_lsu_wb, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_clear()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_clear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_surface_ld_d()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_surface_ld_d, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sys__fb_read_requests()
        {
            return GetValue(RawMetricIdx::sys__fb_read_requests, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_tag_requests_hit()
        {
            return GetValue(RawMetricIdx::lts__t_tag_requests_hit, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_niso_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_niso_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpmsd__cycles_active()
        {
            return GetValue(RawMetricIdx::gpmsd__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_active_gs()
        {
            return GetValue(RawMetricIdx::sm__warps_active_gs, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__d_d2f_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__d_d2f_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_su()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_su, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_pixels_2d()
        {
            return GetValue(RawMetricIdx::gpmsd__input_pixels_2d, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crd_ni_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crd_ni_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active()
        {
            return GetValue(RawMetricIdx::sm__cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__w_output_wavefronts()
        {
            return GetValue(RawMetricIdx::l1tex__w_output_wavefronts, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__threads_launched_tcs()
        {
            return GetValue(RawMetricIdx::sm__threads_launched_tcs, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_surface_red()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_surface_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crd_ni_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crd_ni_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_samp2x_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__samp_samp2x_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_mmu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_mmu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_binner_binned_op_tiled_cache_barriers()
        {
            return GetValue(RawMetricIdx::swdx__tc_binner_binned_op_tiled_cache_barriers, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_short_scoreboard_1()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_short_scoreboard_1, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__output_stalled()
        {
            return GetValue(RawMetricIdx::swdx__output_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_sectors_miss()
        {
            return GetValue(RawMetricIdx::l1tex__t_sectors_miss, 16.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_tex_format_2d_mipmap()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_tex_format_2d_mipmap, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_quads_2d()
        {
            return GetValue(RawMetricIdx::gpmsd__input_quads_2d, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__crstr_discover_stalling_setup()
        {
            return GetValue(RawMetricIdx::raster__crstr_discover_stalling_setup, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crop_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crop_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_tcs()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_tcs, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_input_tasks()
        {
            return GetValue(RawMetricIdx::vaf__beta_input_tasks, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__cdp_alpha_blendopt_pixels_fill_override()
        {
            return GetValue(RawMetricIdx::prop__cdp_alpha_blendopt_pixels_fill_override, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_tex_format_no_mipmap()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_tex_format_no_mipmap, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_tex_format_lwbemap()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_tex_format_lwbemap, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_output_subtiles_3_cycle()
        {
            return GetValue(RawMetricIdx::raster__frstr_output_subtiles_3_cycle, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__crstr_input_prims()
        {
            return GetValue(RawMetricIdx::raster__crstr_input_prims, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants_hit()
        {
            return GetValue(RawMetricIdx::scc__load_constants_hit, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_surface_ld_p()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_surface_ld_p, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pel__in_write_requests_stalled_vaf_alpha()
        {
            return GetValue(RawMetricIdx::pel__in_write_requests_stalled_vaf_alpha, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_pe_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_pe_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_reqseq_stalled_pending_store_same_address()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_reqseq_stalled_pending_store_same_address, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_tex_format_1d2d_array()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_tex_format_1d2d_array, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_surface_st()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_surface_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_sm2tex_busy()
        {
            return GetValue(RawMetricIdx::l1tex__texin_sm2tex_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_surface_atom()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_surface_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue tgb__output_prims_point()
        {
            return GetValue(RawMetricIdx::tgb__output_prims_point, 4.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_miss_clear_zbc_zrop()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_miss_clear_zbc_zrop, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrd_i()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrd_i, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue cwd__output_stalled_state_ack()
        {
            return GetValue(RawMetricIdx::cwd__output_stalled_state_ack, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_lg_global_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_lg_global_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__gpcl1tlb_stalled_request_fifo_full()
        {
            return GetValue(RawMetricIdx::gpc__gpcl1tlb_stalled_request_fifo_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crd_i()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crd_i, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_tsl1_requests_hit_texhdr()
        {
            return GetValue(RawMetricIdx::l1tex__texin_tsl1_requests_hit_texhdr, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_rf_free()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_rf_free, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_3d()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_3d, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__icc_requests_miss_to_gcc()
        {
            return GetValue(RawMetricIdx::sm__icc_requests_miss_to_gcc, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_concat_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpmpd__input_tasks()
        {
            return GetValue(RawMetricIdx::gpmpd__input_tasks, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_tex_throttle()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_tex_throttle, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_l1tex_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_l1tex_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_ldc_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_ldc_pred_on, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_l1tex_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_l1tex_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue tgb__cycles_active()
        {
            return GetValue(RawMetricIdx::tgb__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__cycles_wfi_host_scg0()
        {
            return GetValue(RawMetricIdx::fe__cycles_wfi_host_scg0, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_block_stalled_pending_miss()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_block_stalled_pending_miss, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_allocation_stall_1()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_allocation_stall_1, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__gpm_stalled_by_state_processing()
        {
            return GetValue(RawMetricIdx::vaf__gpm_stalled_by_state_processing, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__prop_utlb_requests_sparse()
        {
            return GetValue(RawMetricIdx::gpc__prop_utlb_requests_sparse, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_concat_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_samp2x_active()
        {
            return GetValue(RawMetricIdx::l1tex__samp_samp2x_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__input_stalled_waiting_for_pixel_shader_barrier_release()
        {
            return GetValue(RawMetricIdx::prop__input_stalled_waiting_for_pixel_shader_barrier_release, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_tex_format_1d2d_array()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_tex_format_1d2d_array, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_global_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_global_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_sz_32b()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_sz_32b, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__r_input_fifo_active()
        {
            return GetValue(RawMetricIdx::lts__r_input_fifo_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gcc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gcc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_clear()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_clear, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__imc_requests_miss_no_tags()
        {
            return GetValue(RawMetricIdx::smsp__imc_requests_miss_no_tags, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fp16()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fp16, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_tex_wb()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_tex_wb, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue vpc__clip_output_attrs()
        {
            return GetValue(RawMetricIdx::vpc__clip_output_attrs, 8.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crd_i_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crd_i_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_iso()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_iso, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_fetched_attr_vector_post_cbf()
        {
            return GetValue(RawMetricIdx::vaf__alpha_fetched_attr_vector_post_cbf, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_ltc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_ltc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fmai()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fmai, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_acache_requests_acache_miss()
        {
            return GetValue(RawMetricIdx::vaf__alpha_acache_requests_acache_miss, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_x2t_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__x_x2t_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_l1tex_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_l1tex_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_cbc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_cbc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_output_cycles()
        {
            return GetValue(RawMetricIdx::raster__frstr_output_cycles, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fma64plusplus()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fma64plusplus, 0.03125, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_surface_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_surface_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_t2d_active()
        {
            return GetValue(RawMetricIdx::l1tex__t_t2d_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_cbc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_cbc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_null_surface()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_null_surface, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpu__time_start()
        {
            return GetValue(RawMetricIdx::gpu__time_start, 10.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crd_i()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crd_i, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_miss_clear_zbc_crop()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_miss_clear_zbc_crop, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants_stalled_update_scoreboard_full()
        {
            return GetValue(RawMetricIdx::scc__load_constants_stalled_update_scoreboard_full, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrop()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrop, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__miop_pq_read_active_tex_smp1()
        {
            return GetValue(RawMetricIdx::sm__miop_pq_read_active_tex_smp1, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue rdm__input_data1_active()
        {
            return GetValue(RawMetricIdx::rdm__input_data1_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__zrop_output_quads()
        {
            return GetValue(RawMetricIdx::prop__zrop_output_quads, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue fe__input_methods()
        {
            return GetValue(RawMetricIdx::fe__input_methods, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__crstr_output_tiles_empty()
        {
            return GetValue(RawMetricIdx::raster__crstr_output_tiles_empty, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_cbc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_cbc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_lg_global_red()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_global_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue cbmgr__cycles_active()
        {
            return GetValue(RawMetricIdx::cbmgr__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue rdm__crop_output_active()
        {
            return GetValue(RawMetricIdx::rdm__crop_output_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_high()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_high, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_xu_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_xu_pred_off_all, 0.25, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__prop_utlb_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::gpc__prop_utlb_stalled_on_tag_allocation, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__latez_output_pixels()
        {
            return GetValue(RawMetricIdx::prop__latez_output_pixels, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__miop_adu_replays_smp0()
        {
            return GetValue(RawMetricIdx::sm__miop_adu_replays_smp0, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_reason_scissor()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_reason_scissor, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_comptag_fill()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_comptag_fill, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_gs()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_gs, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__read_returns()
        {
            return GetValue(RawMetricIdx::crop__read_returns, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_tex_format_1d2d_array()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_tex_format_1d2d_array, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__read_stalled()
        {
            return GetValue(RawMetricIdx::crop__read_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__output_cbes()
        {
            return GetValue(RawMetricIdx::vpc__output_cbes, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_iso_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_iso_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__input_subtiles()
        {
            return GetValue(RawMetricIdx::mpc__input_subtiles, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gcc__cycles_active()
        {
            return GetValue(RawMetricIdx::gcc__cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_aamode_2xaa()
        {
            return GetValue(RawMetricIdx::crop__input_requests_aamode_2xaa, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_hit_tex_format_3d()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_hit_tex_format_3d, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_local_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_local_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gcc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gcc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__write_requests()
        {
            return GetValue(RawMetricIdx::zrop__write_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__threads_launched_cs()
        {
            return GetValue(RawMetricIdx::sm__threads_launched_cs, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrd_ni()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrd_ni, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crd_ni_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crd_ni_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrd_ni()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrd_ni, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_cbe_slots_threshold()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_cbe_slots_threshold, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_ps()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_ps, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__issue_active()
        {
            return GetValue(RawMetricIdx::smsp__issue_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_tsl1_requests_hit_samphdr()
        {
            return GetValue(RawMetricIdx::l1tex__texin_tsl1_requests_hit_samphdr, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_block_stalled_same_address()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_block_stalled_same_address, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__out_input_stalled()
        {
            return GetValue(RawMetricIdx::pel__out_input_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_tsl1_requests_miss_texhdr()
        {
            return GetValue(RawMetricIdx::l1tex__texin_tsl1_requests_miss_texhdr, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_surface_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_surface_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__ltcx_read_sectors()
        {
            return GetValue(RawMetricIdx::lts__ltcx_read_sectors, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__processed_subpackets()
        {
            return GetValue(RawMetricIdx::zrop__processed_subpackets, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_gs()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_gs, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_tex_format_1d_tex()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_tex_format_1d_tex, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc1_utlb_requests_hit()
        {
            return GetValue(RawMetricIdx::gpc__tpc1_utlb_requests_hit, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__output_prims()
        {
            return GetValue(RawMetricIdx::vpc__output_prims, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_tcs()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_tcs, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_timeout()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_timeout, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__d_d2f_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__d_d2f_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mme__cycles_stalled()
        {
            return GetValue(RawMetricIdx::mme__cycles_stalled, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc0_utlb_stalled_request_fifo_full()
        {
            return GetValue(RawMetricIdx::gpc__tpc0_utlb_stalled_request_fifo_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrd_i()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrd_i, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__read_isbes()
        {
            return GetValue(RawMetricIdx::vpc__read_isbes, 0.25, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors()
        {
            return GetValue(RawMetricIdx::lts__t_sectors, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_mmu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_mmu_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crop_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crop_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__processed_requests_type_bundle()
        {
            return GetValue(RawMetricIdx::zrop__processed_requests_type_bundle, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_sz_128b_nearest()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_sz_128b_nearest, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__isbe_allocation_stalled_alpha_on_vsc()
        {
            return GetValue(RawMetricIdx::mpc__isbe_allocation_stalled_alpha_on_vsc, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_cbc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_cbc_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__input_prims_single_binned()
        {
            return GetValue(RawMetricIdx::swdx__input_prims_single_binned, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__input_pixels()
        {
            return GetValue(RawMetricIdx::prop__input_pixels, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrop_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrop_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_tex_format_1d_buffer()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_tex_format_1d_buffer, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_sectors_miss_lg()
        {
            return GetValue(RawMetricIdx::l1tex__t_sectors_miss_lg, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_xu()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_xu, 0.25, tpc__cycles_elapsed().avg);
        }

        MetricValue vpc__alpha_cycles_active()
        {
            return GetValue(RawMetricIdx::vpc__alpha_cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_atomic, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pes__stream_output_attrs()
        {
            return GetValue(RawMetricIdx::pes__stream_output_attrs, 8.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_hit_clear_zbc_zrop()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_hit_clear_zbc_zrop, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__latez_shdz_input_pixels()
        {
            return GetValue(RawMetricIdx::prop__latez_shdz_input_pixels, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__subtiles_launched_smp0()
        {
            return GetValue(RawMetricIdx::sm__subtiles_launched_smp0, 0.25, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_iso_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_iso_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_cbe_memory_threshold()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_cbe_memory_threshold, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_pixels_fully_covered()
        {
            return GetValue(RawMetricIdx::gpmsd__input_pixels_fully_covered, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_atomic, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_imc_miss()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_imc_miss, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_comptag_hit()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_comptag_hit, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__alpha_beta_mode_switches()
        {
            return GetValue(RawMetricIdx::mpc__alpha_beta_mode_switches, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_surface_red()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_surface_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sys__cycles_elapsed()
        {
            MetricValue metricValue;
            if (configuring)
            {
                counts[RawMetricIdx::sys__cycles_elapsed] = 1;
                metricValue = {};
            }
            else
            {
                const double unitCount = counts[RawMetricIdx::sys__cycles_elapsed];
                metricValue.sum = values[RawMetricIdx::sys__cycles_elapsed];
                metricValue.avg = metricValue.sum / unitCount;
                metricValue.peak_sustained = 1.0;
                metricValue.cycles_elapsed = metricValue.avg;
            }
            return metricValue;
        }

        MetricValue smsp__inst_exelwted_pipe_lsu_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_lsu_pred_off_all, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__idc_requests_hit()
        {
            return GetValue(RawMetricIdx::sm__idc_requests_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fma64plusplus_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fma64plusplus_pred_off_all, 0.03125, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_lwlled_occluders()
        {
            return GetValue(RawMetricIdx::raster__zlwll_lwlled_occluders, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_640()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_640, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_lg_local_st()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_local_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_clear()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_clear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_membar()
        {
            return GetValue(RawMetricIdx::lts__t_requests_membar, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fbp__cycles_elapsed()
        {
            MetricValue metricValue;
            if (configuring)
            {
                counts[RawMetricIdx::fbp__cycles_elapsed] = 1;
                metricValue = {};
            }
            else
            {
                const double unitCount = counts[RawMetricIdx::fbp__cycles_elapsed];
                metricValue.sum = values[RawMetricIdx::fbp__cycles_elapsed];
                metricValue.avg = metricValue.sum / unitCount;
                metricValue.peak_sustained = 1.0;
                metricValue.cycles_elapsed = metricValue.avg;
            }
            return metricValue;
        }

        MetricValue l1tex__texin_requests_null_tex()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_null_tex, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_output_subtiles()
        {
            return GetValue(RawMetricIdx::raster__frstr_output_subtiles, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue stri__acache_requests_stri()
        {
            return GetValue(RawMetricIdx::stri__acache_requests_stri, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue cbmgr__beta_cbe_allocation_stalled()
        {
            return GetValue(RawMetricIdx::cbmgr__beta_cbe_allocation_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_miss()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_miss, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_completed_ooo_3d_vtg()
        {
            return GetValue(RawMetricIdx::sm__warps_completed_ooo_3d_vtg, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__gpcl1tlb_requests_miss()
        {
            return GetValue(RawMetricIdx::gpc__gpcl1tlb_requests_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sys__gpfifo_dwords()
        {
            return GetValue(RawMetricIdx::sys__gpfifo_dwords, 16.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_lsu_read()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_lsu_read, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_vsa()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_vsa, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_output_occluders_trivial_accept()
        {
            return GetValue(RawMetricIdx::raster__zlwll_output_occluders_trivial_accept, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vaf__gpm_stalled_by_beta_input_fifo()
        {
            return GetValue(RawMetricIdx::vaf__gpm_stalled_by_beta_input_fifo, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sked__cycles_active()
        {
            return GetValue(RawMetricIdx::sked__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crop_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crop_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_448()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_448, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__i2m_write_bytes()
        {
            return GetValue(RawMetricIdx::fe__i2m_write_bytes, 16.0, sys__cycles_elapsed().avg);
        }

        MetricValue stri__acache_requests_stri_miss()
        {
            return GetValue(RawMetricIdx::stri__acache_requests_stri_miss, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue cbmgr__beta_cbe_allocation_stalled_no_space()
        {
            return GetValue(RawMetricIdx::cbmgr__beta_cbe_allocation_stalled_no_space, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue cbmgr__alpha_cbe_allocation_stalled_no_space()
        {
            return GetValue(RawMetricIdx::cbmgr__alpha_cbe_allocation_stalled_no_space, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__output_ops_vert()
        {
            return GetValue(RawMetricIdx::fe__output_ops_vert, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_issued()
        {
            return GetValue(RawMetricIdx::smsp__inst_issued, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_texels()
        {
            return GetValue(RawMetricIdx::l1tex__t_texels, 16.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_su_read_tram_bank_conflict()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_su_read_tram_bank_conflict, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_raster_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_raster_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__input_prims_patch()
        {
            return GetValue(RawMetricIdx::vpc__input_prims_patch, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_small_page_3()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_small_page_3, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_non_binnable_line()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_non_binnable_line, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__ps_quads_sent_to_pixout()
        {
            return GetValue(RawMetricIdx::sm__ps_quads_sent_to_pixout, 2.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_hit_tex_format_lwbemap()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_hit_tex_format_lwbemap, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_active_tes()
        {
            return GetValue(RawMetricIdx::sm__warps_active_tes, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_global_red()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_global_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue cwd__feedback_mpc_messages()
        {
            return GetValue(RawMetricIdx::cwd__feedback_mpc_messages, 0.5, sys__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_hit_clear_zbc()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_hit_clear_zbc, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crop_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crop_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gcc()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gcc, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_output_subtiles_2d()
        {
            return GetValue(RawMetricIdx::raster__frstr_output_subtiles_2d, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_array()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_array, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crop_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crop_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__input_samples_part2()
        {
            return GetValue(RawMetricIdx::zrop__input_samples_part2, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__latez_rstrz_input_samples()
        {
            return GetValue(RawMetricIdx::prop__latez_rstrz_input_samples, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants()
        {
            return GetValue(RawMetricIdx::scc__load_constants, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue vpc__clip_output_prims()
        {
            return GetValue(RawMetricIdx::vpc__clip_output_prims, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_dispatch_stall()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_dispatch_stall, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_cbc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_cbc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrd_i_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrd_i_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_concat_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__zrop_output_active()
        {
            return GetValue(RawMetricIdx::prop__zrop_output_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_quads()
        {
            return GetValue(RawMetricIdx::prop__csb_output_quads, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vaf__gpm_stalled_by_state_fifo()
        {
            return GetValue(RawMetricIdx::vaf__gpm_stalled_by_state_fifo, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue scc__cycles_active()
        {
            return GetValue(RawMetricIdx::scc__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue prop__cdp_alpha_to_coverage_killed_pixels()
        {
            return GetValue(RawMetricIdx::prop__cdp_alpha_to_coverage_killed_pixels, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crd_i()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crd_i, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_lg_global_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_lg_global_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests()
        {
            return GetValue(RawMetricIdx::crop__input_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__output_prims_triangle()
        {
            return GetValue(RawMetricIdx::vpc__output_prims_triangle, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_global_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_global_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_tsl1_requests_miss_samphdr()
        {
            return GetValue(RawMetricIdx::l1tex__texin_tsl1_requests_miss_samphdr, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__d_d2f_active()
        {
            return GetValue(RawMetricIdx::l1tex__d_d2f_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue fe__output_ops_bundle_scg0_go_idle()
        {
            return GetValue(RawMetricIdx::fe__output_ops_bundle_scg0_go_idle, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_quads_3d()
        {
            return GetValue(RawMetricIdx::prop__csb_output_quads_3d, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__tc_output_tiles()
        {
            return GetValue(RawMetricIdx::raster__tc_output_tiles, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__output_attrs_scalar()
        {
            return GetValue(RawMetricIdx::vpc__output_attrs_scalar, 8.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_sz_64b_nearest()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_sz_64b_nearest, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mme__cycles_active()
        {
            return GetValue(RawMetricIdx::mme__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants_page_same()
        {
            return GetValue(RawMetricIdx::scc__load_constants_page_same, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue prop__input_quads_2d()
        {
            return GetValue(RawMetricIdx::prop__input_quads_2d, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_tex()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_tex, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_atomic, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crd_i_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crd_i_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crd_ni()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crd_ni, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mme__output_method_dwords()
        {
            return GetValue(RawMetricIdx::mme__output_method_dwords, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__threads_launched_tes()
        {
            return GetValue(RawMetricIdx::sm__threads_launched_tes, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_tex_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_tex_pred_on, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue stri__cycles_active()
        {
            return GetValue(RawMetricIdx::stri__cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_mmu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_mmu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_cctlt()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_cctlt, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__read_requests()
        {
            return GetValue(RawMetricIdx::crop__read_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_big_page_2()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_big_page_2, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue pdb__input_stalled_alpha()
        {
            return GetValue(RawMetricIdx::pdb__input_stalled_alpha, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_level_0_threshold()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_level_0_threshold, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crd_i_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crd_i_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__miop_pq_read_active_lsu_smp1()
        {
            return GetValue(RawMetricIdx::sm__miop_pq_read_active_lsu_smp1, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_t2d_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__t_t2d_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__output_ops_bundle_scg1_go_idle()
        {
            return GetValue(RawMetricIdx::fe__output_ops_bundle_scg1_go_idle, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_tcs()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_tcs, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__tc_flushes()
        {
            return GetValue(RawMetricIdx::raster__tc_flushes, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue fe__output_ops_bundle()
        {
            return GetValue(RawMetricIdx::fe__output_ops_bundle, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_tex_format_no_mipmap()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_tex_format_no_mipmap, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_cbc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_cbc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_reqseq_input_stalled_fifo_full()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_reqseq_input_stalled_fifo_full, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_cycles_bank_conflict()
        {
            return GetValue(RawMetricIdx::lts__d_cycles_bank_conflict, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_aamode_4xaa()
        {
            return GetValue(RawMetricIdx::crop__input_requests_aamode_4xaa, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gcc__l15_requests_miss()
        {
            return GetValue(RawMetricIdx::gcc__l15_requests_miss, 0.5, gpc__cycles_elapsed().avg);
        }

        MetricValue mmu__cycles_active()
        {
            return GetValue(RawMetricIdx::mmu__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrop_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrop_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_surface_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_surface_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pel__out_read_stalled_vaf_beta()
        {
            return GetValue(RawMetricIdx::pel__out_read_stalled_vaf_beta, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_output_subtiles_1_cycle()
        {
            return GetValue(RawMetricIdx::raster__frstr_output_subtiles_1_cycle, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_lg_local_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_lg_local_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__input_prims_alpha_patches()
        {
            return GetValue(RawMetricIdx::mpc__input_prims_alpha_patches, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__write_requests_compressed_2to1()
        {
            return GetValue(RawMetricIdx::crop__write_requests_compressed_2to1, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__input_prims_dual_binned()
        {
            return GetValue(RawMetricIdx::swdx__input_prims_dual_binned, 2.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_vsa()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_vsa, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_big_page_0()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_big_page_0, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue raster__crstr_discover_working_no_tile_stalling_setup()
        {
            return GetValue(RawMetricIdx::raster__crstr_discover_working_no_tile_stalling_setup, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_samp2x_busy()
        {
            return GetValue(RawMetricIdx::l1tex__samp_samp2x_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__csb_input_quads_3d()
        {
            return GetValue(RawMetricIdx::prop__csb_input_quads_3d, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue crop__write_subpackets()
        {
            return GetValue(RawMetricIdx::crop__write_subpackets, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrd_i_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrd_i_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_surface_atom()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_surface_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_to_acache_stalled_on_read()
        {
            return GetValue(RawMetricIdx::vaf__beta_to_acache_stalled_on_read, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__xbar_output_active()
        {
            return GetValue(RawMetricIdx::lts__xbar_output_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue cbmgr__alpha_cbe_allocation_stalled_max_tasks()
        {
            return GetValue(RawMetricIdx::cbmgr__alpha_cbe_allocation_stalled_max_tasks, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_retiring_ps_stalled_out_of_order()
        {
            return GetValue(RawMetricIdx::sm__warps_retiring_ps_stalled_out_of_order, 128.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__input_pixels_3d()
        {
            return GetValue(RawMetricIdx::gpmsd__input_pixels_3d, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__input_prims_beta()
        {
            return GetValue(RawMetricIdx::mpc__input_prims_beta, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pel__in_write_requests_stalled_vaf_beta()
        {
            return GetValue(RawMetricIdx::pel__in_write_requests_stalled_vaf_beta, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_big_page_1()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_big_page_1, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_acache_requests_local_hit()
        {
            return GetValue(RawMetricIdx::vaf__beta_acache_requests_local_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_surface_ld()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_surface_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_local_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_local_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue fe__input_method_dwords_i2m()
        {
            return GetValue(RawMetricIdx::fe__input_method_dwords_i2m, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue tgb__output_prims_line()
        {
            return GetValue(RawMetricIdx::tgb__output_prims_line, 4.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__ltcx_read_stalled_fifo_full()
        {
            return GetValue(RawMetricIdx::lts__ltcx_read_stalled_fifo_full, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrd_i_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrd_i_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_in_warp_surface_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_in_warp_surface_ld, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_launched()
        {
            return GetValue(RawMetricIdx::smsp__warps_launched, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_x2w_busy()
        {
            return GetValue(RawMetricIdx::l1tex__x_x2w_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_tex_format_1d_buffer()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_tex_format_1d_buffer, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__w_w2d_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__w_w2d_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpu__time_end()
        {
            return GetValue(RawMetricIdx::gpu__time_end, 10.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrd_ni_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrd_ni_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__processed_samples_part2()
        {
            return GetValue(RawMetricIdx::crop__processed_samples_part2, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc1_utlb_requests_hit_under_miss()
        {
            return GetValue(RawMetricIdx::gpc__tpc1_utlb_requests_hit_under_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue fe__output_ops_ld_const()
        {
            return GetValue(RawMetricIdx::fe__output_ops_ld_const, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_lg_global_ld()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_lg_global_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_cycles_active()
        {
            return GetValue(RawMetricIdx::vaf__alpha_cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__setup_input_prims()
        {
            return GetValue(RawMetricIdx::raster__setup_input_prims, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue zrop__processed_requests_type_plane_eq()
        {
            return GetValue(RawMetricIdx::zrop__processed_requests_type_plane_eq, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_lsu_read_bank_conflict()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_lsu_read_bank_conflict, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_mio_throttle()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_mio_throttle, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pdb__output_stalled_alpha()
        {
            return GetValue(RawMetricIdx::pdb__output_stalled_alpha, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_input_task_fifo_full()
        {
            return GetValue(RawMetricIdx::vaf__beta_input_task_fifo_full, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gcc__l15_requests_miss_constant()
        {
            return GetValue(RawMetricIdx::gcc__l15_requests_miss_constant, 0.5, gpc__cycles_elapsed().avg);
        }

        MetricValue zrop__write_subpackets_depth()
        {
            return GetValue(RawMetricIdx::zrop__write_subpackets_depth, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_pe_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_pe_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__cycles_wfi_ctxsw()
        {
            return GetValue(RawMetricIdx::fe__cycles_wfi_ctxsw, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crop_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crop_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__threads_launched_ps_not_killed()
        {
            return GetValue(RawMetricIdx::sm__threads_launched_ps_not_killed, 8.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__output_ops_bundle_scg1_wfi_host()
        {
            return GetValue(RawMetricIdx::fe__output_ops_bundle_scg1_wfi_host, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_blend_blendopt_fill_over()
        {
            return GetValue(RawMetricIdx::crop__input_requests_blend_blendopt_fill_over, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__gpcl1tlb_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::gpc__gpcl1tlb_stalled_on_tag_allocation, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_tes()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_tes, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_raster_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_raster_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_cycles_active()
        {
            return GetValue(RawMetricIdx::vpc__lwll_cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_bru_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_bru_pred_off_all, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__sm2gpmsd_pixout_cdp0_active_color()
        {
            return GetValue(RawMetricIdx::gpmsd__sm2gpmsd_pixout_cdp0_active_color, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__processed_samples_part1()
        {
            return GetValue(RawMetricIdx::crop__processed_samples_part1, 63.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrd_ni()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrd_ni, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_adu_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_adu_pred_on, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrd_ni_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrd_ni_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue tgb__output_verts_simple()
        {
            return GetValue(RawMetricIdx::tgb__output_verts_simple, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_reason_zero_area()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_reason_zero_area, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_sz_96b()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_sz_96b, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_atomic, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_output_wavefronts()
        {
            return GetValue(RawMetricIdx::l1tex__t_output_wavefronts, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_iso_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_iso_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_long_scoreboard_0()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_long_scoreboard_0, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crop_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crop_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_short_scoreboard_0()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_short_scoreboard_0, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__warps_launched_gs_fast_beta()
        {
            return GetValue(RawMetricIdx::sm__warps_launched_gs_fast_beta, 0.125, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_lg_global_st()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_lg_global_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pda__input_prims_point()
        {
            return GetValue(RawMetricIdx::pda__input_prims_point, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue prop__latez_output_quads()
        {
            return GetValue(RawMetricIdx::prop__latez_output_quads, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_bar_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_bar_pred_off_all, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__setup_output_prims_point()
        {
            return GetValue(RawMetricIdx::raster__setup_output_prims_point, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_input_tiles()
        {
            return GetValue(RawMetricIdx::raster__zlwll_input_tiles, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__input_prims_alpha()
        {
            return GetValue(RawMetricIdx::mpc__input_prims_alpha, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_resseq_stalled_backpressure()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_resseq_stalled_backpressure, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_long_scoreboard_1()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_long_scoreboard_1, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__threads_launched()
        {
            return GetValue(RawMetricIdx::sm__threads_launched, 8.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_input_prims()
        {
            return GetValue(RawMetricIdx::vaf__alpha_input_prims, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fxu()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fxu, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__input_samples()
        {
            return GetValue(RawMetricIdx::prop__input_samples, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_drain()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_drain, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_write_stalled()
        {
            return GetValue(RawMetricIdx::lts__mccif_write_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__alpha_batches_active()
        {
            return GetValue(RawMetricIdx::vpc__alpha_batches_active, 63.0, gpc__cycles_elapsed().avg);
        }

        MetricValue fe__cycles_active()
        {
            return GetValue(RawMetricIdx::fe__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_raster_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_raster_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_input_batches_pre_cbf()
        {
            return GetValue(RawMetricIdx::vaf__alpha_input_batches_pre_cbf, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue tga__output_tasks_primary()
        {
            return GetValue(RawMetricIdx::tga__output_tasks_primary, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_output_occluders_zfail_no_lwll()
        {
            return GetValue(RawMetricIdx::raster__zlwll_output_occluders_zfail_no_lwll, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_write_cycles_16()
        {
            return GetValue(RawMetricIdx::lts__mccif_write_cycles_16, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crd_i_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crd_i_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_wait()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_wait, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__rg_utlb_stalled_write_buffer_full()
        {
            return GetValue(RawMetricIdx::gpc__rg_utlb_stalled_write_buffer_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crop_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crop_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrop_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrop_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_filter_aniso()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_filter_aniso, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_request_latency_384()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_request_latency_384, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_vsb()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_vsb, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_misc_1()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_misc_1, 6.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__prop_utlb_requests_hit()
        {
            return GetValue(RawMetricIdx::gpc__prop_utlb_requests_hit, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gpc__prop_utlb_requests_miss()
        {
            return GetValue(RawMetricIdx::gpc__prop_utlb_requests_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_cs()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_cs, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_batches_active()
        {
            return GetValue(RawMetricIdx::vaf__alpha_batches_active, 8.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_cctlt()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_cctlt, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_array()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_array, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mmu__pde_requests_miss()
        {
            return GetValue(RawMetricIdx::mmu__pde_requests_miss, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_cycles_32()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_cycles_32, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_concat_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gcc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gcc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_mmu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_mmu_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue wwdx__input_cbes()
        {
            return GetValue(RawMetricIdx::wwdx__input_cbes, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue pda__input_prims()
        {
            return GetValue(RawMetricIdx::pda__input_prims, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue stri__acache_requests_stri_hit()
        {
            return GetValue(RawMetricIdx::stri__acache_requests_stri_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__clip_output_verts()
        {
            return GetValue(RawMetricIdx::vpc__clip_output_verts, 0.88, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__tram_startxy_fifo_stalled()
        {
            return GetValue(RawMetricIdx::mpc__tram_startxy_fifo_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_lg_global_red()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_lg_global_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pda__input_prims_triadj()
        {
            return GetValue(RawMetricIdx::pda__input_prims_triadj, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue crop__processed_requests_uncompressed()
        {
            return GetValue(RawMetricIdx::crop__processed_requests_uncompressed, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_reason_frustum()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_reason_frustum, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue pel__in_write_requests()
        {
            return GetValue(RawMetricIdx::pel__in_write_requests, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrop_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrop_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_samples_part2()
        {
            return GetValue(RawMetricIdx::crop__input_samples_part2, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_surface_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_surface_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_reason_rotated_grid()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_reason_rotated_grid, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_in_warp_lg_local_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_in_warp_lg_local_st, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_noncpu_membar()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_noncpu_membar, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__mios_datapath_active()
        {
            return GetValue(RawMetricIdx::sm__mios_datapath_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_l1tex_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_requests_l1tex_atomic, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mme__input_methods_shadow_filtered()
        {
            return GetValue(RawMetricIdx::mme__input_methods_shadow_filtered, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrd_ni_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrd_ni_prefetch, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrd_ni_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrd_ni_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__zlwll_cycles_active()
        {
            return GetValue(RawMetricIdx::zrop__zlwll_cycles_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__d_cycles_bank_conflict_2x2()
        {
            return GetValue(RawMetricIdx::l1tex__d_cycles_bank_conflict_2x2, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_tex_format_1d_buffer()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_tex_format_1d_buffer, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_niso_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_niso_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__cycles_elapsed_beta()
        {
            return GetValue(RawMetricIdx::mpc__cycles_elapsed_beta, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_write_sectors_lg_local_st()
        {
            return GetValue(RawMetricIdx::l1tex__m_write_sectors_lg_local_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_point()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_point, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_fetched_attr_vector_pre_cbf()
        {
            return GetValue(RawMetricIdx::vaf__alpha_fetched_attr_vector_pre_cbf, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__gpcl1tlb_requests_hit_under_miss()
        {
            return GetValue(RawMetricIdx::gpc__gpcl1tlb_requests_hit_under_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gr__cycles_active()
        {
            return GetValue(RawMetricIdx::gr__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_membar()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_membar, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pda__output_batches()
        {
            return GetValue(RawMetricIdx::pda__output_batches, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue zrop__write_subpackets_coalesced()
        {
            return GetValue(RawMetricIdx::zrop__write_subpackets_coalesced, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_tcs()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_tcs, 1.5, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_in_warp_lg_global_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_in_warp_lg_global_st, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_pe_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_pe_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__warps_active_ps()
        {
            return GetValue(RawMetricIdx::sm__warps_active_ps, 128.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrop_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrop_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__output_ops_bundle_scg0_wfi_host()
        {
            return GetValue(RawMetricIdx::fe__output_ops_bundle_scg0_wfi_host, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue stri__cycles_busy()
        {
            return GetValue(RawMetricIdx::stri__cycles_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue tga__output_tasks_complex_interior()
        {
            return GetValue(RawMetricIdx::tga__output_tasks_complex_interior, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__inst_exelwted_pipe_adu_divergent_smp1()
        {
            return GetValue(RawMetricIdx::sm__inst_exelwted_pipe_adu_divergent_smp1, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__tex_write_back_active()
        {
            return GetValue(RawMetricIdx::smsp__tex_write_back_active, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue rdm__zrop_output_active()
        {
            return GetValue(RawMetricIdx::rdm__zrop_output_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__cbc_requests_comptag_writeback()
        {
            return GetValue(RawMetricIdx::lts__cbc_requests_comptag_writeback, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__cycles_active()
        {
            return GetValue(RawMetricIdx::smsp__cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue tga__input_isbes()
        {
            return GetValue(RawMetricIdx::tga__input_isbes, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_cbc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_cbc_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests()
        {
            return GetValue(RawMetricIdx::lts__t_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrd_i_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrd_i_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__clip_input_prims_clipped_single_plane()
        {
            return GetValue(RawMetricIdx::vpc__clip_input_prims_clipped_single_plane, 0.014492753623188406, gpc__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_cycles_stalled_on_mpc()
        {
            return GetValue(RawMetricIdx::vaf__beta_cycles_stalled_on_mpc, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_non_binnable_state()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_non_binnable_state, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_lg_global_atom()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_lg_global_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_zrd_ni_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_zrd_ni_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_tag_requests_miss()
        {
            return GetValue(RawMetricIdx::lts__t_tag_requests_miss, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crop()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crop, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_tex_trilinear()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_tex_trilinear, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_gs()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_gs, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__cycles_active()
        {
            return GetValue(RawMetricIdx::crop__cycles_active, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_miss_tex_format_3d()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_miss_tex_format_3d, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__input_samples_3d()
        {
            return GetValue(RawMetricIdx::prop__input_samples_3d, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__clip_cycles_stalled()
        {
            return GetValue(RawMetricIdx::vpc__clip_cycles_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_l1tex_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_l1tex_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants_stalled_no_gcc_credits()
        {
            return GetValue(RawMetricIdx::scc__load_constants_stalled_no_gcc_credits, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crop()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crop, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc0_utlb_stalled_write_buffer_full()
        {
            return GetValue(RawMetricIdx::gpc__tpc0_utlb_stalled_write_buffer_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_killed_subtiles_output()
        {
            return GetValue(RawMetricIdx::raster__frstr_killed_subtiles_output, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrd_i_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrd_i_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_selected()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_selected, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crd_i_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crd_i_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_hub_membar()
        {
            return GetValue(RawMetricIdx::lts__t_requests_hub_membar, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpmpd__output_stalled_task()
        {
            return GetValue(RawMetricIdx::gpmpd__output_stalled_task, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__isbe_allocation_stalled_beta_on_vsc()
        {
            return GetValue(RawMetricIdx::mpc__isbe_allocation_stalled_beta_on_vsc, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__cycles_wfi_subch_switch_scg0()
        {
            return GetValue(RawMetricIdx::fe__cycles_wfi_subch_switch_scg0, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_pre_nop_wavefronts()
        {
            return GetValue(RawMetricIdx::l1tex__samp_pre_nop_wavefronts, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_samples_3d()
        {
            return GetValue(RawMetricIdx::prop__csb_output_samples_3d, 128.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_surface_atom()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_surface_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_atomic_address_conflicts_surface_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__t_atomic_address_conflicts_surface_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_lg_local_ld()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_local_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pdb__input_tasks()
        {
            return GetValue(RawMetricIdx::pdb__input_tasks, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__subtiles_launched_smp1()
        {
            return GetValue(RawMetricIdx::sm__subtiles_launched_smp1, 0.25, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__d_d2f_busy()
        {
            return GetValue(RawMetricIdx::l1tex__d_d2f_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__icc_prefetches()
        {
            return GetValue(RawMetricIdx::sm__icc_prefetches, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_block_stalled_priority()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_block_stalled_priority, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_sm2tex_active()
        {
            return GetValue(RawMetricIdx::l1tex__texin_sm2tex_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_global_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_global_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_output_subtiles_fully_covered()
        {
            return GetValue(RawMetricIdx::raster__frstr_output_subtiles_fully_covered, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_active_cs()
        {
            return GetValue(RawMetricIdx::sm__warps_active_cs, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_input_patches()
        {
            return GetValue(RawMetricIdx::vaf__beta_input_patches, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_lwlled_occluders_stencil()
        {
            return GetValue(RawMetricIdx::raster__zlwll_lwlled_occluders_stencil, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fe()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fe, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_lg_global_atom()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_global_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_local_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_local_st, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_z_mode_transition()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_z_mode_transition, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__thread_inst_exelwted_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__thread_inst_exelwted_pred_on, 48.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__sm2gpmsd_pixout_cdp1_active_shdz()
        {
            return GetValue(RawMetricIdx::gpmsd__sm2gpmsd_pixout_cdp1_active_shdz, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crd_i_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crd_i_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__gpcl1tlb_requests_hit()
        {
            return GetValue(RawMetricIdx::gpc__gpcl1tlb_requests_hit, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__tc_input_transactions()
        {
            return GetValue(RawMetricIdx::raster__tc_input_transactions, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_pe_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_pe_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_to_acache_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::vaf__beta_to_acache_stalled_on_tag_allocation, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue host__chsw_switches()
        {
            return GetValue(RawMetricIdx::host__chsw_switches, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue pdb__input_stalled_beta()
        {
            return GetValue(RawMetricIdx::pdb__input_stalled_beta, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc0_utlb_stalled_on_tag_allocation()
        {
            return GetValue(RawMetricIdx::gpc__tpc0_utlb_stalled_on_tag_allocation, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_crop_requests_color()
        {
            return GetValue(RawMetricIdx::prop__csb_output_crop_requests_color, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crd_ni_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crd_ni_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_no_instructions_1()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_no_instructions_1, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_l1tex()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_l1tex, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crop_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crop_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__cycles_active_beta()
        {
            return GetValue(RawMetricIdx::mpc__cycles_active_beta, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pel__out_l2_requests()
        {
            return GetValue(RawMetricIdx::pel__out_l2_requests, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__miop_ldc_replays_smp1()
        {
            return GetValue(RawMetricIdx::sm__miop_ldc_replays_smp1, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__gpm_stalled_by_alpha_input_fifo()
        {
            return GetValue(RawMetricIdx::vaf__gpm_stalled_by_alpha_input_fifo, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_blend_blendopt_read_avoid()
        {
            return GetValue(RawMetricIdx::crop__input_requests_blend_blendopt_read_avoid, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_pe()
        {
            return GetValue(RawMetricIdx::lts__t_requests_pe, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_adu_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_adu_pred_off_all, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_input_verts()
        {
            return GetValue(RawMetricIdx::vaf__alpha_input_verts, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrd_ni_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrd_ni_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue tga__isbes_active()
        {
            return GetValue(RawMetricIdx::tga__isbes_active, 32.0, sys__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_reason_diamond_exit_rule()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_reason_diamond_exit_rule, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_input_occluders()
        {
            return GetValue(RawMetricIdx::raster__zlwll_input_occluders, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_x2t_active()
        {
            return GetValue(RawMetricIdx::l1tex__x_x2t_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_noncpu_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_noncpu_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crd_ni_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crd_ni_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_tex()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_tex, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_ldc_pred_off_all()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_ldc_pred_off_all, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__write_subpackets()
        {
            return GetValue(RawMetricIdx::zrop__write_subpackets, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_su_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_su_pred_on, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__input_pixels_2d()
        {
            return GetValue(RawMetricIdx::prop__input_pixels_2d, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_write_request_latency_24()
        {
            return GetValue(RawMetricIdx::lts__mccif_write_request_latency_24, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_cbc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_cbc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_membar()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_membar, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__d_decomp_processed_sectors()
        {
            return GetValue(RawMetricIdx::lts__d_decomp_processed_sectors, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crop()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crop, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__write_requests_compressed_8to1_or_fastclear()
        {
            return GetValue(RawMetricIdx::crop__write_requests_compressed_8to1_or_fastclear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__input_samples_2d()
        {
            return GetValue(RawMetricIdx::prop__input_samples_2d, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_sm2tex_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__texin_sm2tex_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpmpd__input_batches()
        {
            return GetValue(RawMetricIdx::gpmpd__input_batches, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gpc__rg_utlb_requests_hit_under_miss()
        {
            return GetValue(RawMetricIdx::gpc__rg_utlb_requests_hit_under_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_niso()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_niso, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc1_utlb_requests_miss()
        {
            return GetValue(RawMetricIdx::gpc__tpc1_utlb_requests_miss, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrd_i()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrd_i, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mmu__pde_requests_hit()
        {
            return GetValue(RawMetricIdx::mmu__pde_requests_hit, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue vpc__input_prims()
        {
            return GetValue(RawMetricIdx::vpc__input_prims, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue pel__in_write_requests_stalled_tg()
        {
            return GetValue(RawMetricIdx::pel__in_write_requests_stalled_tg, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__cycles_active_alpha()
        {
            return GetValue(RawMetricIdx::mpc__cycles_active_alpha, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_hit()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_hit, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__ctas_active()
        {
            return GetValue(RawMetricIdx::sm__ctas_active, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_resseq_stalled_output_arbitration()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_resseq_stalled_output_arbitration, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_pe()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_pe, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_host_cpu_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_host_cpu_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpmsd__output_quads()
        {
            return GetValue(RawMetricIdx::gpmsd__output_quads, 4.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin2m_fifo_input_stalled()
        {
            return GetValue(RawMetricIdx::l1tex__texin2m_fifo_input_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__x_output_wavefronts()
        {
            return GetValue(RawMetricIdx::l1tex__x_output_wavefronts, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue sm__idc_requests_miss_covered()
        {
            return GetValue(RawMetricIdx::sm__idc_requests_miss_covered, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_l1tex_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_l1tex_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_noncpu_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_noncpu_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crd_ni_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crd_ni_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue fe__input_methods_stalled()
        {
            return GetValue(RawMetricIdx::fe__input_methods_stalled, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue pda__cycles_active()
        {
            return GetValue(RawMetricIdx::pda__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_lsu_write_bank_conflict()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_lsu_write_bank_conflict, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_requests()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__imc_requests_hit()
        {
            return GetValue(RawMetricIdx::smsp__imc_requests_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc1_utlb_stalled_request_fifo_full()
        {
            return GetValue(RawMetricIdx::gpc__tpc1_utlb_stalled_request_fifo_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_tex_format_1d2d_tex()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_tex_format_1d2d_tex, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_lg_local_ld()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_lg_local_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__output_barriers()
        {
            return GetValue(RawMetricIdx::swdx__output_barriers, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_l1tex_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_l1tex_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__beta_acache_requests_acache_miss()
        {
            return GetValue(RawMetricIdx::vaf__beta_acache_requests_acache_miss, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_filter_bilinear()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_filter_bilinear, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_cycles_stalled()
        {
            return GetValue(RawMetricIdx::vpc__lwll_cycles_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_ltc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_ltc_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_state_full()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_state_full, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue raster__crstr_discover_working_no_tile_output()
        {
            return GetValue(RawMetricIdx::raster__crstr_discover_working_no_tile_output, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_fma64plusplus_pred_on()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_fma64plusplus_pred_on, 0.03125, tpc__cycles_elapsed().avg);
        }

        MetricValue cbmgr__alpha_cbe_allocation_stalled()
        {
            return GetValue(RawMetricIdx::cbmgr__alpha_cbe_allocation_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__sm2gpmsd_pixout_cdp1_active_color()
        {
            return GetValue(RawMetricIdx::gpmsd__sm2gpmsd_pixout_cdp1_active_color, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__cycles_active()
        {
            return GetValue(RawMetricIdx::pel__cycles_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crd_i_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crd_i_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrd_i_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrd_i_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue wwdx__input_prims()
        {
            return GetValue(RawMetricIdx::wwdx__input_prims, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_pixels_3d()
        {
            return GetValue(RawMetricIdx::prop__csb_output_pixels_3d, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_ltc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_ltc_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_output_wavefronts()
        {
            return GetValue(RawMetricIdx::l1tex__samp_output_wavefronts, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_niso_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_niso_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_sectors()
        {
            return GetValue(RawMetricIdx::l1tex__t_sectors, 16.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_l1tex_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_l1tex_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__out_l2_requests_write_256b()
        {
            return GetValue(RawMetricIdx::pel__out_l2_requests_write_256b, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__read_subpackets()
        {
            return GetValue(RawMetricIdx::zrop__read_subpackets, 8.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crop_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crop_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crop_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crop_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__warp_launch_stalled_vtg()
        {
            return GetValue(RawMetricIdx::mpc__warp_launch_stalled_vtg, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue tga__batches_active()
        {
            return GetValue(RawMetricIdx::tga__batches_active, 32.0, sys__cycles_elapsed().avg);
        }

        MetricValue zrop__input_requests_containing_stencil()
        {
            return GetValue(RawMetricIdx::zrop__input_requests_containing_stencil, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__zlwll_lwlled_occluders_depth()
        {
            return GetValue(RawMetricIdx::raster__zlwll_lwlled_occluders_depth, 32.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gpu__draw_count()
        {
            return GetValue(RawMetricIdx::gpu__draw_count, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__input_verts_beta()
        {
            return GetValue(RawMetricIdx::mpc__input_verts_beta, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_in_warp_surface_st()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_in_warp_surface_st, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__latez_rstrz_input_pixels()
        {
            return GetValue(RawMetricIdx::prop__latez_rstrz_input_pixels, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_barrier()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_barrier, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue scc__load_constants_page_pool_allocations()
        {
            return GetValue(RawMetricIdx::scc__load_constants_page_pool_allocations, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_surface_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_surface_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__d_cycles_bank_conflict_bilerp()
        {
            return GetValue(RawMetricIdx::l1tex__d_cycles_bank_conflict_bilerp, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_atomic_address_conflicts_lg_global_atom()
        {
            return GetValue(RawMetricIdx::l1tex__t_atomic_address_conflicts_lg_global_atom, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_fetched_attr_scalar_pre_cbf()
        {
            return GetValue(RawMetricIdx::vaf__alpha_fetched_attr_scalar_pre_cbf, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sys__fb_write_dwords()
        {
            return GetValue(RawMetricIdx::sys__fb_write_dwords, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims_reason_zero_length()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims_reason_zero_length, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__write_sectors()
        {
            return GetValue(RawMetricIdx::vpc__write_sectors, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_aamode_1xaa()
        {
            return GetValue(RawMetricIdx::crop__input_requests_aamode_1xaa, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mmu__pte_requests_small_page_2()
        {
            return GetValue(RawMetricIdx::mmu__pte_requests_small_page_2, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue scc__input_state_changes()
        {
            return GetValue(RawMetricIdx::scc__input_state_changes, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue crop__processed_requests()
        {
            return GetValue(RawMetricIdx::crop__processed_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_crop_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_crop_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue prop__zrop_output_stalled()
        {
            return GetValue(RawMetricIdx::prop__zrop_output_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_retiring_ps_stalled_backpressure()
        {
            return GetValue(RawMetricIdx::sm__warps_retiring_ps_stalled_backpressure, 128.0, tpc__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_3d()
        {
            return GetValue(RawMetricIdx::crop__input_requests_3d, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__icc_requests_hit()
        {
            return GetValue(RawMetricIdx::sm__icc_requests_hit, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_conflicts_lg_local_cctl()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_conflicts_lg_local_cctl, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_miss_tag_hit()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_miss_tag_hit, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crop_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crop_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__lwll_lwlled_prims()
        {
            return GetValue(RawMetricIdx::vpc__lwll_lwlled_prims, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue vpc__input_prims_line()
        {
            return GetValue(RawMetricIdx::vpc__input_prims_line, 0.83, gpc__cycles_elapsed().avg);
        }

        MetricValue tga__output_tasks()
        {
            return GetValue(RawMetricIdx::tga__output_tasks, 0.5, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crd_ni_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crd_ni_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__miop_ldc_replays_smp0()
        {
            return GetValue(RawMetricIdx::sm__miop_ldc_replays_smp0, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue smsp__warps_cant_issue_membar()
        {
            return GetValue(RawMetricIdx::smsp__warps_cant_issue_membar, 32.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pes__stream_output_verts()
        {
            return GetValue(RawMetricIdx::pes__stream_output_verts, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue gpmsd__sm2gpmsd_pixout_cdp0_active_shdz()
        {
            return GetValue(RawMetricIdx::gpmsd__sm2gpmsd_pixout_cdp0_active_shdz, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__m_read_sectors_lg_global_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__m_read_sectors_lg_global_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_wr()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_wr, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mpc__input_tasks()
        {
            return GetValue(RawMetricIdx::mpc__input_tasks, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue swdx__output_prims()
        {
            return GetValue(RawMetricIdx::swdx__output_prims, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__input_quads_3d()
        {
            return GetValue(RawMetricIdx::prop__input_quads_3d, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue mpc__cycles_elapsed_alpha()
        {
            return GetValue(RawMetricIdx::mpc__cycles_elapsed_alpha, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crd_i_rd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crd_i_rd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_lg_global_atom_cas()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_global_atom_cas, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_iso_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_iso_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_noncpu_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_noncpu_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__binner_active()
        {
            return GetValue(RawMetricIdx::swdx__binner_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_active_vtg()
        {
            return GetValue(RawMetricIdx::sm__warps_active_vtg, 64.0, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__input_samples_part1()
        {
            return GetValue(RawMetricIdx::zrop__input_samples_part1, 63.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__tc_replayer_bin_flushes_reason_bin_memory_threshold()
        {
            return GetValue(RawMetricIdx::swdx__tc_replayer_bin_flushes_reason_bin_memory_threshold, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_gpc_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_gpc_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue raster__frstr_processed_3_edges()
        {
            return GetValue(RawMetricIdx::raster__frstr_processed_3_edges, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_global_cctl()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_global_cctl, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pel__out_read_stalled_stri()
        {
            return GetValue(RawMetricIdx::pel__out_read_stalled_stri, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_sz_128b()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_sz_128b, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_wr()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_wr, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue mme__call_count()
        {
            return GetValue(RawMetricIdx::mme__call_count, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_iso()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_iso, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue tga__input_batches()
        {
            return GetValue(RawMetricIdx::tga__input_batches, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue crop__write_stalled()
        {
            return GetValue(RawMetricIdx::crop__write_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_mmu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_mmu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue smsp__inst_exelwted_pipe_bar()
        {
            return GetValue(RawMetricIdx::smsp__inst_exelwted_pipe_bar, 0.5, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_host_cpu_cas()
        {
            return GetValue(RawMetricIdx::lts__t_requests_host_cpu_cas, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_raster()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_raster, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__samp_input_quads_sz_64b()
        {
            return GetValue(RawMetricIdx::l1tex__samp_input_quads_sz_64b, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gcc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gcc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_crop()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_crop, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__input_requests_expanded_to_samples()
        {
            return GetValue(RawMetricIdx::zrop__input_requests_expanded_to_samples, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_zrd_ni_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_zrd_ni_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__mios_shmem_accesses_su_read_tram()
        {
            return GetValue(RawMetricIdx::sm__mios_shmem_accesses_su_read_tram, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue gpc__rg_utlb_requests_sparse()
        {
            return GetValue(RawMetricIdx::gpc__rg_utlb_requests_sparse, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue vaf__alpha_cycles_stalled_on_mpc()
        {
            return GetValue(RawMetricIdx::vaf__alpha_cycles_stalled_on_mpc, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_zrd_i()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_zrd_i, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__texin_requests_lg_global_ld()
        {
            return GetValue(RawMetricIdx::l1tex__texin_requests_lg_global_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__isbe_allocation_stalled()
        {
            return GetValue(RawMetricIdx::mpc__isbe_allocation_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__zrop_output_samples()
        {
            return GetValue(RawMetricIdx::prop__zrop_output_samples, 64.0, gpc__cycles_elapsed().avg);
        }

        MetricValue prop__earlyz_output_quads()
        {
            return GetValue(RawMetricIdx::prop__earlyz_output_quads, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crop_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crop_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue gpc__tpc1_utlb_requests_sparse()
        {
            return GetValue(RawMetricIdx::gpc__tpc1_utlb_requests_sparse, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_niso()
        {
            return GetValue(RawMetricIdx::lts__t_requests_niso, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__write_requests_compressed_1to1()
        {
            return GetValue(RawMetricIdx::crop__write_requests_compressed_1to1, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__f_output_tex2sm_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__f_output_tex2sm_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_niso_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_niso_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_pitch_linear()
        {
            return GetValue(RawMetricIdx::crop__input_requests_pitch_linear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_gpc_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_gpc_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_host_cpu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_host_cpu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__f_output_tex2sm_busy()
        {
            return GetValue(RawMetricIdx::l1tex__f_output_tex2sm_busy, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_hub_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_hub_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_requests_atomic, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_write_requests()
        {
            return GetValue(RawMetricIdx::lts__mccif_write_requests, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__out_l2_requests_ilwalidate_256b()
        {
            return GetValue(RawMetricIdx::pel__out_l2_requests_ilwalidate_256b, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__mccif_read_stalled()
        {
            return GetValue(RawMetricIdx::lts__mccif_read_stalled, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_crd_i_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_requests_crd_i_condrd, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_zrop_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_requests_zrop_ilwal, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue swdx__input_stalled()
        {
            return GetValue(RawMetricIdx::swdx__input_stalled, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_in_warp_lg_local_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_in_warp_lg_local_ld, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_miss_ltc_rd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_miss_ltc_rd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_gpc_clear()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_gpc_clear, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_atomic_block_stalled_backpressure()
        {
            return GetValue(RawMetricIdx::lts__d_atomic_block_stalled_backpressure, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_in_warp_lg_global_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_in_warp_lg_global_ld, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_cbc()
        {
            return GetValue(RawMetricIdx::lts__t_requests_cbc, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_host_cpu_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_host_cpu_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue pel__in_write_requests_active()
        {
            return GetValue(RawMetricIdx::pel__in_write_requests_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue zrop__input_requests_write_compression_ok()
        {
            return GetValue(RawMetricIdx::zrop__input_requests_write_compression_ok, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_mmu()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_mmu, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_global_red()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_global_red, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue mpc__isbe_allocation_stalled_beta()
        {
            return GetValue(RawMetricIdx::mpc__isbe_allocation_stalled_beta, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__f_output_tex2sm_active()
        {
            return GetValue(RawMetricIdx::l1tex__f_output_tex2sm_active, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue pda__input_restart_indices()
        {
            return GetValue(RawMetricIdx::pda__input_restart_indices, 12.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_requests_gpc_clear()
        {
            return GetValue(RawMetricIdx::lts__t_requests_gpc_clear, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hub_concat_cas()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hub_concat_cas, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__d_decomp_input_sectors()
        {
            return GetValue(RawMetricIdx::lts__d_decomp_input_sectors, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__threads_launched_vsa()
        {
            return GetValue(RawMetricIdx::sm__threads_launched_vsa, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_t2d_backpressured()
        {
            return GetValue(RawMetricIdx::l1tex__t_t2d_backpressured, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_lg_global_ld()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_lg_global_ld, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_crop_condrd()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_crop_condrd, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue zrop__processed_requests_type_plane_eq_fast()
        {
            return GetValue(RawMetricIdx::zrop__processed_requests_type_plane_eq_fast, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue tgb__tasks_active()
        {
            return GetValue(RawMetricIdx::tgb__tasks_active, 8.0, sys__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_prefetch()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_prefetch, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue sm__cycles_active_vsb()
        {
            return GetValue(RawMetricIdx::sm__cycles_active_vsb, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__csb_output_pixels()
        {
            return GetValue(RawMetricIdx::prop__csb_output_pixels, 16.0, gpc__cycles_elapsed().avg);
        }

        MetricValue sm__warps_completed_ooo_3d_ps()
        {
            return GetValue(RawMetricIdx::sm__warps_completed_ooo_3d_ps, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue prop__csb_killed_quadquads()
        {
            return GetValue(RawMetricIdx::prop__csb_killed_quadquads, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vpc__beta_cycles_active()
        {
            return GetValue(RawMetricIdx::vpc__beta_cycles_active, 1.0, gpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_l1tex_atomic()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_l1tex_atomic, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue vaf__cycles_stalled()
        {
            return GetValue(RawMetricIdx::vaf__cycles_stalled, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_pe_ilwal()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_pe_ilwal, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue lts__t_sectors_hit_hub_membar()
        {
            return GetValue(RawMetricIdx::lts__t_sectors_hit_hub_membar, 4.0, fbp__cycles_elapsed().avg);
        }

        MetricValue crop__input_requests_blend_blendopt_killed()
        {
            return GetValue(RawMetricIdx::crop__input_requests_blend_blendopt_killed, 1.0, fbp__cycles_elapsed().avg);
        }

        MetricValue stri__attrs()
        {
            return GetValue(RawMetricIdx::stri__attrs, 1.0, tpc__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_set_accesses_hit_tex_format_lwbemap()
        {
            return GetValue(RawMetricIdx::l1tex__t_set_accesses_hit_tex_format_lwbemap, 4.0, tpc__cycles_elapsed().avg);
        }

        MetricValue tga__cycles_active()
        {
            return GetValue(RawMetricIdx::tga__cycles_active, 1.0, sys__cycles_elapsed().avg);
        }

        MetricValue l1tex__t_atomic_address_conflicts_lg_global_red()
        {
            return GetValue(RawMetricIdx::l1tex__t_atomic_address_conflicts_lg_global_red, 1.0, tpc__cycles_elapsed().avg);
        }

    };

    struct AllMetrics : public RawMetrics
    {
        MetricValue attrs_per_vector()
        {
            const double value = 4;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue bytes_per_sector()
        {
            const double value = 32;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue cbmgr__throughput()
        {
            return MaxPercent(cbmgr__alpha_cbe_allocations(), cbmgr__beta_cbe_allocations());
        }

        MetricValue crop__throughput()
        {
            return MaxPercent(crop__input_requests(), crop__processed_subpackets(), crop__read_subpackets(), crop__write_subpackets());
        }

        MetricValue cwd__throughput()
        {
            return MaxPercent(cwd__output_ctas(), cwd__feedback_mpc_messages());
        }

        MetricValue fe__throughput()
        {
            return fe__output_ops();
        }

        MetricValue four_attrs_per_vert()
        {
            const double value = 4;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue gcc__l15_requests()
        {
            MetricValue result = gcc__l15_requests_hit() + gcc__l15_requests_miss();
            result.peak_sustained = 0.5;
            return result;
        }

        MetricValue gcc__l15_requests_hit_rate_pct()
        {
            MetricValue result = 100.0 * (gcc__l15_requests_hit() / gcc__l15_requests());
            result.peak_sustained = 0.5;
            return result;
        }

        MetricValue gcc__read_sectors()
        {
            MetricValue result = gcc__l15_requests() * 2 + gcc__tsl2_requests();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue gcc__throughput()
        {
            return MaxPercent(gcc__l15_requests(), gcc__read_sectors(), gcc__tsl2_requests());
        }

        MetricValue gcc__tsl2_requests_hit_rate_pct()
        {
            return 100.0 * (gcc__tsl2_requests_hit() / gcc__tsl2_requests());
        }

        MetricValue gpc__gpcl1tlb_requests()
        {
            MetricValue result = gpc__gpcl1tlb_requests_hit() + gpc__gpcl1tlb_requests_miss();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue gpc__prop_utlb_requests()
        {
            MetricValue result = gpc__prop_utlb_requests_hit() + gpc__prop_utlb_requests_miss();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue gpc__rg_utlb_requests()
        {
            MetricValue result = gpc__rg_utlb_requests_hit() + gpc__rg_utlb_requests_miss();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue gpc__tlb_throughput()
        {
            return MaxPercent(gpc__tpc0_utlb_requests(), gpc__tpc1_utlb_requests(), gpc__rg_utlb_requests(), gpc__prop_utlb_requests(), gpc__gpcl1tlb_requests());
        }

        MetricValue gpc__tpc0_utlb_requests()
        {
            MetricValue result = gpc__tpc0_utlb_requests_hit() + gpc__tpc0_utlb_requests_miss();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue gpc__tpc1_utlb_requests()
        {
            MetricValue result = gpc__tpc1_utlb_requests_hit() + gpc__tpc1_utlb_requests_miss();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue gpmpd__throughput()
        {
            return gpmpd__output_packets();
        }

        MetricValue gpmsd__throughput()
        {
            return MaxPercent(gpmsd__input_active_3d(), gpmsd__input_active_compute());
        }

        MetricValue host__throughput()
        {
            return MaxPercent(sys__fb_read_dwords(), sys__fb_write_dwords(), sys__pushbuffer_dwords());
        }

        MetricValue l1tex__t_sectors_hit()
        {
            return l1tex__t_sectors() - l1tex__t_sectors_miss();
        }

        MetricValue l1tex__t_sectors_hit_rate_pct()
        {
            return 100.0 * (l1tex__t_sectors_hit() / l1tex__t_sectors());
        }

        MetricValue l1tex__throughput()
        {
            return MaxPercent(l1tex__texin_sm2tex_active(), l1tex__lod_output_wavefronts(), l1tex__samp_output_wavefronts(), l1tex__x_output_wavefronts(), l1tex__w_output_wavefronts(), l1tex__t_output_wavefronts(), l1tex__m_read_request_active(), l1tex__d_output_wavefronts(), l1tex__d_sectors_fill(), l1tex__f_output_tex2sm_active());
        }

        MetricValue lts__cbc_requests_comptag()
        {
            MetricValue result = lts__cbc_requests_comptag_hit() + lts__cbc_requests_comptag_miss();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue lts__d_sectors()
        {
            MetricValue result = lts__t_sectors_rd() + lts__t_sectors_atomic() * 2 + lts__t_sectors_wr() + lts__t_sectors_fill() + lts__d_decomp_input_sectors();
            result.peak_sustained = 4.0;
            return result;
        }

        MetricValue lts__ltcx_throughput()
        {
            return MaxPercent(lts__ltcx_read_requests(), lts__ltcx_read_sectors(), lts__t_sectors_fill());
        }

        MetricValue lts__mccif_read_bytes()
        {
            MetricValue result = lts__mccif_read_sectors() * bytes_per_sector();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue lts__mccif_read_sectors()
        {
            MetricValue result = lts__mccif_read_cycles_32() + 2 * lts__mccif_read_cycles_64();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue lts__mccif_throughput()
        {
            return MaxPercent(lts__mccif_read_requests(), lts__mccif_read_sectors(), lts__mccif_write_requests(), lts__mccif_write_bytes());
        }

        MetricValue lts__mccif_write_bytes()
        {
            MetricValue result = lts__mccif_write_cycles_16() * 16 + lts__mccif_write_sectors_excluding_16() * bytes_per_sector();
            result.peak_sustained = 32.0;
            return result;
        }

        MetricValue lts__mccif_write_bytes_appx()
        {
            MetricValue result = lts__mccif_write_sectors_excluding_16() * bytes_per_sector();
            result.peak_sustained = 32.0;
            return result;
        }

        MetricValue lts__rop_input_active()
        {
            MetricValue result = lts__t_sectors_crop_wr() + lts__t_sectors_zrop_wr();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue lts__t_sectors_hit_rate_pct()
        {
            return 100.0 * (lts__t_sectors_hit() / lts__t_sectors());
        }

        MetricValue lts__t_tag_requests()
        {
            MetricValue result = lts__t_tag_requests_hit() + lts__t_tag_requests_miss();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue lts__throughput()
        {
            return MaxPercent(lts__r_input_fifo_active(), lts__cbc_requests_comptag(), lts__t_tag_requests(), lts__t_sectors(), lts__xbar_input_active(), lts__xbar_output_active(), lts__rop_input_active(), lts__d_decomp_input_sectors(), lts__d_sectors());
        }

        MetricValue lts__xbar_input_active()
        {
            MetricValue result = (lts__t_sectors_gpc_wr() + lts__t_sectors_gpc_atomic() + lts__t_sectors_hub_wr() + lts__t_sectors_hub_atomic()) + (lts__t_requests_gpc_rd() + lts__t_requests_gpc_membar() + lts__t_requests_hub_rd() + lts__t_requests_hub_membar());
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue mme__throughput()
        {
            return mme__output_method_dwords();
        }

        MetricValue mmu__hubtlb_requests()
        {
            return mmu__hubtlb_requests_hit() + mmu__hubtlb_requests_miss();
        }

        MetricValue mmu__hubtlb_requests_hit_rate_pct()
        {
            return 100.0 * (mmu__hubtlb_requests_hit() / mmu__hubtlb_requests());
        }

        MetricValue mmu__pde_requests()
        {
            return mmu__pde_requests_hit() + mmu__pde_requests_miss();
        }

        MetricValue mmu__pte_requests()
        {
            return mmu__pte_requests_hit() + mmu__pte_requests_miss();
        }

        MetricValue mmu__throughput()
        {
            return MaxPercent(mmu__hubtlb_requests(), mmu__pte_requests());
        }

        MetricValue mpc__throughput()
        {
            return MaxPercent(mpc__isbe_allocations(), sm__ps_quads_launched(), sm__warps_launched_ps());
        }

        MetricValue pda__throughput()
        {
            return MaxPercent(pda__input_prims(), pda__input_verts());
        }

        MetricValue pdb__throughput()
        {
            return pdb__output_pkts();
        }

        MetricValue pel__throughput()
        {
            return MaxPercent(pel__in_write_requests(), pel__out_l2_requests());
        }

        MetricValue pes__throughput()
        {
            return pes__stream_output_attrs();
        }

        MetricValue pixels_per_quad()
        {
            const double value = 4;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue prop__throughput()
        {
            return MaxPercent(prop__csb_output_active(), prop__csb_output_crop_requests_color(), prop__gnic_port0_active(), prop__gnic_port1_active(), gpmsd__sm2gpmsd_pixout_cdp0_active_color(), gpmsd__sm2gpmsd_pixout_cdp1_active_color(), gpmsd__sm2gpmsd_pixout_cdp0_active_shdz(), gpmsd__sm2gpmsd_pixout_cdp1_active_shdz(), prop__input_active(), prop__zrop_output_active());
        }

        MetricValue raster__throughput()
        {
            return MaxPercent(raster__crstr_output_tiles(), raster__frstr_output_cycles(), raster__setup_input_prims(), raster__tc_output_subtiles(), raster__zlwll_input_tiles());
        }

        MetricValue rdm__throughput()
        {
            return MaxPercent(rdm__crop_output_active(), rdm__input_data0_active(), rdm__input_data1_active(), rdm__zrop_output_active());
        }

        MetricValue rop__read_sectors()
        {
            MetricValue result = crop__read_subpackets() + zrop__read_subpackets();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue rop__throughput()
        {
            return rop__read_sectors();
        }

        MetricValue samples_per_z_occluder()
        {
            const double value = 8;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue scc__throughput()
        {
            return scc__input_state_changes();
        }

        MetricValue sked__throughput()
        {
            return sked__dispatch_count();
        }

        MetricValue sm__cycles_active_3d_vtg_only()
        {
            return sm__cycles_active_3d() - sm__cycles_active_3d_ps();
        }

        MetricValue sm__cycles_active_compute()
        {
            return sm__cycles_active() - sm__cycles_active_3d();
        }

        MetricValue sm__inst_exelwted_pipe_fma64plus_pred_on()
        {
            const MetricValue input = smsp__inst_exelwted_pipe_fma64plus_pred_on();
            return MetricValue{ input.sum, input.avg * 4, 0.125, input.cycles_elapsed };
        }

        MetricValue sm__inst_exelwted_pipe_fma64plusplus_pred_on()
        {
            const MetricValue input = smsp__inst_exelwted_pipe_fma64plusplus_pred_on();
            return MetricValue{ input.sum, input.avg * 4, 0.03125, input.cycles_elapsed };
        }

        MetricValue sm__inst_exelwted_pipe_lsu_pred_on()
        {
            const MetricValue input = smsp__inst_exelwted_pipe_lsu_pred_on();
            return MetricValue{ input.sum, input.avg * 4, 1.0, input.cycles_elapsed };
        }

        MetricValue sm__inst_exelwted_pipe_su_pred_on()
        {
            const MetricValue input = smsp__inst_exelwted_pipe_su_pred_on();
            return MetricValue{ input.sum, input.avg * 4, 1.0, input.cycles_elapsed };
        }

        MetricValue sm__mios_active()
        {
            MetricValue result = sm__mios_datapath_active() + sm__mios_shmem_accesses_lsu() + sm__mios_shmem_active_tram();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue sm__mios_inst_exelwted()
        {
            MetricValue result = smsp__inst_exelwted_pipe_fma64plus() + smsp__inst_exelwted_pipe_fma64plusplus() + smsp__inst_exelwted_pipe_lsu() + smsp__inst_exelwted_pipe_su();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue sm__mios_shmem_accesses_lsu()
        {
            MetricValue result = sm__mios_shmem_accesses_lsu_read() + sm__mios_shmem_accesses_lsu_write();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue sm__mios_shmem_active_tram()
        {
            MetricValue result = sm__mios_shmem_accesses_pe_write_tram() + sm__mios_shmem_accesses_su_read_tram();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue sm__ps_quads_sent_to_pixout_pct()
        {
            return 100.0 * (sm__ps_quads_sent_to_pixout() / sm__ps_quads_launched());
        }

        MetricValue sm__threads_launched_ps()
        {
            return sm__ps_quads_launched() * threads_per_quad();
        }

        MetricValue sm__throughput()
        {
            return MaxPercent(smsp__inst_exelwted(), smsp__issue_active(), smsp__inst_issued_coupled(), smsp__inst_issued_decoupled(), smp__tex_requests(), smp__tex_write_back_active(), smp__lsu_write_back_active(), smp__miop_pq_read_active(), sm__mios_inst_exelwted(), smp__inst_exelwted_pipe_adu_pred_on(), smp__inst_exelwted_pipe_bru(), sm__inst_exelwted_pipe_fma64plus_pred_on(), sm__inst_exelwted_pipe_fma64plusplus_pred_on(), smsp__inst_exelwted_pipe_fmai(), smsp__inst_exelwted_pipe_fxu(), sm__inst_exelwted_pipe_lsu_pred_on(), sm__inst_exelwted_pipe_su_pred_on(), smsp__inst_exelwted_pipe_xu(), smp__inst_exelwted_pipe_tex());
        }

        MetricValue smp__inst_exelwted_pipe_adu_pred_on()
        {
            const MetricValue input = smsp__inst_exelwted_pipe_adu_pred_on();
            return MetricValue{ input.sum, input.avg * 2, 0.5, input.cycles_elapsed };
        }

        MetricValue smp__inst_exelwted_pipe_bru()
        {
            const MetricValue input = smsp__inst_exelwted_pipe_bru();
            return MetricValue{ input.sum, input.avg * 2, 1.0, input.cycles_elapsed };
        }

        MetricValue smp__inst_exelwted_pipe_tex()
        {
            const MetricValue input = smsp__inst_exelwted_pipe_tex();
            return MetricValue{ input.sum, input.avg * 2, 0.5, input.cycles_elapsed };
        }

        MetricValue smp__lsu_write_back_active()
        {
            const MetricValue input = smsp__lsu_write_back_active();
            return MetricValue{ input.sum, input.avg * 2, 1.0, input.cycles_elapsed };
        }

        MetricValue smp__miop_pq_read_active()
        {
            MetricValue result = sm__miop_pq_read_active_smp0() + sm__miop_pq_read_active_smp1();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue smp__tex_requests()
        {
            const MetricValue input = smsp__tex_requests();
            return MetricValue{ input.sum, input.avg * 2, 1.0, input.cycles_elapsed };
        }

        MetricValue smp__tex_write_back_active()
        {
            const MetricValue input = smsp__tex_write_back_active();
            return MetricValue{ input.sum, input.avg * 2, 1.0, input.cycles_elapsed };
        }

        MetricValue smsp__inst_issued_coupled()
        {
            MetricValue result = smsp__inst_exelwted_pipe_fe() + smsp__inst_exelwted_pipe_fmai() + smsp__inst_exelwted_pipe_fxu() + smsp__inst_exelwted_pipe_fp16();
            result.peak_sustained = 1.0;
            return result;
        }

        MetricValue smsp__inst_issued_decoupled()
        {
            MetricValue result = smsp__inst_exelwted_pipe_adu() + smsp__inst_exelwted_pipe_bru() + smsp__inst_exelwted_pipe_fma64plus() + smsp__inst_exelwted_pipe_fma64plusplus() + smsp__inst_exelwted_pipe_lsu() + smsp__inst_exelwted_pipe_su() + smsp__inst_exelwted_pipe_tex() + smsp__inst_exelwted_pipe_xu();
            result.peak_sustained = 0.5;
            return result;
        }

        MetricValue stri__throughput()
        {
            return stri__attrs();
        }

        MetricValue swdx__throughput()
        {
            return MaxPercent(swdx__binner_active(), swdx__input_messages(), swdx__input_prims_dual_binned(), swdx__input_prims_single_binned(), swdx__output_active());
        }

        MetricValue tga__throughput()
        {
            return tga__output_tasks();
        }

        MetricValue tgb__output_verts()
        {
            MetricValue result = tgb__output_verts_complex_boundary() + tgb__output_verts_complex_interior() + tgb__output_verts_simple();
            result.peak_sustained = 32.0;
            return result;
        }

        MetricValue tgb__throughput()
        {
            return tgb__output_verts();
        }

        MetricValue threads_per_quad()
        {
            const double value = 4;
            return MetricValue{ value, value, value, 0};
        }

        MetricValue vaf__alpha_fetched_attr_scalar()
        {
            MetricValue result = vaf__alpha_fetched_attr_scalar_pre_cbf() + vaf__alpha_fetched_attr_scalar_post_cbf();
            result.peak_sustained = 4.0;
            return result;
        }

        MetricValue vaf__throughput()
        {
            return MaxPercent(vaf__alpha_fetched_attr_scalar_post_cbf(), vaf__alpha_fetched_attr_scalar_pre_cbf(), vaf__alpha_fetched_attr_vector_post_cbf(), vaf__alpha_fetched_attr_vector_pre_cbf());
        }

        MetricValue vpc__output_attrs()
        {
            MetricValue result = vpc__output_attrs_scalar() + vpc__output_verts() * 4;
            result.peak_sustained = 8.0;
            return result;
        }

        MetricValue vpc__throughput()
        {
            return MaxPercent(vpc__clip_cycles_active(), vpc__clip_output_prims(), vpc__clip_output_verts(), vpc__output_attrs(), vpc__read_isbes());
        }

        MetricValue wwdx__throughput()
        {
            return MaxPercent(wwdx__input_prims(), wwdx__input_tasks());
        }

        MetricValue zrop__input_samples()
        {
            return zrop__input_samples_part1() + zrop__input_samples_part2();
        }

        MetricValue zrop__throughput()
        {
            return MaxPercent(zrop__input_requests(), zrop__processed_subpackets(), zrop__read_subpackets(), zrop__write_subpackets(), prop__zrop_output_samples());
        }

    };

#define LW_GM20B_RAW_METRIC_DESCS(f_) \
    f_(cbmgr__alpha_cbe_allocation_stalled, "# of cycles where alpha CBE allocation stalled") \
    f_(cbmgr__alpha_cbe_allocation_stalled_max_tasks, "# of cycles where alpha CBE allocation stalled: max tasks") \
    f_(cbmgr__alpha_cbe_allocation_stalled_no_space, "# of cycles where alpha CBE allocation stalled: no space") \
    f_(cbmgr__alpha_cbe_allocations, "# of alpha CBEs allocated") \
    f_(cbmgr__beta_cbe_allocation_stalled, "# of cycles where beta CBE allocation stalled") \
    f_(cbmgr__beta_cbe_allocation_stalled_max_tasks, "# of cycles where beta CBE allocation stalled: max tasks") \
    f_(cbmgr__beta_cbe_allocation_stalled_no_space, "# of cycles where beta CBE allocation stalled: no space") \
    f_(cbmgr__beta_cbe_allocations, "# of alpha CBEs allocated") \
    f_(cbmgr__cycles_active, "# of cycles where CBMGR was active") \
    f_(crop__cycles_active, "# of cycles where CROP was active") \
    f_(crop__input_requests, "# of requests sent to CROP") \
    f_(crop__input_requests_2d, "# of 2d requests sent to CROP") \
    f_(crop__input_requests_3d, "# of 3d requests sent to CROP") \
    f_(crop__input_requests_aamode_1xaa, "# of 1xaa requests sent to CROP") \
    f_(crop__input_requests_aamode_2xaa, "# of 2xaa requests sent to CROP") \
    f_(crop__input_requests_aamode_4xaa, "# of 4xaa requests sent to CROP") \
    f_(crop__input_requests_aamode_8xaa, "# of 8xaa requests sent to CROP") \
    f_(crop__input_requests_all_color_channels, "# of requests where all color channels were enabled") \
    f_(crop__input_requests_blend_blendopt_fill_over, "# of requests optimized as fill") \
    f_(crop__input_requests_blend_blendopt_killed, "# of requests killed by blendopt (Cd==1 && Cs*SRC==0)") \
    f_(crop__input_requests_blend_blendopt_read_avoid, "# of requests optimized with read-avoid") \
    f_(crop__input_requests_blend_enabled, "# of requests with blend enabled") \
    f_(crop__input_requests_clear, "# of clear requests sent to CROP") \
    f_(crop__input_requests_pitch_linear, "# of requests targeting pitch-linear memory") \
    f_(crop__input_samples_part1, "# of samples sent to CROP, part 1") \
    f_(crop__input_samples_part2, "# of samples sent to CROP, part 2") \
    f_(crop__input_stalled, "# of cycles where CROP input stage was stalled") \
    f_(crop__input_stalled_upstream_fifo_full, "# of cycles where the input fifo was full, causing upstream stall") \
    f_(crop__processed_requests, "# of requests processed by CROP") \
    f_(crop__processed_requests_compressed, "# of requests processed by CROP, as compressed packets") \
    f_(crop__processed_requests_uncompressed, "# of requests processed by CROP, as uncompressed subpackets") \
    f_(crop__processed_samples_part1, "# of samples processed by CROP, part 1") \
    f_(crop__processed_samples_part2, "# of samples processed by CROP, part 2") \
    f_(crop__processed_subpackets, "# of subpackets processed by CROP") \
    f_(crop__read_requests, "# of requests of any kind that generated 1 or more subpackets") \
    f_(crop__read_returns, "# of requests of any kind that generated 1 or more read subpackets") \
    f_(crop__read_stalled, "# of cycles where read returns were stalled") \
    f_(crop__read_subpackets, "# of subpacket reads of any kind") \
    f_(crop__write_requests, "# of ZROP requests of any kind that wrote 1 or more subpackets") \
    f_(crop__write_requests_compressed_1to1, "# of write requests with compression ratio 1:1 (uncompressed)") \
    f_(crop__write_requests_compressed_2to1, "# of write requests with compression ratio 2:1") \
    f_(crop__write_requests_compressed_4to1, "# of write requests with compression ratio 4:1") \
    f_(crop__write_requests_compressed_8to1_or_fastclear, "# of write requests with compression ratio 8:1") \
    f_(crop__write_stalled, "# of cycles where subpacket writes were stalled on L2") \
    f_(crop__write_subpackets, "# of subpacket writes of any kind") \
    f_(cwd__cycles_active, "# of cycles where CWD was active") \
    f_(cwd__feedback_mpc_messages, "# of feedback messages sent by MPC to CWD") \
    f_(cwd__output_ctas, "# of CTAs launched by CWD") \
    f_(cwd__output_stalled_no_free_slots, "# of cycles where CWD was stalled by MPC: no free slots") \
    f_(cwd__output_stalled_state_ack, "# of cycles where CWD was stalled by MPC: wait for state ack") \
    f_(fbp__cycles_elapsed, "# of cycles elapsed on FBP") \
    f_(fe__cycles_active, "# of cycles where FE was active") \
    f_(fe__cycles_wfi_ctxsw, "# of cycles where FE waited for GR idle, for CTXSW") \
    f_(fe__cycles_wfi_host_scg0, "# of cycles where FE waited for GR idle, for explicit WFI method") \
    f_(fe__cycles_wfi_subch_switch_scg0, "# of cycles where FE waited for GR idle, for subchannel switch") \
    f_(fe__i2m_write_bytes, "# of bytes written by I2M") \
    f_(fe__i2m_write_requests, "# of write requests sent by I2M") \
    f_(fe__i2m_write_stalled_data_buffer_full, "# of cycles where FE was stalled on I2M write data buffer full") \
    f_(fe__i2m_write_stalled_request_fifo_full, "# of cycles where FE was stalled on I2M write request FIFO full") \
    f_(fe__input_method_dwords_i2m, "# of I2M method dwords sent to FE") \
    f_(fe__input_methods, "# of methods sent to FE") \
    f_(fe__input_methods_stalled, "# of cycles where method fetch from HOST to FE was stalled") \
    f_(fe__output_ops, "# of operations sent by FE") \
    f_(fe__output_ops_bundle, "# of bundles sent by FE") \
    f_(fe__output_ops_bundle_scg0_go_idle, "# of GO_IDLE bundles sent by FE (scg0)") \
    f_(fe__output_ops_bundle_scg0_wfi_host, "# of GO_IDLE bundles sent by FE (scg0), for explicit WFI method") \
    f_(fe__output_ops_bundle_scg1_go_idle, "# of GO_IDLE bundles sent by FE (scg1)") \
    f_(fe__output_ops_bundle_scg1_wfi_host, "# of GO_IDLE bundles sent by FE (scg1), for explicit WFI method") \
    f_(fe__output_ops_ld_const, "# of inline constant updates sent by FE") \
    f_(fe__output_ops_vert, "# of vertex indices sent by FE") \
    f_(gcc__cycles_active, "# of cycles where GCC was active") \
    f_(gcc__l15_requests_constant, "# of GCC requests to L1.5 for constant buffer") \
    f_(gcc__l15_requests_hit, "# of GCC hits in L1.5") \
    f_(gcc__l15_requests_hit_constant, "# of GCC hits to L1.5 for constant buffer") \
    f_(gcc__l15_requests_hit_instruction, "# of GCC hits to L1.5 for instructions") \
    f_(gcc__l15_requests_instruction, "# of GCC requests to L1.5 for instructions") \
    f_(gcc__l15_requests_miss, "# of GCC misses in L1.5") \
    f_(gcc__l15_requests_miss_constant, "# of GCC misses to L1.5 for constant buffer") \
    f_(gcc__l15_requests_miss_instruction, "# of GCC misses to L1.5 for instructions") \
    f_(gcc__tsl2_requests, "# of GCC requests for texture/sampler header (TSL2)") \
    f_(gcc__tsl2_requests_hit, "# of GCC hits for texture/sampler header (TSL2)") \
    f_(gcc__tsl2_requests_miss, "# of GCC misses for texture/sampler header (TSL2)") \
    f_(gpc__cycles_elapsed, "# of cycles elapsed on GPC") \
    f_(gpc__gpcl1tlb_requests_hit, "# of PTE requests that hit in GPC L1 TLB") \
    f_(gpc__gpcl1tlb_requests_hit_under_miss, "# of PTE requests that hit on pending-miss in GPC L1 TLB") \
    f_(gpc__gpcl1tlb_requests_miss, "# of PTE requests that missed in GPC L1 TLB") \
    f_(gpc__gpcl1tlb_stalled_on_tag_allocation, "# of cycles where GPC L1 TLB stalled on tag allocation") \
    f_(gpc__gpcl1tlb_stalled_request_fifo_full, "# of cycles where GPC L1 TLB stalled on request FIFO") \
    f_(gpc__prop_utlb_requests_hit, "# of PTE requests that hit in PROP L0 TLB") \
    f_(gpc__prop_utlb_requests_hit_under_miss, "# of PTE requests that hit on pending-miss in PROP L0 TLB") \
    f_(gpc__prop_utlb_requests_miss, "# of PTE requests that missed in PROP L0 TLB") \
    f_(gpc__prop_utlb_requests_sparse, "# of PTE requests to unmapped sparse texture pages") \
    f_(gpc__prop_utlb_stalled_on_tag_allocation, "# of cycles where PROP L0 TLB stalled on tag allocation") \
    f_(gpc__prop_utlb_stalled_request_fifo_full, "# of cycles where PROP L0 TLB stalled on request FIFO") \
    f_(gpc__rg_utlb_requests_hit, "# of PTE requests that hit in RASTER+GCC L0 TLB") \
    f_(gpc__rg_utlb_requests_hit_under_miss, "# of PTE requests that hit on pending-miss in RASTER+GCC L0 TLB") \
    f_(gpc__rg_utlb_requests_miss, "# of PTE requests that missed in RASTER+GCC L0 TLB") \
    f_(gpc__rg_utlb_requests_sparse, "# of PTE requests to unmapped sparse texture pages") \
    f_(gpc__rg_utlb_stalled_on_tag_allocation, "# of cycles where RASTER+GCC L0 TLB stalled on tag allocation") \
    f_(gpc__rg_utlb_stalled_request_fifo_full, "# of cycles where RASTER+GCC L0 TLB stalled on request FIFO") \
    f_(gpc__rg_utlb_stalled_write_buffer_full, "# of cycles where RASTER+GCC L0 TLB write data buffer was full") \
    f_(gpc__tpc0_utlb_requests_hit, "# of PTE requests that hit in TPC0 L0 TLB") \
    f_(gpc__tpc0_utlb_requests_hit_under_miss, "# of PTE requests that hit on pending-miss in TPC0 L0 TLB") \
    f_(gpc__tpc0_utlb_requests_miss, "# of PTE requests that missed in TPC0 L0 TLB") \
    f_(gpc__tpc0_utlb_requests_sparse, "# of PTE requests to unmapped sparse texture pages") \
    f_(gpc__tpc0_utlb_stalled_on_tag_allocation, "# of cycles where TPC0 L0 TLB stalled on tag allocation") \
    f_(gpc__tpc0_utlb_stalled_request_fifo_full, "# of cycles where TPC0 L0 TLB stalled on request FIFO") \
    f_(gpc__tpc0_utlb_stalled_write_buffer_full, "# of cycles where TPC0 L0 TLB write data buffer was full") \
    f_(gpc__tpc1_utlb_requests_hit, "# of PTE requests that hit in TPC1 L0 TLB") \
    f_(gpc__tpc1_utlb_requests_hit_under_miss, "# of PTE requests that hit on pending-miss in TPC1 L0 TLB") \
    f_(gpc__tpc1_utlb_requests_miss, "# of PTE requests that missed in TPC1 L0 TLB") \
    f_(gpc__tpc1_utlb_requests_sparse, "# of PTE requests to unmapped sparse texture pages") \
    f_(gpc__tpc1_utlb_stalled_on_tag_allocation, "# of cycles where TPC1 L0 TLB stalled on tag allocation") \
    f_(gpc__tpc1_utlb_stalled_request_fifo_full, "# of cycles where TPC1 L0 TLB stalled on request FIFO") \
    f_(gpc__tpc1_utlb_stalled_write_buffer_full, "# of cycles where TPC1 L0 TLB write data buffer was full") \
    f_(gpmpd__cycles_active, "# of cycles where GPMPD was active") \
    f_(gpmpd__input_batches, "# of batches sent to GPMPD from PDA") \
    f_(gpmpd__input_tasks, "# of beta tasks sent to GPMPD from PDB") \
    f_(gpmpd__output_packets, "# of packets output by GPMPD") \
    f_(gpmpd__output_stalled_batch, "# of cycles where batch output to MPC was stalled") \
    f_(gpmpd__output_stalled_task, "# of cycles where task output to MPC was stalled") \
    f_(gpmsd__cycles_active, "# of cycles where GPMSD was active") \
    f_(gpmsd__input_active_3d, "# of cycles where 3D input was sent to GPMSD") \
    f_(gpmsd__input_active_compute, "# of cycles where compute input was sent to GPMSD") \
    f_(gpmsd__input_pixels, "# of pixels sent to GPMSD") \
    f_(gpmsd__input_pixels_2d, "# of 2D pixels sent to GPMSD") \
    f_(gpmsd__input_pixels_3d, "# of 3D pixels sent to GPMSD") \
    f_(gpmsd__input_pixels_fully_covered, "# of pixels sent to GPMSD with full sample coverage") \
    f_(gpmsd__input_quads, "# of pixel-quads sent to GPMSD") \
    f_(gpmsd__input_quads_2d, "# of 2D pixel-quads sent to GPMSD") \
    f_(gpmsd__input_quads_3d, "# of 3D pixel-quads sent to GPMSD") \
    f_(gpmsd__input_samples, "# of samples sent to GPMSD") \
    f_(gpmsd__input_samples_2d, "# of 2D samples sent to GPMSD") \
    f_(gpmsd__input_samples_3d, "# of 3D samples sent to GPMSD") \
    f_(gpmsd__input_stalled, "# of cycles where input to GPMSD was stalled") \
    f_(gpmsd__output_quads, "# of quads output by GPMSD") \
    f_(gpmsd__sm2gpmsd_pixout_cdp0_active_color, "# of cycles where PIXOUT to CDP0 sent color data") \
    f_(gpmsd__sm2gpmsd_pixout_cdp0_active_shdz, "# of cycles where PIXOUT to CDP0 sent shader-z data") \
    f_(gpmsd__sm2gpmsd_pixout_cdp1_active_color, "# of cycles where PIXOUT to CDP1 sent color data") \
    f_(gpmsd__sm2gpmsd_pixout_cdp1_active_shdz, "# of cycles where PIXOUT to CDP1 sent shader-z data") \
    f_(gpu__dispatch_count, "# of compute dispatches sent to GPU") \
    f_(gpu__draw_count, "# of draw calls sent to GPU") \
    f_(gpu__time_active, "total duration in nanoseconds") \
    f_(gpu__time_duration, "incremental duration in nanoseconds; isolated measurement is same as gpu__time_active") \
    f_(gpu__time_end, "end timestamp, relative to start of frame") \
    f_(gpu__time_start, "start timestamp, relative to start of frame") \
    f_(gr__cycles_active, "# of cycles where GR was active") \
    f_(gr__cycles_idle, "# of cycles where GR was idle") \
    f_(host__chsw_switches, "# of channel switches") \
    f_(l1tex__d_cycles_bank_conflict_2x2, "# of extra cycles L1TEX D-Stage spent on bank-conflicts within 2x2 bilerps") \
    f_(l1tex__d_cycles_bank_conflict_bilerp, "# of extra cycles L1TEX D-Stage spent on bank-conflicts across 2x2 bilerps") \
    f_(l1tex__d_d2f_active, "# of cycles where interface from L1TEX D-Stage to L1TEX F-Stage was active") \
    f_(l1tex__d_d2f_backpressured, "# of cycles where interface from L1TEX D-Stage to L1TEX F-Stage was backpressured") \
    f_(l1tex__d_d2f_busy, "# of cycles where interface from L1TEX D-Stage to L1TEX F-Stage was busy") \
    f_(l1tex__d_d2f_stalled, "# of cycles where interface from L1TEX D-Stage to L1TEX F-Stage was stalled") \
    f_(l1tex__d_output_wavefronts, "# of wavefronts output by L1TEX D-Stage") \
    f_(l1tex__d_sectors_fill, "# of sectors read by L1TEX D-Stage from L2") \
    f_(l1tex__f_output_tex2sm_active, "# of cycles where interface carrying return data from L1TEX F-Stage back to SM was active") \
    f_(l1tex__f_output_tex2sm_backpressured, "# of cycles where interface carrying return data from L1TEX F-Stage back to SM was backpressured") \
    f_(l1tex__f_output_tex2sm_busy, "# of cycles where interface carrying return data from L1TEX F-Stage back to SM was busy") \
    f_(l1tex__f_output_tex2sm_stalled, "# of cycles where interface carrying return data from L1TEX F-Stage back to SM was stalled") \
    f_(l1tex__lod_output_wavefronts, "# of wavefronts output by L1TEX LOD-Stage") \
    f_(l1tex__m_read_request_active, "# of cycles where L1TEX M-Stage sent requests to L2") \
    f_(l1tex__m_read_request_stalled, "# of cycles where L1TEX M-Stage requests to L2 were stalled") \
    f_(l1tex__m_read_sector_lwmulative_miss_latency, "cumulative # of latency cycles for sector read misses") \
    f_(l1tex__m_read_sectors, "# of sector reads issued by L1TEX M-Stage to L2") \
    f_(l1tex__m_read_sectors_lg_global_atom, "# of sector reads issued by L1TEX M-Stage to L2 for global atomic non-CAS instructions (ATOM)") \
    f_(l1tex__m_read_sectors_lg_global_atom_cas, "# of sector reads issued by L1TEX M-Stage to L2 for global atomic CAS instructions (ATOM.CAS)") \
    f_(l1tex__m_read_sectors_lg_global_ld, "# of sector reads issued by L1TEX M-Stage to L2 for global load instructions (LDG, LD)") \
    f_(l1tex__m_read_sectors_lg_local_ld, "# of sector reads issued by L1TEX M-Stage to L2 for local load instructions (LDL, LD)") \
    f_(l1tex__m_read_sectors_surface_atom, "# of sector reads issued by L1TEX M-Stage to L2 for surface atomic non-CAS instructions (SUATOM)") \
    f_(l1tex__m_read_sectors_surface_atom_cas, "# of sector reads issued by L1TEX M-Stage to L2 for surface atomic CAS instructions (SUATOM.CAS)") \
    f_(l1tex__m_read_sectors_surface_ld_d, "# of sector reads issued by L1TEX M-Stage to L2 for surface load bytes instructions (SULD.D)") \
    f_(l1tex__m_read_sectors_surface_ld_p, "# of sector reads issued by L1TEX M-Stage to L2 for surface load pixel instructions (SULD.P)") \
    f_(l1tex__m_read_sectors_tex_format_1d2d_array, "# of sector reads issued by L1TEX M-Stage to L2 for 1D or 2D array") \
    f_(l1tex__m_read_sectors_tex_format_1d2d_tex, "# of sector reads issued by L1TEX M-Stage to L2 for 1D or 2D texture") \
    f_(l1tex__m_read_sectors_tex_format_1d_buffer, "# of sector reads issued by L1TEX M-Stage to L2 for 1D buffer") \
    f_(l1tex__m_read_sectors_tex_format_3d, "# of sector reads issued by L1TEX M-Stage to L2 for 3D texture") \
    f_(l1tex__m_read_sectors_tex_format_lwbemap, "# of sector reads issued by L1TEX M-Stage to L2 for lwbemap") \
    f_(l1tex__m_read_sectors_tex_format_no_mipmap, "# of sector reads issued by L1TEX M-Stage to L2 for non-mipmapped texture") \
    f_(l1tex__m_stalled_on_tag_allocation, "# of cycles where L1TEX M-Stage stalled on tag allocation") \
    f_(l1tex__m_write_sectors_lg_global_atom, "# of sector writes issued by L1TEX M-Stage to L2 for global atomic non-CAS instructions (ATOM)") \
    f_(l1tex__m_write_sectors_lg_global_atom_cas, "# of sector writes issued by L1TEX M-Stage to L2 for global atomic CAS instructions (ATOM.CAS)") \
    f_(l1tex__m_write_sectors_lg_global_red, "# of sector writes issued by L1TEX M-Stage to L2 for global reduction instructions (RED)") \
    f_(l1tex__m_write_sectors_lg_global_st, "# of sector writes issued by L1TEX M-Stage to L2 for global store instructions (STG, ST)") \
    f_(l1tex__m_write_sectors_lg_local_st, "# of sector writes issued by L1TEX M-Stage to L2 for local store instructions (STL, ST)") \
    f_(l1tex__m_write_sectors_surface_atom, "# of sector writes issued by L1TEX M-Stage to L2 for surface atomic non-CAS instructions (SUATOM)") \
    f_(l1tex__m_write_sectors_surface_atom_cas, "# of sector writes issued by L1TEX M-Stage to L2 for surface atomic CAS instructions (SUATOM.CAS)") \
    f_(l1tex__m_write_sectors_surface_red, "# of sector writes issued by L1TEX M-Stage to L2 for surface reduction instructions (SURED)") \
    f_(l1tex__m_write_sectors_surface_st, "# of sector writes issued by L1TEX M-Stage to L2 for surface store instructions (SUST)") \
    f_(l1tex__samp_input_quads_filter_aniso, "# of quads sent to L1TEX SAMP-Stage with aniso filter mode") \
    f_(l1tex__samp_input_quads_filter_bilinear, "# of quads sent to L1TEX SAMP-Stage with bilinear filter mode (includes nearest-neighbor filtering)") \
    f_(l1tex__samp_input_quads_filter_trilinear, "# of quads sent to L1TEX SAMP-Stage with trilinear filter mode") \
    f_(l1tex__samp_input_quads_sz_128b, "# of 128-bit quads sent to L1TEX SAMP-Stage with any filter mode") \
    f_(l1tex__samp_input_quads_sz_128b_nearest, "# of 128-bit quads sent to L1TEX SAMP-Stage with nearest-neighbor filter mode") \
    f_(l1tex__samp_input_quads_sz_32b, "# of 32-bit quads sent to L1TEX SAMP-Stage with any filter mode") \
    f_(l1tex__samp_input_quads_sz_32b_nearest, "# of 32-bit quads sent to L1TEX SAMP-Stage with nearest-neighbor filter mode") \
    f_(l1tex__samp_input_quads_sz_64b, "# of 64-bit quads sent to L1TEX SAMP-Stage with any filter mode") \
    f_(l1tex__samp_input_quads_sz_64b_nearest, "# of 64-bit quads sent to L1TEX SAMP-Stage with nearest-neighbor filter mode") \
    f_(l1tex__samp_input_quads_sz_96b, "# of 96-bit quads sent to L1TEX SAMP-Stage with any filter mode") \
    f_(l1tex__samp_input_quads_sz_96b_nearest, "# of 96-bit quads sent to L1TEX SAMP-Stage with nearest-neighbor filter mode") \
    f_(l1tex__samp_input_quads_tex_format_1d_buffer, "# of quads sent to L1TEX SAMP-Stage for 1D buffer") \
    f_(l1tex__samp_input_quads_tex_format_1d_tex, "# of quads sent to L1TEX SAMP-Stage for 1D texture") \
    f_(l1tex__samp_input_quads_tex_format_2d_mipmap, "# of quads sent to L1TEX SAMP-Stage for 2d mipmapped texture") \
    f_(l1tex__samp_input_quads_tex_format_2d_nomipmap, "# of quads sent to L1TEX SAMP-Stage for 2d non-mipmapped texture") \
    f_(l1tex__samp_input_quads_tex_format_3d, "# of quads sent to L1TEX SAMP-Stage for 3d texture") \
    f_(l1tex__samp_input_quads_tex_format_lwbemap, "# of quads sent to L1TEX SAMP-Stage for lwbemap") \
    f_(l1tex__samp_output_wavefronts, "# of wavefronts output by L1TEX SAMP-Stage") \
    f_(l1tex__samp_pre_nop_wavefronts, "# of wavefronts in L1TEX SAMP-Stage before the NOP optimization") \
    f_(l1tex__samp_samp2mipb_active, "# of cycles where interface from L1TEX SAMP-Stage to L1TEX MIPB-Stage was active") \
    f_(l1tex__samp_samp2mipb_backpressured, "# of cycles where interface from L1TEX SAMP-Stage to L1TEX MIPB-Stage was backpressured") \
    f_(l1tex__samp_samp2mipb_busy, "# of cycles where interface from L1TEX SAMP-Stage to L1TEX MIPB-Stage was busy") \
    f_(l1tex__samp_samp2mipb_stalled, "# of cycles where interface from L1TEX SAMP-Stage to L1TEX MIPB-Stage was stalled") \
    f_(l1tex__samp_samp2x_active, "# of cycles where interface from L1TEX SAMP-Stage to L1TEX X-Stage was active") \
    f_(l1tex__samp_samp2x_backpressured, "# of cycles where interface from L1TEX SAMP-Stage to L1TEX X-Stage was backpressured") \
    f_(l1tex__samp_samp2x_busy, "# of cycles where interface from L1TEX SAMP-Stage to L1TEX X-Stage was busy") \
    f_(l1tex__samp_samp2x_stalled, "# of cycles where interface from L1TEX SAMP-Stage to L1TEX X-Stage was stalled") \
    f_(l1tex__t_atomic_address_conflicts_lg_global_atom, "# of cycles where an address conflict oclwrred for global atomic non-CAS instructions (ATOM)") \
    f_(l1tex__t_atomic_address_conflicts_lg_global_atom_cas, "# of cycles where an address conflict oclwrred for global atomic CAS instructions (ATOM.CAS)") \
    f_(l1tex__t_atomic_address_conflicts_lg_global_red, "# of cycles where an address conflict oclwrred for global reduction instructions (RED)") \
    f_(l1tex__t_atomic_address_conflicts_surface_atom, "# of cycles where an address conflict oclwrred for surface atomic non-CAS instructions (SUATOM)") \
    f_(l1tex__t_atomic_address_conflicts_surface_atom_cas, "# of cycles where an address conflict oclwrred for surface atomic CAS instructions (SUATOM.CAS)") \
    f_(l1tex__t_atomic_address_conflicts_surface_red, "# of cycles where an address conflict oclwrred for surface reduction instructions (SURED)") \
    f_(l1tex__t_output_wavefronts, "# of wavefronts output by L1TEX T-Stage") \
    f_(l1tex__t_sectors, "# of sector requests to L1TEX T-Stage") \
    f_(l1tex__t_sectors_miss, "# of sector requests to L1TEX T-Stage that missed") \
    f_(l1tex__t_sectors_miss_lg, "# of sector requests to L1TEX T-Stage that missed for local or global instructions") \
    f_(l1tex__t_sectors_promoted, "# of sector requests from L1TEX T-Stage to L2 generated for line fills (sectors not addressed in provoking request)") \
    f_(l1tex__t_set_accesses, "# of cache set accesses") \
    f_(l1tex__t_set_accesses_cctlt, "# of cache set accesses for texture cache ilwalidate instructions (CCTLT)") \
    f_(l1tex__t_set_accesses_hit, "# of cache set accesses with tag-hit and all sector-hits") \
    f_(l1tex__t_set_accesses_hit_in_warp_lg_global_ld, "# of cache set accesses that hit-in-warp for global load instructions") \
    f_(l1tex__t_set_accesses_hit_in_warp_lg_global_st, "# of cache set accesses that hit-in-warp for global store instructions") \
    f_(l1tex__t_set_accesses_hit_in_warp_lg_local_ld, "# of cache set accesses that hit-in-warp for local load instructions") \
    f_(l1tex__t_set_accesses_hit_in_warp_lg_local_st, "# of cache set accesses that hit-in-warp for local store instructions") \
    f_(l1tex__t_set_accesses_hit_in_warp_surface_ld, "# of cache set accesses that hit-in-warp for surface load instructions") \
    f_(l1tex__t_set_accesses_hit_in_warp_surface_st, "# of cache set accesses that hit-in-warp for surface store instructions") \
    f_(l1tex__t_set_accesses_hit_lg_global_ld, "# of cache set accesses that hit for global load instructions (LDG, LD)") \
    f_(l1tex__t_set_accesses_hit_lg_local_ld, "# of cache set accesses that hit for local load instructions (LDL, LD)") \
    f_(l1tex__t_set_accesses_hit_tex_format_1d2d_array, "# of cache set accesses with tag-hit and all sector-hits, for 1D or 2D array") \
    f_(l1tex__t_set_accesses_hit_tex_format_1d2d_tex, "# of cache set accesses with tag-hit and all sector-hits, for 1D or 2D texture") \
    f_(l1tex__t_set_accesses_hit_tex_format_1d_buffer, "# of cache set accesses with tag-hit and all sector-hits, for 1D buffer") \
    f_(l1tex__t_set_accesses_hit_tex_format_3d, "# of cache set accesses with tag-hit and all sector-hits, for 3D texture") \
    f_(l1tex__t_set_accesses_hit_tex_format_lwbemap, "# of cache set accesses with tag-hit and all sector-hits, for lwbemap") \
    f_(l1tex__t_set_accesses_hit_tex_format_no_mipmap, "# of cache set accesses with tag-hit and all sector-hits, for non-mipmapped texture") \
    f_(l1tex__t_set_accesses_lg_global_atom, "# of cache set accesses for global atomic non-CAS instructions (ATOM)") \
    f_(l1tex__t_set_accesses_lg_global_atom_cas, "# of cache set accesses for global atomic CAS instructions (ATOM.CAS)") \
    f_(l1tex__t_set_accesses_lg_global_cctl, "# of cache set accesses for global cache control instructions (CCTL)") \
    f_(l1tex__t_set_accesses_lg_global_ld, "# of cache set accesses for global load instructions (LDG, LD)") \
    f_(l1tex__t_set_accesses_lg_global_red, "# of cache set accesses for global reduction instructions (RED)") \
    f_(l1tex__t_set_accesses_lg_global_st, "# of cache set accesses for global store instructions (STG, ST)") \
    f_(l1tex__t_set_accesses_lg_local_cctl, "# of cache set accesses for local cache ilwalidate instructions (CCTLL)") \
    f_(l1tex__t_set_accesses_lg_local_ld, "# of cache set accesses for local load instructions (LDL, LD)") \
    f_(l1tex__t_set_accesses_lg_local_st, "# of cache set accesses for local store instructions (STL, ST)") \
    f_(l1tex__t_set_accesses_miss_lg_global_ld, "# of cache set accesses that missed for global load instructions (LDG, LD)") \
    f_(l1tex__t_set_accesses_miss_lg_local_ld, "# of cache set accesses that missed for local load instructions (LDL, LD)") \
    f_(l1tex__t_set_accesses_miss_surface_ld, "# of cache set accesses that missed for surface load instructions (SULD)") \
    f_(l1tex__t_set_accesses_miss_tag_hit, "# of cache set accesses with tag-hit containing sector-misses") \
    f_(l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_array, "# of cache set accesses with tag-hit containing sector-misses, for 1D or 2D array") \
    f_(l1tex__t_set_accesses_miss_tag_hit_tex_format_1d2d_tex, "# of cache set accesses with tag-hit containing sector-misses, for 1D or 2D texture") \
    f_(l1tex__t_set_accesses_miss_tag_hit_tex_format_1d_buffer, "# of cache set accesses with tag-hit containing sector-misses, for 1D buffer") \
    f_(l1tex__t_set_accesses_miss_tag_hit_tex_format_3d, "# of cache set accesses with tag-hit containing sector-misses, for 3D texture") \
    f_(l1tex__t_set_accesses_miss_tag_hit_tex_format_lwbemap, "# of cache set accesses with tag-hit containing sector-misses, for lwbemap") \
    f_(l1tex__t_set_accesses_miss_tag_hit_tex_format_no_mipmap, "# of cache set accesses with tag-hit containing sector-misses, for non-mipmapped texture") \
    f_(l1tex__t_set_accesses_miss_tag_miss, "# of cache set accesses with tag-miss (all sector-misses)") \
    f_(l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_array, "# of cache set accesses with tag-miss (all sector-misses), for 1D or 2D array") \
    f_(l1tex__t_set_accesses_miss_tag_miss_tex_format_1d2d_tex, "# of cache set accesses with tag-miss (all sector-misses), for 1D or 2D texture") \
    f_(l1tex__t_set_accesses_miss_tag_miss_tex_format_3d, "# of cache set accesses with tag-miss (all sector-misses), for 3D texture") \
    f_(l1tex__t_set_accesses_miss_tag_miss_tex_format_lwbemap, "# of cache set accesses with tag-miss (all sector-misses), for lwbemap") \
    f_(l1tex__t_set_accesses_miss_tag_miss_tex_format_no_mipmap, "# of cache set accesses with tag-miss (all sector-misses), for non-mipmapped texture") \
    f_(l1tex__t_set_accesses_surface_atom, "# of cache set accesses for surface atomic non-CAS instructions (SUATOM)") \
    f_(l1tex__t_set_accesses_surface_atom_cas, "# of cache set accesses for surface atomic CAS instructions (SUATOM.CAS)") \
    f_(l1tex__t_set_accesses_surface_ld, "# of cache set accesses for surface load instructions (SULD)") \
    f_(l1tex__t_set_accesses_surface_red, "# of cache set accesses for surface reduction instructions (SURED)") \
    f_(l1tex__t_set_accesses_surface_st, "# of cache set accesses for surface store instructions (SUST)") \
    f_(l1tex__t_set_accesses_tex_format_1d2d_array, "# of cache set accesses for 1D or 2D array") \
    f_(l1tex__t_set_accesses_tex_format_1d2d_tex, "# of cache set accesses for 1D or 2D texture") \
    f_(l1tex__t_set_accesses_tex_format_1d_buffer, "# of cache set accesses for 1D buffer") \
    f_(l1tex__t_set_accesses_tex_format_3d, "# of cache set accesses for 3D texture") \
    f_(l1tex__t_set_accesses_tex_format_lwbemap, "# of cache set accesses for lwbemap") \
    f_(l1tex__t_set_accesses_tex_format_no_mipmap, "# of cache set accesses for non-mipmapped texture") \
    f_(l1tex__t_set_conflicts_cctlt, "# of extra cycles spent on set conflicts in T-Stage for texture cache ilwalidate instructions (CCTLT)") \
    f_(l1tex__t_set_conflicts_lg_global_atom, "# of extra cycles spent on set conflicts in T-Stage for global atomic non-CAS instructions (ATOM)") \
    f_(l1tex__t_set_conflicts_lg_global_atom_cas, "# of extra cycles spent on set conflicts in T-Stage for global atomic CAS instructions (ATOM.CAS)") \
    f_(l1tex__t_set_conflicts_lg_global_cctl, "# of extra cycles spent on set conflicts in T-Stage for global cache control instructions (CCTL)") \
    f_(l1tex__t_set_conflicts_lg_global_ld, "# of extra cycles spent on set conflicts in T-Stage for global load instructions (LDG, LD)") \
    f_(l1tex__t_set_conflicts_lg_global_red, "# of extra cycles spent on set conflicts in T-Stage for global reduction instructions (RED)") \
    f_(l1tex__t_set_conflicts_lg_global_st, "# of extra cycles spent on set conflicts in T-Stage for global store instructions (STG, ST)") \
    f_(l1tex__t_set_conflicts_lg_local_cctl, "# of extra cycles spent on set conflicts in T-Stage for local cache ilwalidate instructions (CCTLL)") \
    f_(l1tex__t_set_conflicts_lg_local_ld, "# of extra cycles spent on set conflicts in T-Stage for local load instructions (LDL, LD)") \
    f_(l1tex__t_set_conflicts_lg_local_st, "# of extra cycles spent on set conflicts in T-Stage for local store instructions (STL, ST)") \
    f_(l1tex__t_set_conflicts_surface_atom, "# of extra cycles spent on set conflicts in T-Stage for surface atomic non-CAS instructions (SUATOM)") \
    f_(l1tex__t_set_conflicts_surface_atom_cas, "# of extra cycles spent on set conflicts in T-Stage for surface atomic CAS instructions (SUATOM.CAS)") \
    f_(l1tex__t_set_conflicts_surface_ld, "# of extra cycles spent on set conflicts in T-Stage for surface load instructions (SULD)") \
    f_(l1tex__t_set_conflicts_surface_red, "# of extra cycles spent on set conflicts in T-Stage for surface reduction instructions (SURED)") \
    f_(l1tex__t_set_conflicts_surface_st, "# of extra cycles spent on set conflicts in T-Stage for surface store instructions (SUST)") \
    f_(l1tex__t_set_conflicts_tex_bilinear, "# of extra cycles spent on set conflicts in T-Stage for bilinear wavefronts") \
    f_(l1tex__t_set_conflicts_tex_trilinear, "# of extra cycles spent on set conflicts in T-Stage for trilinear wavefronts") \
    f_(l1tex__t_t2d_active, "# of cycles where interface from L1TEX T-Stage to L1TEX D-Stage input FIFO was active") \
    f_(l1tex__t_t2d_backpressured, "# of cycles where interface from L1TEX T-Stage to L1TEX D-Stage input FIFO was backpressured") \
    f_(l1tex__t_t2d_busy, "# of cycles where interface from L1TEX T-Stage to L1TEX D-Stage input FIFO was busy") \
    f_(l1tex__t_t2d_stalled, "# of cycles where interface from L1TEX T-Stage to L1TEX D-Stage input FIFO was stalled") \
    f_(l1tex__t_texels, "# of valid texel queries at L1TEX T-Stage") \
    f_(l1tex__texin2m_fifo_input_stalled, "# of cycles where TEXIN-to-M-Stage FIFO's input interface was stalled") \
    f_(l1tex__texin2m_fifo_output_active, "# of cycles where TEXIN-to-M-Stage FIFO's output interface was active") \
    f_(l1tex__texin2m_fifo_output_backpressured, "# of cycles where TEXIN-to-M-Stage FIFO's output interface was backpressured") \
    f_(l1tex__texin2m_fifo_output_busy, "# of cycles where TEXIN-to-M-Stage FIFO's output interface was busy") \
    f_(l1tex__texin2m_fifo_output_stalled, "# of cycles where TEXIN-to-M-Stage FIFO's output interface was stalled") \
    f_(l1tex__texin_requests_cctlt, "# of requests sent to TEXIN for texture cache ilwalidate instructions (CCTLT)") \
    f_(l1tex__texin_requests_lg_global_atom, "# of requests sent to TEXIN for global atomic non-CAS instructions (ATOM)") \
    f_(l1tex__texin_requests_lg_global_atom_cas, "# of requests sent to TEXIN for global atomic CAS instructions (ATOM.CAS)") \
    f_(l1tex__texin_requests_lg_global_cctl, "# of requests sent to TEXIN for global cache control instructions (CCTL)") \
    f_(l1tex__texin_requests_lg_global_ld, "# of requests sent to TEXIN for global load instructions (LDG, LD)") \
    f_(l1tex__texin_requests_lg_global_red, "# of requests sent to TEXIN for global reduction instructions (RED)") \
    f_(l1tex__texin_requests_lg_global_st, "# of requests sent to TEXIN for global store instructions (STG, ST)") \
    f_(l1tex__texin_requests_lg_local_cctl, "# of requests sent to TEXIN for local cache ilwalidate instructions (CCTLL)") \
    f_(l1tex__texin_requests_lg_local_ld, "# of requests sent to TEXIN for local load instructions (LDL, LD)") \
    f_(l1tex__texin_requests_lg_local_st, "# of requests sent to TEXIN for local store instructions (STL, ST)") \
    f_(l1tex__texin_requests_membar, "# of requests sent to TEXIN for memory barrier instructions (MEMBAR)") \
    f_(l1tex__texin_requests_null_lg, "# of requests sent to TEXIN for null local or global instructions") \
    f_(l1tex__texin_requests_null_surface, "# of requests sent to TEXIN for null surface instructions") \
    f_(l1tex__texin_requests_null_tex, "# of requests sent to TEXIN for null texture instructions") \
    f_(l1tex__texin_requests_surface_atom, "# of requests sent to TEXIN for surface atomic non-CAS instructions (SUATOM)") \
    f_(l1tex__texin_requests_surface_atom_cas, "# of requests sent to TEXIN for surface atomic CAS instructions (SUATOM.CAS)") \
    f_(l1tex__texin_requests_surface_ld, "# of requests sent to TEXIN for surface load instructions (SULD)") \
    f_(l1tex__texin_requests_surface_red, "# of requests sent to TEXIN for surface reduction instructions (SURED)") \
    f_(l1tex__texin_requests_surface_st, "# of requests sent to TEXIN for surface store instructions (SUST)") \
    f_(l1tex__texin_requests_tex, "# of texture requests (quads) sent to TEXIN") \
    f_(l1tex__texin_sm2tex_active, "# of cycles where interface carrying requests from SM to L1TEX TEXIN-Stage was active") \
    f_(l1tex__texin_sm2tex_backpressured, "# of cycles where interface carrying requests from SM to L1TEX TEXIN-Stage was backpressured") \
    f_(l1tex__texin_sm2tex_busy, "# of cycles where interface carrying requests from SM to L1TEX TEXIN-Stage was busy") \
    f_(l1tex__texin_sm2tex_stalled, "# of cycles where interface carrying requests from SM to L1TEX TEXIN-Stage was stalled") \
    f_(l1tex__texin_stalled_on_tsl2_miss, "# of cycles where TEXIN was stalled on TSL2 cache miss, requesting texture or sampler header") \
    f_(l1tex__texin_tsl1_requests_hit_samphdr, "# of sampler header requests sent to TSL1 that hit") \
    f_(l1tex__texin_tsl1_requests_hit_texhdr, "# of texture header requests sent to TSL1 that hit") \
    f_(l1tex__texin_tsl1_requests_miss_samphdr, "# of sampler header requests sent to TSL1 that missed") \
    f_(l1tex__texin_tsl1_requests_miss_texhdr, "# of texture header requests sent to TSL1 that missed") \
    f_(l1tex__w_output_wavefronts, "# of wavefronts output by L1TEX W-Stage") \
    f_(l1tex__w_w2d_active, "# of cycles where interface from L1TEX W-Stage to L1TEX D-Stage input FIFO was active") \
    f_(l1tex__w_w2d_backpressured, "# of cycles where interface from L1TEX W-Stage to L1TEX D-Stage input FIFO was backpressured") \
    f_(l1tex__w_w2d_busy, "# of cycles where interface from L1TEX W-Stage to L1TEX D-Stage input FIFO was busy") \
    f_(l1tex__w_w2d_stalled, "# of cycles where interface from L1TEX W-Stage to L1TEX D-Stage input FIFO was stalled") \
    f_(l1tex__x_output_wavefronts, "# of wavefronts output by L1TEX X-Stage") \
    f_(l1tex__x_x2t_active, "# of cycles where interface from L1TEX X-Stage to L1TEX T-Stage was active") \
    f_(l1tex__x_x2t_backpressured, "# of cycles where interface from L1TEX X-Stage to L1TEX T-Stage was backpressured") \
    f_(l1tex__x_x2t_busy, "# of cycles where interface from L1TEX X-Stage to L1TEX T-Stage was busy") \
    f_(l1tex__x_x2t_stalled, "# of cycles where interface from L1TEX X-Stage to L1TEX T-Stage was stalled") \
    f_(l1tex__x_x2w_active, "# of cycles where interface from L1TEX X-Stage to L1TEX W-Stage was active") \
    f_(l1tex__x_x2w_backpressured, "# of cycles where interface from L1TEX X-Stage to L1TEX W-Stage was backpressured") \
    f_(l1tex__x_x2w_busy, "# of cycles where interface from L1TEX X-Stage to L1TEX W-Stage was busy") \
    f_(l1tex__x_x2w_stalled, "# of cycles where interface from L1TEX X-Stage to L1TEX W-Stage was stalled") \
    f_(lts__cbc_requests_comptag_fill, "# of requests causing a CBC comptagline fill") \
    f_(lts__cbc_requests_comptag_hit, "# of requests that hit in CBC") \
    f_(lts__cbc_requests_comptag_miss, "# of requests that missed in CBC") \
    f_(lts__cbc_requests_comptag_writeback, "# of requests causing a CBC comptagline writeback") \
    f_(lts__cbc_requests_hit_clear_zbc, "# of requests for zero-bandwidth clear that hit in CBC") \
    f_(lts__cbc_requests_hit_clear_zbc_crop, "# of requests for zero-bandwidth clear for CROP that hit in CBC") \
    f_(lts__cbc_requests_hit_clear_zbc_zrop, "# of requests for zero-bandwidth clear for ZROP that hit in CBC") \
    f_(lts__cbc_requests_miss_clear_zbc, "# of requests for zero-bandwidth clear that missed in CBC") \
    f_(lts__cbc_requests_miss_clear_zbc_crop, "# of requests for zero-bandwidth clear for CROP that missed in CBC") \
    f_(lts__cbc_requests_miss_clear_zbc_zrop, "# of requests for zero-bandwidth clear for ZROP that missed in CBC") \
    f_(lts__cycles_active, "# of cycles where LTS was active") \
    f_(lts__d_atomic_block_cycles_serialized, "# of cycles where atomic block was busy on a multi-cycle atomic command, preventing subsequent instruction from starting") \
    f_(lts__d_atomic_block_stalled_backpressure, "# of cycles where atomic block stalled due to backpressure from response sequencer or XBAR") \
    f_(lts__d_atomic_block_stalled_pending_miss, "# of cycles where atomic block stalled due to pending cache miss") \
    f_(lts__d_atomic_block_stalled_priority, "# of cycles where atomic block stalled due to higher priority memory accesses") \
    f_(lts__d_atomic_block_stalled_same_address, "# of cycles where atomic block stalled due to previous request accessing same address") \
    f_(lts__d_atomic_reqseq_cycles_bank_conflict, "# of additional cycles the atomic request sequencer spent on bank conflict serialization") \
    f_(lts__d_atomic_reqseq_input_stalled_fifo_full, "# of cycles where atomic request sequencer's input stalled due to FIFO full") \
    f_(lts__d_atomic_reqseq_stalled_pending_store_same_address, "# of cycles where atomic request sequencer stalled due to pending partial sector store to same address") \
    f_(lts__d_atomic_reqseq_stalled_source_not_ready, "# of cycles where atomic request sequencer stalled due to source not ready") \
    f_(lts__d_atomic_resseq_cycles_bank_conflict, "# of additional cycles the atomic response sequencer spent on bank conflict serialization") \
    f_(lts__d_atomic_resseq_stalled_backpressure, "# of cycles the atomic response sequencer stalled due to backpressure from XBAR") \
    f_(lts__d_atomic_resseq_stalled_output_arbitration, "# of cycles the atomic response sequencer stalled because a conlwrrent load arbitrated for XBAR") \
    f_(lts__d_cycles_bank_conflict, "# of cycles where a data bank conflict oclwrred, causing serialization") \
    f_(lts__d_decomp_input_sectors, "# of sectors sent to decompressor") \
    f_(lts__d_decomp_input_stalled, "# of cycles where decompressor input was stalled") \
    f_(lts__d_decomp_processed_sectors, "# of sectors processed by decompressor") \
    f_(lts__ltcx_read_requests, "# of read requests sent to LTCX") \
    f_(lts__ltcx_read_sectors, "# of sectors read from LTCX") \
    f_(lts__ltcx_read_stalled_fifo_full, "# of cycles where LTCX read pipe stalled due to FIFO full") \
    f_(lts__ltcx_write_stalled_fifo_full, "# of cycles where LTCX write pipe stalled due to FIFO full") \
    f_(lts__mccif_read_cycles_32, "# of cycles where a 32B read from MCCIF was initiated") \
    f_(lts__mccif_read_cycles_64, "# of cycles where a 64B read from MCCIF was initiated") \
    f_(lts__mccif_read_request_latency_128, "# of MC read requests with latency of   1 .. 128 cycles") \
    f_(lts__mccif_read_request_latency_256, "# of MC read requests with latency of 129 .. 256 cycles") \
    f_(lts__mccif_read_request_latency_320, "# of MC read requests with latency of 257 .. 320 cycles") \
    f_(lts__mccif_read_request_latency_384, "# of MC read requests with latency of 321 .. 384 cycles") \
    f_(lts__mccif_read_request_latency_448, "# of MC read requests with latency of 385 .. 448 cycles") \
    f_(lts__mccif_read_request_latency_512, "# of MC read requests with latency of 449 .. 512 cycles") \
    f_(lts__mccif_read_request_latency_576, "# of MC read requests with latency of 513 .. 576 cycles") \
    f_(lts__mccif_read_request_latency_640, "# of MC read requests with latency of 577 .. 640 cycles") \
    f_(lts__mccif_read_request_latency_high, "# of MC read requests with latency of 641 or more cycles") \
    f_(lts__mccif_read_requests, "# of read requests sent from LTCX to MCCIF") \
    f_(lts__mccif_read_stalled, "# of cycles where LTCX read pipe stalled on MCCIF") \
    f_(lts__mccif_write_cycles_16, "# of cycles where a 16B write to MC was initiated") \
    f_(lts__mccif_write_request_latency_24, "# of MC write requests with latency of  1 .. 24 cycles") \
    f_(lts__mccif_write_request_latency_48, "# of MC write requests with latency of 25 .. 48 cycles") \
    f_(lts__mccif_write_request_latency_64, "# of MC write requests with latency of 49 .. 64 cycles") \
    f_(lts__mccif_write_request_latency_high, "# of MC write requests with latency of 65 or more cycles") \
    f_(lts__mccif_write_requests, "# of write requests sent to MCCIF") \
    f_(lts__mccif_write_sectors_excluding_16, "# of sectors written from LTCX to MC, excluding 16B writes") \
    f_(lts__mccif_write_stalled, "# of cycles where LTCX write pipe stalled on MCCIF") \
    f_(lts__r_input_fifo_active, "# of cycles where LTS request input FIFO was active") \
    f_(lts__r_input_fifo_stalled, "# of cycles where LTS request input FIFO was stalled") \
    f_(lts__t_requests, "# of requests sent to LTS") \
    f_(lts__t_requests_atomic, "# of requests for any type of atomic operation") \
    f_(lts__t_requests_cas, "# of LTS requests issued for op cas") \
    f_(lts__t_requests_cbc, "# of LTS requests issued by CBC") \
    f_(lts__t_requests_cbc_rd, "# of LTS requests issued by CBC for op rd") \
    f_(lts__t_requests_cbc_wr, "# of LTS requests issued by CBC for op wr") \
    f_(lts__t_requests_clear, "# of LTS requests issued for op clear") \
    f_(lts__t_requests_concat_cas, "# of LTS requests issued for op concat_cas") \
    f_(lts__t_requests_condrd, "# of LTS requests issued for op condrd") \
    f_(lts__t_requests_crd_i, "# of LTS requests issued by CROP read interlocked") \
    f_(lts__t_requests_crd_i_condrd, "# of LTS requests issued by CROP read interlocked for op condrd") \
    f_(lts__t_requests_crd_i_prefetch, "# of LTS requests issued by CROP read interlocked for op prefetch") \
    f_(lts__t_requests_crd_i_rd, "# of LTS requests issued by CROP read interlocked for op rd") \
    f_(lts__t_requests_crd_ni, "# of LTS requests issued by CROP read non-interlocked") \
    f_(lts__t_requests_crd_ni_condrd, "# of LTS requests issued by CROP read non-interlocked for op condrd") \
    f_(lts__t_requests_crd_ni_prefetch, "# of LTS requests issued by CROP read non-interlocked for op prefetch") \
    f_(lts__t_requests_crd_ni_rd, "# of LTS requests issued by CROP read non-interlocked for op rd") \
    f_(lts__t_requests_crop, "# of LTS requests issued by CROP") \
    f_(lts__t_requests_crop_condrd, "# of LTS requests issued by CROP for op condrd") \
    f_(lts__t_requests_crop_ilwal, "# of LTS requests issued by CROP for op ilwal") \
    f_(lts__t_requests_crop_prefetch, "# of LTS requests issued by CROP for op prefetch") \
    f_(lts__t_requests_crop_rd, "# of LTS requests issued by CROP for op rd") \
    f_(lts__t_requests_crop_wr, "# of LTS requests issued by CROP for op wr") \
    f_(lts__t_requests_gcc, "# of LTS requests issued by GCC") \
    f_(lts__t_requests_gcc_rd, "# of LTS requests issued by GCC for op rd") \
    f_(lts__t_requests_gpc, "# of LTS requests issued by GPC") \
    f_(lts__t_requests_gpc_atomic, "# of LTS requests issued by GPC for op atomic") \
    f_(lts__t_requests_gpc_cas, "# of LTS requests issued by GPC for op cas") \
    f_(lts__t_requests_gpc_clear, "# of LTS requests issued by GPC for op clear") \
    f_(lts__t_requests_gpc_concat_cas, "# of LTS requests issued by GPC for op concat_cas") \
    f_(lts__t_requests_gpc_condrd, "# of LTS requests issued by GPC for op condrd") \
    f_(lts__t_requests_gpc_ilwal, "# of LTS requests issued by GPC for op ilwal") \
    f_(lts__t_requests_gpc_membar, "# of LTS requests issued by GPC for op membar") \
    f_(lts__t_requests_gpc_prefetch, "# of LTS requests issued by GPC for op prefetch") \
    f_(lts__t_requests_gpc_rd, "# of LTS requests issued by GPC for op rd") \
    f_(lts__t_requests_gpc_wr, "# of LTS requests issued by GPC for op wr") \
    f_(lts__t_requests_host_cpu, "# of LTS requests issued by HOST CPU") \
    f_(lts__t_requests_host_cpu_atomic, "# of LTS requests issued by HOST CPU for op atomic") \
    f_(lts__t_requests_host_cpu_cas, "# of LTS requests issued by HOST CPU for op cas") \
    f_(lts__t_requests_host_cpu_clear, "# of LTS requests issued by HOST CPU for op clear") \
    f_(lts__t_requests_host_cpu_concat_cas, "# of LTS requests issued by HOST CPU for op concat_cas") \
    f_(lts__t_requests_host_cpu_condrd, "# of LTS requests issued by HOST CPU for op condrd") \
    f_(lts__t_requests_host_cpu_ilwal, "# of LTS requests issued by HOST CPU for op ilwal") \
    f_(lts__t_requests_host_cpu_membar, "# of LTS requests issued by HOST CPU for op membar") \
    f_(lts__t_requests_host_cpu_prefetch, "# of LTS requests issued by HOST CPU for op prefetch") \
    f_(lts__t_requests_host_cpu_rd, "# of LTS requests issued by HOST CPU for op rd") \
    f_(lts__t_requests_host_cpu_wr, "# of LTS requests issued by HOST CPU for op wr") \
    f_(lts__t_requests_host_noncpu, "# of LTS requests issued by HOST non-CPU") \
    f_(lts__t_requests_host_noncpu_atomic, "# of LTS requests issued by HOST non-CPU for op atomic") \
    f_(lts__t_requests_host_noncpu_cas, "# of LTS requests issued by HOST non-CPU for op cas") \
    f_(lts__t_requests_host_noncpu_clear, "# of LTS requests issued by HOST non-CPU for op clear") \
    f_(lts__t_requests_host_noncpu_concat_cas, "# of LTS requests issued by HOST non-CPU for op concat_cas") \
    f_(lts__t_requests_host_noncpu_condrd, "# of LTS requests issued by HOST non-CPU for op condrd") \
    f_(lts__t_requests_host_noncpu_ilwal, "# of LTS requests issued by HOST non-CPU for op ilwal") \
    f_(lts__t_requests_host_noncpu_membar, "# of LTS requests issued by HOST non-CPU for op membar") \
    f_(lts__t_requests_host_noncpu_prefetch, "# of LTS requests issued by HOST non-CPU for op prefetch") \
    f_(lts__t_requests_host_noncpu_rd, "# of LTS requests issued by HOST non-CPU for op rd") \
    f_(lts__t_requests_host_noncpu_wr, "# of LTS requests issued by HOST non-CPU for op wr") \
    f_(lts__t_requests_hub, "# of LTS requests issued by HUB") \
    f_(lts__t_requests_hub_atomic, "# of LTS requests issued by HUB for op atomic") \
    f_(lts__t_requests_hub_cas, "# of LTS requests issued by HUB for op cas") \
    f_(lts__t_requests_hub_clear, "# of LTS requests issued by HUB for op clear") \
    f_(lts__t_requests_hub_concat_cas, "# of LTS requests issued by HUB for op concat_cas") \
    f_(lts__t_requests_hub_condrd, "# of LTS requests issued by HUB for op condrd") \
    f_(lts__t_requests_hub_ilwal, "# of LTS requests issued by HUB for op ilwal") \
    f_(lts__t_requests_hub_membar, "# of LTS requests issued by HUB for op membar") \
    f_(lts__t_requests_hub_prefetch, "# of LTS requests issued by HUB for op prefetch") \
    f_(lts__t_requests_hub_rd, "# of LTS requests issued by HUB for op rd") \
    f_(lts__t_requests_hub_wr, "# of LTS requests issued by HUB for op wr") \
    f_(lts__t_requests_ilwal, "# of LTS requests issued for op ilwal") \
    f_(lts__t_requests_iso, "# of LTS requests issued by all isochronous clients") \
    f_(lts__t_requests_iso_rd, "# of LTS requests issued by all isochronous clients for op rd") \
    f_(lts__t_requests_iso_wr, "# of LTS requests issued by all isochronous clients for op wr") \
    f_(lts__t_requests_l1tex, "# of LTS requests issued by L1TEX") \
    f_(lts__t_requests_l1tex_atomic, "# of LTS requests issued by L1TEX for op atomic") \
    f_(lts__t_requests_l1tex_rd, "# of LTS requests issued by L1TEX for op rd") \
    f_(lts__t_requests_l1tex_wr, "# of LTS requests issued by L1TEX for op wr") \
    f_(lts__t_requests_ltc, "# of LTS requests issued by LTC subunits") \
    f_(lts__t_requests_ltc_atomic, "# of LTS requests issued by LTC subunits for op atomic") \
    f_(lts__t_requests_ltc_cas, "# of LTS requests issued by LTC subunits for op cas") \
    f_(lts__t_requests_ltc_clear, "# of LTS requests issued by LTC subunits for op clear") \
    f_(lts__t_requests_ltc_concat_cas, "# of LTS requests issued by LTC subunits for op concat_cas") \
    f_(lts__t_requests_ltc_condrd, "# of LTS requests issued by LTC subunits for op condrd") \
    f_(lts__t_requests_ltc_ilwal, "# of LTS requests issued by LTC subunits for op ilwal") \
    f_(lts__t_requests_ltc_membar, "# of LTS requests issued by LTC subunits for op membar") \
    f_(lts__t_requests_ltc_prefetch, "# of LTS requests issued by LTC subunits for op prefetch") \
    f_(lts__t_requests_ltc_rd, "# of LTS requests issued by LTC subunits for op rd") \
    f_(lts__t_requests_ltc_wr, "# of LTS requests issued by LTC subunits for op wr") \
    f_(lts__t_requests_membar, "# of LTS requests issued for op membar") \
    f_(lts__t_requests_mmu, "# of LTS requests issued by MMU") \
    f_(lts__t_requests_mmu_rd, "# of LTS requests issued by MMU for op rd") \
    f_(lts__t_requests_mmu_wr, "# of LTS requests issued by MMU for op wr") \
    f_(lts__t_requests_niso, "# of LTS requests issued by all non-isochronous clients") \
    f_(lts__t_requests_niso_rd, "# of LTS requests issued by all non-isochronous clients for op rd") \
    f_(lts__t_requests_niso_wr, "# of LTS requests issued by all non-isochronous clients for op wr") \
    f_(lts__t_requests_pe, "# of LTS requests issued by PEL+PES") \
    f_(lts__t_requests_pe_ilwal, "# of LTS requests issued by PEL+PES for op ilwal") \
    f_(lts__t_requests_pe_rd, "# of LTS requests issued by PEL+PES for op rd") \
    f_(lts__t_requests_pe_wr, "# of LTS requests issued by PEL+PES for op wr") \
    f_(lts__t_requests_prefetch, "# of LTS requests issued for op prefetch") \
    f_(lts__t_requests_raster, "# of LTS requests issued by RASTER") \
    f_(lts__t_requests_raster_rd, "# of LTS requests issued by RASTER for op rd") \
    f_(lts__t_requests_raster_wr, "# of LTS requests issued by RASTER for op wr") \
    f_(lts__t_requests_rd, "# of LTS requests issued for op rd") \
    f_(lts__t_requests_wr, "# of LTS requests issued for op wr") \
    f_(lts__t_requests_zrd_i, "# of LTS requests issued by ZROP read interlocked") \
    f_(lts__t_requests_zrd_i_condrd, "# of LTS requests issued by ZROP read interlocked for op condrd") \
    f_(lts__t_requests_zrd_i_prefetch, "# of LTS requests issued by ZROP read interlocked for op prefetch") \
    f_(lts__t_requests_zrd_i_rd, "# of LTS requests issued by ZROP read interlocked for op rd") \
    f_(lts__t_requests_zrd_ni, "# of LTS requests issued by ZROP read non-interlocked") \
    f_(lts__t_requests_zrd_ni_condrd, "# of LTS requests issued by ZROP read non-interlocked for op condrd") \
    f_(lts__t_requests_zrd_ni_prefetch, "# of LTS requests issued by ZROP read non-interlocked for op prefetch") \
    f_(lts__t_requests_zrd_ni_rd, "# of LTS requests issued by ZROP read non-interlocked for op rd") \
    f_(lts__t_requests_zrop, "# of LTS requests issued by ZROP") \
    f_(lts__t_requests_zrop_condrd, "# of LTS requests issued by ZROP for op condrd") \
    f_(lts__t_requests_zrop_ilwal, "# of LTS requests issued by ZROP for op ilwal") \
    f_(lts__t_requests_zrop_prefetch, "# of LTS requests issued by ZROP for op prefetch") \
    f_(lts__t_requests_zrop_rd, "# of LTS requests issued by ZROP for op rd") \
    f_(lts__t_requests_zrop_wr, "# of LTS requests issued by ZROP for op wr") \
    f_(lts__t_sectors, "# of sectors accessed within cachelines") \
    f_(lts__t_sectors_atomic, "# of sectors accessed for op atomic") \
    f_(lts__t_sectors_cas, "# of sectors accessed for op cas") \
    f_(lts__t_sectors_cbc, "# of sectors accessed by CBC") \
    f_(lts__t_sectors_cbc_rd, "# of sectors accessed by CBC for op rd") \
    f_(lts__t_sectors_cbc_wr, "# of sectors accessed by CBC for op wr") \
    f_(lts__t_sectors_clear, "# of sectors accessed for op clear") \
    f_(lts__t_sectors_concat_cas, "# of sectors accessed for op concat_cas") \
    f_(lts__t_sectors_condrd, "# of sectors accessed for op condrd") \
    f_(lts__t_sectors_crd_i, "# of sectors accessed by CROP read interlocked") \
    f_(lts__t_sectors_crd_i_condrd, "# of sectors accessed by CROP read interlocked for op condrd") \
    f_(lts__t_sectors_crd_i_prefetch, "# of sectors accessed by CROP read interlocked for op prefetch") \
    f_(lts__t_sectors_crd_i_rd, "# of sectors accessed by CROP read interlocked for op rd") \
    f_(lts__t_sectors_crd_ni, "# of sectors accessed by CROP read non-interlocked") \
    f_(lts__t_sectors_crd_ni_condrd, "# of sectors accessed by CROP read non-interlocked for op condrd") \
    f_(lts__t_sectors_crd_ni_prefetch, "# of sectors accessed by CROP read non-interlocked for op prefetch") \
    f_(lts__t_sectors_crd_ni_rd, "# of sectors accessed by CROP read non-interlocked for op rd") \
    f_(lts__t_sectors_crop, "# of sectors accessed by CROP") \
    f_(lts__t_sectors_crop_condrd, "# of sectors accessed by CROP for op condrd") \
    f_(lts__t_sectors_crop_ilwal, "# of sectors accessed by CROP for op ilwal") \
    f_(lts__t_sectors_crop_prefetch, "# of sectors accessed by CROP for op prefetch") \
    f_(lts__t_sectors_crop_rd, "# of sectors accessed by CROP for op rd") \
    f_(lts__t_sectors_crop_wr, "# of sectors accessed by CROP for op wr") \
    f_(lts__t_sectors_fill, "# of sector fills from memory") \
    f_(lts__t_sectors_gcc, "# of sectors accessed by GCC") \
    f_(lts__t_sectors_gcc_rd, "# of sectors accessed by GCC for op rd") \
    f_(lts__t_sectors_gpc, "# of sectors accessed by GPC") \
    f_(lts__t_sectors_gpc_atomic, "# of sectors accessed by GPC for op atomic") \
    f_(lts__t_sectors_gpc_cas, "# of sectors accessed by GPC for op cas") \
    f_(lts__t_sectors_gpc_clear, "# of sectors accessed by GPC for op clear") \
    f_(lts__t_sectors_gpc_concat_cas, "# of sectors accessed by GPC for op concat_cas") \
    f_(lts__t_sectors_gpc_condrd, "# of sectors accessed by GPC for op condrd") \
    f_(lts__t_sectors_gpc_ilwal, "# of sectors accessed by GPC for op ilwal") \
    f_(lts__t_sectors_gpc_membar, "# of sectors accessed by GPC for op membar") \
    f_(lts__t_sectors_gpc_prefetch, "# of sectors accessed by GPC for op prefetch") \
    f_(lts__t_sectors_gpc_rd, "# of sectors accessed by GPC for op rd") \
    f_(lts__t_sectors_gpc_wr, "# of sectors accessed by GPC for op wr") \
    f_(lts__t_sectors_hit, "# of sector hits within cachelines") \
    f_(lts__t_sectors_hit_atomic, "# of sectors hit for op atomic") \
    f_(lts__t_sectors_hit_cas, "# of sectors hit for op cas") \
    f_(lts__t_sectors_hit_cbc, "# of sectors hit by CBC") \
    f_(lts__t_sectors_hit_cbc_rd, "# of sectors hit by CBC for op rd") \
    f_(lts__t_sectors_hit_cbc_wr, "# of sectors hit by CBC for op wr") \
    f_(lts__t_sectors_hit_clear, "# of sectors hit for op clear") \
    f_(lts__t_sectors_hit_concat_cas, "# of sectors hit for op concat_cas") \
    f_(lts__t_sectors_hit_condrd, "# of sectors hit for op condrd") \
    f_(lts__t_sectors_hit_crd_i, "# of sectors hit by CROP read interlocked") \
    f_(lts__t_sectors_hit_crd_i_condrd, "# of sectors hit by CROP read interlocked for op condrd") \
    f_(lts__t_sectors_hit_crd_i_prefetch, "# of sectors hit by CROP read interlocked for op prefetch") \
    f_(lts__t_sectors_hit_crd_i_rd, "# of sectors hit by CROP read interlocked for op rd") \
    f_(lts__t_sectors_hit_crd_ni, "# of sectors hit by CROP read non-interlocked") \
    f_(lts__t_sectors_hit_crd_ni_condrd, "# of sectors hit by CROP read non-interlocked for op condrd") \
    f_(lts__t_sectors_hit_crd_ni_prefetch, "# of sectors hit by CROP read non-interlocked for op prefetch") \
    f_(lts__t_sectors_hit_crd_ni_rd, "# of sectors hit by CROP read non-interlocked for op rd") \
    f_(lts__t_sectors_hit_crop, "# of sectors hit by CROP") \
    f_(lts__t_sectors_hit_crop_condrd, "# of sectors hit by CROP for op condrd") \
    f_(lts__t_sectors_hit_crop_ilwal, "# of sectors hit by CROP for op ilwal") \
    f_(lts__t_sectors_hit_crop_prefetch, "# of sectors hit by CROP for op prefetch") \
    f_(lts__t_sectors_hit_crop_rd, "# of sectors hit by CROP for op rd") \
    f_(lts__t_sectors_hit_crop_wr, "# of sectors hit by CROP for op wr") \
    f_(lts__t_sectors_hit_gcc, "# of sectors hit by GCC") \
    f_(lts__t_sectors_hit_gcc_rd, "# of sectors hit by GCC for op rd") \
    f_(lts__t_sectors_hit_gpc, "# of sectors hit by GPC") \
    f_(lts__t_sectors_hit_gpc_atomic, "# of sectors hit by GPC for op atomic") \
    f_(lts__t_sectors_hit_gpc_cas, "# of sectors hit by GPC for op cas") \
    f_(lts__t_sectors_hit_gpc_clear, "# of sectors hit by GPC for op clear") \
    f_(lts__t_sectors_hit_gpc_concat_cas, "# of sectors hit by GPC for op concat_cas") \
    f_(lts__t_sectors_hit_gpc_condrd, "# of sectors hit by GPC for op condrd") \
    f_(lts__t_sectors_hit_gpc_ilwal, "# of sectors hit by GPC for op ilwal") \
    f_(lts__t_sectors_hit_gpc_membar, "# of sectors hit by GPC for op membar") \
    f_(lts__t_sectors_hit_gpc_prefetch, "# of sectors hit by GPC for op prefetch") \
    f_(lts__t_sectors_hit_gpc_rd, "# of sectors hit by GPC for op rd") \
    f_(lts__t_sectors_hit_gpc_wr, "# of sectors hit by GPC for op wr") \
    f_(lts__t_sectors_hit_host_cpu, "# of sectors hit by HOST CPU") \
    f_(lts__t_sectors_hit_host_cpu_atomic, "# of sectors hit by HOST CPU for op atomic") \
    f_(lts__t_sectors_hit_host_cpu_cas, "# of sectors hit by HOST CPU for op cas") \
    f_(lts__t_sectors_hit_host_cpu_clear, "# of sectors hit by HOST CPU for op clear") \
    f_(lts__t_sectors_hit_host_cpu_concat_cas, "# of sectors hit by HOST CPU for op concat_cas") \
    f_(lts__t_sectors_hit_host_cpu_condrd, "# of sectors hit by HOST CPU for op condrd") \
    f_(lts__t_sectors_hit_host_cpu_ilwal, "# of sectors hit by HOST CPU for op ilwal") \
    f_(lts__t_sectors_hit_host_cpu_membar, "# of sectors hit by HOST CPU for op membar") \
    f_(lts__t_sectors_hit_host_cpu_prefetch, "# of sectors hit by HOST CPU for op prefetch") \
    f_(lts__t_sectors_hit_host_cpu_rd, "# of sectors hit by HOST CPU for op rd") \
    f_(lts__t_sectors_hit_host_cpu_wr, "# of sectors hit by HOST CPU for op wr") \
    f_(lts__t_sectors_hit_host_noncpu, "# of sectors hit by HOST non-CPU") \
    f_(lts__t_sectors_hit_host_noncpu_atomic, "# of sectors hit by HOST non-CPU for op atomic") \
    f_(lts__t_sectors_hit_host_noncpu_cas, "# of sectors hit by HOST non-CPU for op cas") \
    f_(lts__t_sectors_hit_host_noncpu_clear, "# of sectors hit by HOST non-CPU for op clear") \
    f_(lts__t_sectors_hit_host_noncpu_concat_cas, "# of sectors hit by HOST non-CPU for op concat_cas") \
    f_(lts__t_sectors_hit_host_noncpu_condrd, "# of sectors hit by HOST non-CPU for op condrd") \
    f_(lts__t_sectors_hit_host_noncpu_ilwal, "# of sectors hit by HOST non-CPU for op ilwal") \
    f_(lts__t_sectors_hit_host_noncpu_membar, "# of sectors hit by HOST non-CPU for op membar") \
    f_(lts__t_sectors_hit_host_noncpu_prefetch, "# of sectors hit by HOST non-CPU for op prefetch") \
    f_(lts__t_sectors_hit_host_noncpu_rd, "# of sectors hit by HOST non-CPU for op rd") \
    f_(lts__t_sectors_hit_host_noncpu_wr, "# of sectors hit by HOST non-CPU for op wr") \
    f_(lts__t_sectors_hit_hub, "# of sectors hit by HUB") \
    f_(lts__t_sectors_hit_hub_atomic, "# of sectors hit by HUB for op atomic") \
    f_(lts__t_sectors_hit_hub_cas, "# of sectors hit by HUB for op cas") \
    f_(lts__t_sectors_hit_hub_clear, "# of sectors hit by HUB for op clear") \
    f_(lts__t_sectors_hit_hub_concat_cas, "# of sectors hit by HUB for op concat_cas") \
    f_(lts__t_sectors_hit_hub_condrd, "# of sectors hit by HUB for op condrd") \
    f_(lts__t_sectors_hit_hub_ilwal, "# of sectors hit by HUB for op ilwal") \
    f_(lts__t_sectors_hit_hub_membar, "# of sectors hit by HUB for op membar") \
    f_(lts__t_sectors_hit_hub_prefetch, "# of sectors hit by HUB for op prefetch") \
    f_(lts__t_sectors_hit_hub_rd, "# of sectors hit by HUB for op rd") \
    f_(lts__t_sectors_hit_hub_wr, "# of sectors hit by HUB for op wr") \
    f_(lts__t_sectors_hit_ilwal, "# of sectors hit for op ilwal") \
    f_(lts__t_sectors_hit_iso, "# of sectors hit by all isochronous clients") \
    f_(lts__t_sectors_hit_iso_rd, "# of sectors hit by all isochronous clients for op rd") \
    f_(lts__t_sectors_hit_iso_wr, "# of sectors hit by all isochronous clients for op wr") \
    f_(lts__t_sectors_hit_l1tex, "# of sectors hit by L1TEX") \
    f_(lts__t_sectors_hit_l1tex_atomic, "# of sectors hit by L1TEX for op atomic") \
    f_(lts__t_sectors_hit_l1tex_rd, "# of sectors hit by L1TEX for op rd") \
    f_(lts__t_sectors_hit_l1tex_wr, "# of sectors hit by L1TEX for op wr") \
    f_(lts__t_sectors_hit_ltc, "# of sectors hit by LTC subunits") \
    f_(lts__t_sectors_hit_ltc_atomic, "# of sectors hit by LTC subunits for op atomic") \
    f_(lts__t_sectors_hit_ltc_cas, "# of sectors hit by LTC subunits for op cas") \
    f_(lts__t_sectors_hit_ltc_clear, "# of sectors hit by LTC subunits for op clear") \
    f_(lts__t_sectors_hit_ltc_concat_cas, "# of sectors hit by LTC subunits for op concat_cas") \
    f_(lts__t_sectors_hit_ltc_condrd, "# of sectors hit by LTC subunits for op condrd") \
    f_(lts__t_sectors_hit_ltc_ilwal, "# of sectors hit by LTC subunits for op ilwal") \
    f_(lts__t_sectors_hit_ltc_membar, "# of sectors hit by LTC subunits for op membar") \
    f_(lts__t_sectors_hit_ltc_prefetch, "# of sectors hit by LTC subunits for op prefetch") \
    f_(lts__t_sectors_hit_ltc_rd, "# of sectors hit by LTC subunits for op rd") \
    f_(lts__t_sectors_hit_ltc_wr, "# of sectors hit by LTC subunits for op wr") \
    f_(lts__t_sectors_hit_membar, "# of sectors hit for op membar") \
    f_(lts__t_sectors_hit_mmu, "# of sectors hit by MMU") \
    f_(lts__t_sectors_hit_mmu_rd, "# of sectors hit by MMU for op rd") \
    f_(lts__t_sectors_hit_mmu_wr, "# of sectors hit by MMU for op wr") \
    f_(lts__t_sectors_hit_niso, "# of sectors hit by all non-isochronous clients") \
    f_(lts__t_sectors_hit_niso_rd, "# of sectors hit by all non-isochronous clients for op rd") \
    f_(lts__t_sectors_hit_niso_wr, "# of sectors hit by all non-isochronous clients for op wr") \
    f_(lts__t_sectors_hit_pe, "# of sectors hit by PEL+PES") \
    f_(lts__t_sectors_hit_pe_ilwal, "# of sectors hit by PEL+PES for op ilwal") \
    f_(lts__t_sectors_hit_pe_rd, "# of sectors hit by PEL+PES for op rd") \
    f_(lts__t_sectors_hit_pe_wr, "# of sectors hit by PEL+PES for op wr") \
    f_(lts__t_sectors_hit_prefetch, "# of sectors hit for op prefetch") \
    f_(lts__t_sectors_hit_raster, "# of sectors hit by RASTER") \
    f_(lts__t_sectors_hit_raster_rd, "# of sectors hit by RASTER for op rd") \
    f_(lts__t_sectors_hit_raster_wr, "# of sectors hit by RASTER for op wr") \
    f_(lts__t_sectors_hit_rd, "# of sectors hit for op rd") \
    f_(lts__t_sectors_hit_valid, "# of sector hits where data was valid") \
    f_(lts__t_sectors_hit_wr, "# of sectors hit for op wr") \
    f_(lts__t_sectors_hit_zrd_i, "# of sectors hit by ZROP read interlocked") \
    f_(lts__t_sectors_hit_zrd_i_condrd, "# of sectors hit by ZROP read interlocked for op condrd") \
    f_(lts__t_sectors_hit_zrd_i_prefetch, "# of sectors hit by ZROP read interlocked for op prefetch") \
    f_(lts__t_sectors_hit_zrd_i_rd, "# of sectors hit by ZROP read interlocked for op rd") \
    f_(lts__t_sectors_hit_zrd_ni, "# of sectors hit by ZROP read non-interlocked") \
    f_(lts__t_sectors_hit_zrd_ni_condrd, "# of sectors hit by ZROP read non-interlocked for op condrd") \
    f_(lts__t_sectors_hit_zrd_ni_prefetch, "# of sectors hit by ZROP read non-interlocked for op prefetch") \
    f_(lts__t_sectors_hit_zrd_ni_rd, "# of sectors hit by ZROP read non-interlocked for op rd") \
    f_(lts__t_sectors_hit_zrop, "# of sectors hit by ZROP") \
    f_(lts__t_sectors_hit_zrop_condrd, "# of sectors hit by ZROP for op condrd") \
    f_(lts__t_sectors_hit_zrop_ilwal, "# of sectors hit by ZROP for op ilwal") \
    f_(lts__t_sectors_hit_zrop_prefetch, "# of sectors hit by ZROP for op prefetch") \
    f_(lts__t_sectors_hit_zrop_rd, "# of sectors hit by ZROP for op rd") \
    f_(lts__t_sectors_hit_zrop_wr, "# of sectors hit by ZROP for op wr") \
    f_(lts__t_sectors_host_cpu, "# of sectors accessed by HOST CPU") \
    f_(lts__t_sectors_host_cpu_atomic, "# of sectors accessed by HOST CPU for op atomic") \
    f_(lts__t_sectors_host_cpu_cas, "# of sectors accessed by HOST CPU for op cas") \
    f_(lts__t_sectors_host_cpu_clear, "# of sectors accessed by HOST CPU for op clear") \
    f_(lts__t_sectors_host_cpu_concat_cas, "# of sectors accessed by HOST CPU for op concat_cas") \
    f_(lts__t_sectors_host_cpu_condrd, "# of sectors accessed by HOST CPU for op condrd") \
    f_(lts__t_sectors_host_cpu_ilwal, "# of sectors accessed by HOST CPU for op ilwal") \
    f_(lts__t_sectors_host_cpu_membar, "# of sectors accessed by HOST CPU for op membar") \
    f_(lts__t_sectors_host_cpu_prefetch, "# of sectors accessed by HOST CPU for op prefetch") \
    f_(lts__t_sectors_host_cpu_rd, "# of sectors accessed by HOST CPU for op rd") \
    f_(lts__t_sectors_host_cpu_wr, "# of sectors accessed by HOST CPU for op wr") \
    f_(lts__t_sectors_host_noncpu, "# of sectors accessed by HOST non-CPU") \
    f_(lts__t_sectors_host_noncpu_atomic, "# of sectors accessed by HOST non-CPU for op atomic") \
    f_(lts__t_sectors_host_noncpu_cas, "# of sectors accessed by HOST non-CPU for op cas") \
    f_(lts__t_sectors_host_noncpu_clear, "# of sectors accessed by HOST non-CPU for op clear") \
    f_(lts__t_sectors_host_noncpu_concat_cas, "# of sectors accessed by HOST non-CPU for op concat_cas") \
    f_(lts__t_sectors_host_noncpu_condrd, "# of sectors accessed by HOST non-CPU for op condrd") \
    f_(lts__t_sectors_host_noncpu_ilwal, "# of sectors accessed by HOST non-CPU for op ilwal") \
    f_(lts__t_sectors_host_noncpu_membar, "# of sectors accessed by HOST non-CPU for op membar") \
    f_(lts__t_sectors_host_noncpu_prefetch, "# of sectors accessed by HOST non-CPU for op prefetch") \
    f_(lts__t_sectors_host_noncpu_rd, "# of sectors accessed by HOST non-CPU for op rd") \
    f_(lts__t_sectors_host_noncpu_wr, "# of sectors accessed by HOST non-CPU for op wr") \
    f_(lts__t_sectors_hub, "# of sectors accessed by HUB") \
    f_(lts__t_sectors_hub_atomic, "# of sectors accessed by HUB for op atomic") \
    f_(lts__t_sectors_hub_cas, "# of sectors accessed by HUB for op cas") \
    f_(lts__t_sectors_hub_clear, "# of sectors accessed by HUB for op clear") \
    f_(lts__t_sectors_hub_concat_cas, "# of sectors accessed by HUB for op concat_cas") \
    f_(lts__t_sectors_hub_condrd, "# of sectors accessed by HUB for op condrd") \
    f_(lts__t_sectors_hub_ilwal, "# of sectors accessed by HUB for op ilwal") \
    f_(lts__t_sectors_hub_membar, "# of sectors accessed by HUB for op membar") \
    f_(lts__t_sectors_hub_prefetch, "# of sectors accessed by HUB for op prefetch") \
    f_(lts__t_sectors_hub_rd, "# of sectors accessed by HUB for op rd") \
    f_(lts__t_sectors_hub_wr, "# of sectors accessed by HUB for op wr") \
    f_(lts__t_sectors_ilwal, "# of sectors accessed for op ilwal") \
    f_(lts__t_sectors_iso, "# of sectors accessed by all isochronous clients") \
    f_(lts__t_sectors_iso_rd, "# of sectors accessed by all isochronous clients for op rd") \
    f_(lts__t_sectors_iso_wr, "# of sectors accessed by all isochronous clients for op wr") \
    f_(lts__t_sectors_l1tex, "# of sectors accessed by L1TEX") \
    f_(lts__t_sectors_l1tex_atomic, "# of sectors accessed by L1TEX for op atomic") \
    f_(lts__t_sectors_l1tex_rd, "# of sectors accessed by L1TEX for op rd") \
    f_(lts__t_sectors_l1tex_wr, "# of sectors accessed by L1TEX for op wr") \
    f_(lts__t_sectors_ltc, "# of sectors accessed by LTC subunits") \
    f_(lts__t_sectors_ltc_atomic, "# of sectors accessed by LTC subunits for op atomic") \
    f_(lts__t_sectors_ltc_cas, "# of sectors accessed by LTC subunits for op cas") \
    f_(lts__t_sectors_ltc_clear, "# of sectors accessed by LTC subunits for op clear") \
    f_(lts__t_sectors_ltc_concat_cas, "# of sectors accessed by LTC subunits for op concat_cas") \
    f_(lts__t_sectors_ltc_condrd, "# of sectors accessed by LTC subunits for op condrd") \
    f_(lts__t_sectors_ltc_ilwal, "# of sectors accessed by LTC subunits for op ilwal") \
    f_(lts__t_sectors_ltc_membar, "# of sectors accessed by LTC subunits for op membar") \
    f_(lts__t_sectors_ltc_prefetch, "# of sectors accessed by LTC subunits for op prefetch") \
    f_(lts__t_sectors_ltc_rd, "# of sectors accessed by LTC subunits for op rd") \
    f_(lts__t_sectors_ltc_wr, "# of sectors accessed by LTC subunits for op wr") \
    f_(lts__t_sectors_membar, "# of sectors accessed for op membar") \
    f_(lts__t_sectors_miss, "# of sector misses within cachelines") \
    f_(lts__t_sectors_miss_atomic, "# of sectors missed for op atomic") \
    f_(lts__t_sectors_miss_cas, "# of sectors missed for op cas") \
    f_(lts__t_sectors_miss_cbc, "# of sectors missed by CBC") \
    f_(lts__t_sectors_miss_cbc_rd, "# of sectors missed by CBC for op rd") \
    f_(lts__t_sectors_miss_cbc_wr, "# of sectors missed by CBC for op wr") \
    f_(lts__t_sectors_miss_clear, "# of sectors missed for op clear") \
    f_(lts__t_sectors_miss_concat_cas, "# of sectors missed for op concat_cas") \
    f_(lts__t_sectors_miss_condrd, "# of sectors missed for op condrd") \
    f_(lts__t_sectors_miss_crd_i, "# of sectors missed by CROP read interlocked") \
    f_(lts__t_sectors_miss_crd_i_condrd, "# of sectors missed by CROP read interlocked for op condrd") \
    f_(lts__t_sectors_miss_crd_i_prefetch, "# of sectors missed by CROP read interlocked for op prefetch") \
    f_(lts__t_sectors_miss_crd_i_rd, "# of sectors missed by CROP read interlocked for op rd") \
    f_(lts__t_sectors_miss_crd_ni, "# of sectors missed by CROP read non-interlocked") \
    f_(lts__t_sectors_miss_crd_ni_condrd, "# of sectors missed by CROP read non-interlocked for op condrd") \
    f_(lts__t_sectors_miss_crd_ni_prefetch, "# of sectors missed by CROP read non-interlocked for op prefetch") \
    f_(lts__t_sectors_miss_crd_ni_rd, "# of sectors missed by CROP read non-interlocked for op rd") \
    f_(lts__t_sectors_miss_crop, "# of sectors missed by CROP") \
    f_(lts__t_sectors_miss_crop_condrd, "# of sectors missed by CROP for op condrd") \
    f_(lts__t_sectors_miss_crop_ilwal, "# of sectors missed by CROP for op ilwal") \
    f_(lts__t_sectors_miss_crop_prefetch, "# of sectors missed by CROP for op prefetch") \
    f_(lts__t_sectors_miss_crop_rd, "# of sectors missed by CROP for op rd") \
    f_(lts__t_sectors_miss_crop_wr, "# of sectors missed by CROP for op wr") \
    f_(lts__t_sectors_miss_gcc, "# of sectors missed by GCC") \
    f_(lts__t_sectors_miss_gcc_rd, "# of sectors missed by GCC for op rd") \
    f_(lts__t_sectors_miss_gpc, "# of sectors missed by GPC") \
    f_(lts__t_sectors_miss_gpc_atomic, "# of sectors missed by GPC for op atomic") \
    f_(lts__t_sectors_miss_gpc_cas, "# of sectors missed by GPC for op cas") \
    f_(lts__t_sectors_miss_gpc_clear, "# of sectors missed by GPC for op clear") \
    f_(lts__t_sectors_miss_gpc_concat_cas, "# of sectors missed by GPC for op concat_cas") \
    f_(lts__t_sectors_miss_gpc_condrd, "# of sectors missed by GPC for op condrd") \
    f_(lts__t_sectors_miss_gpc_ilwal, "# of sectors missed by GPC for op ilwal") \
    f_(lts__t_sectors_miss_gpc_membar, "# of sectors missed by GPC for op membar") \
    f_(lts__t_sectors_miss_gpc_prefetch, "# of sectors missed by GPC for op prefetch") \
    f_(lts__t_sectors_miss_gpc_rd, "# of sectors missed by GPC for op rd") \
    f_(lts__t_sectors_miss_gpc_wr, "# of sectors missed by GPC for op wr") \
    f_(lts__t_sectors_miss_host_cpu, "# of sectors missed by HOST CPU") \
    f_(lts__t_sectors_miss_host_cpu_atomic, "# of sectors missed by HOST CPU for op atomic") \
    f_(lts__t_sectors_miss_host_cpu_cas, "# of sectors missed by HOST CPU for op cas") \
    f_(lts__t_sectors_miss_host_cpu_clear, "# of sectors missed by HOST CPU for op clear") \
    f_(lts__t_sectors_miss_host_cpu_concat_cas, "# of sectors missed by HOST CPU for op concat_cas") \
    f_(lts__t_sectors_miss_host_cpu_condrd, "# of sectors missed by HOST CPU for op condrd") \
    f_(lts__t_sectors_miss_host_cpu_ilwal, "# of sectors missed by HOST CPU for op ilwal") \
    f_(lts__t_sectors_miss_host_cpu_membar, "# of sectors missed by HOST CPU for op membar") \
    f_(lts__t_sectors_miss_host_cpu_prefetch, "# of sectors missed by HOST CPU for op prefetch") \
    f_(lts__t_sectors_miss_host_cpu_rd, "# of sectors missed by HOST CPU for op rd") \
    f_(lts__t_sectors_miss_host_cpu_wr, "# of sectors missed by HOST CPU for op wr") \
    f_(lts__t_sectors_miss_host_noncpu, "# of sectors missed by HOST non-CPU") \
    f_(lts__t_sectors_miss_host_noncpu_atomic, "# of sectors missed by HOST non-CPU for op atomic") \
    f_(lts__t_sectors_miss_host_noncpu_cas, "# of sectors missed by HOST non-CPU for op cas") \
    f_(lts__t_sectors_miss_host_noncpu_clear, "# of sectors missed by HOST non-CPU for op clear") \
    f_(lts__t_sectors_miss_host_noncpu_concat_cas, "# of sectors missed by HOST non-CPU for op concat_cas") \
    f_(lts__t_sectors_miss_host_noncpu_condrd, "# of sectors missed by HOST non-CPU for op condrd") \
    f_(lts__t_sectors_miss_host_noncpu_ilwal, "# of sectors missed by HOST non-CPU for op ilwal") \
    f_(lts__t_sectors_miss_host_noncpu_membar, "# of sectors missed by HOST non-CPU for op membar") \
    f_(lts__t_sectors_miss_host_noncpu_prefetch, "# of sectors missed by HOST non-CPU for op prefetch") \
    f_(lts__t_sectors_miss_host_noncpu_rd, "# of sectors missed by HOST non-CPU for op rd") \
    f_(lts__t_sectors_miss_host_noncpu_wr, "# of sectors missed by HOST non-CPU for op wr") \
    f_(lts__t_sectors_miss_hub, "# of sectors missed by HUB") \
    f_(lts__t_sectors_miss_hub_atomic, "# of sectors missed by HUB for op atomic") \
    f_(lts__t_sectors_miss_hub_cas, "# of sectors missed by HUB for op cas") \
    f_(lts__t_sectors_miss_hub_clear, "# of sectors missed by HUB for op clear") \
    f_(lts__t_sectors_miss_hub_concat_cas, "# of sectors missed by HUB for op concat_cas") \
    f_(lts__t_sectors_miss_hub_condrd, "# of sectors missed by HUB for op condrd") \
    f_(lts__t_sectors_miss_hub_ilwal, "# of sectors missed by HUB for op ilwal") \
    f_(lts__t_sectors_miss_hub_membar, "# of sectors missed by HUB for op membar") \
    f_(lts__t_sectors_miss_hub_prefetch, "# of sectors missed by HUB for op prefetch") \
    f_(lts__t_sectors_miss_hub_rd, "# of sectors missed by HUB for op rd") \
    f_(lts__t_sectors_miss_hub_wr, "# of sectors missed by HUB for op wr") \
    f_(lts__t_sectors_miss_ilwal, "# of sectors missed for op ilwal") \
    f_(lts__t_sectors_miss_iso, "# of sectors missed by all isochronous clients") \
    f_(lts__t_sectors_miss_iso_rd, "# of sectors missed by all isochronous clients for op rd") \
    f_(lts__t_sectors_miss_iso_wr, "# of sectors missed by all isochronous clients for op wr") \
    f_(lts__t_sectors_miss_l1tex, "# of sectors missed by L1TEX") \
    f_(lts__t_sectors_miss_l1tex_atomic, "# of sectors missed by L1TEX for op atomic") \
    f_(lts__t_sectors_miss_l1tex_rd, "# of sectors missed by L1TEX for op rd") \
    f_(lts__t_sectors_miss_l1tex_wr, "# of sectors missed by L1TEX for op wr") \
    f_(lts__t_sectors_miss_ltc, "# of sectors missed by LTC subunits") \
    f_(lts__t_sectors_miss_ltc_atomic, "# of sectors missed by LTC subunits for op atomic") \
    f_(lts__t_sectors_miss_ltc_cas, "# of sectors missed by LTC subunits for op cas") \
    f_(lts__t_sectors_miss_ltc_clear, "# of sectors missed by LTC subunits for op clear") \
    f_(lts__t_sectors_miss_ltc_concat_cas, "# of sectors missed by LTC subunits for op concat_cas") \
    f_(lts__t_sectors_miss_ltc_condrd, "# of sectors missed by LTC subunits for op condrd") \
    f_(lts__t_sectors_miss_ltc_ilwal, "# of sectors missed by LTC subunits for op ilwal") \
    f_(lts__t_sectors_miss_ltc_membar, "# of sectors missed by LTC subunits for op membar") \
    f_(lts__t_sectors_miss_ltc_prefetch, "# of sectors missed by LTC subunits for op prefetch") \
    f_(lts__t_sectors_miss_ltc_rd, "# of sectors missed by LTC subunits for op rd") \
    f_(lts__t_sectors_miss_ltc_wr, "# of sectors missed by LTC subunits for op wr") \
    f_(lts__t_sectors_miss_membar, "# of sectors missed for op membar") \
    f_(lts__t_sectors_miss_mmu, "# of sectors missed by MMU") \
    f_(lts__t_sectors_miss_mmu_rd, "# of sectors missed by MMU for op rd") \
    f_(lts__t_sectors_miss_mmu_wr, "# of sectors missed by MMU for op wr") \
    f_(lts__t_sectors_miss_niso, "# of sectors missed by all non-isochronous clients") \
    f_(lts__t_sectors_miss_niso_rd, "# of sectors missed by all non-isochronous clients for op rd") \
    f_(lts__t_sectors_miss_niso_wr, "# of sectors missed by all non-isochronous clients for op wr") \
    f_(lts__t_sectors_miss_pe, "# of sectors missed by PEL+PES") \
    f_(lts__t_sectors_miss_pe_ilwal, "# of sectors missed by PEL+PES for op ilwal") \
    f_(lts__t_sectors_miss_pe_rd, "# of sectors missed by PEL+PES for op rd") \
    f_(lts__t_sectors_miss_pe_wr, "# of sectors missed by PEL+PES for op wr") \
    f_(lts__t_sectors_miss_prefetch, "# of sectors missed for op prefetch") \
    f_(lts__t_sectors_miss_raster, "# of sectors missed by RASTER") \
    f_(lts__t_sectors_miss_raster_rd, "# of sectors missed by RASTER for op rd") \
    f_(lts__t_sectors_miss_raster_wr, "# of sectors missed by RASTER for op wr") \
    f_(lts__t_sectors_miss_rd, "# of sectors missed for op rd") \
    f_(lts__t_sectors_miss_wr, "# of sectors missed for op wr") \
    f_(lts__t_sectors_miss_zrd_i, "# of sectors missed by ZROP read interlocked") \
    f_(lts__t_sectors_miss_zrd_i_condrd, "# of sectors missed by ZROP read interlocked for op condrd") \
    f_(lts__t_sectors_miss_zrd_i_prefetch, "# of sectors missed by ZROP read interlocked for op prefetch") \
    f_(lts__t_sectors_miss_zrd_i_rd, "# of sectors missed by ZROP read interlocked for op rd") \
    f_(lts__t_sectors_miss_zrd_ni, "# of sectors missed by ZROP read non-interlocked") \
    f_(lts__t_sectors_miss_zrd_ni_condrd, "# of sectors missed by ZROP read non-interlocked for op condrd") \
    f_(lts__t_sectors_miss_zrd_ni_prefetch, "# of sectors missed by ZROP read non-interlocked for op prefetch") \
    f_(lts__t_sectors_miss_zrd_ni_rd, "# of sectors missed by ZROP read non-interlocked for op rd") \
    f_(lts__t_sectors_miss_zrop, "# of sectors missed by ZROP") \
    f_(lts__t_sectors_miss_zrop_condrd, "# of sectors missed by ZROP for op condrd") \
    f_(lts__t_sectors_miss_zrop_ilwal, "# of sectors missed by ZROP for op ilwal") \
    f_(lts__t_sectors_miss_zrop_prefetch, "# of sectors missed by ZROP for op prefetch") \
    f_(lts__t_sectors_miss_zrop_rd, "# of sectors missed by ZROP for op rd") \
    f_(lts__t_sectors_miss_zrop_wr, "# of sectors missed by ZROP for op wr") \
    f_(lts__t_sectors_mmu, "# of sectors accessed by MMU") \
    f_(lts__t_sectors_mmu_rd, "# of sectors accessed by MMU for op rd") \
    f_(lts__t_sectors_mmu_wr, "# of sectors accessed by MMU for op wr") \
    f_(lts__t_sectors_niso, "# of sectors accessed by all non-isochronous clients") \
    f_(lts__t_sectors_niso_rd, "# of sectors accessed by all non-isochronous clients for op rd") \
    f_(lts__t_sectors_niso_wr, "# of sectors accessed by all non-isochronous clients for op wr") \
    f_(lts__t_sectors_pe, "# of sectors accessed by PEL+PES") \
    f_(lts__t_sectors_pe_ilwal, "# of sectors accessed by PEL+PES for op ilwal") \
    f_(lts__t_sectors_pe_rd, "# of sectors accessed by PEL+PES for op rd") \
    f_(lts__t_sectors_pe_wr, "# of sectors accessed by PEL+PES for op wr") \
    f_(lts__t_sectors_prefetch, "# of sectors accessed for op prefetch") \
    f_(lts__t_sectors_raster, "# of sectors accessed by RASTER") \
    f_(lts__t_sectors_raster_rd, "# of sectors accessed by RASTER for op rd") \
    f_(lts__t_sectors_raster_wr, "# of sectors accessed by RASTER for op wr") \
    f_(lts__t_sectors_rd, "# of sectors accessed for op rd") \
    f_(lts__t_sectors_wr, "# of sectors accessed for op wr") \
    f_(lts__t_sectors_zrd_i, "# of sectors accessed by ZROP read interlocked") \
    f_(lts__t_sectors_zrd_i_condrd, "# of sectors accessed by ZROP read interlocked for op condrd") \
    f_(lts__t_sectors_zrd_i_prefetch, "# of sectors accessed by ZROP read interlocked for op prefetch") \
    f_(lts__t_sectors_zrd_i_rd, "# of sectors accessed by ZROP read interlocked for op rd") \
    f_(lts__t_sectors_zrd_ni, "# of sectors accessed by ZROP read non-interlocked") \
    f_(lts__t_sectors_zrd_ni_condrd, "# of sectors accessed by ZROP read non-interlocked for op condrd") \
    f_(lts__t_sectors_zrd_ni_prefetch, "# of sectors accessed by ZROP read non-interlocked for op prefetch") \
    f_(lts__t_sectors_zrd_ni_rd, "# of sectors accessed by ZROP read non-interlocked for op rd") \
    f_(lts__t_sectors_zrop, "# of sectors accessed by ZROP") \
    f_(lts__t_sectors_zrop_condrd, "# of sectors accessed by ZROP for op condrd") \
    f_(lts__t_sectors_zrop_ilwal, "# of sectors accessed by ZROP for op ilwal") \
    f_(lts__t_sectors_zrop_prefetch, "# of sectors accessed by ZROP for op prefetch") \
    f_(lts__t_sectors_zrop_rd, "# of sectors accessed by ZROP for op rd") \
    f_(lts__t_sectors_zrop_wr, "# of sectors accessed by ZROP for op wr") \
    f_(lts__t_tag_requests_hit, "# of LTS requests with tag hit") \
    f_(lts__t_tag_requests_miss, "# of LTS requests with tag miss") \
    f_(lts__t_tags_writeback_tier1_issued, "# of sector writebacks, tier1 issued") \
    f_(lts__t_tags_writeback_tier1_killed, "# of sector writebacks, tier1 killed") \
    f_(lts__t_tags_writeback_tier2_issued, "# of sector writebacks, tier2") \
    f_(lts__xbar_output_active, "# of cycles where interface from LTS to XBAR was active") \
    f_(mme__call_count, "# of MME macros called") \
    f_(mme__cycles_active, "# of cycles where MME was active") \
    f_(mme__cycles_stalled, "# of cycles where MME was stalled") \
    f_(mme__input_method_dwords, "# of method dwords sent to MME") \
    f_(mme__input_methods_shadow_filtered, "# of method filtered by MME as redundant state changes") \
    f_(mme__output_method_dwords, "# of method dwords output by MME") \
    f_(mmu__cycles_active, "# of cycles where MMU was active") \
    f_(mmu__hubtlb_requests_hit, "# of PTE requests that hit in HUB TLB") \
    f_(mmu__hubtlb_requests_hit_under_miss, "# of PTE requests that hit on pending-miss in HUB TLB") \
    f_(mmu__hubtlb_requests_miss, "# of PTE requests that missed in HUB TLB") \
    f_(mmu__hubtlb_stalled_on_tag_allocation, "# of cycles where HUB TLB stalled on tag allocation") \
    f_(mmu__hubtlb_stalled_request_fifo_full, "# of cycles where HUB TLB stalled on request FIFO") \
    f_(mmu__pde_requests_hit, "# of PDE requests that hit in MMU cache") \
    f_(mmu__pde_requests_miss, "# of PDE requests that missed in MMU cache") \
    f_(mmu__pte_requests_big_page_0, "# of PTE requests for big pages (64kB or 128kB)") \
    f_(mmu__pte_requests_big_page_1, "# of PTE requests for big pages (64kB or 128kB)") \
    f_(mmu__pte_requests_big_page_2, "# of PTE requests for big pages (64kB or 128kB)") \
    f_(mmu__pte_requests_big_page_3, "# of PTE requests for big pages (64kB or 128kB)") \
    f_(mmu__pte_requests_hit, "# of PTE requests that hit in MMU cache") \
    f_(mmu__pte_requests_miss, "# of PTE requests that missed in MMU cache") \
    f_(mmu__pte_requests_small_page_0, "# of PTE requests for 4kB pages") \
    f_(mmu__pte_requests_small_page_1, "# of PTE requests for 4kB pages") \
    f_(mmu__pte_requests_small_page_2, "# of PTE requests for 4kB pages") \
    f_(mmu__pte_requests_small_page_3, "# of PTE requests for 4kB pages") \
    f_(mpc__alpha_beta_mode_switches, "# of times MPC switched between alpha/beta/unpartitioned modes") \
    f_(mpc__cycles_active_alpha, "# of cycles active in alpha mode") \
    f_(mpc__cycles_active_beta, "# of cycles active in beta mode") \
    f_(mpc__cycles_elapsed_alpha, "# of cycles elapsed in alpha mode") \
    f_(mpc__cycles_elapsed_beta, "# of cycles elapsed in beta mode") \
    f_(mpc__input_batches, "# of batches sent to MPC") \
    f_(mpc__input_prims_alpha, "# of primitives sent to MPC in alpha mode") \
    f_(mpc__input_prims_alpha_patches, "# of patches sent to MPC") \
    f_(mpc__input_prims_beta, "# of primitives sent to MPC in beta mode") \
    f_(mpc__input_subtiles, "# of subtiles sent to MPC from FRSTR") \
    f_(mpc__input_tasks, "# of beta tasks sent to MPC") \
    f_(mpc__input_verts_alpha, "# of vertices sent to MPC for alpha workloads") \
    f_(mpc__input_verts_beta, "# of vertices sent to MPC for beta workloads") \
    f_(mpc__isbe_allocation_stalled, "# of cycles where ISBE allocation was stalled") \
    f_(mpc__isbe_allocation_stalled_alpha, "# of cycles where ISBE allocation was stalled in alpha mode") \
    f_(mpc__isbe_allocation_stalled_alpha_on_vsc, "# of cycles where ISBE allocation was stalled on VSC in alpha mode") \
    f_(mpc__isbe_allocation_stalled_beta, "# of cycles where ISBE allocation was stalled in beta mode") \
    f_(mpc__isbe_allocation_stalled_beta_on_vsc, "# of cycles where ISBE allocation was stalled on VSC in beta mode") \
    f_(mpc__isbe_allocations, "# of ISBE allocations") \
    f_(mpc__isbe_allocations_alpha, "# of ISBE allocations in alpha mode") \
    f_(mpc__isbe_allocations_beta, "# of ISBE allocations in beta mode") \
    f_(mpc__output_batches, "# of batches output by MPC") \
    f_(mpc__tram_allocation_stalled, "# of cycles where TRAM Allocation was stalled") \
    f_(mpc__tram_fill_fifo_stalled, "# of cycles where TRAM Fill FIFO was stalled") \
    f_(mpc__tram_startxy_fifo_stalled, "# of cycles where TRAM StartXY FIFO was stalled") \
    f_(mpc__warp_launch_stalled_gs, "# of cycles where GS warp launch was stalled") \
    f_(mpc__warp_launch_stalled_gs_fast_alpha, "# of cycles where FGS warp launch was stalled in alpha mode") \
    f_(mpc__warp_launch_stalled_gs_fast_beta, "# of cycles where FGS warp launch was stalled in beta mode") \
    f_(mpc__warp_launch_stalled_ps, "# of cycles where PS warp launch was stalled") \
    f_(mpc__warp_launch_stalled_rf_free, "# of cycles where warp launch was stalled despite sufficient free register file space") \
    f_(mpc__warp_launch_stalled_tcs, "# of cycles where TCS warp launch was stalled") \
    f_(mpc__warp_launch_stalled_tes, "# of cycles where TES warp launch was stalled") \
    f_(mpc__warp_launch_stalled_vsa, "# of cycles where VSa warp launch was stalled") \
    f_(mpc__warp_launch_stalled_vsb, "# of cycles where VSb warp launch was stalled") \
    f_(mpc__warp_launch_stalled_vtg, "# of cycles where VTG warp launch was stalled") \
    f_(pda__cycles_active, "# of cycles where PDA was active") \
    f_(pda__input_prims, "# of primitives sent to PDA") \
    f_(pda__input_prims_line, "# of lines sent to PDA") \
    f_(pda__input_prims_lineadj, "# of lines+adjacancies sent to PDA") \
    f_(pda__input_prims_patch, "# of patches sent to PDA") \
    f_(pda__input_prims_point, "# of points sent to PDA") \
    f_(pda__input_prims_tri, "# of triangles sent to PDA") \
    f_(pda__input_prims_triadj, "# of triangles+adjacencies sent to PDA") \
    f_(pda__input_prims_triflat, "# of flat-shaded triangles sent to PDA") \
    f_(pda__input_restart_indices, "# of restart indices sent to PDA") \
    f_(pda__input_stalled_index_fetch, "# of cycles where PDA index fetch was stalled") \
    f_(pda__input_verts, "# of vertices sent to PDA") \
    f_(pda__output_batches, "# of batches output by PDA") \
    f_(pda__output_verts, "# of vertices output by PDA into batches") \
    f_(pdb__cycles_active, "# of cycles where PDB was active") \
    f_(pdb__input_batches, "# of batches sent to PDB") \
    f_(pdb__input_stalled_alpha, "# of cycles where input to PDB from SCC was stalled by alpha pipe") \
    f_(pdb__input_stalled_beta, "# of cycles where input to PDB from SCC was stalled by beta pipe") \
    f_(pdb__input_tasks, "# of beta tasks output by PDB") \
    f_(pdb__output_pkts, "# of packets output by PDB") \
    f_(pdb__output_stalled, "# of cycles where PDB output was stalled on XBAR") \
    f_(pdb__output_stalled_alpha, "# of cycles where PDB output was stalled by alpha pipe") \
    f_(pdb__output_stalled_beta, "# of cycles where PDB output was stalled by beta pipe") \
    f_(pel__cycles_active, "# of cycles where PEL was active") \
    f_(pel__in_active, "# of cycles where PEL in stages write unit was active ") \
    f_(pel__in_write_requests, "# of cycles where PEL in-stage was active") \
    f_(pel__in_write_requests_active, "# of cycles where PEL in-stage's write interface was active") \
    f_(pel__in_write_requests_stalled, "# of cycles where PEL in-stage's write interface was stalled") \
    f_(pel__in_write_requests_stalled_tg, "# of cycles where PEL in-stage's write interface was stalled, for TG request") \
    f_(pel__in_write_requests_stalled_vaf_alpha, "# of cycles where PEL in-stage's write interface was stalled, for VAF alpha request") \
    f_(pel__in_write_requests_stalled_vaf_beta, "# of cycles where PEL in-stage's write interface was stalled, for VAF beta request") \
    f_(pel__out_active, "# of cycles where PEL out stage was active") \
    f_(pel__out_input_stalled, "# of cycles where PEL out stages input interface stalled, stalling ACACHE") \
    f_(pel__out_l2_requests, "# of L2 requests sent to PEL out stage from ACACHE, VSC") \
    f_(pel__out_l2_requests_ilwalidate_256b, "# of 32B L2 ilwalidate requests sent to PEL out stage") \
    f_(pel__out_l2_requests_read, "# of 32B L2 read requests sent to PEL out stage from ACACHE") \
    f_(pel__out_l2_requests_write_256b, "# of 32B L2 write requests sent to PEL out stage from VSC") \
    f_(pel__out_l2_requests_write_512b, "# of 64B L2 write requests sent to PEL out stage from VSC") \
    f_(pel__out_read_stalled_stri, "# of cycles where PEL out stages read interface stalled, for STRI request") \
    f_(pel__out_read_stalled_vaf_alpha, "# of cycles where PEL out stages read interface stalled, for VAF alpha request") \
    f_(pel__out_read_stalled_vaf_beta, "# of cycles where PEL out stages read interface stalled, for VAF beta request") \
    f_(pes__cycles_active, "# of cycles where PES was active") \
    f_(pes__stream_output_attrs, "# of attributes output by STREAM") \
    f_(pes__stream_output_prims, "# of primitives output by STREAM") \
    f_(pes__stream_output_verts, "# of vertices output by STREAM") \
    f_(prop__cdp_alpha_blendopt_killed_pixels, "# of pixels killed by blendopt (Cd==1 && Cs*SRC==0)") \
    f_(prop__cdp_alpha_blendopt_pixels_fill_override, "# of pixels optimized to use fast constant color fill") \
    f_(prop__cdp_alpha_blendopt_read_avoid, "# of pixels optimized by read-avoid") \
    f_(prop__cdp_alpha_test_killed_pixels, "# of pixels killed by alpha testing") \
    f_(prop__cdp_alpha_to_coverage_killed_pixels, "# of pixels killed by alpha-to-coverage") \
    f_(prop__cdp_alpha_to_coverage_output_pixels, "# of pixels output by alpha-to-coverage") \
    f_(prop__csb_input_pixels_3d, "# of 3d pixels sent to CSB") \
    f_(prop__csb_input_quads_3d, "# of pixel-quads sent to CSB") \
    f_(prop__csb_input_samples_3d, "# of samples sent to CSB") \
    f_(prop__csb_killed_quadquads, "# of quad-quads fully killed by color sample mask") \
    f_(prop__csb_output_active, "# of cycles where CSB sent output to CROP") \
    f_(prop__csb_output_crop_requests, "# of packets output by CSB, sent to CROP") \
    f_(prop__csb_output_crop_requests_color, "# of color packets output by CSB, sent to CROP") \
    f_(prop__csb_output_pixels, "# of pixels output by CSB, sent to CROP") \
    f_(prop__csb_output_pixels_3d, "# of 3d pixels output by CSB, sent to CROP") \
    f_(prop__csb_output_quads, "# of pixel-quads output by CSB, sent to CROP") \
    f_(prop__csb_output_quads_3d, "# of 3d pixel-quads output by CSB, sent to CROP") \
    f_(prop__csb_output_samples_3d, "# of 3d samples output by CSB, sent to CROP") \
    f_(prop__csb_output_stalled, "# of cycles where CSB output to CROP was stalled") \
    f_(prop__cycles_active, "# of cycles where PROP was active") \
    f_(prop__earlyz_killed_pixels, "# of EarlyZ pixels killed by ZROP") \
    f_(prop__earlyz_killed_quads, "# of EarlyZ pixel-quads killed by ZROP") \
    f_(prop__earlyz_killed_samples, "# of EarlyZ samples killed by ZROP") \
    f_(prop__earlyz_output_pixels, "# of EarlyZ pixels output by ZROP") \
    f_(prop__earlyz_output_quads, "# of EarlyZ pixel-quads output by ZROP") \
    f_(prop__earlyz_output_samples, "# of EarlyZ samples output by ZROP") \
    f_(prop__gnic_port0_active, "# of cycles where PROP sent a GNIC request to port0") \
    f_(prop__gnic_port0_stalled, "# of cycles where PROP was stalled on GNIC port0") \
    f_(prop__gnic_port1_active, "# of cycles where PROP sent a GNIC request to port1") \
    f_(prop__gnic_port1_stalled, "# of cycles where PROP was stalled on GNIC port1") \
    f_(prop__input_active, "# of cycles where input was sent to PROP") \
    f_(prop__input_pixels, "# of pixels sent to PROP") \
    f_(prop__input_pixels_2d, "# of 2D pixels sent to PROP") \
    f_(prop__input_pixels_3d, "# of 3D pixels sent to PROP") \
    f_(prop__input_quads, "# of pixel-quads sent to PROP") \
    f_(prop__input_quads_2d, "# of 2D pixel-quads sent to PROP") \
    f_(prop__input_quads_3d, "# of 3D pixel-quads sent to PROP") \
    f_(prop__input_samples, "# of samples sent to PROP") \
    f_(prop__input_samples_2d, "# of 2D samples sent to PROP") \
    f_(prop__input_samples_3d, "# of 3D samples sent to PROP") \
    f_(prop__input_stalled, "# of cycles where input was stalled") \
    f_(prop__input_stalled_waiting_for_pixel_shader_barrier_release, "# of cycles the PROP input stage was stalled waiting for a pixel shader barrier to complete") \
    f_(prop__latez_output_pixels, "# of LateZ pixels output by ZROP") \
    f_(prop__latez_output_quads, "# of LateZ pixel-quads output by ZROP") \
    f_(prop__latez_output_samples, "# of LateZ samples output by ZROP") \
    f_(prop__latez_rstrz_input_pixels, "# of LateZ pixels sent to ZROP using rasterZ") \
    f_(prop__latez_rstrz_input_quads, "# of LateZ pixel-quads sent to ZROP using rasterZ") \
    f_(prop__latez_rstrz_input_samples, "# of LateZ samples sent to ZROP using rasterZ") \
    f_(prop__latez_shdz_input_pixels, "# of LateZ pixels sent to ZROP using shaderZ") \
    f_(prop__latez_shdz_input_quads, "# of LateZ pixel-quads sent to ZROP using shaderZ") \
    f_(prop__latez_shdz_input_samples, "# of LateZ samples sent to ZROP using shaderZ") \
    f_(prop__pixel_shader_barriers, "# of pixel shader barriers") \
    f_(prop__zrop_output_active, "# of cycles where PROP sent output to ZROP") \
    f_(prop__zrop_output_quads, "# of quads that passed Z-Test in ZROP (EarlyZ and LateZ)") \
    f_(prop__zrop_output_samples, "# of samples that passed Z-Test in ZROP (EarlyZ and LateZ)") \
    f_(prop__zrop_output_stalled, "# of cycles where PROP output to ZROP was stalled") \
    f_(raster__crstr_lwlled_prims_no_tile_coverage, "# of primitives lwlled by CRSTR for reason: no tile coverage") \
    f_(raster__crstr_discover_stalling_setup, "# of cycles the CRSTR Discover stage is stalling SETUP, and SETUP has a primitive ready") \
    f_(raster__crstr_discover_working_no_tile_output, "# of cycles the CRSTR Discover stage working but not outputting") \
    f_(raster__crstr_discover_working_no_tile_stalling_setup, "# of cycles the CRSTR Discover stage is working and stalling SETUP") \
    f_(raster__crstr_input_prims, "# of primitives sent to CRSTR") \
    f_(raster__crstr_output_tiles, "# of 16x16 coarse-raster tiles output by CRSTR") \
    f_(raster__crstr_output_tiles_empty, "# of empty 16x16 coarse-raster tiles output by CRSTR") \
    f_(raster__crstr_search_stalling_input, "# of cycles where CRSTR Search stage is stalling Input, a primitive is waiting in Input") \
    f_(raster__frstr_killed_subtiles_input, "# of 8x8 fine raster tiles killed by FRSTR at input stage") \
    f_(raster__frstr_killed_subtiles_output, "# of 8x8 fine raster tiles killed by FRSTR after processing") \
    f_(raster__frstr_output_cycles, "# of cycles FRSTR spent callwlating an output subtile") \
    f_(raster__frstr_output_subtiles, "# of 8x8 fine raster tiles output by FRSTR") \
    f_(raster__frstr_output_subtiles_1_cycle, "# of 8x8 fine raster tiles processed in 1 cycle") \
    f_(raster__frstr_output_subtiles_2_cycle, "# of 8x8 fine raster tiles processed in 2 cycle") \
    f_(raster__frstr_output_subtiles_2d, "# of 8x8 fine raster tiles output by FRSTR for 2D") \
    f_(raster__frstr_output_subtiles_3_cycle, "# of 8x8 fine raster tiles processed in 3 cycle") \
    f_(raster__frstr_output_subtiles_4_cycle, "# of 8x8 fine raster tiles processed in 4 cycle") \
    f_(raster__frstr_output_subtiles_fully_covered, "# of 8x8 fine raster tiles output by FRSTR with full coverage") \
    f_(raster__frstr_processed_3_edges, "# of 8x8 fine raster tiles that processed 3 edges of a triangle (indicating small triangles)") \
    f_(raster__setup_lwlled_prims, "# of primitives lwlled by RASTER SETUP") \
    f_(raster__setup_input_prims, "# or primitives sent to RASTER SETUP") \
    f_(raster__setup_output_prims, "# of primitives output by RASTER SETUP") \
    f_(raster__setup_output_prims_line, "# of lines output by RASTER SETUP") \
    f_(raster__setup_output_prims_point, "# of points output by RASTER SETUP") \
    f_(raster__setup_output_prims_triangle, "# of triangles output by RASTER SETUP") \
    f_(raster__tc_flushes, "# of TC flushes") \
    f_(raster__tc_input_transactions, "# of transactions sent to TC from FRSTR") \
    f_(raster__tc_output_subtiles, "# of 8x8 subtiles output by TC") \
    f_(raster__tc_output_tiles, "# of 16x16 tiles output by TC") \
    f_(raster__zlwll_lwlled_occluders, "# of z-occluders (4x2s) lwlled by ZLWLL") \
    f_(raster__zlwll_lwlled_occluders_depth, "# of z-occluders (4x2s) lwlled by ZLWLL for reason: depth") \
    f_(raster__zlwll_lwlled_occluders_depth_bounds, "# of z-occluders (4x2s) lwlled by ZLWLL for reason: depth bounds") \
    f_(raster__zlwll_lwlled_occluders_near_far_clipped, "# of z-occluders (4x2s) lwlled by ZLWLL for reason: near/far clip") \
    f_(raster__zlwll_lwlled_occluders_stencil, "# of z-occluders (4x2s) lwlled by ZLWLL for reason: stencil") \
    f_(raster__zlwll_input_occluders, "# of z-occluders (4x2s) sent to ZLWLL by CRSTR") \
    f_(raster__zlwll_input_tiles, "# of 16x16 coarse-raster tiles sent to ZLWLL") \
    f_(raster__zlwll_occluders_zfail, "# of z-occluders (4x2s) lwlled by ZLWLL that failed z-test, and may have also failed other reasons (except near/far)") \
    f_(raster__zlwll_output_occluders_trivial_accept, "# of z-occluders (4x2s) output by ZLWLL that were trivially accepted") \
    f_(raster__zlwll_output_occluders_zfail_no_lwll, "# of z-occluders (4x2s) output by ZLWLL that failed z-test, but cannot be lwlled for stencil output") \
    f_(rdm__crop_output_active, "# of cycles where RDM sent data to CROP") \
    f_(rdm__crop_output_stalled, "# of cycles where RDM output to CROP was stalled") \
    f_(rdm__cycles_active, "# of cycles where RDM was active") \
    f_(rdm__input_data0_active, "# of cycles where PROP sent data to RDm on data0") \
    f_(rdm__input_data1_active, "# of cycles where PROP sent data to RDm on data1") \
    f_(rdm__zrop_output_active, "# of cycles where RDM sent data to ZROP") \
    f_(scc__cycles_active, "# of cycles where SCC was active") \
    f_(scc__input_state_changes, "# of state changes sent to SCC") \
    f_(scc__load_constants, "# of constant updates sent to SCC") \
    f_(scc__load_constants_hit, "# of constant updates that hit in cache") \
    f_(scc__load_constants_miss, "# of constant updates that missed in cache") \
    f_(scc__load_constants_page_first_update, "# of constant updates that were first in page, after a state change") \
    f_(scc__load_constants_page_pool_allocations, "# of SCC page pool allocations triggered by constant updates") \
    f_(scc__load_constants_page_same, "# of constant updates oclwring after a first write") \
    f_(scc__load_constants_stalled_max_versions, "# of cycles where SCC stalled on constant updates: max versions hit") \
    f_(scc__load_constants_stalled_no_gcc_credits, "# of cycles where SCC stalled on constant updates: no gcc credits") \
    f_(scc__load_constants_stalled_update_scoreboard_full, "# of cycles where SCC stalled on constant updates: update scoreboard full") \
    f_(sked__cycles_active, "# of cycles where SKED was active") \
    f_(sked__dispatch_active_scg0, "# of cycles where SKED has in-flight dispatch") \
    f_(sked__dispatch_count, "# of compute dispatches sent from SKED to CWD") \
    f_(sm__ctas_active, "sum of per-cycle # of CTAs in flight") \
    f_(sm__ctas_launched, "# of CTAs launched") \
    f_(sm__cycles_active, "# of cycles with at least 1 warp in flight") \
    f_(sm__cycles_active_3d, "# of cycles with any 3D warp in flight, or pixel drainout pending") \
    f_(sm__cycles_active_3d_ps, "# of cycles with PS warps in flight, or pixel drainout pending") \
    f_(sm__cycles_active_3d_vtg, "# of cycles with VTG warps in flight") \
    f_(sm__cycles_active_cs, "# of cycles with CS warps in flight") \
    f_(sm__cycles_active_gs, "# of cycles with GS warps in flight") \
    f_(sm__cycles_active_ps, "# of cycles with PS warps in flight") \
    f_(sm__cycles_active_tcs, "# of cycles with TCS warps in flight") \
    f_(sm__cycles_active_tes, "# of cycles with TES warps in flight") \
    f_(sm__cycles_active_vs, "# of cycles with VSa or VSb warps in flight") \
    f_(sm__cycles_active_vsa, "# of cycles with VSa warps in flight") \
    f_(sm__cycles_active_vsb, "# of cycles with VSb warps in flight") \
    f_(sm__icc_prefetches, "# of ICC prefetch requests to GCC") \
    f_(sm__icc_requests_hit, "# of ICC requests that hit") \
    f_(sm__icc_requests_miss_covered, "# of ICC requests that missed and were covered by a pending miss") \
    f_(sm__icc_requests_miss_no_tags, "# of ICC requests that missed and were not completed due to no available miss tags to GCC") \
    f_(sm__icc_requests_miss_to_gcc, "# of ICC requests that missed and were sent to GCC") \
    f_(sm__idc_requests_hit, "# of IDC requests that hit") \
    f_(sm__idc_requests_miss_covered, "# of IDC requests that missed and were covered by a pending miss") \
    f_(sm__idc_requests_miss_no_tags, "# of IDC requests that missed and could not be completed due to no available miss tags to GCC") \
    f_(sm__idc_requests_miss_to_gcc, "# of IDC requests that missed and were sent to GCC") \
    f_(sm__inst_exelwted_pipe_adu_divergent_smp0, "# of LDC/AL2P instructions with divergent addresses") \
    f_(sm__inst_exelwted_pipe_adu_divergent_smp1, "# of LDC/AL2P instructions with divergent addresses") \
    f_(sm__miop_adu_replays_smp0, "# of LDC/AL2P instruction replays to handle divergent addresses") \
    f_(sm__miop_adu_replays_smp1, "# of LDC/AL2P instruction replays to handle divergent addresses") \
    f_(sm__miop_ldc_replays_smp0, "# of LDC instruction replays due to cache misses") \
    f_(sm__miop_ldc_replays_smp1, "# of LDC instruction replays due to cache misses") \
    f_(sm__miop_pq_read_active_lsu_smp0, "# of cycles where MIOP PQ sent register operands to LSU pipe") \
    f_(sm__miop_pq_read_active_lsu_smp1, "# of cycles where MIOP PQ sent register operands to LSU pipe") \
    f_(sm__miop_pq_read_active_pixout_smp0, "# of cycles where MIOP PQ sent register operands to PIXOUT") \
    f_(sm__miop_pq_read_active_pixout_smp1, "# of cycles where MIOP PQ sent register operands to PIXOUT") \
    f_(sm__miop_pq_read_active_smp0, "# of cycles where MIOP PQ sent register operands to a pipeline") \
    f_(sm__miop_pq_read_active_smp1, "# of cycles where MIOP PQ sent register operands to a pipeline") \
    f_(sm__miop_pq_read_active_tex_smp0, "# of cycles where MIOP PQ sent register operands to TEX pipe") \
    f_(sm__miop_pq_read_active_tex_smp1, "# of cycles where MIOP PQ sent register operands to TEX pipe") \
    f_(sm__mios_datapath_active, "# of cycles MIOS was active exelwting non-shared memory operations such as FP64, INT, and shuffle instructions.") \
    f_(sm__mios_shmem_accesses_lsu_read, "# of LSU read accesses from MIOS shared-memory data RAM for load instructions (ALD, LDS, ATOMS, ATOMS.CAS)") \
    f_(sm__mios_shmem_accesses_lsu_read_bank_conflict, "# of LSU read accesses from MIOS shared-memory data RAM due to bank conflicts from load instructions (ALD, LDS, ATOMS, ATOMS.CAS) shared-memory data bank conflicts for LSU reads") \
    f_(sm__mios_shmem_accesses_lsu_write, "# of LSU write accesses to MIOS shared-memory data RAM for store instructions (AST, STS, ATOMS, ATOMS.CAS)") \
    f_(sm__mios_shmem_accesses_lsu_write_bank_conflict, "# of LSU write accesses to MIOS shared-memory data RAM due to bank conflicts from store instructions (AST, STS, ATOMS, ATOMS.CAS) shared-memory data bank conflicts for LSU reads") \
    f_(sm__mios_shmem_accesses_pe_read_isbe, "# of PE ISBE read accesses from MIOS shared-memory data RAM") \
    f_(sm__mios_shmem_accesses_pe_write_isbe, "# of PE ISBE write accesses to MIOS shared-memory data RAM") \
    f_(sm__mios_shmem_accesses_pe_write_tram, "# of PE TRAM write accesses from MIOS shared-memory data RAM") \
    f_(sm__mios_shmem_accesses_su_read_tram, "# of LSU read accesses from MIOS shared-memory data RAM from IPA instructions") \
    f_(sm__mios_shmem_accesses_su_read_tram_bank_conflict, "# of LSU read accesses from MIOS shared-memory data RAM due to bank conflicts from IPA instructions") \
    f_(sm__ps_quads_killed, "# of PS pixel quads where all threads were killed by shader") \
    f_(sm__ps_quads_launched, "# of PS pixel quads launched") \
    f_(sm__ps_quads_sent_to_pixout, "# of PS pixel quads sent from SM to PIXOUT") \
    f_(sm__ps_warps_killed, "# of PS warps where all threads were killed by shader") \
    f_(sm__subtiles_active, "sum of per-cycle # of subtiles") \
    f_(sm__subtiles_launched_smp0, "# of subtiles launched") \
    f_(sm__subtiles_launched_smp1, "# of subtiles launched") \
    f_(sm__threads_launched, "# of shader threads launched") \
    f_(sm__threads_launched_cs, "# of CS threads launched") \
    f_(sm__threads_launched_gs, "# of GS threads launched") \
    f_(sm__threads_launched_ps_not_killed, "# of PS threads launched in not killed state (with sample coverage)") \
    f_(sm__threads_launched_tcs, "# of TCS threads launched") \
    f_(sm__threads_launched_tes, "# of TES threads launched") \
    f_(sm__threads_launched_vsa, "# of VSa threads launched") \
    f_(sm__threads_launched_vsb, "# of VSb threads launched") \
    f_(sm__warps_active, "sum of per-cycle # of warps in flight") \
    f_(sm__warps_active_cs, "sum of per-cycle # of CS warps in flight") \
    f_(sm__warps_active_gs, "sum of per-cycle # of GS warps in flight") \
    f_(sm__warps_active_ps, "sum of per-cycle # of PS warps in flight") \
    f_(sm__warps_active_tcs, "sum of per-cycle # of TCS warps in flight") \
    f_(sm__warps_active_tes, "sum of per-cycle # of TES warps in flight") \
    f_(sm__warps_active_vsa, "sum of per-cycle # of VSa warps in flight") \
    f_(sm__warps_active_vsb, "sum of per-cycle # of VSb warps in flight") \
    f_(sm__warps_active_vtg, "sum of per-cycle # of VTG warps in flight") \
    f_(sm__warps_completed_ooo_3d_ps, "# of PS warps that completed out-of-order") \
    f_(sm__warps_completed_ooo_3d_vtg, "# of VTG warps that completed out-of-order") \
    f_(sm__warps_draining_ps, "sum of per-cycle # of PS warps waiting to be drained") \
    f_(sm__warps_launched_cs, "# of CS warps launched") \
    f_(sm__warps_launched_gs, "# of GS warps launched") \
    f_(sm__warps_launched_gs_fast_alpha, "# of FastGS warps launched in alpha mode") \
    f_(sm__warps_launched_gs_fast_beta, "# of FastGS warps launched in beta mode") \
    f_(sm__warps_launched_ps, "# of PS warps launched") \
    f_(sm__warps_launched_tcs, "# of TCS warps launched") \
    f_(sm__warps_launched_tes, "# of TES warps launched") \
    f_(sm__warps_launched_vsa, "# of VSa warps launched") \
    f_(sm__warps_launched_vsb, "# of VSb warps launched") \
    f_(sm__warps_retiring_ps_stalled_backpressure, "sum of per-cycle # of completed pixel warps that could not drain because of back pressure") \
    f_(sm__warps_retiring_ps_stalled_not_selected, "sum of per-cycle # of completed pixel warps that could not drain because a different in-order warp was selected") \
    f_(sm__warps_retiring_ps_stalled_out_of_order, "sum of per-cycle # of completed pixel warps that could not drain because of out of order completion") \
    f_(smsp__cycles_active, "# of cycles with at least 1 warp in flight") \
    f_(smsp__imc_requests_hit, "# of IMC requests that hit") \
    f_(smsp__imc_requests_miss_covered, "# of IMC requests that missed and were covered by a pending miss") \
    f_(smsp__imc_requests_miss_no_tags, "# of IMC requests that missed and could not be completed due to no available miss tags to GCC") \
    f_(smsp__imc_requests_miss_to_gcc, "# of IMC requests that missed and were sent to GCC") \
    f_(smsp__inst_exelwted, "# of warp instructions exelwted") \
    f_(smsp__inst_exelwted_cs, "# of warp instructions exelwted by CS") \
    f_(smsp__inst_exelwted_gs, "# of warp instructions exelwted by PS") \
    f_(smsp__inst_exelwted_lsu_wb, "# of warp instructions exelwted by LSU pipe that wrote back to the register file") \
    f_(smsp__inst_exelwted_lsu_wb_pending, "sum of per-cycle # of warp instructions pending by LSU pipe that wrote back to the register file") \
    f_(smsp__inst_exelwted_pipe_adu, "# of instructions exelwted by the adu pipe") \
    f_(smsp__inst_exelwted_pipe_adu_pred_off_all, "# of instructions exelwted by the adu pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_adu_pred_on, "# of instructions exelwted by the adu pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_bar, "# of instructions exelwted by the bar pipe") \
    f_(smsp__inst_exelwted_pipe_bar_pred_off_all, "# of instructions exelwted by the bar pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_bar_pred_on, "# of instructions exelwted by the bar pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_bru, "# of instructions exelwted by the bru pipe") \
    f_(smsp__inst_exelwted_pipe_bru_pred_off_all, "# of instructions exelwted by the bru pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_bru_pred_on, "# of instructions exelwted by the bru pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_fe, "# of instructions exelwted by the fe pipe") \
    f_(smsp__inst_exelwted_pipe_fma64plus, "# of instructions exelwted by the fma64plus pipe") \
    f_(smsp__inst_exelwted_pipe_fma64plus_pred_off_all, "# of instructions exelwted by the fma64plus pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_fma64plus_pred_on, "# of instructions exelwted by the fma64plus pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_fma64plusplus, "# of instructions exelwted by the fma64plusplus pipe") \
    f_(smsp__inst_exelwted_pipe_fma64plusplus_pred_off_all, "# of instructions exelwted by the fma64plusplus pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_fma64plusplus_pred_on, "# of instructions exelwted by the fma64plusplus pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_fmai, "# of instructions exelwted by the fmai pipe") \
    f_(smsp__inst_exelwted_pipe_fp16, "# of instructions exelwted by the fp16 pipe") \
    f_(smsp__inst_exelwted_pipe_fxu, "# of instructions exelwted by the fxu pipe") \
    f_(smsp__inst_exelwted_pipe_ldc, "# of instructions exelwted by the ldc pipe") \
    f_(smsp__inst_exelwted_pipe_ldc_pred_off_all, "# of instructions exelwted by the ldc pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_ldc_pred_on, "# of instructions exelwted by the ldc pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_lsu, "# of instructions exelwted by the lsu pipe") \
    f_(smsp__inst_exelwted_pipe_lsu_pred_off_all, "# of instructions exelwted by the lsu pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_lsu_pred_on, "# of instructions exelwted by the lsu pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_su, "# of instructions exelwted by the su pipe") \
    f_(smsp__inst_exelwted_pipe_su_pred_off_all, "# of instructions exelwted by the su pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_su_pred_on, "# of instructions exelwted by the su pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_tex, "# of instructions exelwted by the tex pipe") \
    f_(smsp__inst_exelwted_pipe_tex_pred_off_all, "# of instructions exelwted by the tex pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_tex_pred_on, "# of instructions exelwted by the tex pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_pipe_xu, "# of instructions exelwted by the xu pipe") \
    f_(smsp__inst_exelwted_pipe_xu_pred_off_all, "# of instructions exelwted by the xu pipe where all active threads' guard-predicated was false") \
    f_(smsp__inst_exelwted_pipe_xu_pred_on, "# of instructions exelwted by the xu pipe where at least 1 active thread's guard-predicate was true") \
    f_(smsp__inst_exelwted_ps, "# of warp instructions exelwted by GS") \
    f_(smsp__inst_exelwted_tcs, "# of warp instructions exelwted by TCS") \
    f_(smsp__inst_exelwted_tes, "# of warp instructions exelwted by TES") \
    f_(smsp__inst_exelwted_tex_wb, "# of warp instructions exelwted by TEX pipe that wrote back to the register file") \
    f_(smsp__inst_exelwted_tex_wb_pending, "sum of per-cycle # of warp instructions pending by TEX pipe that wrote back to the register file") \
    f_(smsp__inst_exelwted_vs, "# of warp instructions exelwted by VSa or VSb") \
    f_(smsp__inst_exelwted_vsa, "# of warp instructions exelwted by VSa") \
    f_(smsp__inst_exelwted_vsb, "# of warp instructions exelwted by VSb") \
    f_(smsp__inst_issued, "# of warp instructions issued") \
    f_(smsp__issue_active, "# of cycles where at least 1 instruction was issued") \
    f_(smsp__lsu_write_back_active, "# of cycles where the interface from LSU to register file was active") \
    f_(smsp__miop_pq_write_active, "# of cycles where SMSP sent register operands to MIOP PQ") \
    f_(smsp__ps_threads_killed, "# of PS threads killed by shader") \
    f_(smsp__tex_requests, "# of requests sent from SM to TEX") \
    f_(smsp__tex_write_back_active, "# of cycles where the interface from TEX to register file was active") \
    f_(smsp__thread_inst_exelwted, "# of thread instructions exelwted") \
    f_(smsp__thread_inst_exelwted_pred_on, "# of thread instructions exelwted where at least 1 thread was predicated on") \
    f_(smsp__warps_active, "sum of per-cycle # of active warps") \
    f_(smsp__warps_cant_issue_allocation_stall_0, "sum of per-cycle # of warps stalled waiting for a branch to resolve, waiting for all memory operations to retire, or waiting to be allocated to the micro-scheduler") \
    f_(smsp__warps_cant_issue_allocation_stall_1, "sum of per-cycle # of warps stalled waiting for a branch to resolve, waiting for all memory operations to retire, or waiting to be allocated to the micro-scheduler") \
    f_(smsp__warps_cant_issue_barrier, "sum of per-cycle # of warps stalled waiting for sibling warps at a CTA barrier") \
    f_(smsp__warps_cant_issue_dispatch_stall, "sum of per-cycle # of warps stalled waiting on a dispatch stall") \
    f_(smsp__warps_cant_issue_drain, "sum of per-cycle # of warps stalled waiting after EXIT for all memory instructions to complete so that warp resources can be freed") \
    f_(smsp__warps_cant_issue_imc_miss, "sum of per-cycle # of warps stalled waiting for an immediate constant cache (IMC) miss") \
    f_(smsp__warps_cant_issue_long_scoreboard_0, "sum of per-cycle # of warps stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, tex) operation") \
    f_(smsp__warps_cant_issue_long_scoreboard_1, "sum of per-cycle # of warps stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, tex) operation") \
    f_(smsp__warps_cant_issue_math_pipe_throttle, "sum of per-cycle # of warps stalled waiting for the exelwtion pipe to be available") \
    f_(smsp__warps_cant_issue_membar, "sum of per-cycle # of warps stalled waiting on a memory barrier") \
    f_(smsp__warps_cant_issue_mio_throttle, "sum of per-cycle # of warps stalled waiting for the MIO instruction queue to be not full") \
    f_(smsp__warps_cant_issue_misc_0, "sum of per-cycle # of warps stalled on a miscellaneous hardware reason") \
    f_(smsp__warps_cant_issue_misc_1, "sum of per-cycle # of warps stalled on a miscellaneous hardware reason") \
    f_(smsp__warps_cant_issue_no_instructions_0, "sum of per-cycle # of warps stalled waiting to be selected to fetch an instruction or waiting on an icache miss") \
    f_(smsp__warps_cant_issue_no_instructions_1, "sum of per-cycle # of warps stalled waiting to be selected to fetch an instruction or waiting on an icache miss") \
    f_(smsp__warps_cant_issue_not_selected, "sum of per-cycle # of warps stalled waiting for the microscheduler to select the warp to issue") \
    f_(smsp__warps_cant_issue_selected, "sum of per-cycle # of warps selected by the microscheduler and issued an instruction") \
    f_(smsp__warps_cant_issue_short_scoreboard_0, "sum of per-cycle # of warps stalled waiting for a scoreboard dependency on a MIO operation (not to L1TEX)") \
    f_(smsp__warps_cant_issue_short_scoreboard_1, "sum of per-cycle # of warps stalled waiting for a scoreboard dependency on a MIO operation (not to L1TEX)") \
    f_(smsp__warps_cant_issue_tex_throttle, "sum of per-cycle # of warps stalled waiting for the TEX/L1 instruction queue to be not full") \
    f_(smsp__warps_cant_issue_tile_allocation_stall, "sum of per-cycle # of warps stalled waiting for sibling warps at a batch barrier") \
    f_(smsp__warps_cant_issue_wait, "sum of per-cycle # of warps stalled waiting on a fixed latency exelwtion dependency") \
    f_(smsp__warps_eligible, "sum of per-cycle # of warps eligible to issue an instruction") \
    f_(smsp__warps_launched, "# of warps launched") \
    f_(stri__acache_requests_stri, "# of A$ requests generated by attribute fetch in STRI") \
    f_(stri__acache_requests_stri_hit, "# of A$ requests that hit in A$ proper") \
    f_(stri__acache_requests_stri_local_hit, "# of A$ requests that hit in STRI-local-cache") \
    f_(stri__acache_requests_stri_miss, "# of A$ requests that missed in A$ and went to L2") \
    f_(stri__attrs, "# of attribute plane-equations output by STRI") \
    f_(stri__cycles_active, "# of cycles where STRI was active") \
    f_(stri__cycles_busy, "# of cycles where STRI was busy") \
    f_(stri__cycles_stalled, "# of cycles where STRI was stalled") \
    f_(stri__to_acache_stalled_on_read, "# of cycles where STRI-to-A$ interface was stalled in beta mode, on read") \
    f_(stri__to_acache_stalled_on_tag_allocation, "# of cycles where STRI-to-A$ interface was stalled in beta mode, on tag allocation") \
    f_(swdx__binner_active, "# of cycles where binner was active") \
    f_(swdx__cycles_active, "# of cycles where SWDX was active") \
    f_(swdx__input_active, "# of cycles where SWDX input was active") \
    f_(swdx__input_messages, "# of messages sent to SWDX") \
    f_(swdx__input_pixel_shader_barriers, "# of pixel shader barriers sent to SWDX") \
    f_(swdx__input_prims_dual_binned, "# of input primitives that were part of a replay section") \
    f_(swdx__input_prims_single_binned, "# of input primitives that were part of a replay section") \
    f_(swdx__input_stalled, "# of cycles where SWDX input was stalled") \
    f_(swdx__output_active, "# of cycles where SWDX sent output to SETUP") \
    f_(swdx__output_barriers, "# of barriers sent from SWDX to SETUP") \
    f_(swdx__output_prims, "# of primitives sent to SETUP") \
    f_(swdx__output_stalled, "# of cycles where SWDX stalled sending output to SETUP") \
    f_(swdx__tc_binner_binned_op_tiled_cache_barriers, "# of binnable barriers recorded by tiled cache barriers") \
    f_(swdx__tc_replayer_bin_flushes, "# of tiled-cache bin flushes") \
    f_(swdx__tc_replayer_bin_flushes_reason_bin_memory_threshold, "# of tiled-cache bin flushes due to bin memory threshold") \
    f_(swdx__tc_replayer_bin_flushes_reason_cbe_memory_threshold, "# of tiled-cache bin flushes due to CBE memory threshold") \
    f_(swdx__tc_replayer_bin_flushes_reason_cbe_slots_threshold, "# of tiled-cache bin flushes due to CBE slots threshold") \
    f_(swdx__tc_replayer_bin_flushes_reason_clear, "# of tiled-cache bin flushes due to clears") \
    f_(swdx__tc_replayer_bin_flushes_reason_constant_table_full, "# of tiled-cache bin flushes due to constant table full") \
    f_(swdx__tc_replayer_bin_flushes_reason_drain_timeout, "# of tiled-cache bin flushes due to drain timeout") \
    f_(swdx__tc_replayer_bin_flushes_reason_explicit, "# of tiled-cache bin flushes due to explicit bin flush method") \
    f_(swdx__tc_replayer_bin_flushes_reason_level_0_threshold, "# of tiled-cache bin flushes due to level 0 buffer full (bundle memory)") \
    f_(swdx__tc_replayer_bin_flushes_reason_level_1_threshold, "# of tiled-cache bin flushes due to level 1 buffer full (bbox memory)") \
    f_(swdx__tc_replayer_bin_flushes_reason_non_binnable_line, "# of tiled-cache bin flushes due to non-binnable line primitive") \
    f_(swdx__tc_replayer_bin_flushes_reason_non_binnable_state, "# of tiled-cache bin flushes due to non-binnable state change") \
    f_(swdx__tc_replayer_bin_flushes_reason_pagepool_full, "# of tiled-cache bin flushes due to constant buffer pagepool threshold") \
    f_(swdx__tc_replayer_bin_flushes_reason_state_full, "# of tiled-cache bin flushes due to state buffer threshold") \
    f_(swdx__tc_replayer_bin_flushes_reason_timeout, "# of tiled-cache bin flushes due to timeout") \
    f_(swdx__tc_replayer_bin_flushes_reason_z_mode_transition, "# of tiled-cache bin flushes due to depth mode transition") \
    f_(swdx__tc_replayer_bin_flushes_replay_not_tiled, "# of tiled-cache bin flushes that were not replayed per tile") \
    f_(swdx__tc_replayer_bin_flushes_replay_tiled, "# of tiled-cache bin flushes that were replayed per tile") \
    f_(sys__cycles_elapsed, "# of cycles elapsed on SYS") \
    f_(sys__fb_read_dwords, "# of dwords read by HOST") \
    f_(sys__fb_read_requests, "# of read requests sent by HOST") \
    f_(sys__fb_write_dwords, "# of dwords written by HOST") \
    f_(sys__fb_write_requests, "# of write requests sent by HOST") \
    f_(sys__gpfifo_dwords, "# of GPFIFO dwords read by HOST, sent to PBDMA") \
    f_(sys__pushbuffer_dwords, "# of pushbuffer dwords read by HOST, sent to PBDMA") \
    f_(tga__batches_active, "sum of per-cycle # of batches in flight") \
    f_(tga__cycles_active, "# of cycles where TGA had at least one batch in flight") \
    f_(tga__input_batches, "# of batches sent to TGA") \
    f_(tga__input_isbes, "# of ISBEs sent to TGA") \
    f_(tga__input_patches, "# of patches sent to TGA") \
    f_(tga__isbes_active, "sum of per-cycle # of ISBEs in flight") \
    f_(tga__output_prims, "# of primitives output by TGA") \
    f_(tga__output_tasks, "# of beta tasks output by TGA") \
    f_(tga__output_tasks_complex_boundary, "# of beta tasks output by TGA: complex boundary task") \
    f_(tga__output_tasks_complex_interior, "# of beta tasks output by TGA: complex interior task") \
    f_(tga__output_tasks_primary, "# of beta tasks output by TGA where InstanceID starts at 0") \
    f_(tgb__cycles_active, "# of cycles where TGB had at least one beta task in flight") \
    f_(tgb__output_prims_line, "# of lines output by TGB") \
    f_(tgb__output_prims_point, "# of points output by TGB") \
    f_(tgb__output_prims_triangle, "# of triangles output by TGB") \
    f_(tgb__output_verts_complex_boundary, "# of vertices output by TGB: complex boundary task") \
    f_(tgb__output_verts_complex_interior, "# of vertices output by TGB: complex interior task") \
    f_(tgb__output_verts_simple, "# of vertices output by TGB: simple task") \
    f_(tgb__tasks_active, "sum of per-cycle # of tasks in flight") \
    f_(tpc__cycles_elapsed, "# of cycles elapsed on TPC") \
    f_(vaf__alpha_acache_requests, "# of A$ requests generated by attribute fetch in VAF alpha mode") \
    f_(vaf__alpha_acache_requests_acache_hit, "# of A$ requests that hit in A$ proper in VAF alpha mode") \
    f_(vaf__alpha_acache_requests_acache_miss, "# of A$ requests that missed in A$ and went to L2 in VAF alpha mode") \
    f_(vaf__alpha_acache_requests_local_hit, "# of A$ requests that hit in VAF-local-cache in VAF alpha mode") \
    f_(vaf__alpha_batches_active, "sum of per-cycle # of active batches") \
    f_(vaf__alpha_cycles_active, "# of cycles where VAF has at least one batch in flight") \
    f_(vaf__alpha_cycles_elapsed, "# of cycles where VAF was in alpha mode") \
    f_(vaf__alpha_cycles_stalled_on_mpc, "# of cycles where VAF was stalled in alpha mode, waiting for ISBE from MPC") \
    f_(vaf__alpha_fetched_attr_scalar_indexed, "# of scalar indexed attributes fetched by VAF") \
    f_(vaf__alpha_fetched_attr_scalar_indexed_constant, "# of scalar indexed constant attributes fetched by VAF") \
    f_(vaf__alpha_fetched_attr_scalar_post_cbf, "# of scalar attributes fetched by VAF post-VSa shader") \
    f_(vaf__alpha_fetched_attr_scalar_pre_cbf, "# of scalar attributes fetched by VAF pre-VSa shader") \
    f_(vaf__alpha_fetched_attr_vector_post_cbf, "# of attribute-vectors fetched by VAF for VSb shader") \
    f_(vaf__alpha_fetched_attr_vector_pre_cbf, "# of attribute-vectors fetched by VAF for VSa shader") \
    f_(vaf__alpha_input_batches_post_cbf, "# of batches sent to VSb shader") \
    f_(vaf__alpha_input_batches_pre_cbf, "# of batches sent to VSa shader") \
    f_(vaf__alpha_input_prims, "# of primitives sent to VAF") \
    f_(vaf__alpha_input_verts, "# of vertices sent to VAF") \
    f_(vaf__alpha_to_acache_stalled_on_read, "# of cycles where VAF-to-A$ interface was stalled in alpha mode, on read") \
    f_(vaf__alpha_to_acache_stalled_on_tag_allocation, "# of cycles where VAF-to-A$ interface was stalled in alpha mode, on tag allocation") \
    f_(vaf__beta_acache_requests_acache_hit, "# of A$ requests that hit in A$ proper in VAF beta mode") \
    f_(vaf__beta_acache_requests_acache_miss, "# of A$ requests that missed in A$ and went to L2 in VAF beta mode") \
    f_(vaf__beta_acache_requests_local_hit, "# of A$ requests that hit in VAF-local-cache in VAF beta mode") \
    f_(vaf__beta_cycles_active, "# of cycles where VAF has at least one task in flight") \
    f_(vaf__beta_cycles_elapsed, "# of cycles where VAF was in beta mode") \
    f_(vaf__beta_cycles_stalled_on_mpc, "# of cycles where VAF was stalled in beta mode, waiting for ISBE from MPC") \
    f_(vaf__beta_input_patches, "# of tess patches sent to VAF") \
    f_(vaf__beta_input_task_fifo_full, "# of cycles where VAF input task fifo is full") \
    f_(vaf__beta_input_tasks, "# of tess tasks sent to VAF") \
    f_(vaf__beta_tasks_active, "sum of per-cycle # of active tasks") \
    f_(vaf__beta_to_acache_stalled_on_read, "# of cycles where VAF-to-A$ interface was stalled in beta mode, on read") \
    f_(vaf__beta_to_acache_stalled_on_tag_allocation, "# of cycles where VAF-to-A$ interface was stalled in beta mode, on tag allocation") \
    f_(vaf__cycles_active, "# of cycles where VAF was active") \
    f_(vaf__cycles_stalled, "# of cycles where VAF was stalled") \
    f_(vaf__gpm_stalled_by_alpha_input_fifo, "# of cycles where GPM was stalled by VAF, due to VAF alpha input fifo full") \
    f_(vaf__gpm_stalled_by_beta_input_fifo, "# of cycles where GPM was stalled by VAF, due to VAF beta input fifo full") \
    f_(vaf__gpm_stalled_by_state_fifo, "# of cycles where GPM was stalled by VAF, due to VAF state fifo full") \
    f_(vaf__gpm_stalled_by_state_processing, "# of cycles where GPM was stalled by VAF, due to VAF state processing (Map/PrimID/VertexID)") \
    f_(vpc__alpha_batches_active, "sum of per-cycle # of active batches") \
    f_(vpc__alpha_cycles_active, "# of cycles where VPC had at least one batch in flight") \
    f_(vpc__beta_cycles_active, "# of cycles where VPC had at least one beta task in flight") \
    f_(vpc__beta_tasks_active, "sum of per-cycle # of active beta tasks") \
    f_(vpc__clip_cycles_active, "# of cycles where VPC Clip was active") \
    f_(vpc__clip_cycles_stalled, "# of cycles where VPC Clip was stalled") \
    f_(vpc__clip_input_prims_clipped, "# of primitives sent to VPC Clip, clipped by one or more planes") \
    f_(vpc__clip_input_prims_clipped_multi_plane, "# of primitives sent to VPC Clip, clipped by multiple planes") \
    f_(vpc__clip_input_prims_clipped_single_plane, "# of primitives sent to VPC Clip, clipped by a single plane") \
    f_(vpc__clip_output_attrs, "# of attributes output by VPC Clip for clipped primitives") \
    f_(vpc__clip_output_prims, "# of primitives output by VPC Clip") \
    f_(vpc__clip_output_verts, "# of vertices output by VPC Clip") \
    f_(vpc__lwll_lwlled_prims, "# of lwlled primitives of any type") \
    f_(vpc__lwll_lwlled_prims_line, "# of lwlled lines") \
    f_(vpc__lwll_lwlled_prims_point, "# of lwlled points") \
    f_(vpc__lwll_lwlled_prims_reason_backfacing, "# of primitives lwlled for reason: back-facing") \
    f_(vpc__lwll_lwlled_prims_reason_bounding_box, "# of primitives lwlled for reason: bounding box") \
    f_(vpc__lwll_lwlled_prims_reason_diamond_exit_rule, "# of lines lwlled for reason: diamond exit rule") \
    f_(vpc__lwll_lwlled_prims_reason_frustum, "# of primitives lwlled for reason: frustum lwlling and user-clip plane") \
    f_(vpc__lwll_lwlled_prims_reason_rotated_grid, "# of primitives lwlled for reason: rotated grid") \
    f_(vpc__lwll_lwlled_prims_reason_scissor, "# of primitives lwlled for reason: scissor") \
    f_(vpc__lwll_lwlled_prims_reason_zero_area, "# of primitives lwlled for reason: zero area") \
    f_(vpc__lwll_lwlled_prims_reason_zero_length, "# of primitives lwlled for reason: zero length") \
    f_(vpc__lwll_lwlled_prims_triangle, "# of lwlled triangles") \
    f_(vpc__lwll_cycles_active, "# of cycles where VPC Lwll was active") \
    f_(vpc__lwll_cycles_stalled, "# of cycles where VPC Lwll was stalled") \
    f_(vpc__cycles_active, "# of cycles where VPC had at least one batch or beta task in flight") \
    f_(vpc__input_isbes, "# of ISBEs sent to VPC") \
    f_(vpc__input_isbes_prim, "# of ISBEs sent to VPC at a primitive boundary") \
    f_(vpc__input_prims, "# of primitives of any type, before clipping and lwlling") \
    f_(vpc__input_prims_line, "# of lines before clipping and lwlling") \
    f_(vpc__input_prims_patch, "# of patches before clipping and lwlling") \
    f_(vpc__input_prims_point, "# of points before clipping and lwlling") \
    f_(vpc__input_prims_triangle, "# of triangles before clipping and lwlling") \
    f_(vpc__output_attrs_scalar, "# of scalar attributes output by VPC, excluding POSITION") \
    f_(vpc__output_cbes, "# of CBEs output by VPC") \
    f_(vpc__output_prims, "# of primitives of any type, after clipping and lwlling") \
    f_(vpc__output_prims_line, "# of lines after clipping and lwlling") \
    f_(vpc__output_prims_point, "# of points after clipping and lwlling") \
    f_(vpc__output_prims_stippled_line, "# of stippled lines after clipping and lwlling") \
    f_(vpc__output_prims_triangle, "# of triangles after clipping and lwlling") \
    f_(vpc__output_verts, "# of vertices after clipping and lwlling") \
    f_(vpc__read_isbes, "# of ISBEs read by VPC from any connected SM in the GPC") \
    f_(vpc__write_sectors, "# of sectors written by VPC into CBEs") \
    f_(wwdx__cycles_active, "# of cycles where WWDX was active") \
    f_(wwdx__input_cbes, "# of CBEs sent to WWDX from PE") \
    f_(wwdx__input_prims, "# of primitives sent to WWDX") \
    f_(wwdx__input_tasks, "# of beta tasks sent to WWDX") \
    f_(wwdx__output_stalled, "# of cycles where WWDX output was stalled") \
    f_(zrop__cycles_active, "# of cycles where ZROP was active") \
    f_(zrop__input_requests, "# of requests of any kind sent to ZROP") \
    f_(zrop__input_requests_containing_stencil, "# of requests containing stencil") \
    f_(zrop__input_requests_expanded_to_samples, "# of requests requiring full per-sample expansion") \
    f_(zrop__input_requests_type_bundles, "# of bundle biquad requests sent to ZROP") \
    f_(zrop__input_requests_type_plane_eq, "# of plane-equation requests sent to ZROP") \
    f_(zrop__input_requests_type_shdz_biquad, "# of shader-z pixel biquad requests sent to ZROP") \
    f_(zrop__input_requests_write_compression_ok, "# of requests that were allowed to write compressed") \
    f_(zrop__input_samples_part1, "# of samples sent to ZROP, part 1") \
    f_(zrop__input_samples_part2, "# of samples sent to ZROP, part 2") \
    f_(zrop__input_stalled, "# of cycles where input to ZROP was stalled") \
    f_(zrop__processed_requests, "# of requests of any kind processed by ZROP") \
    f_(zrop__processed_requests_type_bundle, "# of bundle biquad requests processed by ZROP") \
    f_(zrop__processed_requests_type_plane_eq, "# of plane-equation requests that reached the processing stage in ZROP") \
    f_(zrop__processed_requests_type_plane_eq_fast, "# of plane-equation requests that were processed as planes in ZROP") \
    f_(zrop__processed_requests_type_shdz_biquad, "# of shader-z pixel biquad requests processed by ZROP") \
    f_(zrop__processed_subpackets, "# of subpackets processed by ZROP") \
    f_(zrop__processed_subpackets_stencil, "# of depth subpackets processed by ZROP") \
    f_(zrop__read_requests, "# of requests of any kind that generated 1 or more subpackets") \
    f_(zrop__read_returns, "# of requests of any kind that generated 1 or more read subpackets") \
    f_(zrop__read_subpackets, "# of subpacket reads of any kind") \
    f_(zrop__read_subpackets_stencil, "# of subpacket reads for stencil") \
    f_(zrop__write_requests, "# of ZROP requests of any kind that wrote 1 or more subpackets") \
    f_(zrop__write_subpackets, "# of subpacket writes of any kind") \
    f_(zrop__write_subpackets_coalesced, "# of subpacket writes coalesced across requests") \
    f_(zrop__write_subpackets_depth, "# of subpacket writes for depth") \
    f_(zrop__write_subpackets_stencil, "# of subpacket writes for stencil") \
    f_(zrop__zlwll_cycles_active, "# of cycles where ZLWLL was active") \

#define LW_GM20B_ALL_METRIC_DESCS(f_) \
    LW_GM20B_RAW_METRIC_DESCS(f_) \
    f_(cbmgr__throughput, "CBMGR throughput") \
    f_(crop__throughput, "CROP throughput") \
    f_(cwd__throughput, "CWD throughput") \
    f_(fe__throughput, "FE throughput") \
    f_(gcc__l15_requests, "# of GCC requests to L1.5") \
    f_(gcc__l15_requests_hit_rate_pct, "% of GCC requests that hit in L1.5") \
    f_(gcc__read_sectors, "# of L2 sectors read by GCC from L2") \
    f_(gcc__throughput, "GCC throughput") \
    f_(gcc__tsl2_requests_hit_rate_pct, "% of texture/sampler header requests that hit in TSL2") \
    f_(gpc__gpcl1tlb_requests, "# of PTE requests sent to GPC L1 TLB") \
    f_(gpc__prop_utlb_requests, "# of PTE requests sent to PROP L0 TLB") \
    f_(gpc__rg_utlb_requests, "# of PTE requests sent to RASTER+GCC L0 TLB") \
    f_(gpc__tlb_throughput, "GPC TLB throughput") \
    f_(gpc__tpc0_utlb_requests, "# of PTE requests sent to TPC0 L0 TLB") \
    f_(gpc__tpc1_utlb_requests, "# of PTE requests sent to TPC1 L0 TLB") \
    f_(gpmpd__throughput, "GPMPD throughput") \
    f_(gpmsd__throughput, "GPMSD throughput") \
    f_(host__throughput, "HOST throughput") \
    f_(l1tex__t_sectors_hit, "# of sector requests to L1TEX T-Stage that hit") \
    f_(l1tex__t_sectors_hit_rate_pct, "% of sector requests that hit in L1TEX") \
    f_(l1tex__throughput, "L1TEX throughput") \
    f_(lts__cbc_requests_comptag, "# of compressible requests handled by CBC") \
    f_(lts__d_sectors, "# of sectors accessed in data banks") \
    f_(lts__ltcx_throughput, "LTCX throughput") \
    f_(lts__mccif_read_bytes, "# of bytes read from MC into MCCIF") \
    f_(lts__mccif_read_sectors, "# of sectors read from MC into MCCIF") \
    f_(lts__mccif_throughput, "MCCIF throughput") \
    f_(lts__mccif_write_bytes, "# of bytes written from LTCX to MCCIF") \
    f_(lts__mccif_write_bytes_appx, "# of bytes written from LTCX to MCCIF") \
    f_(lts__rop_input_active, "# of cycles where CROP or ZROP sent requests to LTS") \
    f_(lts__t_sectors_hit_rate_pct, "% of sector requests that hit in LTS") \
    f_(lts__t_tag_requests, "# of tag requests sent to LTS") \
    f_(lts__throughput, "LTS throughput") \
    f_(lts__xbar_input_active, "# of cycles where interface from XBAR to LTS was active") \
    f_(mme__throughput, "MME throughput") \
    f_(mmu__hubtlb_requests, "# of PTE requests sent to HUB TLB") \
    f_(mmu__hubtlb_requests_hit_rate_pct, "% of PTE requests that hit in HUB TLB") \
    f_(mmu__pde_requests, "# of PDE requests sent to MMU") \
    f_(mmu__pte_requests, "# of PTE requests sent to MMU") \
    f_(mmu__throughput, "MMU throughput") \
    f_(mpc__throughput, "MPC throughput") \
    f_(pda__throughput, "PDA throughput") \
    f_(pdb__throughput, "PDB throughput") \
    f_(pel__throughput, "PEL throughput") \
    f_(pes__throughput, "PES throughput") \
    f_(prop__throughput, "PROP throughput") \
    f_(raster__throughput, "RASTER throughput") \
    f_(rdm__throughput, "RDM throughput") \
    f_(rop__read_sectors, "# of cycles where RDM output to ZROP was stalled") \
    f_(rop__throughput, "RDM throughput") \
    f_(scc__throughput, "SCC throughput") \
    f_(sked__throughput, "SKED throughput") \
    f_(sm__cycles_active_3d_vtg_only, "# of cycles with VTG warps in flight, and zero PS warps in flight") \
    f_(sm__cycles_active_compute, "# of cycles with CS warps in flight") \
    f_(sm__inst_exelwted_pipe_fma64plus_pred_on, "# of instructions exelwted by the fma64plus pipe where all active threads' guard-predicated was false") \
    f_(sm__inst_exelwted_pipe_fma64plusplus_pred_on, "# of instructions exelwted by the fma64plusplus pipe where all active threads' guard-predicated was false") \
    f_(sm__inst_exelwted_pipe_lsu_pred_on, "# of instructions exelwted by the lsu pipe where all active threads' guard-predicated was false") \
    f_(sm__inst_exelwted_pipe_su_pred_on, "# of instructions exelwted by the su pipe where all active threads' guard-predicated was false") \
    f_(sm__mios_active, "# of cycles MIOS was active") \
    f_(sm__mios_inst_exelwted, "# of instructions exelwted by MIOS") \
    f_(sm__mios_shmem_accesses_lsu, "# of memory accesses to LSU accessed MIOS shared-memory data RAM") \
    f_(sm__mios_shmem_active_tram, "# of cycles PE and LSU accessed MIOS shared-memory data RAM for TRAM") \
    f_(sm__ps_quads_sent_to_pixout_pct, "% of PS quads launched with at least one surviving pixel") \
    f_(sm__threads_launched_ps, "# of PS threads launched (killed and not-killed") \
    f_(sm__throughput, "SM throughput") \
    f_(smp__inst_exelwted_pipe_adu_pred_on, "# of instructions exelwted by the adu pipe where all active threads' guard-predicated was false") \
    f_(smp__inst_exelwted_pipe_bru, "# of instructions exelwted by the bru pipe") \
    f_(smp__inst_exelwted_pipe_tex, "# of instructions exelwted by the tex pipe") \
    f_(smp__lsu_write_back_active, "# of cycles where the interface from LSU to register file was active") \
    f_(smp__miop_pq_read_active, "# of cycles where MIOP PQ sent register operands to a pipeline") \
    f_(smp__tex_requests, "# of requests sent from SM to TEX") \
    f_(smp__tex_write_back_active, "# of cycles where the interface from TEX to register file was active") \
    f_(smsp__inst_issued_coupled, "# of coupled instructions issued") \
    f_(smsp__inst_issued_decoupled, "# of decoupled instructions issued") \
    f_(stri__throughput, "STRI throughput") \
    f_(swdx__throughput, "SWDX throughput") \
    f_(tga__throughput, "TGA throughput") \
    f_(tgb__output_verts, "# of vertices output by TGB") \
    f_(tgb__throughput, "TGB throughput") \
    f_(vaf__alpha_fetched_attr_scalar, "# of scalar attributes fetched by VAF") \
    f_(vaf__throughput, "VAF throughput") \
    f_(vpc__output_attrs, "# of attributes output by VPC") \
    f_(vpc__throughput, "VPC throughput") \
    f_(wwdx__throughput, "WWDX throughput") \
    f_(zrop__input_samples, "# of samples sent to ZROP") \
    f_(zrop__throughput, "ZROP Throughput") \


}}} // namespace
