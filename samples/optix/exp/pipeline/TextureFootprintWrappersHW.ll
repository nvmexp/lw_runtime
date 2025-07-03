;
; Copyright (c) 2020 LWPU Corporation.  All rights reserved.
;
; LWPU Corporation and its licensors retain all intellectual property and proprietary
; rights in and to this software, related documentation and any modifications thereto.
; Any use, reproduction, disclosure or distribution of this software and related
; documentation without an express license agreement from LWPU Corporation is strictly
; prohibited.
;
; TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
; AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
; INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
; PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
; SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
; LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
; BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
; INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
; SUCH DAMAGES
;

target datalayout = "e-p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
target triple = "lwptx64-lwpu-lwca"

; "Dummy" direct callable, so rtcore won't complain when compiling
define i32 @__direct_callable__optix_sparse_textures( i32 ) {
  ret i32 0
}

declare { <4 x i32>, i1 }
@llvm.lwvm.tex.footprint.v4i32(i32 %mode, i64 %tex, i64 %samp,
                               float %coord_x, float %coord_y,
                               float %coord_z, i32 %array_idx,
                               float %bias_or_lod, i32 %granularity)

define { i32, i32, i32, i32 } @_lw_optix_lwstom_abi_optix_internal_tex_footprint_2d(  i64 %tex, float %xCoord, float %yCoord, i32 %granularity, i32* %singleMipLevel ) {
  ; WAR 3007016: D2IR backend outputs LOD selection flag when no LOD is used
  ; Mode: 0x3=3 [2D texture]
  ; %res = call { <4 x i32>, i1 } @llvm.lwvm.tex.footprint.v4i32( i32 3, 
  ;                                                               i64 %tex, 
  ;                                                               i64 undef, 
  ;                                                               float %xCoord, 
  ;                                                               float %yCoord, 
  ;                                                               float undef, 
  ;                                                               i32 undef, 
  ;                                                               float undef, 
  ;                                                               i32 %granularity )

  ; Mode: 0x23=35 [2D texture, absolute LOD adjust]
  %res = call { <4 x i32>, i1 } @llvm.lwvm.tex.footprint.v4i32( i32 35, 
                                                                i64 %tex, 
                                                                i64 undef, 
                                                                float %xCoord, 
                                                                float %yCoord, 
                                                                float undef, 
                                                                i32 undef, 
                                                                float 0.0, 
                                                                i32 %granularity )

  %resSingleMipLevel = extractvalue { <4 x i32>, i1 } %res, 1
  %resSingleMipLevelI32 = zext i1 %resSingleMipLevel to i32
  store i32 %resSingleMipLevelI32, i32* %singleMipLevel

  %resVec = extractvalue { <4 x i32>, i1 } %res, 0
  %res0 = extractelement <4 x i32> %resVec, i32 0
  %res1 = extractelement <4 x i32> %resVec, i32 1
  %res2 = extractelement <4 x i32> %resVec, i32 2
  %res3 = extractelement <4 x i32> %resVec, i32 3

  %r0 = insertvalue { i32, i32, i32, i32 } undef, i32 %res0, 0
  %r1 = insertvalue { i32, i32, i32, i32 } %r0, i32 %res1, 1
  %r2 = insertvalue { i32, i32, i32, i32 } %r1, i32 %res2, 2
  %r3 = insertvalue { i32, i32, i32, i32 } %r2, i32 %res3, 3
  ret { i32, i32, i32, i32 } %r3
}



declare { <4 x i32>, i1 }
@llvm.lwvm.tex.footprint.grad.v4i32(i32 %mode, i64 %tex, i64 %samp,
                                    float %coord_x, float %coord_y,
                                    float %coord_z, i32 %array_idx,
                                    float %dsdx, float %dsdy, float %dsdz,
                                    float %dtdx, float %dtdy, float %dtdz,
                                    i32 %granularity)

define { i32, i32, i32, i32 } @_lw_optix_lwstom_abi_optix_internal_tex_footprint_2d_grad_coarse(  i64 %tex, 
                                                                                                  float %xCoord, 
                                                                                                  float %yCoord, 
                                                                                                  float %dPdx_x, 
                                                                                                  float %dPdx_y, 
                                                                                                  float %dPdy_x,
                                                                                                  float %dPdy_y,
                                                                                                  i32 %granularity,
                                                                                                  i32* %singleMipLevel ) {
  ; Mode: 0x4003=16387 [2D texture, coarse LOD]
  %res = call { <4 x i32>, i1 } @llvm.lwvm.tex.footprint.grad.v4i32( i32 16387, 
                                                                     i64 %tex, 
                                                                     i64 undef, 
                                                                     float %xCoord, 
                                                                     float %yCoord, 
                                                                     float undef, 
                                                                     i32 undef, 
                                                                     float %dPdx_x,
                                                                     float %dPdx_y,
                                                                     float undef,
                                                                     float %dPdy_x,
                                                                     float %dPdy_y,
                                                                     float undef,
                                                                     i32 %granularity )
                                                                              
  %resSingleMipLevel = extractvalue { <4 x i32>, i1 } %res, 1
  %resSingleMipLevelI32 = zext i1 %resSingleMipLevel to i32
  store i32 %resSingleMipLevelI32, i32* %singleMipLevel

  %resVec = extractvalue { <4 x i32>, i1 } %res, 0
  %res0 = extractelement <4 x i32> %resVec, i32 0
  %res1 = extractelement <4 x i32> %resVec, i32 1
  %res2 = extractelement <4 x i32> %resVec, i32 2
  %res3 = extractelement <4 x i32> %resVec, i32 3

  %r0 = insertvalue { i32, i32, i32, i32 } undef, i32 %res0, 0
  %r1 = insertvalue { i32, i32, i32, i32 } %r0, i32 %res1, 1
  %r2 = insertvalue { i32, i32, i32, i32 } %r1, i32 %res2, 2
  %r3 = insertvalue { i32, i32, i32, i32 } %r2, i32 %res3, 3
  ret { i32, i32, i32, i32 } %r3
}

define { i32, i32, i32, i32 } @_lw_optix_lwstom_abi_optix_internal_tex_footprint_2d_grad_fine(  i64 %tex, 
                                                                                                float %xCoord, 
                                                                                                float %yCoord, 
                                                                                                float %dPdx_x, 
                                                                                                float %dPdx_y, 
                                                                                                float %dPdy_x,
                                                                                                float %dPdy_y,
                                                                                                i32 %granularity,
                                                                                                i32* %singleMipLevel ) {
  ; Mode: 0x3=3 [2D texture]
  %res = call { <4 x i32>, i1 } @llvm.lwvm.tex.footprint.grad.v4i32( i32 3, 
                                                                     i64 %tex, 
                                                                     i64 undef, 
                                                                     float %xCoord, 
                                                                     float %yCoord, 
                                                                     float undef, 
                                                                     i32 undef, 
                                                                     float %dPdx_x,
                                                                     float %dPdx_y,
                                                                     float undef,
                                                                     float %dPdy_x,
                                                                     float %dPdy_y,
                                                                     float undef,
                                                                     i32 %granularity )

  %resSingleMipLevel = extractvalue { <4 x i32>, i1 } %res, 1
  %resSingleMipLevelI32 = zext i1 %resSingleMipLevel to i32
  store i32 %resSingleMipLevelI32, i32* %singleMipLevel

  %resVec = extractvalue { <4 x i32>, i1 } %res, 0
  %res0 = extractelement <4 x i32> %resVec, i32 0
  %res1 = extractelement <4 x i32> %resVec, i32 1
  %res2 = extractelement <4 x i32> %resVec, i32 2
  %res3 = extractelement <4 x i32> %resVec, i32 3

  %r0 = insertvalue { i32, i32, i32, i32 } undef, i32 %res0, 0
  %r1 = insertvalue { i32, i32, i32, i32 } %r0, i32 %res1, 1
  %r2 = insertvalue { i32, i32, i32, i32 } %r1, i32 %res2, 2
  %r3 = insertvalue { i32, i32, i32, i32 } %r2, i32 %res3, 3
  ret { i32, i32, i32, i32 } %r3
}


define { i32, i32, i32, i32 } @_lw_optix_lwstom_abi_optix_internal_tex_footprint_2d_lod_coarse( i64 %tex, 
                                                                                                float %xCoord, 
                                                                                                float %yCoord, 
                                                                                                float %level, 
                                                                                                i32 %granularity,
                                                                                                i32* %singleMipLevel ) {
  ; Mode: 0x4023=16419 [2D texture, absolute LOD adjust, coarse LOD]
  %res = call { <4 x i32>, i1 } @llvm.lwvm.tex.footprint.v4i32( i32 16419, 
                                                                i64 %tex, 
                                                                i64 undef, 
                                                                float %xCoord, 
                                                                float %yCoord, 
                                                                float undef, 
                                                                i32 undef, 
                                                                float %level, 
                                                                i32 %granularity )

  %resSingleMipLevel = extractvalue { <4 x i32>, i1 } %res, 1
  %resSingleMipLevelI32 = zext i1 %resSingleMipLevel to i32
  store i32 %resSingleMipLevelI32, i32* %singleMipLevel

  %resVec = extractvalue { <4 x i32>, i1 } %res, 0
  %res0 = extractelement <4 x i32> %resVec, i32 0
  %res1 = extractelement <4 x i32> %resVec, i32 1
  %res2 = extractelement <4 x i32> %resVec, i32 2
  %res3 = extractelement <4 x i32> %resVec, i32 3

  %r0 = insertvalue { i32, i32, i32, i32 } undef, i32 %res0, 0
  %r1 = insertvalue { i32, i32, i32, i32 } %r0, i32 %res1, 1
  %r2 = insertvalue { i32, i32, i32, i32 } %r1, i32 %res2, 2
  %r3 = insertvalue { i32, i32, i32, i32 } %r2, i32 %res3, 3
  ret { i32, i32, i32, i32 } %r3
}

define { i32, i32, i32, i32 } @_lw_optix_lwstom_abi_optix_internal_tex_footprint_2d_lod_fine( i64 %tex, 
                                                                                              float %xCoord, 
                                                                                              float %yCoord, 
                                                                                              float %level, 
                                                                                              i32 %granularity,
                                                                                              i32* %singleMipLevel ) {
  ; Mode: 0x23=35 [2D texture, absolute LOD adjust]
  %res = call { <4 x i32>, i1 } @llvm.lwvm.tex.footprint.v4i32( i32 35, 
                                                                i64 %tex, 
                                                                i64 undef, 
                                                                float %xCoord, 
                                                                float %yCoord, 
                                                                float undef, 
                                                                i32 undef, 
                                                                float %level, 
                                                                i32 %granularity )

  %resSingleMipLevel = extractvalue { <4 x i32>, i1 } %res, 1
  %resSingleMipLevelI32 = zext i1 %resSingleMipLevel to i32
  store i32 %resSingleMipLevelI32, i32* %singleMipLevel

  %resVec = extractvalue { <4 x i32>, i1 } %res, 0
  %res0 = extractelement <4 x i32> %resVec, i32 0
  %res1 = extractelement <4 x i32> %resVec, i32 1
  %res2 = extractelement <4 x i32> %resVec, i32 2
  %res3 = extractelement <4 x i32> %resVec, i32 3

  %r0 = insertvalue { i32, i32, i32, i32 } undef, i32 %res0, 0
  %r1 = insertvalue { i32, i32, i32, i32 } %r0, i32 %res1, 1
  %r2 = insertvalue { i32, i32, i32, i32 } %r1, i32 %res2, 2
  %r3 = insertvalue { i32, i32, i32, i32 } %r2, i32 %res3, 3
  ret { i32, i32, i32, i32 } %r3
}
