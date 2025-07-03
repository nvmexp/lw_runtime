;
; Copyright (c) 2021 LWPU Corporation.  All rights reserved.
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
define i32 @__direct_callable__optix_texture_footprint_wrappers_sw( i32 ) {
  ret i32 0
}

; Returns {width, height, undef, numMipLevels}.  However, the number of miplevels is incorrect for for non-mipmapped textures.
; See https://p4viewer.lwpu.com/get/sw/compiler/docs/UnifiedLWVMIR/asciidoc/html/lwvmIR-1x.html#_texture_surface_query_intrinsics
declare <4 x i32> @llvm.lwvm.tex.query.composite(i32 %mode, i64 %tex, i32 %idx) readnone

define <4 x i32> @optix_internal_tex_query_composite( i64 %tex )
{
  ; Mode: 0 = bindless
  %res = call <4 x i32> @llvm.lwvm.tex.query.composite(i32 0, i64 %tex, i32 undef)
  ret <4 x i32> %res
}
